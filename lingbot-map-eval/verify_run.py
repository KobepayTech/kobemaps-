"""Headless verification run: infer, then check the outputs are geometrically sane.

Runs streaming inference on a folder of frames, then reports depth/confidence
statistics and the recovered camera trajectory, and writes a PLY point cloud
plus depth previews so the reconstruction can be inspected without viser.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.geometry import (
    unproject_depth_map_to_point_map,
    closed_form_inverse_se3_general,
)
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.models.gct_stream import GCTStream


def write_ply(path, xyz, rgb):
    """Minimal binary-free PLY writer (ascii, fine at these point counts)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--first_k", type=int, default=24)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--keyframe_interval", type=int, default=1)
    ap.add_argument("--out_dir", default="verify_out")
    ap.add_argument("--conf_threshold", type=float, default=1.5)
    ap.add_argument("--downsample", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    paths = sorted(
        os.path.join(args.image_folder, p)
        for p in os.listdir(args.image_folder)
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )[: args.first_k]
    print(f"Loading {len(paths)} frames from {args.image_folder}")
    # Must match the model's img_size/patch_size — the helper defaults to 512/16.
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=518, patch_size=14
    ).to(device)
    S, _, H, W = images.shape
    print(f"Input tensor: {tuple(images.shape)}")

    print("Building model (SDPA backend, fp32 CPU)...")
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    )
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e9:.3f} B")
    model = model.to(device).eval()

    t0 = time.time()
    with torch.no_grad():
        pred = model.inference_streaming(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
        )
    dt = time.time() - t0
    print(f"\nInference: {dt:.1f}s total, {dt/S:.2f}s/frame")

    # ── Unpack ───────────────────────────────────────────────────────────────
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} {tuple(v.shape)} {v.dtype}")

    # The model emits a pose encoding; decode it to OpenCV w2c extrinsics + intrinsics.
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])

    depth = pred["depth"].squeeze(0).cpu().numpy()          # [S,H,W,1]
    depth_conf = pred["depth_conf"].squeeze(0).cpu().numpy()  # [S,H,W]
    extrinsic = extri.squeeze(0).cpu().numpy()                # [S,3,4] w2c
    intrinsic = intri.squeeze(0).cpu().numpy()                # [S,3,3]

    d = depth[..., 0]
    print("\n── Depth ─────────────────────────────────────────")
    print(f"  finite: {np.isfinite(d).all()}  min={d.min():.4f}  "
          f"max={d.max():.4f}  median={np.median(d):.4f}")
    print(f"  confidence: min={depth_conf.min():.3f} max={depth_conf.max():.3f} "
          f"mean={depth_conf.mean():.3f}")
    frac = (depth_conf > args.conf_threshold).mean()
    print(f"  fraction above conf threshold {args.conf_threshold}: {frac*100:.1f}%")

    print("\n── Intrinsics (frame 0) ──────────────────────────")
    print(f"  fx={intrinsic[0,0,0]:.1f}  fy={intrinsic[0,1,1]:.1f}  "
          f"cx={intrinsic[0,0,2]:.1f}  cy={intrinsic[0,1,2]:.1f}  (image {W}x{H})")

    # ── Trajectory ───────────────────────────────────────────────────────────
    ext44 = np.zeros((S, 4, 4), dtype=np.float64)
    ext44[:, :3, :] = extrinsic
    ext44[:, 3, 3] = 1.0
    c2w = closed_form_inverse_se3_general(torch.from_numpy(ext44)).numpy()
    centers = c2w[:, :3, 3]

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    total = steps.sum()
    span = np.linalg.norm(centers.max(0) - centers.min(0))
    print("\n── Camera trajectory ─────────────────────────────")
    print(f"  frame 0 center: {np.round(centers[0], 4)}")
    print(f"  frame {S-1} center: {np.round(centers[-1], 4)}")
    print(f"  path length: {total:.4f}   bbox diagonal: {span:.4f}")
    print(f"  per-frame step: min={steps.min():.4f} max={steps.max():.4f} "
          f"mean={steps.mean():.4f} std={steps.std():.4f}")

    # Rotation matrices must stay orthonormal — a good check that poses are valid.
    R = c2w[:, :3, :3]
    orth_err = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    dets = np.linalg.det(R)
    print(f"  max |R Rᵀ - I|: {orth_err:.2e}   det(R) range: "
          f"[{dets.min():.6f}, {dets.max():.6f}]")

    # ── Point cloud ──────────────────────────────────────────────────────────
    world_pts = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # [S,H,W,3]

    mask = depth_conf > args.conf_threshold
    xyz = world_pts[mask]
    rgb = (imgs_np[mask] * 255).clip(0, 255).astype(np.uint8)
    print("\n── Point cloud ───────────────────────────────────")
    print(f"  points above threshold: {len(xyz):,} of {mask.size:,}")
    if len(xyz):
        print(f"  bbox min: {np.round(xyz.min(0), 3)}")
        print(f"  bbox max: {np.round(xyz.max(0), 3)}")
        sel = np.arange(0, len(xyz), max(1, args.downsample))
        ply = os.path.join(args.out_dir, "cloud.ply")
        write_ply(ply, xyz[sel], rgb[sel])
        print(f"  wrote {len(sel):,} points -> {ply}")

    # ── Depth previews ───────────────────────────────────────────────────────
    for i in (0, S // 2, S - 1):
        di = d[i]
        lo, hi = np.percentile(di, 2), np.percentile(di, 98)
        norm = np.clip((di - lo) / max(hi - lo, 1e-8), 0, 1)
        # simple turbo-ish ramp: near = warm, far = cool
        rgbmap = np.stack([norm, 1 - np.abs(norm - 0.5) * 2, 1 - norm], -1)
        strip = np.concatenate(
            [(imgs_np[i] * 255).astype(np.uint8),
             (rgbmap * 255).astype(np.uint8)], axis=0
        )
        p = os.path.join(args.out_dir, f"frame_{i:04d}_rgb_depth.png")
        Image.fromarray(strip).save(p)
        print(f"  wrote {p}")

    np.savez_compressed(
        os.path.join(args.out_dir, "predictions.npz"),
        depth=depth, depth_conf=depth_conf,
        extrinsic=extrinsic, intrinsic=intrinsic, centers=centers,
    )
    print(f"\nSaved raw predictions -> {os.path.join(args.out_dir, 'predictions.npz')}")


if __name__ == "__main__":
    main()
