"""Prove the live/incremental driver matches the offline batch path.

Runs the same frames two ways on one model instance:
  1. model.inference_streaming(all_frames)      <- what demo.py does
  2. LiveReconstructor.push(frame) per frame    <- what live_map.py does

If the incremental driver is a faithful use of the causal API, the recovered
camera poses and depth maps should agree to numerical noise.
"""

import argparse
import os

import numpy as np
import torch

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

from live_map import LiveReconstructor, preprocess_frame
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cpu")
    paths = sorted(
        os.path.join(args.image_folder, p)
        for p in os.listdir(args.image_folder)
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )[: args.n]

    print("Building model...")
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
        kv_cache_sliding_window=64, kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        use_sdpa=True, camera_num_iterations=4,
    )
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = model.to(device).eval()

    # ── Path 1: offline batch ────────────────────────────────────────────────
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    print(f"\n[batch] inference_streaming on {tuple(images.shape)}")
    with torch.no_grad():
        batch = model.inference_streaming(images, num_scale_frames=args.num_scale_frames)
    b_extri, _ = pose_encoding_to_extri_intri(batch["pose_enc"], images.shape[-2:])
    b_extri = b_extri[0].cpu().numpy()
    b_depth = batch["depth"][0].cpu().numpy()

    # ── Path 2: incremental / live ───────────────────────────────────────────
    print(f"[live ] pushing {len(paths)} frames one at a time")
    rec = LiveReconstructor(model, device, num_scale_frames=args.num_scale_frames)
    l_extri, l_depth = [], []
    for p in paths:
        res = rec.push(preprocess_frame(Image.open(p)))
        if res is None:
            continue
        for j in range(res["extrinsic"].shape[0]):
            l_extri.append(res["extrinsic"][j])
            l_depth.append(res["depth"][j])
    l_extri = np.stack(l_extri)
    l_depth = np.stack(l_depth)

    # ── Compare ──────────────────────────────────────────────────────────────
    n = min(len(b_extri), len(l_extri))
    print(f"\nComparing {n} frames")
    pose_err = np.abs(b_extri[:n] - l_extri[:n])
    depth_err = np.abs(b_depth[:n] - l_depth[:n])
    rel = depth_err / np.maximum(np.abs(b_depth[:n]), 1e-6)

    print(f"  extrinsics : max abs diff = {pose_err.max():.3e}  mean = {pose_err.mean():.3e}")
    print(f"  depth      : max abs diff = {depth_err.max():.3e}  mean = {depth_err.mean():.3e}")
    print(f"  depth      : max rel diff = {rel.max():.3e}")

    ok = pose_err.max() < 1e-3 and rel.max() < 1e-2
    print(f"\n{'MATCH — incremental path is equivalent to batch' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
