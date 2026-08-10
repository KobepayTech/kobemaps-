"""Live streaming reconstruction from a camera (or any incremental frame source).

`demo.py` is offline: it loads every frame into one tensor before inference.
The model itself is causal, though — `inference_streaming` just runs a scale
phase over the first N frames and then calls `forward(..., num_frame_per_block=1,
causal_inference=True)` per frame, with the KV cache persisting on the module
between calls. This drives that same step function from a live source, so poses
and depth come out frame-by-frame as you record.

Sources:
    --source webcam --camera 0     capture from an attached camera
    --source replay --replay_dir DIR   replay frames off disk one at a time
                                       (same code path, for testing without a camera)

Ctrl-C stops capture and writes the trajectory + point cloud.
"""

import argparse
import os
import signal
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)

IMAGE_SIZE = 518
PATCH_SIZE = 14


# =============================================================================
# Preprocessing (mirrors load_fn.load_and_preprocess_images "crop" mode,
# but for an in-memory frame instead of a file path)
# =============================================================================

def preprocess_frame(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> [1,3,H,W] float tensor in [0,1], sized for the model."""
    img = img.convert("RGB")
    w, h = img.size
    new_w = IMAGE_SIZE
    new_h = round(h * (new_w / w) / PATCH_SIZE) * PATCH_SIZE
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)

    if new_h > IMAGE_SIZE:  # center-crop tall frames, as the loader does
        y0 = (new_h - IMAGE_SIZE) // 2
        t = t[:, y0:y0 + IMAGE_SIZE, :]
    return t.unsqueeze(0)


# =============================================================================
# Frame sources
# =============================================================================

class WebcamSource:
    def __init__(self, index=0, width=None, height=None):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None
        rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self):
        self.cap.release()


class ReplaySource:
    """Feeds frames off disk one at a time — identical code path, no camera needed."""

    def __init__(self, directory, limit=None, delay=0.0):
        self.paths = sorted(
            os.path.join(directory, p)
            for p in os.listdir(directory)
            if p.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if limit:
            self.paths = self.paths[:limit]
        self.i = 0
        self.delay = delay

    def read(self):
        if self.i >= len(self.paths):
            return None
        p = self.paths[self.i]
        self.i += 1
        if self.delay:
            time.sleep(self.delay)
        return Image.open(p)

    def close(self):
        pass


# =============================================================================
# Live reconstructor
# =============================================================================

class LiveReconstructor:
    """Wraps the model's causal step so frames can be pushed in one at a time."""

    def __init__(self, model, device, num_scale_frames=8, keyframe_interval=1):
        self.model = model
        self.device = device
        self.num_scale_frames = num_scale_frames
        self.keyframe_interval = keyframe_interval

        self.model.clean_kv_cache()
        self._scale_buffer = []
        self.started = False
        self.n_frames = 0
        self.results = []  # per-frame dicts

    @torch.no_grad()
    def push(self, frame: torch.Tensor):
        """Push one preprocessed [1,3,H,W] frame. Returns a result dict once
        streaming has begun, or None while the scale buffer is still filling."""
        frame = frame.to(self.device)

        if not self.started:
            self._scale_buffer.append(frame)
            if len(self._scale_buffer) < self.num_scale_frames:
                return None
            # Scale phase: bidirectional attention across the first N frames.
            scale_images = torch.stack(self._scale_buffer, dim=1)  # [1,N,3,H,W]
            out = self.model.forward(
                scale_images,
                num_frame_for_scale=self.num_scale_frames,
                num_frame_per_block=self.num_scale_frames,
                causal_inference=True,
            )
            self.started = True
            self._scale_buffer = []
            self.n_frames = self.num_scale_frames
            return self._unpack(out, scale_phase=True)

        # Streaming phase: one frame, KV cache carries the history.
        idx = self.n_frames
        is_keyframe = (
            self.keyframe_interval <= 1
            or (idx - self.num_scale_frames) % self.keyframe_interval == 0
        )
        if not is_keyframe:
            self.model._set_skip_append(True)

        out = self.model.forward(
            frame.unsqueeze(1),  # [1,1,3,H,W]
            num_frame_for_scale=self.num_scale_frames,
            num_frame_per_block=1,
            causal_inference=True,
        )

        if not is_keyframe:
            self.model._set_skip_append(False)

        self.n_frames += 1
        return self._unpack(out, scale_phase=False)

    def _unpack(self, out, scale_phase):
        """Decode pose encoding to extrinsics/intrinsics and detach to CPU."""
        pose_enc = out["pose_enc"]                       # [1,n,9]
        depth = out["depth"].detach().cpu()              # [1,n,H,W,1]
        conf = out["depth_conf"].detach().cpu()          # [1,n,H,W]
        hw = depth.shape[2:4]
        extri, intri = pose_encoding_to_extri_intri(pose_enc, hw)

        res = {
            "extrinsic": extri.detach().cpu().numpy()[0],   # [n,3,4] w2c
            "intrinsic": intri.detach().cpu().numpy()[0],   # [n,3,3]
            "depth": depth.numpy()[0],                      # [n,H,W,1]
            "depth_conf": conf.numpy()[0],                  # [n,H,W]
            "scale_phase": scale_phase,
        }
        self.results.append(res)
        return res


def camera_center(extrinsic_3x4):
    """w2c [3,4] -> camera center in world coords."""
    m = np.eye(4)
    m[:3, :] = extrinsic_3x4
    return closed_form_inverse_se3_general(torch.from_numpy(m[None]))[0, :3, 3].numpy()


def write_ply(path, xyz, rgb):
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
    ap.add_argument("--source", choices=["webcam", "replay"], default="webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--replay_dir", default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--keyframe_interval", type=int, default=1,
                    help="Cache every Nth frame. Raise for long live sessions.")
    ap.add_argument("--use_sdpa", action="store_true", default=False)
    ap.add_argument("--camera_num_iterations", type=int, default=4)
    ap.add_argument("--out_dir", default="live_out")
    ap.add_argument("--conf_threshold", type=float, default=1.5)
    ap.add_argument("--save_cloud", action="store_true",
                    help="Accumulate points for a PLY on exit (memory-hungry).")
    ap.add_argument("--cloud_stride", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building model...")
    model = GCTStream(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa or not torch.cuda.is_available(),
        camera_num_iterations=args.camera_num_iterations,
    )
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    if device.type == "cuda":
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)

    if args.source == "webcam":
        src = WebcamSource(args.camera)
    else:
        if not args.replay_dir:
            sys.exit("--replay_dir is required with --source replay")
        src = ReplaySource(args.replay_dir, limit=args.max_frames)

    rec = LiveReconstructor(model, device,
                            num_scale_frames=args.num_scale_frames,
                            keyframe_interval=args.keyframe_interval)

    stop = {"now": False}

    def on_sigint(sig, frm):
        print("\nStopping capture...")
        stop["now"] = True

    signal.signal(signal.SIGINT, on_sigint)

    centers, cloud_xyz, cloud_rgb = [], [], []
    n = 0
    t_start = time.time()
    print(f"Capturing (buffering {args.num_scale_frames} frames for scale phase)... Ctrl-C to stop.")

    while not stop["now"]:
        if args.max_frames and n >= args.max_frames:
            break
        pil = src.read()
        if pil is None:
            print("Source exhausted.")
            break
        frame = preprocess_frame(pil)
        n += 1

        t0 = time.time()
        res = rec.push(frame)
        dt = time.time() - t0
        if res is None:
            continue  # still filling the scale buffer

        for j in range(res["extrinsic"].shape[0]):
            c = camera_center(res["extrinsic"][j])
            centers.append(c)

        d = res["depth"][-1, ..., 0]
        cf = res["depth_conf"][-1]
        tag = "scale" if res["scale_phase"] else "live"
        print(f"[{n:5d}] {tag:5s} {dt*1000:7.1f} ms  "
              f"pos=({centers[-1][0]:+.3f},{centers[-1][1]:+.3f},{centers[-1][2]:+.3f})  "
              f"depth={np.median(d):.3f}  conf>{args.conf_threshold}={100*(cf>args.conf_threshold).mean():.0f}%",
              flush=True)

        if args.save_cloud:
            pts = unproject_depth_map_to_point_map(
                res["depth"][-1:], res["extrinsic"][-1:], res["intrinsic"][-1:]
            )[0]
            m = cf > args.conf_threshold
            xyz = pts[m][::args.cloud_stride]
            rgb = (frame[0].permute(1, 2, 0).numpy()[m][::args.cloud_stride] * 255).astype(np.uint8)
            cloud_xyz.append(xyz)
            cloud_rgb.append(rgb)

    src.close()
    elapsed = time.time() - t_start
    print(f"\nCaptured {n} frames in {elapsed:.1f}s ({n/max(elapsed,1e-6):.2f} FPS end-to-end)")

    if centers:
        centers = np.array(centers)
        np.save(os.path.join(args.out_dir, "trajectory.npy"), centers)
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([0.0])
        print(f"Trajectory: {len(centers)} poses, path length {steps.sum():.3f}, "
              f"step mean {steps.mean():.4f} max {steps.max():.4f}")
        print(f"  saved -> {os.path.join(args.out_dir, 'trajectory.npy')}")

    if args.save_cloud and cloud_xyz:
        xyz = np.concatenate(cloud_xyz)
        rgb = np.concatenate(cloud_rgb)
        p = os.path.join(args.out_dir, "live_cloud.ply")
        write_ply(p, xyz, rgb)
        print(f"Point cloud: {len(xyz):,} points -> {p}")


if __name__ == "__main__":
    main()
