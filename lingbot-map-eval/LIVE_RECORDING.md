# Live recording with LingBot-Map

`demo.py` cannot do live capture. It takes `--image_folder` or `--video_path`,
loads every frame into one tensor, and only then runs inference — so you need
the whole recording before you start.

The **model** has no such limitation. `inference_streaming`
(`lingbot_map/models/gct_stream.py:350`) runs a bidirectional scale phase over
the first N frames and then calls, per frame:

```python
self.forward(frame_image,
             num_frame_for_scale=scale_frames,
             num_frame_per_block=1,
             causal_inference=True)
```

The KV cache persists on the module between calls. That step function is all a
live loop needs — only the demo harness is offline, not the architecture.

`live_map.py` drives that step from a camera.

## Usage

```bash
# From an attached camera, with the map building live in the browser
python live_map.py --model_path /path/to/lingbot-map.pt \
    --source webcam --camera 0 --view

# From a phone or IP camera over the network
python live_map.py --model_path /path/to/lingbot-map.pt \
    --source stream --stream_url rtsp://192.168.1.42:8554/live --view

# Replay frames off disk one at a time (same code path, no camera needed)
python live_map.py --model_path /path/to/lingbot-map.pt \
    --source replay --replay_dir example/loop --max_frames 100 --view
```

### Sources

| Flag | Use |
| :--- | :--- |
| `--source webcam --camera 0` | Attached USB / built-in camera |
| `--source stream --stream_url URL` | RTSP / HTTP / MJPEG — phone camera apps, IP cameras |
| `--source replay --replay_dir DIR` | Frames off disk, one at a time — identical code path, for testing without a camera |

Network streams set `CAP_PROP_BUFFERSIZE=1`. Inference is much slower than the
camera, so without this the driver buffer fills with stale frames and the map
lags further behind reality the longer you record.

### Live view (`--view`)

Serves a viser scene at `http://localhost:8080` that grows as you record:
accumulated points, a camera frustum at the current pose, and the trajectory
so far. The viewer stays up after capture stops so you can inspect the result.

This is the piece upstream does not have. `demo.py` only opens its viewer
*after* processing a complete recording — `PointCloudViewer` takes a finished
prediction dict and renders it once. `LiveViewer` instead pushes each frame's
points in as they are produced, one scene node per frame, so an update costs
O(new points) rather than re-uploading the whole cloud. `--max_frames_shown`
retires the oldest chunks to bound browser memory on long sessions.

Verified two ways:

- **In isolation** — 6 synthetic frames pushed with `--max_frames_shown 3`;
  server returned HTTP 200, oldest chunks correctly retired (3 retained), and
  the frustum and trajectory spline updated without error.
- **During real capture** — 11 replayed frames of `example/loop`. The viewer
  answered HTTP 200 *while capture was still running*, which is the point: the
  map is served as it builds, not after.

### Scale-phase geometry fix

The scale phase returns all N warmup frames in one result, but only the newest
was being unprojected — so 7 of the first 8 frames never reached the map. Input
frames are now carried through the result and every frame contributes.

The point counts confirm it. Despite **fewer frames and a coarser stride**, the
fixed version produces 64% more points:

| | Frames | Stride | Points | Contributing frames | Points/frame (stride-normalised) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Before | 12 | 30 | 23,944 | 5 | 4,789 |
| After | 11 | 40 | 39,387 | 11 | 4,774 |

Per-frame density is unchanged (4,789 vs 4,774, 0.3% apart) — exactly what you
expect if the fix changed *how many frames contribute* rather than the geometry
itself. The resulting cloud stays coherent: all finite, extent
`1.37 × 1.03 × 2.86` (corridor-shaped), radial spread median 0.420 / p95 0.921.

Ctrl-C stops capture and writes the trajectory. Add `--save_cloud` to
accumulate a PLY as you go.

### The map it produces

With `--save_cloud`, each frame's depth is unprojected to world coordinates
using its predicted pose and fused into one growing colored point cloud, written
on exit alongside `trajectory.npy`.

Verified on 12 replayed frames of `example/loop` (indoor corridor):

```
Captured 12 frames in 135.3s
Trajectory: 12 poses, path length 0.470, step mean 0.0427 max 0.0675
Point cloud: 23,944 points -> live_map_out/live_cloud.ply
```

The cloud is geometrically coherent, not just non-empty — all coordinates
finite, bounding box extent `1.08 × 1.05 × 2.39` (2.2× longer along the view
axis than laterally, i.e. corridor-shaped), radial spread from the centroid
tight at median 0.413 / p95 0.850 / max 1.852, so no scattered outliers.

**What it is:** a colored 3D point cloud plus a camera trajectory, openable in
MeshLab or CloudCompare. **What it is not:** a mesh, an occupancy grid, or a
SLAM map with loop closure — and there is no live view while recording (see
below).

Pose and depth are emitted per frame as you record:

```
[    8] scale 88147.5 ms  pos=(+0.030,+0.068,-0.202)  depth=0.892  conf>1.5=93%
[    9] live  12401.9 ms  pos=(+0.032,+0.085,-0.254)  depth=0.883  conf>1.5=94%
[   10] live  12892.3 ms  pos=(+0.038,+0.102,-0.315)  depth=0.879  conf>1.5=95%
```

## How it works

`LiveReconstructor.push(frame)` holds the two phases:

1. **Warmup** — buffers the first `--num_scale_frames` (default 8) frames, then
   runs them as one block with bidirectional attention. Nothing is emitted until
   this fills, so there is a startup latency of 8 frames.
2. **Streaming** — every subsequent frame is a single `forward` call with
   `num_frame_per_block=1`. The KV cache carries all history, so each call sees
   the whole sequence so far.

Preprocessing is replicated from `load_fn.load_and_preprocess_images` for
in-memory frames: resize width to 518, height to the nearest multiple of 14,
center-crop if taller than 518. It must match — the model asserts on patch
divisibility.

## Correctness

`compare_live_vs_batch.py` runs the same frames both ways on one model instance
— `inference_streaming(all_frames)` versus `push()` per frame — and compares
recovered extrinsics and depth:

```bash
python compare_live_vs_batch.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/loop --n 11
```

Measured on 11 frames of `example/loop` (8 scale + 3 streaming), CPU fp32:

```
extrinsics : max abs diff = 0.000e+00  mean = 0.000e+00
depth      : max abs diff = 0.000e+00  mean = 0.000e+00
depth      : max rel diff = 0.000e+00
MATCH — incremental path is equivalent to batch
```

Bitwise identical, not merely close — the same deterministic ops run in the same
order against the same KV cache state, so pushing frames one at a time is the
same computation `inference_streaming` performs internally. The incremental
driver is a faithful use of the causal API, not an approximation of it.

Run it after any change to the preprocessing or the push loop.

## Practical notes for live use

- **You need a GPU.** Measured 10–16 s/frame on 4 CPU cores; the paper's ~20 FPS
  assumes a GPU with bf16 and FlashInfer's paged KV cache. CPU live capture is a
  plumbing test, not a usable capture rate.
- **Raise `--keyframe_interval` for long sessions.** The KV cache grows with
  every frame, and the model was trained with video RoPE on 320 views — quality
  degrades past that. `--keyframe_interval 6` caches every 6th frame while still
  emitting predictions for all of them.
- **There is no state reset.** A long live session will eventually drift beyond
  the training range. Upstream's guidance is to switch to windowed mode, but
  windowed inference needs the full sequence and so is not available live —
  for an unbounded session you would need to restart the stream periodically
  and stitch, which this script does not do.
- **Scale is arbitrary.** The first camera defines the origin and the
  reconstruction is up to an unknown scale factor. Metric output needs a known
  baseline or calibration.
- **Camera intrinsics are predicted, not read** from your device. The model
  estimates focal length per frame.
