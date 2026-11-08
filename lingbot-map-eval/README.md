# Running LingBot-Map on CPU

Notes from getting [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)
running headless, with no GPU. The upstream README assumes CUDA 12.8 + FlashInfer;
this documents what actually works without either.

## Result

It runs. Streaming and windowed inference both produce geometrically valid
reconstructions on CPU, roughly 9 s/frame at 518×294 with 4 cores.

## Setup

The upstream install instructions pin a CUDA build of PyTorch and FlashInfer.
Neither is required — the model has a CPU fallback (`demo.py:421`) and an SDPA
attention path that replaces FlashInfer's paged KV cache.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Skip the `[vis]` extra. Without `viser` installed, `demo.py` runs inference and
prints the prediction keys instead of launching a viewer — which is what you
want headless.

Download the checkpoint (4.63 GB):

```python
from huggingface_hub import hf_hub_download
hf_hub_download("robbyant/lingbot-map", "lingbot-map.pt")
```

## Running

Pass `--use_sdpa`. Without it the model tries FlashInfer and raises
`RuntimeError: FlashInfer is not available`.

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/loop --first_k 24 --use_sdpa --mode streaming
```

Windowed mode works too:

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/university --first_k 20 --use_sdpa \
    --mode windowed --window_size 12 --overlap_size 4
```

## Gotchas

- **`--use_sdpa` is not the default.** FlashInfer is, and it is CUDA-only.
- **`load_and_preprocess_images` defaults do not match the model.** The helper
  defaults to `image_size=512, patch_size=16`, but the model wants `518/14`. Calling
  it bare yields a 288-pixel height and trips
  `AssertionError: Input image height 288 is not a multiple of patch height 14`.
  Call it the way `demo.py` does: `mode="crop", image_size=518, patch_size=14`.
- **`inference_streaming` returns `pose_enc`, not extrinsics.** Decode with
  `pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])`.
- **Extrinsic conventions differ between call sites.**
  `unproject_depth_map_to_point_map` expects **w2c** and inverts internally, while
  `demo.py:278` stores **c2w** under `predictions["extrinsic"]` for the viewer.
  Feeding the viewer's c2w back into unprojection silently gives a wrong cloud.
- **`Failed to load pretrained weights: [Errno 2] ... ''` on startup is harmless.**
  That is the DINOv2 trunk looking for an empty `pretrained_path`; the real
  checkpoint loads immediately after with 0 missing and 0 unexpected keys.
- **`torch.amp.autocast("cuda", ...)` warns on CPU** and disables itself. Benign —
  CPU inference runs in fp32 and the aggregator dtype cast is skipped.
- The `.cuda()` calls in `lingbot_map/utils/geometry.py:430-440` are in
  `induced_flow` / `compute_distance_matrix_flow`, which the demo path never calls.

## Verification

`verify_run.py` runs inference headless and checks the output is actually sane
rather than merely non-crashing — depth statistics, intrinsics, camera trajectory
continuity, rotation orthonormality — then writes a PLY point cloud, depth
previews, and the raw predictions.

```bash
python verify_run.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/loop --first_k 24 --out_dir verify_out
```

### Measured on 24 frames of `example/loop` (indoor office corridor)

| Check | Result |
| :--- | :--- |
| Model parameters | 1.158 B |
| Checkpoint load | 0 missing, 0 unexpected keys |
| Inference | 212.9 s total, 8.87 s/frame |
| Depth | all finite, 0.350 – 3.624, median 0.848 |
| Confidence | 92.6% of pixels above threshold 1.5 |
| Intrinsics (frame 0) | fx=431.1, fy=432.0, cx=259.0, cy=147.0 — principal point exactly centered for 518×294 |
| Trajectory | path length 1.362 over 24 frames, per-frame step 0.0104 – 0.0807 (mean 0.0592, std 0.0207) — smooth, no jumps |
| Rotations | max ‖RRᵀ − I‖ = 5.8e-08, det(R) = 1.000000 |
| Point cloud | 3,383,031 of 3,655,008 points above confidence threshold |

Frame 0 is at the origin, as expected — the first camera defines the canonical
frame. `sample_depth_frame0.png` shows the input frame above its predicted depth
(warm = far, cool = near): the corridor vanishing point is correctly furthest,
near cubicle partitions are closest, and object silhouettes are crisp.

## Performance caveat

~9 s/frame on 4 CPU cores in fp32 is about 180× off the paper's ~20 FPS claim,
which assumes a GPU with bf16 and FlashInfer's paged KV cache. CPU is fine for
correctness checks and small scenes; it is not viable for the long-sequence
workloads the model is built for. `--compile` is GPU-only in practice — it warms
up CUDA graphs.
