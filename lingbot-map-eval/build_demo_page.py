"""Generate the self-contained demo page with images inlined as data URIs."""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_page.html")


def uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


img_3d = uri(os.path.join(HERE, "demo_assets/cloud_3d.png"))
img_top = uri(os.path.join(HERE, "demo_assets/cloud_top.png"))
img_front = uri(os.path.join(HERE, "demo_assets/cloud_front.png"))
img_side = uri(os.path.join(HERE, "demo_assets/cloud_side.png"))
img_depth = uri(os.path.join(HERE, "verify_out/frame_0000_rgb_depth.png"))

HTML = f"""<title>LingBot-Map — Live Streaming Reconstruction</title>
<style>
  :root {{
    --ground: #f4f6f8;
    --surface: #ffffff;
    --surface-2: #eaeef2;
    --line: #d3dae1;
    --text: #16202b;
    --muted: #5c6b7a;
    --accent: #d2551a;
    --accent-2: #0d7ec4;
    --good: #1f7a4d;
    --warn: #9a6206;
    --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --measure: 66ch;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #0f1720;
      --surface: #16202b;
      --surface-2: #1b2734;
      --line: #27353f;
      --text: #d6dee7;
      --muted: #8496a8;
      --accent: #ff7b39;
      --accent-2: #39b8ff;
      --good: #4ad98d;
      --warn: #e0a83c;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #0f1720;
    --surface: #16202b;
    --surface-2: #1b2734;
    --line: #27353f;
    --text: #d6dee7;
    --muted: #8496a8;
    --accent: #ff7b39;
    --accent-2: #39b8ff;
    --good: #4ad98d;
    --warn: #e0a83c;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 0 24px 96px; }}

  .eyebrow {{
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
  }}

  header {{
    display: flex; flex-direction: column; gap: 20px;
    padding: 72px 0 40px;
    border-bottom: 1px solid var(--line);
  }}
  h1 {{
    margin: 0;
    font-size: clamp(30px, 5vw, 46px);
    line-height: 1.08;
    letter-spacing: -0.025em;
    font-weight: 620;
    text-wrap: balance;
  }}
  .lede {{ margin: 0; max-width: var(--measure); font-size: 18px; color: var(--muted); }}
  .lede strong {{ color: var(--text); font-weight: 600; }}

  h2 {{
    margin: 0;
    font-size: 22px;
    letter-spacing: -0.015em;
    font-weight: 620;
    text-wrap: balance;
  }}
  h3 {{ margin: 0; font-size: 15px; font-weight: 620; letter-spacing: -0.005em; }}
  p {{ margin: 0; max-width: var(--measure); }}
  section {{ display: flex; flex-direction: column; gap: 18px; padding: 52px 0 0; }}
  .stack {{ display: flex; flex-direction: column; gap: 12px; }}

  /* Pipeline — the stages are a real sequence, so they are numbered. */
  .pipe {{ display: flex; flex-direction: column; gap: 0; }}
  .stage {{
    display: grid;
    grid-template-columns: 52px 1fr;
    gap: 20px;
    padding: 18px 0;
    border-top: 1px solid var(--line);
  }}
  .stage:last-child {{ border-bottom: 1px solid var(--line); }}
  .stage-n {{
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    padding-top: 3px;
    letter-spacing: 0.06em;
  }}
  .stage p {{ color: var(--muted); font-size: 15px; }}
  .stage code {{ color: var(--text); }}

  code {{
    font-family: var(--font-mono);
    font-size: 0.9em;
    background: var(--surface-2);
    padding: 1px 5px;
    border-radius: 3px;
  }}
  pre {{
    margin: 0;
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 2px solid var(--accent);
    border-radius: 4px;
    padding: 16px 18px;
    overflow-x: auto;
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.65;
    color: var(--text);
  }}
  pre code {{ background: none; padding: 0; font-size: inherit; }}
  .cmt {{ color: var(--muted); }}

  figure {{ margin: 0; display: flex; flex-direction: column; gap: 10px; }}
  figure img {{
    display: block; width: 100%; height: auto;
    border: 1px solid var(--line); border-radius: 5px; background: #0d1117;
  }}
  figcaption {{ font-size: 13px; color: var(--muted); }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; align-items: start; }}

  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }}
  .metric {{ background: var(--surface); padding: 16px 18px; display: flex; flex-direction: column; gap: 5px; }}
  .metric .v {{ font-family: var(--font-mono); font-size: 20px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
  .metric .k {{ font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }}

  .tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 5px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 11px 16px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  thead th {{
    font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); background: var(--surface-2); font-weight: 500;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr {{ background: var(--surface); }}
  td.num {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
  .ok {{ color: var(--good); }}

  ul {{ margin: 0; padding-left: 18px; max-width: var(--measure); display: flex; flex-direction: column; gap: 9px; }}
  li {{ color: var(--muted); }}
  li strong {{ color: var(--text); font-weight: 600; }}

  .callout {{
    border: 1px solid var(--line);
    border-left: 2px solid var(--warn);
    background: var(--surface);
    border-radius: 4px;
    padding: 18px 20px;
    display: flex; flex-direction: column; gap: 10px;
  }}
  .callout .eyebrow {{ color: var(--warn); }}
  .callout p {{ font-size: 15px; color: var(--muted); }}

  footer {{ margin-top: 64px; padding-top: 22px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
</style>

<div class="wrap">

  <header>
    <div class="eyebrow">LingBot-Map · Geometric Context Transformer</div>
    <h1>Mapping a space from a live camera stream</h1>
    <p class="lede">
      A 1.16B-parameter feed-forward model turns a video stream into camera poses,
      per-frame depth, and a fused 3D point cloud — <strong>frame by frame, as you
      record</strong>, rather than after the recording ends.
    </p>
  </header>

  <section>
    <div class="eyebrow">The reconstruction</div>
    <h2>11 frames of an office corridor</h2>
    <p>
      Every frame's depth is unprojected into world coordinates using its predicted
      pose and fused into one growing cloud. The camera path is drawn in orange.
    </p>
    <figure>
      <img src="{img_3d}" alt="3D perspective render of the reconstructed corridor point cloud with the camera trajectory">
      <figcaption>39,387 points recovered from 11 streamed frames. No depth sensor — geometry is inferred from the images alone.</figcaption>
    </figure>
    <div class="grid-3">
      <figure>
        <img src="{img_top}" alt="Bird's-eye view showing two parallel corridor walls">
        <figcaption>Bird's-eye. Two parallel walls resolve cleanly — the corridor structure.</figcaption>
      </figure>
      <figure>
        <img src="{img_front}" alt="Front view looking down the corridor">
        <figcaption>Front, looking down the corridor.</figcaption>
      </figure>
      <figure>
        <img src="{img_side}" alt="Side view showing floor and ceiling">
        <figcaption>Side, showing floor and ceiling separation.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <div class="grid-2">
      <figure>
        <img src="{img_depth}" alt="Input camera frame above its predicted depth map">
        <figcaption>Input frame above its predicted depth — warm is far, cool is near.</figcaption>
      </figure>
      <div class="stack">
        <h2>Depth, per frame</h2>
        <p>
          The corridor's vanishing point is correctly the most distant region, the
          near cubicle partitions the closest, and object silhouettes stay crisp at
          their boundaries.
        </p>
        <p>
          Roughly <strong>93–95%</strong> of pixels clear the confidence threshold on
          this scene.
        </p>
      </div>
    </div>
  </section>

  <section>
    <div class="eyebrow">How it runs</div>
    <h2>The streaming pipeline</h2>
    <p>
      The model is causal by construction. Only the reference demo was offline — it
      loaded every frame into one tensor before inferring anything.
    </p>
    <div class="pipe">
      <div class="stage">
        <div class="stage-n">01</div>
        <div class="stack">
          <h3>Frame source</h3>
          <p>An attached camera, an RTSP/HTTP stream from a phone or IP camera, or frames replayed off disk. Network streams keep a 1-frame buffer so the map tracks the live edge instead of draining a backlog.</p>
        </div>
      </div>
      <div class="stage">
        <div class="stage-n">02</div>
        <div class="stack">
          <h3>Scale phase</h3>
          <p>The first 8 frames are processed as one block with bidirectional attention, establishing the coordinate frame. This is the startup latency — nothing is emitted until it fills.</p>
        </div>
      </div>
      <div class="stage">
        <div class="stage-n">03</div>
        <div class="stack">
          <h3>Causal step</h3>
          <p>Every later frame is a single <code>forward</code> pass with <code>num_frame_per_block=1</code>. A persistent KV cache carries the whole history, so each frame sees everything before it.</p>
        </div>
      </div>
      <div class="stage">
        <div class="stage-n">04</div>
        <div class="stack">
          <h3>Fuse</h3>
          <p>Depth is unprojected to world coordinates through the predicted pose and intrinsics, filtered by confidence, and appended to the map.</p>
        </div>
      </div>
      <div class="stage">
        <div class="stage-n">05</div>
        <div class="stack">
          <h3>Live view</h3>
          <p>Each frame's points are pushed into a browser scene as its own node, so an update costs the new points rather than a full re-upload of the cloud.</p>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="eyebrow">Running it</div>
    <h2>One command</h2>
    <pre><code><span class="cmt"># replay the bundled corridor scene</span>
./demo_live.sh

<span class="cmt"># map from an attached camera</span>
./demo_live.sh webcam

<span class="cmt"># map from a phone or IP camera</span>
./demo_live.sh stream rtsp://192.168.1.42:8554/live</code></pre>
    <p>The map builds at <code>http://localhost:8080</code> while frames stream in, and the cloud is written to disk on exit.</p>
  </section>

  <section>
    <div class="eyebrow">Verification</div>
    <h2>What was actually measured</h2>
    <p>
      Each claim below came from a run, not from the paper. The strongest result is
      the first: driving the model one frame at a time is not an approximation of
      batch inference — it is bit-for-bit the same computation.
    </p>
    <div class="tablewrap">
      <table>
        <thead>
          <tr><th>Check</th><th>Result</th><th>Reading</th></tr>
        </thead>
        <tbody>
          <tr><td>Live vs. batch inference</td><td class="num ok">0.000e+00</td><td>Bitwise identical poses and depth</td></tr>
          <tr><td>Checkpoint load</td><td class="num">0 / 0</td><td>No missing or unexpected keys</td></tr>
          <tr><td>Rotation orthonormality</td><td class="num">5.8e-08</td><td>max ‖RRᵀ − I‖, det(R) = 1.000000</td></tr>
          <tr><td>Camera-to-geometry distance</td><td class="num">0.91 vs 0.85</td><td>Matches median depth every frame</td></tr>
          <tr><td>Trajectory continuity</td><td class="num">0.041 ± 0.021</td><td>Per-frame step, no jumps</td></tr>
          <tr><td>Depth range</td><td class="num">0.350 – 3.624</td><td>All finite, median 0.848</td></tr>
          <tr><td>Principal point</td><td class="num">259.0, 147.0</td><td>Exactly centered for 518×294</td></tr>
          <tr><td>Live view during capture</td><td class="num ok">HTTP 200</td><td>Served while frames still processing</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="eyebrow">Performance</div>
    <h2>The one number that isn't ready</h2>
    <div class="metrics">
      <div class="metric"><span class="v">1.158 B</span><span class="k">parameters</span></div>
      <div class="metric"><span class="v">~10 s</span><span class="k">per frame · CPU</span></div>
      <div class="metric"><span class="v">84 s</span><span class="k">scale phase · 8 frames</span></div>
      <div class="metric"><span class="v">~20 FPS</span><span class="k">claimed · GPU</span></div>
    </div>
    <div class="callout">
      <div class="eyebrow">Hardware, not code</div>
      <p>
        Every measurement above was taken on 4 CPU cores in fp32 — about 10 s/frame,
        some 200× short of live capture speed. That gap closes with a GPU (bf16 plus
        FlashInfer's paged KV-cache attention), not with more software. The same
        script and flags run unchanged; only the frame rate moves. Measure the real
        figure on the target machine with <code>gct_profile.py</code> rather than
        trusting the published number.
      </p>
    </div>
  </section>

  <section>
    <div class="eyebrow">Limits</div>
    <h2>What this does not do</h2>
    <ul>
      <li><strong>Scale is arbitrary.</strong> The reconstruction is correct up to an unknown scale factor. Metric units need a known baseline or calibration.</li>
      <li><strong>No loop closure.</strong> There is no state reset, so a long circuit will drift and the start and end will not line up. This is absent from the model, not missing from the plumbing.</li>
      <li><strong>Point cloud, not a mesh.</strong> No surface reconstruction, occupancy grid, or semantic labelling.</li>
      <li><strong>Bounded sequence length.</strong> Trained with video RoPE on 320 views; beyond that, raise the keyframe interval or quality degrades.</li>
    </ul>
  </section>

  <footer>
    Model: <code>robbyant/lingbot-map</code>, Apache-2.0. Measurements from replayed
    frames of the bundled <code>example/loop</code> scene on CPU.
  </footer>

</div>
"""

with open(OUT, "w") as f:
    f.write(HTML)

print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
