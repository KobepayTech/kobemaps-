"""Render demo stills from a reconstructed map: cloud views + camera trajectory."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_ply(path):
    xyz, rgb = [], []
    with open(path) as f:
        header = True
        for line in f:
            if header:
                if line.startswith("end_header"):
                    header = False
                continue
            v = line.split()
            xyz.append([float(x) for x in v[:3]])
            rgb.append([int(x) for x in v[3:6]])
    return np.array(xyz), np.array(rgb) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--out_dir", default="demo_assets")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    xyz, rgb = load_ply(args.ply)
    print(f"loaded {len(xyz):,} points")

    traj = None
    if args.trajectory and os.path.exists(args.trajectory):
        traj = np.load(args.trajectory)
        print(f"loaded trajectory: {traj.shape}")

    # Trim extreme outliers so the framing follows the scene, not stragglers.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    p, c = xyz[keep], rgb[keep]

    # OpenCV camera convention: X right, Y down, Z forward. So (X,Z) is the
    # bird's-eye plane and (X,Y) faces down the corridor.
    views = [
        ("top",   (0, 2), "X (right)", "Z (forward)", "bird's-eye — corridor walls and camera path"),
        ("front", (0, 1), "X (right)", "Y (down)", "front — looking down the corridor"),
        ("side",  (2, 1), "Z (forward)", "Y (down)", "side — floor and ceiling"),
    ]
    for name, (i, j), xl, yl, title in views:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.scatter(p[:, i], p[:, j], c=c, s=0.25, marker=".", linewidths=0)
        if traj is not None:
            ax.plot(traj[:, i], traj[:, j], "-", color="#ff7b39", lw=2.0,
                    label="camera path")
            ax.scatter(traj[0, i], traj[0, j], color="#39d3ff", s=45,
                       zorder=5, label="start")
            ax.legend(facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#c9d1d9", fontsize=8, loc="upper right")
        ax.set_xlabel(xl, color="#8b949e")
        ax.set_ylabel(yl, color="#8b949e")
        ax.set_title(title, color="#c9d1d9", fontsize=11)
        ax.tick_params(colors="#484f58", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"cloud_{name}.png")
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("wrote", out)

    # 3D perspective view
    fig = plt.figure(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    step = max(1, len(p) // 60000)
    ax.scatter(p[::step, 0], p[::step, 2], -p[::step, 1],
               c=c[::step], s=0.3, marker=".", linewidths=0)
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 2], -traj[:, 1], color="#ff7b39", lw=2.5)
    ax.set_xlabel("X", color="#8b949e", fontsize=8)
    ax.set_ylabel("Z", color="#8b949e", fontsize=8)
    ax.set_zlabel("-Y", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#484f58", labelsize=6)
    ax.xaxis.pane.set_facecolor("#0d1117")
    ax.yaxis.pane.set_facecolor("#0d1117")
    ax.zaxis.pane.set_facecolor("#0d1117")
    ax.view_init(elev=18, azim=-72)
    ax.set_title("reconstructed corridor", color="#c9d1d9", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "cloud_3d.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
