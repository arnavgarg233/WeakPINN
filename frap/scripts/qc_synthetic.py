"""Quality-check figure for the 4 synthetic FRAP stacks.

Renders a 4x4 grid: 4 stacks (clean, noise_low, noise_med, noise_high)
x 4 timepoints (first, 1/3, 2/3, last). Saves to figures/qc_synthetic.png.

Purpose: visually confirm
  1. Bleach disk visible at t=0 with correct depth (0.8 -> floor ~0.2).
  2. Recovery progresses over time (disk fills in).
  3. Noise level grows monotonically clean -> high (visible Poisson).
  4. No edge artifacts (Neumann BC is implemented correctly - no
     wrap-around bleed from one edge to the opposite edge).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    stacks = ["clean", "noise_low", "noise_med", "noise_high"]
    data_dir = Path("data")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(stacks), 4, figsize=(11, 11), constrained_layout=True)
    for r, name in enumerate(stacks):
        d = np.load(data_dir / f"synthetic_{name}.npz")
        stack = d["stack"]
        T = stack.shape[0]
        frame_idxs = [0, T // 3, 2 * T // 3, T - 1]
        for c, fi in enumerate(frame_idxs):
            ax = axes[r, c]
            im = ax.imshow(stack[fi], cmap="gray", vmin=0, vmax=1.1)
            ax.set_title(f"{name}  t={fi}", fontsize=9)
            ax.axis("off")
            if c == 3:
                plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Synthetic FRAP stacks (Neumann BC, bleach=80%, D=0.05)", fontsize=12)
    out = fig_dir / "qc_synthetic.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # Print recovery-curve summary for each stack
    print("\nRecovery curves (mean intensity in central 10x10 bleach region):")
    print(f"  {'stack':12s}  t=0      t=T/3    t=2T/3   t=T-1")
    for name in stacks:
        d = np.load(data_dir / f"synthetic_{name}.npz")
        stack = d["stack"]
        T, H, W = stack.shape
        cy, cx = H // 2, W // 2
        center = stack[:, cy - 5 : cy + 5, cx - 5 : cx + 5].mean(axis=(1, 2))
        idxs = [0, T // 3, 2 * T // 3, T - 1]
        vals = "  ".join(f"{center[i]:.4f}" for i in idxs)
        print(f"  {name:12s}  {vals}")

    print("\nEdge-mean (top-left 8x8 corner) should stay near 1.0 if Neumann is correct:")
    print(f"  {'stack':12s}  t=0      t=T-1    delta")
    for name in stacks:
        d = np.load(data_dir / f"synthetic_{name}.npz")
        stack = d["stack"]
        edge0 = float(stack[0, :8, :8].mean())
        edge_end = float(stack[-1, :8, :8].mean())
        print(f"  {name:12s}  {edge0:.4f}   {edge_end:.4f}   {edge_end - edge0:+.4f}")


if __name__ == "__main__":
    main()
