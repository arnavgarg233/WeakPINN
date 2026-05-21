"""Build publication-quality Figure 4 and Figure 5 for the combined paper.

Figure 4 - Controlled FRAP diffusion recovery (synthetic):
  A. Real prebleach / bleach / postbleach frames as a FRAP task schematic
  B. Strong-form vs weak-form residual schematic (PDE labels)
  C. Synthetic D percent error across noise levels (clean / low / med / high)
  D. Pooled noisy headline (82.1% MAE reduction; noisy only)

Figure 5 - Experimental FRAP microscopy validation (real):
  A. Real 32ww D estimates across 5 seeds, weak vs strong, with LS anchor
  B. Real 56ww D estimates across 5 seeds, weak vs strong, with LS anchor
  C. Cross-seed std comparison (32ww tied; 56ww 68.4% lower under weak)
  D. Held-out reconstruction MSE - sanity panel: "comparable reconstruction error"

Reads: results/*.json, results/ls_reference_D.json,
       data/deepfrap/validation_exp/data/frap_56ww_005.mat
Writes: figures/figure4_synthetic.pdf+.png, figures/figure5_real.pdf+.png
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.io as sio
import seaborn as sns


D_NORM_TRUE = 0.01975  # synthetic ground truth, PINN coords


def _setup_style():
    """Publication-quality defaults via seaborn paper context + whitegrid theme."""
    sns.set_theme(context="paper", style="whitegrid", font="serif")
    mpl.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.45,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


METHOD_COLORS = {"data": "#888888", "strong": "#d95f02", "weak": "#1b9e77"}
METHOD_LABELS = {"data": "data-only", "strong": "strong-form", "weak": "weak-form"}

# Per-run JSONs were moved to an off-repo backup; allow regeneration without
# copying them back into results/.
PHASE9_BACKUP = Path(os.environ.get("WEAKPINN_PHASE9_BACKUP", ""))  # set this env var to enable fallback


def _phase9_glob(pat):
    backup_hits = sorted(PHASE9_BACKUP.glob(pat)) if PHASE9_BACKUP.exists() else []
    if backup_hits:
        return [str(p) for p in backup_hits]
    return sorted(glob.glob(f"results/{pat}"))


def load_D(block_pattern, method):
    """Return D_recovered list across seeds for a block pattern."""
    return [json.load(open(f))["D_recovered"]
            for f in _phase9_glob(block_pattern)
            if f"_{method}_" in f]


def load_mse(block_pattern, method):
    return [json.load(open(f))["val_mse"]
            for f in _phase9_glob(block_pattern)
            if f"_{method}_" in f]


# ============================================================================
# FIGURE 5 - Controlled FRAP diffusion recovery (synthetic)
# ============================================================================

def panel_5A(ax_pre, ax_bleach, ax_post):
    """Three real frames from frap_56ww_005.mat: prebleach, bleach, mid-recovery."""
    m = sio.loadmat("data/deepfrap/validation_exp/data/frap_56ww_005.mat",
                    struct_as_record=False, squeeze_me=True)
    exp = m["experiment"]
    pre = exp.prebleach.image_data[:, :, -1].astype(np.float32) / 65535.0
    bleach = exp.bleach.image_data[:, :, -1].astype(np.float32) / 65535.0
    post_mid = exp.postbleach.image_data[:, :, 30].astype(np.float32) / 65535.0
    vmin, vmax = 0.2, 0.95
    for ax, im, title in [
        (ax_pre, pre, "prebleach (t<0)"),
        (ax_bleach, bleach, "bleach (t≈0)"),
        (ax_post, post_mid, "postbleach (t≈8 s)"),
    ]:
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])


def panel_5B(ax):
    """Schematic: strong-form vs weak-form residual."""
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, r"$\partial_t c \;=\; D\,\nabla^2 c \;-\; k\,c$",
            ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(0.25, 0.74, "strong-form", ha="center", color=METHOD_COLORS["strong"],
            fontsize=11, fontweight="bold")
    ax.text(0.25, 0.58,
            r"$r_{\text{strong}} = \partial_t c_\theta - D_\theta\,(c_{xx}+c_{yy}) + k_\theta c_\theta$",
            ha="center", va="top", fontsize=10)
    ax.text(0.25, 0.36, "noise differentiated\ntwice", ha="center", va="top",
            fontsize=9, style="italic", color="#888")
    ax.text(0.75, 0.74, "weak-form", ha="center", color=METHOD_COLORS["weak"],
            fontsize=11, fontweight="bold")
    ax.text(0.75, 0.58,
            r"$\int \varphi_m \partial_t c\,d\Omega + D \int \nabla\varphi_m\!\cdot\!\nabla c\,d\Omega + k\int\varphi_m c\,d\Omega$",
            ha="center", va="top", fontsize=9)
    ax.text(0.75, 0.36,
            "derivatives moved\nto smooth test\nfunctions $\\varphi_m$",
            ha="center", va="top", fontsize=9, style="italic", color="#888")
    ax.text(0.5, 0.08,
            r"$\varphi_m = G_m(x,y)\cdot(1-x^2)^2(1-y^2)^2 \quad \Rightarrow\quad \varphi_m|_{\partial\Omega}=0$",
            ha="center", va="bottom", fontsize=9, color="#444")


def panel_5C(ax):
    """Synthetic D percent error across noise levels - paired bars + scatter."""
    noise_levels = ["clean", "low", "med", "high"]
    blocks = {"clean": "clean_*.json", "low": "noiselow_*.json",
              "med": "noisemed_*.json", "high": "noisehigh_*.json"}
    x = np.arange(len(noise_levels))
    width = 0.35
    means_w, means_s = [], []
    for nl in noise_levels:
        pat = blocks[nl]
        w_errs = [abs(D - D_NORM_TRUE) / D_NORM_TRUE * 100 for D in load_D(pat, "weak")]
        s_errs = [abs(D - D_NORM_TRUE) / D_NORM_TRUE * 100 for D in load_D(pat, "strong")]
        means_w.append(np.mean(w_errs)); means_s.append(np.mean(s_errs))
        # scatter individual seeds
        ax.scatter([x[noise_levels.index(nl)] - width/2] * len(s_errs), s_errs,
                   color="white", edgecolor=METHOD_COLORS["strong"], s=14, zorder=3)
        ax.scatter([x[noise_levels.index(nl)] + width/2] * len(w_errs), w_errs,
                   color="white", edgecolor=METHOD_COLORS["weak"], s=14, zorder=3)
    ax.bar(x - width/2, means_s, width, label="strong-form",
           color=METHOD_COLORS["strong"], alpha=0.85)
    ax.bar(x + width/2, means_w, width, label="weak-form",
           color=METHOD_COLORS["weak"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(noise_levels)
    ax.set_xlabel("photon-noise level")
    ax.set_ylabel(r"$|D - D_{\mathrm{true}}|\,/\,D_{\mathrm{true}}$  (%)")
    ax.set_title("D-recovery error across noise (3 seeds per cell)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def panel_5D(ax):
    """Pooled noisy headline: 82.1% MAE reduction."""
    # Pool low/med/high only (NOT clean)
    weak_abs, strong_abs = [], []
    for noise in ["noiselow", "noisemed", "noisehigh"]:
        for D in load_D(f"{noise}_*.json", "weak"):
            weak_abs.append(abs(D - D_NORM_TRUE))
        for D in load_D(f"{noise}_*.json", "strong"):
            strong_abs.append(abs(D - D_NORM_TRUE))
    mae_w = float(np.mean(weak_abs))
    mae_s = float(np.mean(strong_abs))
    reduction = 100 * (1 - mae_w / mae_s)
    bars = ax.bar(["strong-form", "weak-form"], [mae_s, mae_w],
                  color=[METHOD_COLORS["strong"], METHOD_COLORS["weak"]],
                  alpha=0.85, width=0.55)
    ax.set_ylabel(r"D-MAE")
    ax.set_title(f"Pooled noisy MAE: {reduction:.1f}% lower under weak-form")
    for b, val in zip(bars, [mae_s, mae_w]):
        ax.text(b.get_x() + b.get_width() / 2, val + max(mae_s, mae_w) * 0.02,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, mae_s * 1.25)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def make_figure5():
    # 7-col grid with col 3 as a buffer between Panel C (cols 0..2) and Panel D
    # (cols 4..6). Shifts Panel D to the right so its y-axis ticks don't
    # overlap with Panel C's bars.
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 7, height_ratios=[1, 1.3, 1.3],
                          hspace=0.55, wspace=0.95)
    # Row 1: Panel A as three side-by-side frames (cols 0..1, 2..3, 4..5)
    ax_pre    = fig.add_subplot(gs[0, 0:2])
    ax_bleach = fig.add_subplot(gs[0, 2:4])
    ax_post   = fig.add_subplot(gs[0, 4:6])
    panel_5A(ax_pre, ax_bleach, ax_post)
    fig.text(0.02, 0.93, "A", fontsize=14, fontweight="bold")
    # Panel A subtitle
    fig.text(0.5, 0.66,
             "Goal: infer the diffusion coefficient $D$ from the recovery dynamics.",
             ha="center", va="top", fontsize=10, style="italic", color="#444")
    # Row 2: Panel B (residual schematic, full width across the 7-col grid)
    ax_B = fig.add_subplot(gs[1, :])
    panel_5B(ax_B)
    fig.text(0.02, 0.61, "B", fontsize=14, fontweight="bold")
    # Row 3: Panels C and D with col 3 left empty as a horizontal buffer
    ax_C = fig.add_subplot(gs[2, 0:3])
    panel_5C(ax_C)
    # Panel labels in axes coords so they sit cleanly above-left of each
    # ax (not on top of the title text)
    ax_C.text(-0.16, 1.08, "C", transform=ax_C.transAxes,
              fontsize=14, fontweight="bold")
    ax_D = fig.add_subplot(gs[2, 4:7])
    panel_5D(ax_D)
    ax_D.text(-0.16, 1.08, "D", transform=ax_D.transAxes,
              fontsize=14, fontweight="bold")
    fig.suptitle("Controlled FRAP diffusion recovery",
                 fontsize=12, y=0.99, fontweight="bold")
    fig.savefig("../figures/main/figure4_synthetic.png", bbox_inches="tight")
    print("wrote figures/figure4_synthetic.png")


# ============================================================================
# FIGURE 6 - Experimental FRAP microscopy validation (real)
# ============================================================================

import re
import csv as _csv

# Top-5 stacks per condition, ranked by composite quality (rank 1 = best)
TOP5_STACKS = {
    "32ww": ["010", "006", "004", "005", "014"],
    "56ww": ["005", "004", "018", "011", "014"],
}
LEGACY_STACK = {"32ww": "010", "56ww": "005"}  # the n=1 stack from the first Phase 9


def _parse_real_fname(path):
    """Return (cond, stack_idx) for a real-FRAP run JSON. Legacy files
    (no stack_idx in name) map to the per-condition LEGACY_STACK."""
    name = Path(path).name
    m = re.match(r"real_(32ww|56ww)_(\d{3})_(strong|weak)_seed\d+\.json", name)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"real_(32ww|56ww)_(strong|weak)_seed\d+\.json", name)
    if m:
        cond = m.group(1)
        return cond, LEGACY_STACK[cond]
    return None, None


def load_per_stack_D(cond, method):
    """Return {stack_idx: [D values across seeds]} for one condition + method."""
    files = [f for f in _phase9_glob(f"real_{cond}_*.json") if f"_{method}_" in f]
    by_stack = {}
    for f in files:
        c, idx = _parse_real_fname(f)
        if c != cond:
            continue
        by_stack.setdefault(idx, []).append(json.load(open(f))["D_recovered"])
    return by_stack


def load_ls_per_stack():
    """Return {f'{cond}_{idx}': D_norm} from the per-stack LS reference CSV."""
    out = {}
    factor = 1.400868e-3
    with open("results/ls_reference_D.csv") as f:
        for r in _csv.DictReader(f):
            key = f"{r['condition']}_{int(r['dataset_index']):03d}"
            out[key] = float(r["D_m2_per_s"]) * 1e12 * factor
    return out


def panel_6_per_stack_strip(ax, cond, story_title):
    """Per-stack strip plot: 5 stacks on x-axis, per-stack strong/weak D values."""
    stacks = TOP5_STACKS[cond]
    s_by = load_per_stack_D(cond, "strong")
    w_by = load_per_stack_D(cond, "weak")
    ls_per = load_ls_per_stack()
    rng = np.random.default_rng(0)

    all_vals = []
    for x, idx in enumerate(stacks):
        s_vals = s_by.get(idx, [])
        w_vals = w_by.get(idx, [])
        # strong on left of x, weak on right; small jitter
        sx = x - 0.18 + rng.uniform(-0.04, 0.04, len(s_vals))
        wx = x + 0.18 + rng.uniform(-0.04, 0.04, len(w_vals))
        ax.scatter(sx, s_vals, color=METHOD_COLORS["strong"], s=55,
                   edgecolor="white", linewidth=0.6, zorder=4,
                   label="strong-form" if x == 0 else None)
        ax.scatter(wx, w_vals, color=METHOD_COLORS["weak"], s=55,
                   edgecolor="white", linewidth=0.6, zorder=4,
                   label="weak-form" if x == 0 else None)
        # per-stack mean bars
        if s_vals:
            ax.plot([x - 0.30, x - 0.06], [np.mean(s_vals)] * 2,
                    color=METHOD_COLORS["strong"], lw=2.2, zorder=5)
        if w_vals:
            ax.plot([x + 0.06, x + 0.30], [np.mean(w_vals)] * 2,
                    color=METHOD_COLORS["weak"], lw=2.2, zorder=5)
        # LS anchor as a small horizontal segment at this stack's x
        ls_v = ls_per.get(f"{cond}_{idx}")
        if ls_v is not None:
            ax.plot([x - 0.34, x + 0.34], [ls_v] * 2,
                    color="#444", linestyle="--", lw=1.0, alpha=0.7, zorder=3,
                    label="LS-fit anchor" if x == 0 else None)
        all_vals.extend(s_vals); all_vals.extend(w_vals)
        if ls_v is not None:
            all_vals.append(ls_v)

    ax.set_xticks(range(len(stacks)))
    ax.set_xticklabels([f"#{i+1}\n({idx})" for i, idx in enumerate(stacks)])
    ax.set_xlabel("stack (quality rank)")
    ax.set_ylabel(r"$D_{\mathrm{norm}}$ recovered")
    ax.set_title(story_title)
    ymin, ymax = min(all_vals), max(all_vals)
    pad = (ymax - ymin) * 0.20 + 1e-9
    ax.set_ylim(ymin - pad, ymax + pad * 1.4)
    ax.set_xlim(-0.5, len(stacks) - 0.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92,
              edgecolor="#bbb", borderpad=0.4)
    sns.despine(ax=ax)


def panel_6C_std_scatter(ax):
    """Scatter of weak_std vs strong_std for all 10 stacks; y=x diagonal.
    All points below diagonal = weak is more stable on every stack."""
    pts = []
    for cond in ("32ww", "56ww"):
        for idx in TOP5_STACKS[cond]:
            s = load_per_stack_D(cond, "strong").get(idx, [])
            w = load_per_stack_D(cond, "weak").get(idx, [])
            if not s or not w:
                continue
            pts.append((cond, idx, float(np.std(s)), float(np.std(w))))

    # plot reference y=x line first
    s_all = [p[2] for p in pts]
    w_all = [p[3] for p in pts]
    lim = max(max(s_all), max(w_all)) * 1.15
    ax.plot([0, lim], [0, lim], color="#888", linestyle="--", lw=1.2,
            alpha=0.85, zorder=1, label="weak = strong")

    # color by condition
    cond_colors = {"32ww": "#4477aa", "56ww": "#aa4477"}
    cond_labels = {"32ww": "32ww (fast)", "56ww": "56ww (slow)"}
    for cond in ("32ww", "56ww"):
        xs = [p[2] for p in pts if p[0] == cond]
        ys = [p[3] for p in pts if p[0] == cond]
        ax.scatter(xs, ys, s=110, color=cond_colors[cond],
                   edgecolor="white", linewidth=0.8,
                   label=cond_labels[cond], zorder=4)

    # annotations
    n_below = sum(1 for _, _, s, w in pts if w < s)
    n_total = len(pts)
    mean_red = float(np.mean([100 * (1 - w / s) for _, _, s, w in pts]))
    ax.text(0.04, 0.95,
            f"{n_below}/{n_total} stacks\nweak more stable\n(mean −{mean_red:.0f}% std)",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888",
                      alpha=0.95, lw=0.6))
    ax.text(lim * 0.65, lim * 0.95, "weak less stable",
            color="#888", fontsize=8.5, style="italic")
    ax.text(lim * 0.45, lim * 0.05, "weak more stable", ha="left",
            color="#222", fontsize=8.5, style="italic")

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("strong-form cross-seed std of $D_{\\mathrm{norm}}$")
    ax.set_ylabel("weak-form cross-seed std of $D_{\\mathrm{norm}}$")
    ax.set_title("All 10 stacks: weak vs strong cross-seed std")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_aspect("equal")
    sns.despine(ax=ax)


def make_figure6():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panel_6_per_stack_strip(
        axes[0], "32ww",
        "32ww — fast diffusion, 5 stacks (top quality-ranked)",
    )
    panel_6_per_stack_strip(
        axes[1], "56ww",
        "56ww — slow diffusion, 5 stacks (weak consistently tighter)",
    )
    panel_6C_std_scatter(axes[2])

    for ax, label in zip(axes, ["A", "B", "C"]):
        ax.text(-0.14, 1.07, label, transform=ax.transAxes,
                fontsize=15, fontweight="bold")

    fig.suptitle("Experimental FRAP validation across 10 stacks (5 per condition)",
                 fontsize=13, y=1.00, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.87, bottom=0.14,
                        wspace=0.32)
    fig.savefig("../figures/main/figure5_real.png", bbox_inches="tight")
    print("wrote figures/figure5_real.png")


if __name__ == "__main__":
    _setup_style()
    np.random.seed(0)
    make_figure5()
    make_figure6()
