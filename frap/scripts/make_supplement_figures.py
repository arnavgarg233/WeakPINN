"""Build supplement Figures S-FRAP1..S-FRAP5 for the combined paper.

Outputs to figures/supplement/.

S-FRAP1  Data audit + representative prebleach/bleach/postbleach frames + quality screen
S-FRAP2  Lambda tuning sweeps (10k clean synthetic) for strong and weak
S-FRAP3  Reconstruction MSE across noise levels - weak does NOT win on MSE
S-FRAP4  Residual diagnostics (median strong + weak residuals per condition)
S-FRAP5  All 66 Phase 9 runs - per-seed D scatter grouped by stack/noise/method
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.io as sio
import seaborn as sns


D_NORM_TRUE = 0.01975
SUPP = Path("../figures/supplement")
SUPP.mkdir(parents=True, exist_ok=True)

# Tune JSONs live in an off-repo backup; main repo only ships the aggregate CSVs.
# Both paths are checked so the figure script remains reproducible.
TUNE_FALLBACK_DIR = Path(os.environ.get("WEAKPINN_PHASE9_BACKUP", ""))  # set this env var to enable fallback

METHOD_COLORS = {"data": "#888888", "strong": "#d95f02", "weak": "#1b9e77"}
METHOD_LABELS = {"data": "data-only", "strong": "strong-form", "weak": "weak-form"}


def _setup_style():
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


def _find_tune_files(pattern: str) -> list[Path]:
    """Locate tune JSONs in repo results/ or off-repo backup; backup wins if both."""
    backup_hits = sorted(TUNE_FALLBACK_DIR.glob(pattern))
    if backup_hits:
        return backup_hits
    return sorted(Path("results").glob(pattern))


def load_D(pat, method):
    return [json.load(open(f))["D_recovered"]
            for f in sorted(glob.glob(f"results/{pat}"))
            if f"_{method}_" in f]


def load_mse(pat, method):
    return [json.load(open(f))["val_mse"]
            for f in sorted(glob.glob(f"results/{pat}"))
            if f"_{method}_" in f]


def load_resid(pat, method, which):
    return [json.load(open(f))[f"median_{which}_residual"]
            for f in sorted(glob.glob(f"results/{pat}"))
            if f"_{method}_" in f]


def save_fig(fig, name):
    fig.savefig(SUPP / f"{name}.png", bbox_inches="tight")
    print(f"wrote {SUPP / name}.png")


# =============================================================================
# S-FRAP1  Data audit
# =============================================================================

def figure_sfrap1():
    sel = json.load(open("results/selected_stacks.json"))
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1], hspace=0.55, wspace=0.4)

    # Top panel: quality screen composite ranks
    ax = fig.add_subplot(gs[0, :])
    with open("results/quality_screen.csv") as f:
        rows = list(csv.DictReader(f))
    rows_32 = [r for r in rows if r["condition"] == "32ww"]
    rows_56 = [r for r in rows if r["condition"] == "56ww"]
    rows_32.sort(key=lambda r: int(r["index"]))
    rows_56.sort(key=lambda r: int(r["index"]))
    score_32 = [float(r["composite_score"]) for r in rows_32]
    score_56 = [float(r["composite_score"]) for r in rows_56]
    x = np.arange(20)
    width = 0.4
    ax.bar(x - width/2, score_32, width, label="32ww (n=20 candidates)", color="#4477aa", alpha=0.8)
    ax.bar(x + width/2, score_56, width, label="56ww (n=20 candidates)", color="#aa4477", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{i+1:02d}" for i in range(20)], fontsize=7)
    ax.set_xlabel("candidate index"); ax.set_ylabel("composite score")
    ax.set_title("Quality screen across 40 candidate DeepFRAP stacks (composite of 5 metrics; lower is better)")
    sel_idx_32 = sel["32ww"]["dataset_index"]
    sel_idx_56 = sel["56ww"]["dataset_index"]
    # mark selected
    ax.axvline(sel_idx_32 - 1 - width/2, color="#4477aa", ls=":", alpha=0.6)
    ax.axvline(sel_idx_56 - 1 + width/2, color="#aa4477", ls=":", alpha=0.6)
    ax.text(sel_idx_32 - 1 - width/2, max(max(score_32), max(score_56)) * 0.95,
            f"selected\n32ww #{sel_idx_32:02d}", color="#4477aa", ha="center", fontsize=8)
    ax.text(sel_idx_56 - 1 + width/2, max(max(score_32), max(score_56)) * 0.85,
            f"selected\n56ww #{sel_idx_56:02d}", color="#aa4477", ha="center", fontsize=8)
    ax.legend(loc="upper right", frameon=False)

    # Middle row: 32ww frames
    m32 = sio.loadmat(f"data/deepfrap/validation_exp/data/{sel['32ww']['file']}",
                      struct_as_record=False, squeeze_me=True)
    e32 = m32["experiment"]
    # Bottom row: 56ww frames
    m56 = sio.loadmat(f"data/deepfrap/validation_exp/data/{sel['56ww']['file']}",
                      struct_as_record=False, squeeze_me=True)
    e56 = m56["experiment"]

    for row, (exp_struct, label) in enumerate([(e32, "32ww"), (e56, "56ww")], start=1):
        for col, (phase, t_idx, title_suffix) in enumerate([
            ("prebleach", -1, "prebleach"),
            ("bleach", -1, "bleach"),
            ("postbleach", 5, "postbleach t≈1.3s"),
            ("postbleach", 30, "postbleach t≈8s"),
            ("postbleach", 60, "postbleach t≈16s"),
            ("postbleach", 99, "postbleach t≈26s"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            img = getattr(exp_struct, phase).image_data[:, :, t_idx].astype(np.float32) / 65535
            ax.imshow(img, cmap="gray", vmin=0.2, vmax=0.95)
            ax.set_title(f"{label}  {title_suffix}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("DeepFRAP data audit and selected experimental stacks",
                 fontsize=12, fontweight="bold", y=0.995)
    save_fig(fig, "S-FRAP1_data_audit")


# =============================================================================
# S-FRAP2  Lambda tuning sweeps (10k)
# =============================================================================

def figure_sfrap2():
    """Per-method λ tuning sweep on clean synthetic, 10k steps.

    Cleaner layout: val_mse on left (log y), D on right (linear). The selected
    λ is highlighted by a vertical shaded band, with the value annotated INSIDE
    the panel's lower-right (not at the top where it collided with the legend).
    The D_true reference is shown both as a horizontal dotted line AND as a
    legend entry, avoiding the right-edge text that was clipping the axis.
    """
    selected = json.load(open("config/lambda.json"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    for ax, method, pattern in [
        (axes[0], "strong", "tune10k_strong_lam*.json"),
        (axes[1], "weak", "tune10k_weak_lam*.json"),
    ]:
        files = _find_tune_files(pattern)
        records = sorted(
            (json.load(open(p)) for p in files),
            key=lambda r: float(r["lambda_phys"]),
        )
        lams = np.array([float(r["lambda_phys"]) for r in records])
        mses = np.array([float(r["val_mse"]) for r in records])
        Ds = np.array([float(r["D_recovered"]) for r in records])
        col = METHOD_COLORS[method]

        ax2 = ax.twinx()
        ax2.grid(False)

        # Selected lambda band first (zorder 0)
        sel_lam = float(selected[method])
        ax.axvspan(sel_lam / 1.5, sel_lam * 1.5, color="#cccccc", alpha=0.30, zorder=0)
        ax.axvline(sel_lam, color="#444", linestyle="-", alpha=0.5, linewidth=1.0, zorder=1)

        # val_mse (left) and D (right)
        sns.lineplot(x=lams, y=mses, ax=ax, color=col, marker="o",
                     markersize=9, linewidth=2.2, label="val_mse (left axis)",
                     zorder=4)
        sns.lineplot(x=lams, y=Ds, ax=ax2, color="#444", marker="s",
                     markersize=7, linewidth=1.6, linestyle="--",
                     label=r"$D$ recovered (right axis)", zorder=4)

        # D_true reference - dotted; rolled into the right-axis legend below
        ax2.axhline(D_NORM_TRUE, color="#2ca02c", linestyle=":", alpha=0.85,
                    linewidth=1.4, zorder=2,
                    label=rf"$D_{{\mathrm{{norm,true}}}} = {D_NORM_TRUE}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_{\mathrm{phys}}$")
        ax.set_ylabel("val_mse")
        ax2.set_ylabel(r"$D_{\mathrm{norm}}$ recovered")
        ax.set_title(f"{METHOD_LABELS[method]} sweep")

        # Selected-lambda annotation: parked in lower-left (away from the
        # data trajectory which descends-then-rises in mse vs lambda)
        ax.text(0.04, 0.06, f"selected  λ = {sel_lam:g}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10.5, fontweight="semibold", color="#222",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#888", alpha=0.95, lw=0.7))

        # Headroom for the legend at top
        ax.set_ylim(mses.min() * 0.5, mses.max() * 3.5)

        # Merge legends from both axes into one in upper-right
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        leg = ax.legend(h1 + h2, l1 + l2, loc="upper right",
                        frameon=True, facecolor="white",
                        edgecolor="#bbb", framealpha=0.95,
                        fontsize=8.5, handlelength=2.4, labelspacing=0.4)
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, right=False)

    fig.suptitle("Per-method $\lambda_{\mathrm{phys}}$ tuning on clean synthetic (10k steps)",
                 fontsize=12.5, y=1.00, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.945, top=0.86, bottom=0.13, wspace=0.36)
    save_fig(fig, "S-FRAP2_lambda_tuning")


# =============================================================================
# S-FRAP3  Reconstruction MSE across noise levels
# =============================================================================

def figure_sfrap3():
    noise_levels = ["clean", "noiselow", "noisemed", "noisehigh"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(noise_levels))
    for method in ["data", "strong", "weak"]:
        means = []
        stds = []
        for nl in noise_levels:
            mses = load_mse(f"{nl}_*.json", method)
            means.append(np.mean(mses)); stds.append(np.std(mses))
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=4,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xticks(x); ax.set_xticklabels(noise_levels)
    ax.set_yscale("log")
    ax.set_xlabel("photon-noise level")
    ax.set_ylabel("held-out val_mse (log scale)")
    ax.set_title("Reconstruction MSE across noise (mean $\\pm$ std over 3 seeds)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.text(0.5, -0.02,
             "All three methods reach the same Poisson noise floor as photon count drops; "
             "weak-form does NOT have lower MSE than strong-form.",
             ha="center", fontsize=9, style="italic", color="#555")
    fig.suptitle("Reconstruction error is comparable across methods",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "S-FRAP3_reconstruction_mse")


# =============================================================================
# S-FRAP4  Residual diagnostics
# =============================================================================

def figure_sfrap4():
    blocks = [("clean", "clean_*.json"),
              ("noiselow", "noiselow_*.json"),
              ("noisemed", "noisemed_*.json"),
              ("noisehigh", "noisehigh_*.json"),
              ("real_32ww", "real_32ww_*.json"),
              ("real_56ww", "real_56ww_*.json")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, which in zip(axes, ("strong", "weak")):
        x = np.arange(len(blocks))
        width = 0.35
        s_med = [np.median(load_resid(pat, "strong", which)) for _, pat in blocks]
        w_med = [np.median(load_resid(pat, "weak", which)) for _, pat in blocks]
        ax.bar(x - width/2, s_med, width, label="strong-form trained model",
               color=METHOD_COLORS["strong"], alpha=0.85)
        ax.bar(x + width/2, w_med, width, label="weak-form trained model",
               color=METHOD_COLORS["weak"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in blocks], rotation=30, ha="right")
        ax.set_ylabel(f"median |{which} residual|")
        ax.set_yscale("log")
        ax.set_title(f"{which}-form residual evaluated on both methods")
        ax.legend(loc="best", frameon=False, fontsize=8)
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.suptitle("Fair residual diagnostics: both metrics on both models",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "S-FRAP4_residual_diagnostics")


# =============================================================================
# S-FRAP5  All 66 runs per-seed D recovery scatter
# =============================================================================

def figure_sfrap5():
    blocks = [("clean", "clean_*.json"),
              ("noise_low", "noiselow_*.json"),
              ("noise_med", "noisemed_*.json"),
              ("noise_high", "noisehigh_*.json"),
              ("real_32ww", "real_32ww_*.json"),
              ("real_56ww", "real_56ww_*.json")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                             gridspec_kw={"width_ratios": [4, 2]})
    # Left: synthetic only (the 4 synthetic blocks); shared y-scale with D_norm_true
    ax = axes[0]
    pos = 0
    xticks = []
    xticklabels = []
    for label, pat in blocks[:4]:
        for j, method in enumerate(("data", "strong", "weak")):
            Ds = load_D(pat, method)
            x_jit = np.random.uniform(-0.1, 0.1, len(Ds))
            ax.scatter([pos + j] * len(Ds) + x_jit, Ds,
                       color=METHOD_COLORS[method], s=50, alpha=0.85,
                       label=METHOD_LABELS[method] if pos == 0 else "")
            xticks.append(pos + j)
            xticklabels.append(method[0])  # first letter of method
        pos += 4
    ax.axhline(D_NORM_TRUE, color="green", ls="--", lw=1.2, alpha=0.7,
               label=f"true D = {D_NORM_TRUE}")
    ax.set_xticks([1 + 4 * i for i in range(4)])
    ax.set_xticklabels(["clean", "low", "med", "high"])
    ax.set_xlabel("noise level (each cluster: data / strong / weak)")
    ax.set_ylabel(r"$D_{\mathrm{norm}}$ recovered")
    ax.set_title("Synthetic blocks (3 seeds per cell)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Right: real only
    ax = axes[1]
    pos = 0
    ls_ref = json.load(open("results/ls_reference_D.json"))
    for label, pat in blocks[4:]:
        cond = label.replace("real_", "")
        for j, method in enumerate(("data", "strong", "weak")):
            Ds = load_D(pat, method)
            x_jit = np.random.uniform(-0.1, 0.1, len(Ds))
            ax.scatter([pos + j] * len(Ds) + x_jit, Ds,
                       color=METHOD_COLORS[method], s=50, alpha=0.85)
        # LS anchor
        anchor = ls_ref[cond]["D_norm"]
        ax.plot([pos - 0.4, pos + 2.4], [anchor, anchor],
                color="#444", ls="--", lw=1, alpha=0.7)
        ax.text(pos + 2.5, anchor, f"LS  {anchor:.3f}", fontsize=8, va="center")
        pos += 4
    ax.set_xticks([1 + 4 * i for i in range(2)])
    ax.set_xticklabels(["32ww", "56ww"])
    ax.set_yscale("log")
    ax.set_xlabel("condition (each cluster: data / strong / weak)")
    ax.set_ylabel(r"$D_{\mathrm{norm}}$ recovered (log)")
    ax.set_title("Real blocks (5 seeds per cell)")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.suptitle("All 66 Phase 9 runs, per-seed",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "S-FRAP5_all_runs_scatter")


if __name__ == "__main__":
    _setup_style()
    np.random.seed(0)
    # Only 2 supplement figures retained for submission.
    # S-FRAP3 (reconstruction MSE), S-FRAP4 (residual diagnostics),
    # and S-FRAP5 (all-runs scatter) were converted to tables instead;
    # their builder functions are kept in this file as dead code in case
    # we want to regenerate the figures for reviewer response.
    figure_sfrap1()
    figure_sfrap2()
