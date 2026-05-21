"""Phase 11 - analysis and 4-panel figure for the FRAP weak-form PINN experiment.

Reads JSON result files written by `scripts/train_frap_pinn.py` (Agent A) plus
the unit / lambda / LS-fit reference config files, and renders:

    Panel A: example FRAP frame (synthetic-noise, mid-recovery snapshot)
    Panel B: held-out MSE vs noise level, per method
    Panel C: D_recovered distribution per method, per condition (boxplot)
    Panel D: fair residual comparison - both residual metrics on both PINN methods

Outputs:
    figures/frap_panel.pdf
    figures/frap_panel.png

Run from the frap/ root (only after S4 is posted by Agent A):

    python scripts/analyze.py [--results-dir results] [--data-dir data] \\
                              [--config-dir config] [--figures-dir figures] \\
                              [--out-basename frap_panel]

Design notes:
  - Strict typing via pydantic: every JSON load is validated into a RunResult
    or one of the small config models. Unknown extra JSON fields are tolerated
    (`extra="ignore"`) so analysis does not break if Agent A's training output
    gains diagnostic columns later.
  - Robust to missing/incomplete inputs: any individual JSON that fails to load
    or validate is skipped with a warning; missing config files yield empty
    overlays (LS-fit anchor, lambda annotations) instead of crashing.
  - CPU only. Agent B never touches MPS; this script reads result JSONs and
    .npz frames, nothing else.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """One training-run JSON written by scripts/train_frap_pinn.py.

    Matches the result dict assembled at the bottom of `train()`:
        method, seed, lambda_phys, init_D, stack,
        D_recovered, k_recovered, true_D,
        val_mse, median_strong_residual, median_weak_residual,
        elapsed_sec, losses

    `losses` (a list of per-log-step dicts) is not used by the figure and is
    dropped via `extra="ignore"` so future schema additions don't break us.
    """

    model_config = ConfigDict(extra="ignore")

    method: str
    seed: int
    lambda_phys: float
    init_D: float
    stack: str
    D_recovered: float
    k_recovered: float = 0.0
    true_D: Optional[float] = None
    val_mse: float
    median_strong_residual: float
    median_weak_residual: float
    elapsed_sec: float = 0.0


class LambdaConfig(BaseModel):
    """config/lambda.json - per-method physics-loss weight from Phase 8 tuning."""

    model_config = ConfigDict(extra="ignore")

    strong: float
    weak: float


class UnitsConfig(BaseModel):
    """config/units.json - unit conversion constants (Phase 8.5)."""

    model_config = ConfigDict(extra="ignore")

    pixel_size_um: float = 0.7598
    time_per_frame_s: float = 0.265
    n_pixels: int = 256
    n_frames: int = 100
    bleach_radius_um: float = 15.0
    D_phys_to_D_norm_factor_per_um2_per_s: float = 1.400868e-3


class LSReferenceRow(BaseModel):
    """One row of the LS-fit anchor table (results/ls_reference_D.csv or .json).

    Provides D in both physical (um^2/s) and normalized (PINN-comparable) units
    so Panel C can overlay a horizontal anchor line in the same axis as
    PINN-recovered D. Compared QUALITATIVELY only - LS-fit is not a training
    target (Phase 8.5).
    """

    model_config = ConfigDict(extra="ignore")

    condition: str
    D_phys_um2_per_s: float
    D_norm: float


# ---------------------------------------------------------------------------
# Loaders - all robust to absent / corrupt files
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[RunResult]:
    """Read every *.json in results_dir; return validated RunResult records.

    Files that fail to parse or fail schema validation are skipped with a
    warning. Files that look like *non*-training-result JSONs (no D_recovered
    field, e.g. ls_reference_D.json or selected_stacks.json) are quietly
    ignored.
    """
    runs: list[RunResult] = []
    if not results_dir.exists():
        warnings.warn(f"results dir {results_dir} does not exist; no runs loaded")
        return runs
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open() as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"skip {path.name}: read error ({e})")
            continue
        if not isinstance(raw, dict) or "D_recovered" not in raw:
            continue
        try:
            run = RunResult.model_validate(raw)
        except ValidationError as e:
            warnings.warn(f"skip {path.name}: schema mismatch ({e.error_count()} errors)")
            continue
        runs.append(run)
    return runs


def load_lambda(config_dir: Path) -> Optional[LambdaConfig]:
    """Return parsed lambda config or None (with warning) if not yet written.

    Annotation-only: the figure stays valid without it; missing-S3 means we
    just omit the lambda values from the figure title.
    """
    path = config_dir / "lambda.json"
    if not path.exists():
        warnings.warn(f"{path} not found - S3 not yet posted; lambda values will be omitted")
        return None
    try:
        with path.open() as f:
            return LambdaConfig.model_validate_json(f.read())
    except (json.JSONDecodeError, ValidationError) as e:
        warnings.warn(f"{path} unreadable: {e}")
        return None


def load_units(config_dir: Path) -> UnitsConfig:
    """Return parsed unit config, falling back to Phase 8.5 defaults."""
    path = config_dir / "units.json"
    if not path.exists():
        warnings.warn(f"{path} not found - using Phase 8.5 defaults")
        return UnitsConfig()
    try:
        with path.open() as f:
            return UnitsConfig.model_validate_json(f.read())
    except (json.JSONDecodeError, ValidationError):
        warnings.warn(f"{path} unreadable; falling back to defaults")
        return UnitsConfig()


def load_ls_reference(results_dir: Path) -> list[LSReferenceRow]:
    """Read LS-fit anchor rows from CSV (preferred) or JSON (fallback).

    Accepts either format because the two Agent A sessions wrote slightly
    different artefacts (.csv vs .json) during the parallel-execution mishap.
    Missing-or-empty input yields an empty list and just suppresses the
    Panel C anchor line.
    """
    rows: list[LSReferenceRow] = []
    csv_path = results_dir / "ls_reference_D.csv"
    if csv_path.exists():
        try:
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        rows.append(LSReferenceRow.model_validate(r))
                    except ValidationError:
                        continue
            if rows:
                return rows
        except OSError as e:
            warnings.warn(f"{csv_path} read error: {e}")
    json_path = results_dir / "ls_reference_D.json"
    if json_path.exists():
        try:
            with json_path.open() as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            raw = None
        if isinstance(raw, dict):
            for cond, vals in raw.items():
                if not isinstance(vals, dict):
                    continue
                try:
                    rows.append(LSReferenceRow.model_validate({"condition": cond, **vals}))
                except ValidationError:
                    continue
    if not rows:
        warnings.warn(f"no LS-fit reference found in {results_dir} - Panel C anchor will be omitted")
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


METHODS: tuple[str, ...] = ("data", "strong", "weak")
METHOD_LABELS: dict[str, str] = {"data": "data-only", "strong": "strong-form", "weak": "weak-form"}
METHOD_COLORS: dict[str, str] = {"data": "#7f7f7f", "strong": "#d62728", "weak": "#1f77b4"}
NOISE_LEVELS: tuple[str, ...] = ("clean", "low", "med", "high")


def _runs_with_stack(runs: list[RunResult], substr: str) -> list[RunResult]:
    return [r for r in runs if substr in r.stack]


def _runs_for(runs: list[RunResult], substr: str, method: str) -> list[RunResult]:
    return [r for r in runs if substr in r.stack and r.method == method]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def panel_a_example_frame(ax: plt.Axes, data_dir: Path) -> None:
    """Show a mid-recovery frame from the most-noisy available synthetic stack.

    Prefers `synthetic_noise_med.npz` for visual punch; falls back through low
    and clean if the heavier stacks aren't generated yet.
    """
    candidates = [
        data_dir / "synthetic_noise_med.npz",
        data_dir / "synthetic_noise_low.npz",
        data_dir / "synthetic_noise_high.npz",
        data_dir / "synthetic_clean.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            arr = np.load(path)
            stack = arr["stack"]
        except (OSError, KeyError) as e:
            warnings.warn(f"{path} unreadable: {e}")
            continue
        T = stack.shape[0]
        frame_idx = T // 3
        ax.imshow(stack[frame_idx], cmap="magma", origin="lower")
        ax.set_title(f"A) Example frame ({path.stem}, t={frame_idx}/{T - 1})")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.text(0.5, 0.5, "no synthetic stack found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("A) Example frame")
    ax.set_xticks([])
    ax.set_yticks([])


def panel_b_mse_vs_noise(ax: plt.Axes, runs: list[RunResult]) -> None:
    """Held-out MSE vs noise level, one line per method, mean +/- std over seeds."""
    x = np.arange(len(NOISE_LEVELS))
    has_any = False
    for method in METHODS:
        means: list[float] = []
        stds: list[float] = []
        for noise in NOISE_LEVELS:
            stack_substr = "synthetic_clean.npz" if noise == "clean" else f"synthetic_noise_{noise}.npz"
            subset = _runs_for(runs, stack_substr, method)
            if not subset:
                means.append(float("nan"))
                stds.append(float("nan"))
                continue
            vals = np.array([r.val_mse for r in subset], dtype=np.float64)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=0)))
            has_any = True
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.errorbar(
            x, means_arr, yerr=stds_arr,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            marker="o", capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_LEVELS)
    ax.set_xlabel("noise level")
    ax.set_ylabel("held-out MSE (chronological split)")
    ax.set_title("B) Held-out MSE vs noise")
    if has_any:
        # log scale only after positive data exists; matplotlib chokes on empty log axes
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no synthetic results yet", ha="center", va="center", transform=ax.transAxes)


def panel_c_D_recovery(
    ax: plt.Axes,
    runs: list[RunResult],
    ls_rows: list[LSReferenceRow],
) -> None:
    """Boxplot of recovered D_norm per (condition, method).

    Synthetic boxes are pooled across noise levels (so the box width
    encodes synthetic-noise robustness); real-data boxes are separated by
    condition (32ww vs 56ww) since the true D differs by ~10x. Reference
    lines: synthetic true D = 0.05, LS-fit anchors from Phase 8.5.
    """
    box_data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []

    syn_runs = _runs_with_stack(runs, "synthetic")
    real_32 = _runs_with_stack(runs, "32ww")
    real_56 = _runs_with_stack(runs, "56ww")

    for method in METHODS:
        group = [r.D_recovered for r in syn_runs if r.method == method]
        if group:
            box_data.append(np.array(group))
            labels.append(f"syn\n{METHOD_LABELS[method]}")
            colors.append(METHOD_COLORS[method])
    for cond_label, cond_runs in (("32ww", real_32), ("56ww", real_56)):
        for method in METHODS:
            group = [r.D_recovered for r in cond_runs if r.method == method]
            if group:
                box_data.append(np.array(group))
                labels.append(f"{cond_label}\n{METHOD_LABELS[method]}")
                colors.append(METHOD_COLORS[method])

    if not box_data:
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C) D recovery (normalized D)")
        return

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    if syn_runs:
        ax.axhline(0.05, color="k", linestyle="--", linewidth=0.7, label="synthetic true D = 0.05")
    for row in ls_rows:
        ax.axhline(
            row.D_norm, color="0.4", linestyle=":", linewidth=0.7,
            label=f"LS-fit {row.condition} (D_norm = {row.D_norm:.3g})",
        )
    ax.set_ylabel("D_norm (PINN parameter)")
    ax.set_title("C) D recovery (normalized D)")
    ax.tick_params(axis="x", labelsize=7)
    handles, labs = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labs, fontsize=7, loc="best")


def panel_d_fair_residuals(ax: plt.Axes, runs: list[RunResult]) -> None:
    """Both residual metrics (strong, weak) on both PINN methods (strong, weak).

    Reviewer-fairness panel: a strong-form-trained model that lowers its
    matched (strong) residual but has a large weak residual is fitting noise.
    A weak-form-trained model with a small weak residual and a comparable
    strong residual is consistent with low-noise physics. We never report
    only the matched residual on the matched method.

    Data-only runs are excluded (no physics term => the comparison is undefined).
    """
    pinn_runs = [r for r in runs if r.method in ("strong", "weak")]
    if not pinn_runs:
        ax.text(0.5, 0.5, "no PINN results yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D) Fair residual comparison")
        return

    methods = ("strong", "weak")
    metric_keys = ("median_strong_residual", "median_weak_residual")
    metric_labels = ("|strong residual|", "|weak residual|")
    width = 0.35
    x = np.arange(len(methods))

    for j, (metric, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        vals: list[float] = []
        errs: list[float] = []
        for method in methods:
            subset = [getattr(r, metric) for r in pinn_runs if r.method == method]
            if subset:
                arr = np.abs(np.array(subset, dtype=np.float64))
                vals.append(float(arr.mean()))
                errs.append(float(arr.std(ddof=0)))
            else:
                vals.append(float("nan"))
                errs.append(0.0)
        ax.bar(x + (j - 0.5) * width, vals, width=width, yerr=errs, capsize=3, label=mlabel, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("median |residual| (pooled across runs)")
    ax.set_title("D) Fair residual comparison")
    # log scale only if at least one positive bar - empty/zero data would crash the locator
    positive = [r.median_strong_residual for r in pinn_runs] + [r.median_weak_residual for r in pinn_runs]
    if any(v > 0 for v in positive):
        ax.set_yscale("log")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def build_figure(
    runs: list[RunResult],
    ls_rows: list[LSReferenceRow],
    lambda_cfg: Optional[LambdaConfig],
    data_dir: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panel_a_example_frame(axes[0, 0], data_dir)
    panel_b_mse_vs_noise(axes[0, 1], runs)
    panel_c_D_recovery(axes[1, 0], runs, ls_rows)
    panel_d_fair_residuals(axes[1, 1], runs)

    suptitle = "FRAP weak-form vs strong-form PINN - cross-domain stress test"
    if lambda_cfg is not None:
        suptitle += f"\nlambda_strong = {lambda_cfg.strong:g}, lambda_weak = {lambda_cfg.weak:g}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase 11 analysis for FRAP weak-form PINN")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--config-dir", type=Path, default=Path("config"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    p.add_argument("--out-basename", type=str, default="frap_panel")
    args = p.parse_args(argv)

    runs = load_results(args.results_dir)
    print(f"loaded {len(runs)} training-run JSONs from {args.results_dir}", file=sys.stderr)

    ls_rows = load_ls_reference(args.results_dir)
    lambda_cfg = load_lambda(args.config_dir)
    _units = load_units(args.config_dir)  # consumed by the supplement table; figure uses D_norm only

    fig = build_figure(runs, ls_rows, lambda_cfg, args.data_dir)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.figures_dir / f"{args.out_basename}.pdf"
    png_path = args.figures_dir / f"{args.out_basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path} and {png_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
