"""Phase 2.5 quality screen across all 40 experimental FRAP stacks.

Computes 5 fixed quality metrics per stack, ranks them, and recommends the
best stack from each molecular-weight condition (32ww and 56ww) as the
representative real-data inputs for Phase 9 Branch A runs.

Metrics (all higher = worse, except bleach_depth where higher = better):

  1. sat_frac          : fraction of postbleach pixels at uint16 max
                         (65535). Indicates clipped highlights.
  2. prebleach_cv      : coefficient of variation of frame-mean intensity
                         across the 10 prebleach frames. Indicates laser /
                         focus instability before any physics.
  3. recovery_nonsmooth: integrated absolute second difference of mean
                         postbleach intensity over time, normalized by the
                         recovery amplitude. High = jagged / noisy curve.
  4. drift             : mean absolute frame-to-frame difference of pixels
                         OUTSIDE the bleach circle, normalized by prebleach
                         intensity. High = sample drift / focus jitter.
  5. bleach_depth      : (prebleach_mean - first_postbleach_mean) /
                         prebleach_mean, computed inside the bleach circle.
                         HIGHER is better (deeper bleach = stronger signal).

A composite score is the unweighted mean of per-metric ranks
(1 = best, 40 = worst). For bleach_depth ranks are reversed.
Lower composite = better. Top pick per condition is the lowest composite.

CSV output: results/quality_screen.csv
Pretty summary printed to stdout and tee'd to results/quality_screen_summary.txt.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class StackMetrics(BaseModel):
    """Per-stack quality metrics (raw values, before ranking)."""
    model_config = ConfigDict(frozen=True)

    path: str
    condition: str
    index: int
    pixel_size_m: float
    time_frame_s: float
    bleach_radius_pixels: float
    sat_frac: float
    prebleach_cv: float
    recovery_nonsmooth: float
    drift: float
    bleach_depth: float


@dataclass(frozen=True)
class StackArrays:
    """Raw image_data arrays + bleach circle, ready for metric computation."""
    prebleach: NDArray  # (H, W, T_pre)
    postbleach: NDArray  # (H, W, T_post)
    bleach_mask: NDArray  # (H, W) bool, True inside circle


def _load_stack(path: Path) -> tuple[StackArrays, dict]:
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    exp = m["experiment"]
    pre = exp.prebleach.image_data.astype(np.float32)
    post = exp.postbleach.image_data.astype(np.float32)
    H, W = pre.shape[:2]
    pixel_size = float(exp.postbleach.pixel_size_x)
    bleach_r_px = 0.5 * float(exp.bleach.bleach_size_x) / pixel_size
    cx = W / 2.0 + float(exp.bleach.bleach_position_y) / pixel_size
    cy = H / 2.0 - float(exp.bleach.bleach_position_x) / pixel_size
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    bleach_mask = rr < bleach_r_px
    meta = {
        "pixel_size_m": pixel_size,
        "time_frame_s": float(exp.postbleach.time_frame),
        "bleach_radius_pixels": bleach_r_px,
        "bit_depth": int(exp.postbleach.bit_depth),
    }
    return StackArrays(prebleach=pre, postbleach=post, bleach_mask=bleach_mask), meta


def _compute_metrics(path: Path, arrays: StackArrays, meta: dict) -> StackMetrics:
    bit_max = float((1 << int(meta["bit_depth"])) - 1)
    post = arrays.postbleach
    pre = arrays.prebleach
    bleach_mask = arrays.bleach_mask
    outside_mask = ~bleach_mask

    sat_frac = float((post >= bit_max).mean())

    pre_frame_means = pre.mean(axis=(0, 1))
    prebleach_cv = float(pre_frame_means.std() / max(pre_frame_means.mean(), 1e-8))

    post_mean_curve = post.mean(axis=(0, 1)).astype(np.float64)
    amp = float(np.ptp(post_mean_curve))
    if amp <= 0:
        recovery_nonsmooth = float("inf")
    else:
        d2 = np.diff(post_mean_curve, n=2)
        recovery_nonsmooth = float(np.abs(d2).sum() / amp)

    drift_per_frame = np.abs(np.diff(post[:, :, :], axis=2))
    drift_outside = drift_per_frame[outside_mask].mean()
    pre_outside = pre[outside_mask].mean()
    drift = float(drift_outside / max(pre_outside, 1e-8))

    pre_in_bleach = pre[bleach_mask].mean()
    first_post_in_bleach = post[:, :, 0][bleach_mask].mean()
    bleach_depth = float((pre_in_bleach - first_post_in_bleach) / max(pre_in_bleach, 1e-8))

    name = path.stem
    if "32ww" in name:
        cond = "32ww"
        idx = int(name.split("_")[-1])
    elif "56ww" in name:
        cond = "56ww"
        idx = int(name.split("_")[-1])
    else:
        cond = "unknown"
        idx = -1

    return StackMetrics(
        path=str(path),
        condition=cond,
        index=idx,
        pixel_size_m=meta["pixel_size_m"],
        time_frame_s=meta["time_frame_s"],
        bleach_radius_pixels=meta["bleach_radius_pixels"],
        sat_frac=sat_frac,
        prebleach_cv=prebleach_cv,
        recovery_nonsmooth=recovery_nonsmooth,
        drift=drift,
        bleach_depth=bleach_depth,
    )


def _rank_ascending(values: list[float]) -> list[float]:
    """Return ranks (1=smallest, len(values)=largest). NaN/inf get max rank."""
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    ranks = np.full(arr.shape, float(len(arr)), dtype=np.float64)
    if finite_mask.any():
        order = np.argsort(arr[finite_mask], kind="stable")
        finite_ranks = np.empty(finite_mask.sum(), dtype=np.float64)
        finite_ranks[order] = np.arange(1, finite_mask.sum() + 1, dtype=np.float64)
        ranks[finite_mask] = finite_ranks
    return ranks.tolist()


def composite_scores(metrics: list[StackMetrics]) -> list[float]:
    """Return per-stack composite score: mean of 5 per-metric ranks, lower=better."""
    n = len(metrics)
    sat_r = _rank_ascending([m.sat_frac for m in metrics])
    pcv_r = _rank_ascending([m.prebleach_cv for m in metrics])
    rs_r = _rank_ascending([m.recovery_nonsmooth for m in metrics])
    dr_r = _rank_ascending([m.drift for m in metrics])
    bd_inv_r = _rank_ascending([-m.bleach_depth for m in metrics])
    out = []
    for i in range(n):
        out.append((sat_r[i] + pcv_r[i] + rs_r[i] + dr_r[i] + bd_inv_r[i]) / 5.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/deepfrap/validation_exp/data"),
    )
    parser.add_argument("--out-csv", type=Path, default=Path("results/quality_screen.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/quality_screen_summary.txt"))
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("frap_*ww_*.mat"))
    if not paths:
        print(f"!! no .mat files found under {args.data_dir}")
        return 1

    print(f">> screening {len(paths)} stacks under {args.data_dir}")
    metrics_list: list[StackMetrics] = []
    for path in paths:
        arr, meta = _load_stack(path)
        m = _compute_metrics(path, arr, meta)
        metrics_list.append(m)
        print(
            f"   {m.condition} idx={m.index:03d}  "
            f"sat={m.sat_frac:.4f}  preCV={m.prebleach_cv:.4f}  "
            f"nonsmooth={m.recovery_nonsmooth:.4f}  drift={m.drift:.4f}  "
            f"bleach_depth={m.bleach_depth:.4f}"
        )

    scores = composite_scores(metrics_list)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "condition", "index",
            "pixel_size_m", "time_frame_s", "bleach_radius_pixels",
            "sat_frac", "prebleach_cv", "recovery_nonsmooth",
            "drift", "bleach_depth", "composite_score",
        ])
        for m, s in zip(metrics_list, scores):
            w.writerow([
                m.path, m.condition, m.index,
                m.pixel_size_m, m.time_frame_s, m.bleach_radius_pixels,
                m.sat_frac, m.prebleach_cv, m.recovery_nonsmooth,
                m.drift, m.bleach_depth, s,
            ])
    print(f">> wrote {args.out_csv} ({len(metrics_list)} rows)")

    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("FRAP stack quality screen - top picks per molecular-weight condition")
    lines.append("=" * 96)
    for cond in ("32ww", "56ww"):
        cond_idx = [i for i, m in enumerate(metrics_list) if m.condition == cond]
        if not cond_idx:
            lines.append(f"\n!! no stacks found for {cond}")
            continue
        cond_idx.sort(key=lambda i: scores[i])
        lines.append(f"\n{cond}  (n={len(cond_idx)})  top 5 by composite rank:")
        lines.append(
            f"  {'idx':>4s}  {'comp':>6s}  {'sat':>8s}  {'preCV':>8s}  "
            f"{'nonsmooth':>9s}  {'drift':>8s}  {'depth':>8s}"
        )
        for i in cond_idx[:5]:
            m = metrics_list[i]
            lines.append(
                f"  {m.index:>4d}  {scores[i]:>6.2f}  {m.sat_frac:>8.5f}  "
                f"{m.prebleach_cv:>8.5f}  {m.recovery_nonsmooth:>9.4f}  "
                f"{m.drift:>8.5f}  {m.bleach_depth:>8.4f}"
            )
        winner = cond_idx[0]
        lines.append(f"  >>>  PICK: frap_{cond}_{metrics_list[winner].index:03d}.mat (composite={scores[winner]:.2f})")

    lines.append("\n" + "=" * 96)
    summary = "\n".join(lines)
    print(summary)
    args.out_summary.write_text(summary + "\n")
    print(f">> wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
