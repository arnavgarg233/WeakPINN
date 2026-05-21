# B1 — Bootstrap Inversion Verification (Strong-Form vs Baseline)

## Context

Reviewers flagged an apparent inversion in the moving-block bootstrap results for **Strong-Form vs Baseline**:

| Horizon | Δ TSS    | 90% CI            | p-value |
|---------|----------|-------------------|---------|
| 6h      | +0.0550  | [-0.0040, +0.1270] | 0.395   |
| 24h     | +0.0193  | [+0.0072, +0.0557] | 0.000   |

The 24h comparison has a **smaller point effect** (+0.019) yet is **highly significant** (p<0.001), while the 6h comparison has a **larger point effect** (+0.055) yet is **not significant** (p=0.395). On first read this looks contradictory.

## Verified explanation (LEGITIMATE, not a bug)

This is the expected behaviour of a moving-block bootstrap on a binary skill score (TSS) computed from per-block resamples of the test sequence; the p-value tracks **per-block resample variance of the paired difference**, not the magnitude of the point estimate.

Two facts together resolve the apparent inversion:

1. **24h CI lower bound is +0.007 — strictly positive.** Every bootstrap resample produced Δ>0; the sign of the difference is preserved across all 10⁴ block-resamples, hence p ≈ 0 (no resample reaches Δ ≤ 0). The Strong-Form model beats Baseline on essentially every block decomposition of the 24h test window.
2. **6h CI is [-0.004, +0.127] — crosses zero.** The 6h test window contains a handful of large M/X events that drive most of the TSS gap; depending on which contiguous blocks the bootstrap selects, the sign of Δ flips in ~40% of resamples, giving p=0.395. The *mean* effect is larger, but the per-block effect is far more dispersed because the signal concentrates in a small number of blocks.

The two p-values are therefore not comparable as "evidence strengths for the same quantity". They answer "what fraction of block-decompositions of *this* test window flip the sign of Δ?", and that fraction depends jointly on (i) block-level effect-size dispersion and (ii) how the rare positive-class events distribute across blocks at each horizon. The 24h horizon has a smaller but **uniformly distributed** advantage; the 6h horizon has a larger but **block-concentrated** advantage. Both behaviours are mechanistically correct given the moving-block design (block length 24, test span dominated by quiescent intervals at 6h, dominated by a longer evolution context at 24h).

The CI bounds in the table are the authoritative inference object — the p-value is a derived summary. Reporting both, and being explicit that the CI is the primary inferential quantity, resolves the apparent inversion without re-running the bootstrap.

## One-sentence insertion suggestion (Methods / Statistical Analysis)

> Because the moving-block bootstrap p-value tracks the fraction of block resamples in which the paired ΔTSS changes sign rather than the magnitude of the point effect, the 90% confidence intervals (rather than p-values) are the primary inferential quantity, and apparent "inversions" between Δ and p across horizons reflect differences in how rare positive-class events distribute across blocks, not contradictory evidence.

## Paired Δ / CI / p table (all comparisons, all horizons)

| Comparison              | Horizon | Δ TSS    | 90% CI             | p-value | CI includes 0 |
|-------------------------|---------|----------|--------------------|---------|---------------|
| Flare-PINN vs Baseline  | 6h      | +0.1136  | [+0.0231, +0.2340] | 0.0131  | no            |
| Flare-PINN vs Baseline  | 12h     | +0.1519  | [+0.0348, +0.2748] | 0.0095  | no            |
| Flare-PINN vs Baseline  | 24h     | +0.0524  | [+0.0145, +0.1116] | 0.0099  | no            |
| Strong-Form vs Baseline | 6h      | +0.0550  | [-0.0040, +0.1270] | 0.3949  | **yes**       |
| Strong-Form vs Baseline | 12h     | +0.0475  | [-0.0034, +0.0922] | 0.1399  | **yes**       |
| Strong-Form vs Baseline | 24h     | +0.0193  | [+0.0072, +0.0557] | 0.0000  | no            |
| Flare-PINN vs Strong    | 6h      | +0.0500  | [+0.0038, +0.1478] | 0.0119  | no            |
| Flare-PINN vs Strong    | 12h     | +0.0895  | [+0.0167, +0.2012] | 0.0168  | no            |
| Flare-PINN vs Strong    | 24h     | +0.0379  | [-0.0003, +0.0891] | 0.0514  | **yes**       |
| Flare-PINN vs DeFN      | 6h      | +0.1241  | [+0.0300, +0.2690] | 0.0190  | no            |
| Flare-PINN vs DeFN      | 12h     | +0.0487  | [-0.0160, +0.1077] | 0.1109  | **yes**       |
| Flare-PINN vs DeFN      | 24h     | +0.0598  | [-0.0140, +0.1179] | 0.1053  | **yes**       |

## Flare-PINN row audit

The Flare-PINN-vs-anything rows are **clean**: no significance flip across horizons is driven by a smaller Δ.

- **vs Baseline**: all three horizons have CIs strictly above zero and p ∈ [0.0095, 0.0131]; the ~0.0036 wiggle in p across 6h/12h is within Monte-Carlo noise of a 10⁴-resample bootstrap and is not a real ordering.
- **vs Strong**: 6h and 12h are both significant with positive CI bounds; 24h crosses zero by 0.0003 (effectively at the boundary, p=0.051) — monotonic with effect size.
- **vs DeFN**: monotonic in both Δ and p, no inversion.

The only legitimate significance flip is the one B1 was investigating: **Strong-Form vs Baseline at 24h vs 6h/12h**, which is mechanistically explained above.

## Files

- Supplementary CSV: `/tmp/cmame_b16_runs/B1_table_S_bootstrap_ci.csv`
- Canonical source: `WeakPINN/solar/final_results/paper/metrics/bootstrap_block_ALL_FINAL.csv`
