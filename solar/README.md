# Flare-PINN

**Weak-form physics-informed neural network for operational solar flare forecasting.**

Solar-domain application within the [WeakPINN](../README.md) methodology repository. Flare-PINN couples a convolutional-GRU magnetogram encoder, a FiLM-conditioned implicit neural field, and a weak-form resistive MHD induction constraint to produce calibrated multi-horizon flare probabilities (6 h / 12 h / 24 h).

## Headline results (chronological test set, corrected integrated weak-moment arm)

| Arm | Scored step | TSS 6 h | TSS 12 h | TSS 24 h | Grand mean |
|---|---|---|---|---|---|
| **Integrated weak moments** | 44,000 | **0.7896 ± 0.0387** | **0.8141 ± 0.0158** | **0.7816 ± 0.0099** | **0.7951 ± 0.0173** |
| Data-only, capped | 40,000 | 0.7123 | 0.6809 | 0.7456 | 0.7130 |
| Difference | | +0.0772 | +0.1332 | +0.0360 | **+0.0822** |

Three seeds, 5,716 test windows, checkpoint and thresholds fixed on validation.

| Arm @ step 44,000 | Physics | Grand-mean test TSS |
|---|---|---|
| Data-only, capped at 40k | no | 0.7130 |
| Continuation control | no | 0.7155 |
| **Integrated weak moments** | **yes** | **0.7951** |

Continued training without the physics term recovers +0.0025 of the +0.0822
gain; the physics term carries +0.0796 at matched training length. The control
scored at step 46,000 gives 0.7176, so this does not depend on where in the
selection window it is read.

**The physics term buys generalization, not fit.** On validation, adjacent to
training, the control scores 0.8048 and the physics arm 0.7918, so the term
costs −0.0130. On the test split, two years further into the declining cycle
with events three times rarer, the same term is worth +0.0796. The swing is
+0.0926, which is regularizer behaviour rather than added capacity. It also
explains the earlier matched continuation: its +0.0022 increment was measured
entirely in distribution and at roughly half this dose, the regime where the
term does least.

Receipts, per-checkpoint score files and the prediction recorded before the run
are in [`../recovery/no_physics_control/`](../recovery/no_physics_control/).

> **Superseded.** Earlier releases headlined 24 h TSS 0.798, multi-horizon
> 0.826 / 0.833 / 0.798, and a 3-seed spread of 0.817 ± 0.011 at 6 h, from a
> local-integrand configuration whose development used test-set information. The
> strong-form and DeFN comparisons published beside them were never re-run under
> the corrected protocol and are not matched comparators.


Test evaluation uses **frozen validation D2C (Distance-to-Corner) thresholds** to prevent any leakage of test-set information into the operating-point choice.

## Dataset (80 / 5 / 15 chronological split, after consolidation)

| Split | Period | Windows | M+ flares | Positive rate |
|---|---|---|---|---|
| Train | Jan 2011 – Aug 2015 | 28,405 | 1,081 | 3.81 % |
| Validation | Aug – Dec 2015 | 1,905 | 98 | 5.14 % |
| Test | Dec 2015 – Dec 2017 | 5,716 | 91 | 1.59 % |

The pre-consolidation train CSV listed 30,481 windows but 61 HARPs (range H725–H997, 2,076 windows) lacked `.npz` magnetogram bundles after the consolidation step and were silently skipped by the dataloader. 28,405 is the actual training-set size the model saw; the deposit and SI are reconciled to this number.

## Repository layout

```
solar/
├── src/                       model code (Flare-PINN architecture)
│   ├── models/                Conv-GRU encoder, implicit field, multi-horizon head
│   ├── data/                  dataloader, windowing
│   ├── configs/               training configs (paper-lock + ablations)
│   ├── baselines/             strong-form variant, DeFN reimpl, no-physics ablation
│   ├── utils/
│   └── train.py               main training entrypoint
├── data_scripts/              JSOC fetch (drms-direct), HARP-NOAA mapping, windowing
├── tools/
│   ├── training/              multi-seed run shells
│   ├── validation/            D2C threshold, bootstrap, ROC metrics
│   ├── analysis/              lead-time, physics-residual diagnostics
│   ├── visualization/         calibration, confusion matrices, ROC-PR-TSS
│   └── defn/                  DeFN feature-table + audit
└── final_results/             curated paper metrics (CSV + JSON; PNG figures)
```

## Reproducibility quickstart

End-to-end reproduction requires the SHARP cutout data from JSOC (≈ 200 GB) and a CUDA / MPS GPU for the multi-day training schedule. For verification, the curated `final_results/` directory contains all per-seed metrics, bootstrap intervals, and physics-residual CSVs used in the manuscript.

```bash
# 1) Fetch SHARP series (requires JSOC email registration; see data_scripts/README)
python data_scripts/fetch_sharp.py --series cea_720s --period 2011-01:2017-12

# 2) Build chronological windows (48 h lookback, 6 h stride, 49 frames @ 1 h cadence)
python data_scripts/build_windows.py --out data/windows.parquet

# 3) Train the weak-form Flare-PINN (seed 1234, paper-lock config)
python -m src.train --config src/configs/flare_pinn_final.yaml --seed 1234

# 4) Compute test metrics under frozen-validation D2C thresholds
python tools/validation/rerun_test_d2c_metrics.sh

# 5) Reproduce paper figures
bash tools/visualization/regenerate_all_figures.sh
```

## Pre-trained checkpoints

The paper-lock checkpoint (seed 1234 @ step 44 k) and the 3-seed and 5-seed multi-seed runs are deposited on figshare alongside the manuscript. Code in this repository will load them via `--checkpoint` at evaluation time.

DeFN baseline pre-trained predictions for the five paper seeds {24, 10, 100, 42, 123} are also on the figshare deposit (`outputs/baselines/defn/`); the DeFN training schedule itself takes ≈ 15 min/seed on CPU.

## Citation

Solar-side citation is the methodology paper (see top-level [README](../README.md)). The DeFN reimplementation follows [Nishizuka et al. 2018](https://doi.org/10.3847/1538-4357/aabd31); the official TF code is at [github.com/komeisugiura/defn18](https://github.com/komeisugiura/defn18).
