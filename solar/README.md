# Flare-PINN

**Weak-form physics-informed neural network for operational solar flare forecasting.**

Solar-domain application within the [WeakPINN](../README.md) methodology repository. Flare-PINN couples a convolutional-GRU magnetogram encoder, a FiLM-conditioned implicit neural field, and a weak-form resistive MHD induction constraint to produce calibrated multi-horizon flare probabilities (6 h / 12 h / 24 h).

## Headline results (24 h horizon, paper-lock checkpoint, seed 1234 @ step 44 k)

| Model | TSS | POD | FAR | Brier | D2C threshold |
|---|---|---|---|---|---|
| **Flare-PINN (weak-form)** | **0.798** | 0.945 | 0.906 | 0.041 | 0.237 |
| Strong-form PINN | 0.765 | 0.890 | 0.897 | 0.039 | 0.241 |
| DeFN (5 seeds, mean ± SD) | 0.730 ± 0.035 | 0.851 | 0.890 | 0.019 | per-seed |
| No-physics ablation | 0.746 | 0.879 | 0.904 | 0.040 | 0.246 |

**Multi-horizon Flare-PINN (paper-lock):** TSS 0.826 (6 h) / 0.833 (12 h) / 0.798 (24 h).

**3-seed robustness:** 0.817 ± 0.011 (6 h) / 0.811 ± 0.021 (12 h) / 0.790 ± 0.007 (24 h).

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
