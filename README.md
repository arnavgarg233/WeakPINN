# WeakPINN

**Weak-form physics-informed learning for noisy scientific image dynamics.**

This repository implements and validates a single methodological idea — moving derivatives off noisy observational data and onto smooth, boundary-vanishing test functions via integration by parts — across two scientifically distinct second-order PDE inverse problems:

| Domain | Application | Code | Underlying PDE |
|---|---|---|---|
| Heliophysics | **Flare-PINN** — operational solar flare forecasting | [`solar/`](solar/) | Resistive MHD induction |
| Microscopy | **FRAP weak-form PINN** — diffusion-coefficient recovery | [`frap/`](frap/) | Linear reaction-diffusion |

WeakPINN is a methodology repository, not a single application. Each domain has its own README, training pipeline, and reproducibility instructions.

## Headline results

**Flare-PINN (solar, 24 h horizon, paper-lock checkpoint)**

| Model | TSS | POD | FAR |
|---|---|---|---|
| **Flare-PINN (weak-form)** | **0.798** | 0.945 | 0.906 |
| Strong-form PINN | 0.765 | 0.890 | 0.897 |
| DeFN (5 seeds, mean ± SD) | 0.730 ± 0.039 | 0.851 | 0.890 |
| Benchmark (no physics) | 0.746 | 0.879 | 0.904 |

**FRAP cross-domain validation (synthetic, pooled noisy)**

| Method | D-MAE (norm units) | % err vs. D_norm_true |
|---|---|---|
| **Weak-form PINN** | **3.88e-4** | **1.96 %** |
| Strong-form PINN | 2.16e-3 | 10.94 % |
| Data-only ablation | — (no D gradient) | n/a |

Weak-form reduces D-MAE by **82.1 %** relative to strong-form across the noisy synthetic sweep. The same matched comparison on **10 experimental DeepFRAP stacks** (top-5 quality-ranked per molecular-weight condition, 5 training seeds per stack per method) shows weak-form is more stable than strong-form on **10/10 stacks** (median cross-seed std reduction **60.7%**, mean **49.0%**). The effect is particularly large in the slow-diffusion 56ww condition (59.6–77.0% reduction on all 5 stacks; mean −66.7%).

## Repository layout

```
WeakPINN/
├── figures/                 all publication figures (one canonical location)
│   ├── main/                figures 1-5 (main paper)
│   └── supplement/          SI figures (S-FRAP, solar SI, methodology benchmarks)
├── solar/                   Flare-PINN: solar flare forecasting
│   ├── src/                 model, training, evaluation, DeFN baseline
│   ├── data_scripts/        JSOC fetch, windowing, splits
│   ├── tools/               analysis, validation, viz, methodology_benchmarks
│   ├── final_results/       curated paper-ready metrics + B1/B4/B5 results,
│   │                        plus benchmarks/ (B2 heat-eq sweep, B3 autodiff)
│   └── README.md
├── frap/                    FRAP weak-form PINN cross-domain validation
│   ├── src/                 models + losses
│   ├── scripts/             data, training, analysis, figure generation
│   ├── tests/
│   ├── results/             curated supplement tables + B6 test-fn sensitivity
│   └── README.md
├── docs/                    shared methodology notes + B1-B6 synthesis
├── environment.yml          conda environment (PyTorch + MPS)
└── LICENSE
```

## Installation

```bash
conda env create -f environment.yml
conda activate weakpinn
```

Tested on macOS / Apple-silicon (MPS) and Linux CUDA. Solar pipeline assumes ≥ 32 GB system RAM during data prep; training itself runs on a single GPU.

## Reproducing the paper

Each subdirectory README walks through its domain's pipeline end-to-end:

- [`solar/README.md`](solar/README.md) — full Flare-PINN pipeline: JSOC fetch → windowing → training → evaluation
- [`frap/README.md`](frap/README.md) — FRAP weak-form: synthetic generation → λ tuning → main matrix → figures

Larger artifacts (windowed magnetogram data, training checkpoints, raw DeepFRAP unzip) are not in this repo; each domain README points to the corresponding figshare / Zenodo deposit.

## Citation

If you use WeakPINN in your research, please cite the methodology paper:

```bibtex
@article{weakpinn2026,
  title   = {Weak-Form Physics-Informed Learning for Noise-Limited Scientific Image Dynamics},
  author  = {Garg, Arnav},
  journal = {Under review},
  year    = {2026}
}
```

The solar dataset draws on SDO/HMI SHARP cutouts; the FRAP experimental stacks are from the public [DeepFRAP dataset](https://doi.org/10.5281/zenodo.3874218) (Röding et al., 2020).

## License

MIT.
