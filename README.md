# WeakPINN

**Weak-form physics-informed learning for noisy scientific image dynamics.**

This repository implements and validates a single methodological idea — moving derivatives off noisy observational data and onto smooth, boundary-vanishing test functions via integration by parts — across two scientifically distinct second-order PDE inverse problems:

| Domain | Application | Code | Underlying PDE |
|---|---|---|---|
| Heliophysics | **Flare-PINN** — operational solar flare forecasting | [`solar/`](solar/) | Resistive MHD induction |
| Microscopy | **FRAP weak-form PINN** — diffusion-coefficient recovery | [`frap/`](frap/) | Linear reaction-diffusion |

WeakPINN is a methodology repository, not a single application. Each domain has its own README, training pipeline, and reproducibility instructions.

## Headline results

**Flare-PINN (solar, chronological test set, corrected integrated weak-moment arm)**

| Arm | TSS 6 h | TSS 12 h | TSS 24 h | Grand mean |
|---|---|---|---|---|
| **Integrated weak moments** | **0.7896 ± 0.0387** | **0.8141 ± 0.0158** | **0.7816 ± 0.0099** | **0.7951 ± 0.0173** |
| Data-only, capped at 40 k | 0.7123 | 0.6809 | 0.7456 | 0.7130 |
| Difference | +0.0772 | +0.1332 | +0.0360 | **+0.0822** |

A two-stage recipe: 40,000 data-only steps, then an integrated weak-moment
continuation scored at step 44,000. A control trained for the same steps
**without** the physics term reaches only 0.7155, so continued training accounts
for +0.0025 of the gain and the physics term for +0.0796. Read in distribution
instead, on the validation window adjacent to training, the same term is worth
−0.0130: it buys out-of-distribution generalization rather than fit. Receipts in
[`recovery/no_physics_control/`](recovery/no_physics_control/).

> **Superseded.** Earlier releases headlined 24 h TSS 0.798 from a
> local-integrand configuration whose development used test-set information,
> beside a strong-form configuration never re-run under the corrected protocol.


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
  author  = {Author details withheld for double-anonymized review},
  journal = {Under review},
  year    = {2026}
}
```

The solar dataset draws on SDO/HMI SHARP cutouts; the FRAP experimental stacks are from the public [DeepFRAP dataset](https://doi.org/10.5281/zenodo.3874218) (Röding et al., 2020).

## License

MIT.
