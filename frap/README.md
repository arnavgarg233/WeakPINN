# FRAP weak-form PINN

**Cross-domain validation of the weak-form PINN methodology on FRAP microscopy.**

FRAP application within the [WeakPINN](../README.md) methodology repository. The PDE is the linear reaction-diffusion equation `∂_t c = D ∇² c − k c`; the recovery target is the molecular diffusion coefficient `D` from a noisy postbleach image stack.

## Headline results

**Synthetic (D_norm_true = 0.01975, 3 seeds per cell):**

| Block | Strong D-MAE | Weak D-MAE | Strong % err | Weak % err |
|---|---|---|---|---|
| clean | 6.59e-4 | 4.10e-4 | 3.34 | 2.08 |
| noise_low | 9.67e-4 | 2.79e-4 | 4.90 | 1.41 |
| noise_med | 2.12e-3 | 5.93e-4 | 10.75 | 3.00 |
| noise_high | 3.39e-3 | 2.91e-4 | 17.17 | 1.48 |
| **POOLED NOISY** | **2.16e-3** | **3.88e-4** | **10.94** | **1.96** |

**Pooled noisy result: weak-form reduces D-MAE by 82.1 % relative to strong-form.**

**Experimental DeepFRAP (top-5 stacks per condition, 5 seeds per stack per method, 100 runs total):**

| Condition | n stacks | Weak σ (mean over stacks) | Strong σ (mean over stacks) | Stack-level σ reduction range | Mean stack-level reduction |
|---|---|---|---|---|---|
| 32ww (fast) | 5 | 0.00211 | 0.00322 | 0.5 – 69.8% | **31.4%** |
| 56ww (slow) | 5 | 0.00031 | 0.00095 | **59.6 – 77.0%** | **66.7%** |
| **Both conditions (10 stacks)** | 10 | 0.00121 | 0.00208 | — | **49.0%** mean / **60.7%** median |

**Weak-form is more stable than strong-form on 10/10 experimental stacks. The effect is particularly large and consistent in the slow-diffusion 56ww condition (60–77% std reduction on all 5 stacks).** Reconstruction MSE is comparable across methods. Per-stack values are in [`results/supplement_tables/S-FRAP3_experimental_aggregate.csv`](results/supplement_tables/S-FRAP3_experimental_aggregate.csv).

## Selected hyperparameters (pilot-calibrated, frozen before main matrix)

`λ_phys` weights tuned on the clean synthetic development stack at the matched 10 k-step training length:

- **`λ_strong = 1.0`** (chosen as conservative D-recovery baseline; lowest val_mse at `λ = 0.1` but with worse D recovery)
- **`λ_weak = 30.0`**

Noisy synthetic and experimental stacks were not used during hyperparameter selection.

## Layout

```
frap/
├── src/
│   ├── models.py             PINN_FRAP architecture (D, optional k learnable)
│   └── losses.py             strong-form + weak-form residuals (boundary-vanishing test functions)
├── scripts/
│   ├── audit_mat_shapes.py            Phase 2: data audit
│   ├── quality_screen.py              Phase 2.5: rank 40 candidate DeepFRAP stacks
│   ├── extract_ls_reference.py        LS-fit D anchor extraction
│   ├── generate_synthetic_frap.py     Phase 3: bounded-domain FD simulator (Neumann BC)
│   ├── preprocess.py                  Phase 4: normalize, coords, chronological split
│   ├── convert_real_mat_to_npz.py     Branch-A: real .mat -> .npz
│   ├── train_frap_pinn.py             Phase 7: trainer (data / strong / weak)
│   ├── select_lambda.py               Phase 8: per-method λ selection rule
│   ├── tune_lambda.sh                 Phase 8a: 5k pilot tuning
│   ├── tune_weak_10k.sh               Phase 8b: 10k weak retune
│   ├── tune_strong_10k.sh             Phase 8c: 10k strong retune (symmetry)
│   ├── run_main_experiment.sh         Phase 9: 66-run main matrix
│   ├── qc_synthetic.py                synthetic data QC figure
│   ├── make_main_figures.py           Figures 4 + 5 (main text, FRAP)
│   ├── make_supplement_figures.py     Supplement figures S-FRAP1, S-FRAP2
│   └── make_supplement_tables.py      Supplement tables S-FRAP1..S-FRAP5
├── tests/
│   ├── test_models.py
│   └── test_losses.py                 analytic-solution residual checks
├── data/                              small synthetic .npz only
│   └── synthetic_{clean,noise_{low,med,high}}.npz
├── config/                            lambda.json + units.json
└── results/                           curated CSVs (per-stack quality, LS reference, supplement tables, B6 test-fn sweep)

(figures live at the repo top-level: ../figures/main + ../figures/supplement)
```

## Reproducing the matrix

```bash
# (One-time) Download DeepFRAP experimental data from Zenodo (1.2 GB, into data/deepfrap/)
curl -L -o data/deepfrap.zip "https://zenodo.org/api/records/3874218/files/deepfrap.zip/content"
unzip -q data/deepfrap.zip -d data/deepfrap && rm data/deepfrap.zip

# 1) Audit + select stacks
python scripts/audit_mat_shapes.py $(find data/deepfrap -name "*.mat")
python scripts/quality_screen.py
python scripts/extract_ls_reference.py

# 2) Convert the two selected real stacks to .npz (creates data/real_{32,56}ww.npz)
python scripts/convert_real_mat_to_npz.py

# 3) Generate synthetic stacks (small npz files are shipped in data/; this overwrites)
python scripts/generate_synthetic_frap.py

# 4) Tune lambda at the matched 10k training length
bash scripts/tune_strong_10k.sh
bash scripts/tune_weak_10k.sh
# (writes results/supplement_tables/S-FRAP4_lambda_tuning.csv + config/lambda.json)

# 5) Run the full 66-run main matrix (~3.5 h on M2 Pro MPS)
bash scripts/run_main_experiment.sh

# 6) Build figures + tables
python scripts/make_main_figures.py
python scripts/make_supplement_figures.py
python scripts/make_supplement_tables.py
```

Per-run training JSONs are not shipped in the repo (66 × ~5 KB each); they are produced by step 5 and consumed by steps 6.

## Data sources

- **Experimental stacks**: Röding et al. 2020, *Journal of Microscopy*. Zenodo DOI [10.5281/zenodo.3874218](https://doi.org/10.5281/zenodo.3874218).
- **Synthetic stacks**: generated by `scripts/generate_synthetic_frap.py` (bounded-domain FD with Neumann boundary, calibrated photon-shot noise + PSF + imaging-bleach).

## Citation

See the top-level [WeakPINN README](../README.md). Experimental data citation: Röding et al. 2020 (DeepFRAP), Zenodo DOI 10.5281/zenodo.3874218.
