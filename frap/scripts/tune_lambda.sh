#!/bin/bash
# Phase 8 - Per-method lambda_phys sweep on clean synthetic data (seed 0).
#
# Strong-form and weak-form residuals live on different numerical scales
# (weak is mean-integrated over Gaussian test functions, naturally 10^3-10^4
# smaller than per-point strong residual squared). Lambdas tuned independently.
#
# Grids:
#   strong: 0.01  0.1  1     10    100
#   weak:   1     10   100   1000  10000   100000
#
# Selection rule (see scripts/select_lambda.py):
#   Among lambdas with stable training and val_mse within 2x of best for that
#   method, choose the lambda with the lowest |D_recovered - 0.05| / 0.05.
#   On tie, lower val_mse wins.
#
# Run AFTER Agent B's patched S2 (boundary-vanishing test functions).

set -euo pipefail

PY=${PY:-python}
cd "$(dirname "$0")/.."
mkdir -p results

STRONG_LAMBDAS=(0.01 0.1 1.0 10.0 100.0)
WEAK_LAMBDAS=(1.0 10.0 100.0 1000.0 10000.0 100000.0)

for lam in "${STRONG_LAMBDAS[@]}"; do
  out="results/tune_strong_lam${lam}.json"
  if [ -f "$out" ]; then
    echo ">> skip existing $out"
    continue
  fi
  echo ">> tune strong lambda=$lam"
  "$PY" scripts/train_frap_pinn.py \
    --stack data/synthetic_clean.npz \
    --method strong \
    --seed 0 \
    --steps 5000 \
    --lambda_phys "$lam" \
    --init_D 0.05 \
    --out "$out"
done

for lam in "${WEAK_LAMBDAS[@]}"; do
  out="results/tune_weak_lam${lam}.json"
  if [ -f "$out" ]; then
    echo ">> skip existing $out"
    continue
  fi
  echo ">> tune weak lambda=$lam"
  "$PY" scripts/train_frap_pinn.py \
    --stack data/synthetic_clean.npz \
    --method weak \
    --seed 0 \
    --steps 5000 \
    --lambda_phys "$lam" \
    --init_D 0.05 \
    --out "$out"
done

echo
echo ">> selecting lambdas"
"$PY" scripts/select_lambda.py \
  --results-dir results \
  --true-D 0.05 \
  --val-mse-tol 2.0 \
  --out-csv results/lambda_tuning_summary.csv \
  --out-config config/lambda.json
