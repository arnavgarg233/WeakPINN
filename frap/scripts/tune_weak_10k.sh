#!/bin/bash
# Phase 8.5 - Focused weak-form lambda retune at 10k steps (matches Phase 9 length).
#
# The 5k weak tunes gave lambda=10000 with D=0.0196 (0.9% err), but at 10k
# steps that same lambda over-smooths to D~0.0113 (~43% err). Re-tuning at
# the actual training length to find a lambda that holds across the matrix.
#
# Strong sweep is NOT re-run - clean strong at 10k recovered D within ~5%
# at lambda=1, which the user has accepted.
#
# Writes:
#   results/tune10k_weak_lam{1,10,30,100,300,1000,3000,10000}.json
#
# Selection (downstream): scripts/select_lambda.py with the dual-pattern
# convention - strong from tune_strong_*.json, weak from tune10k_weak_*.json.

set -euo pipefail

PY=${PY:-python}
cd "$(dirname "$0")/.."
mkdir -p results

WEAK_LAMBDAS=(1.0 10.0 30.0 100.0 300.0 1000.0 3000.0 10000.0)

for lam in "${WEAK_LAMBDAS[@]}"; do
  out="results/tune10k_weak_lam${lam}.json"
  if [ -f "$out" ]; then
    echo ">> skip existing $out"
    continue
  fi
  echo ">> tune weak 10k lambda=$lam"
  "$PY" scripts/train_frap_pinn.py \
    --stack data/synthetic_clean.npz \
    --method weak \
    --seed 0 \
    --steps 10000 \
    --lambda_phys "$lam" \
    --init_D 0.05 \
    --out "$out"
done

echo
echo ">> selecting lambdas (strong from 5k tune_strong_*, weak from 10k tune10k_weak_*)"
"$PY" scripts/select_lambda.py \
  --val-mse-tol 2.0 \
  --strong-pattern "tune_strong_lam*.json" \
  --weak-pattern "tune10k_weak_lam*.json" \
  --out-csv results/lambda_tuning_summary.csv \
  --out-config config/lambda.json
