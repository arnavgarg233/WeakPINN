#!/bin/bash
# Phase 8.6 - Symmetry retune of strong-form lambda at 10k steps.
#
# Matches scripts/tune_weak_10k.sh so the supplement can report
# "both lambdas tuned at the 10k training length on clean synthetic".
#
# Writes results/tune10k_strong_lam{0.01,0.1,1,10,100}.json. The 5k
# tune_strong_*.json files are preserved but no longer used for selection.
#
# After this finishes, select_lambda.py is re-run with both 10k patterns.

set -euo pipefail

PY=${PY:-python}
cd "$(dirname "$0")/.."
mkdir -p results

STRONG_LAMBDAS=(0.01 0.1 1.0 10.0 100.0)

for lam in "${STRONG_LAMBDAS[@]}"; do
  out="results/tune10k_strong_lam${lam}.json"
  if [ -f "$out" ]; then
    echo ">> skip existing $out"
    continue
  fi
  echo ">> tune strong 10k lambda=$lam"
  "$PY" scripts/train_frap_pinn.py \
    --stack data/synthetic_clean.npz \
    --method strong \
    --seed 0 \
    --steps 10000 \
    --lambda_phys "$lam" \
    --init_D 0.05 \
    --out "$out"
done

echo
echo ">> selecting lambdas from 10k tunes (both methods)"
"$PY" scripts/select_lambda.py \
  --val-mse-tol 2.0 \
  --strong-pattern "tune10k_strong_lam*.json" \
  --weak-pattern "tune10k_weak_lam*.json" \
  --out-csv results/lambda_tuning_summary.csv \
  --out-config config/lambda.json
