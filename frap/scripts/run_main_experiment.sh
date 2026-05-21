#!/bin/bash
# Phase 9 - Main experiment matrix (synthetic clean + 3 noise regimes + Branch A real).
# Reads tuned lambdas from config/lambda.json (written by scripts/tune_lambda.sh).
#
# Total runs: 9 (clean) + 27 (noise) + 30 (real) = 66
# Estimated wall clock on M2 MPS: ~12-15 hr. Launch overnight.

set -euo pipefail

PY=${PY:-python}
cd "$(dirname "$0")/.."

if [ ! -f config/lambda.json ]; then
  echo "!! config/lambda.json missing. Run scripts/tune_lambda.sh first."
  exit 1
fi

LAMBDA_STRONG=$("$PY" -c "import json; print(json.load(open('config/lambda.json'))['strong'])")
LAMBDA_WEAK=$("$PY" -c "import json; print(json.load(open('config/lambda.json'))['weak'])")
echo "LAMBDA_STRONG=$LAMBDA_STRONG  LAMBDA_WEAK=$LAMBDA_WEAK"

mkdir -p results

run_one() {
  local stack="$1" method="$2" seed="$3" steps="$4" learn_k_flag="$5" tag="$6"
  local lam
  if [ "$method" == "strong" ]; then lam="$LAMBDA_STRONG"
  elif [ "$method" == "weak" ]; then lam="$LAMBDA_WEAK"
  else lam=0.0
  fi
  local out="results/${tag}_${method}_seed${seed}.json"
  if [ -f "$out" ]; then
    echo "   skip existing $out"
    return
  fi
  echo "   run $tag/$method/seed$seed lambda=$lam"
  "$PY" scripts/train_frap_pinn.py \
    --stack "$stack" \
    --method "$method" \
    --seed "$seed" \
    --steps "$steps" \
    --lambda_phys "$lam" \
    --init_D 0.05 \
    $learn_k_flag \
    --out "$out"
}

# Clean synthetic (sanity)
echo ">>> Clean synthetic"
for method in data strong weak; do
  for seed in 0 1 2; do
    run_one data/synthetic_clean.npz "$method" "$seed" 10000 "" clean
  done
done

# Noisy synthetic, three regimes
echo ">>> Noisy synthetic"
for noise in low med high; do
  for method in data strong weak; do
    for seed in 0 1 2; do
      run_one "data/synthetic_noise_${noise}.npz" "$method" "$seed" 10000 "" "noise${noise}"
    done
  done
done

# Real stacks (Branch A) - use --learn_k because imaging-bleach over 26.5 s is non-negligible
echo ">>> Real (Branch A)"
for cond in 32ww 56ww; do
  for method in data strong weak; do
    for seed in 0 1 2 3 4; do
      run_one "data/real_${cond}.npz" "$method" "$seed" 15000 "--learn_k" "real_${cond}"
    done
  done
done

echo ">>> Main matrix done."
