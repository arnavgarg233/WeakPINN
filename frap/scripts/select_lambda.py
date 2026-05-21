"""Phase 8 - Read tune_*.json files, score them, select best lambda per method.

D_norm_true (the PINN's TRUE normalized D) is computed PER STACK from the
.npz metadata, NOT hardcoded. For a synthetic stack:

    D_norm_true = D_phys * 2 * T_phys / L^2

where D_phys is the simulator coefficient, T_phys = (T - 1) * dt is the
physical recovery duration, and L = 2 is the PINN's normalized spatial
extent ([-1, 1]). This matches scripts/preprocess.physical_D_to_normalized.

For data/synthetic_clean.npz (D_phys=0.05, dt=0.01, T=80):
    D_norm_true = 0.05 * 2 * 0.79 / 4 = 0.01975

Selection rule (per method):
  1. Drop runs with NaN/inf in val_mse or D_recovered (mark as failed).
  2. From the remaining stable runs, find the best val_mse.
  3. Filter to runs with val_mse <= val_mse_tol * best_val_mse for that method.
  4. Among those, choose the run with smallest |D - D_norm_true| / D_norm_true.
  5. Ties broken by lower val_mse, then by lower lambda (cheaper to constrain).

Writes:
  results/lambda_tuning_summary.csv  : one row per tune run
  config/lambda.json                 : {"strong": <lam>, "weak": <lam>}
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


METHOD_RE = re.compile(r"tune(?:10k)?_(strong|weak)_lam([0-9eE\.\+\-]+)\.json$")
L_NORM = 2.0  # PINN normalized spatial extent is [-1, 1]


def compute_D_norm_true(stack_path: str) -> float | None:
    """Compute the PINN-coord true D for a synthetic stack. Returns None if real data."""
    p = Path(stack_path)
    if not p.exists():
        return None
    d = np.load(p)
    if "D" not in d.files:
        return None
    D_phys = float(d["D"])
    if D_phys <= 0:
        return None
    dt = float(d["dt"])
    T = int(d["stack"].shape[0])
    T_phys = (T - 1) * dt
    return D_phys * 2.0 * T_phys / (L_NORM * L_NORM)


def parse_one(path: Path, override_true_D: float | None) -> dict:
    j = json.load(open(path))
    name = path.name
    m = METHOD_RE.match(name)
    if m is None:
        raise ValueError(f"can't parse method/lambda from {name}")
    method = m.group(1)
    lam = float(j["lambda_phys"])
    D = float(j["D_recovered"])
    val_mse = float(j["val_mse"])
    nan_flag = (math.isnan(D) or math.isinf(D)
                or math.isnan(val_mse) or math.isinf(val_mse))
    # Compute D_norm_true from stack metadata, unless caller overrode it
    if override_true_D is not None:
        D_norm_true = override_true_D
    else:
        D_norm_true = compute_D_norm_true(j["stack"])
    if D_norm_true and D_norm_true > 0:
        rel_D_err = abs(D - D_norm_true) / D_norm_true
    else:
        rel_D_err = float("inf")
    return {
        "method": method,
        "lambda_phys": lam,
        "val_mse": val_mse,
        "D_recovered": D,
        "D_norm_true": D_norm_true,
        "D_phys_in_stack": j.get("true_D"),
        "rel_D_err": rel_D_err,
        "median_strong_residual": float(j["median_strong_residual"]),
        "median_weak_residual": float(j["median_weak_residual"]),
        "elapsed_sec": float(j["elapsed_sec"]),
        "failed": bool(nan_flag),
        "source": str(path),
    }


def select_for_method(rows: list[dict], val_mse_tol: float) -> dict | None:
    stable = [r for r in rows if not r["failed"]]
    if not stable:
        return None
    best_mse = min(r["val_mse"] for r in stable)
    threshold = best_mse * val_mse_tol
    candidates = [r for r in stable if r["val_mse"] <= threshold]
    # Sort: (rel_D_err asc, val_mse asc, lambda_phys asc)
    candidates.sort(key=lambda r: (r["rel_D_err"], r["val_mse"], r["lambda_phys"]))
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--true-D-override", type=float, default=None,
                    help="(optional) override D_norm_true; default = derive per-stack from .npz metadata")
    ap.add_argument("--val-mse-tol", type=float, default=2.0,
                    help="filter to runs with val_mse <= tol * best_val_mse")
    ap.add_argument("--strong-pattern", type=str, default="tune_strong_lam*.json",
                    help="glob (relative to results-dir) for strong-method tune files")
    ap.add_argument("--weak-pattern", type=str, default="tune_weak_lam*.json",
                    help="glob (relative to results-dir) for weak-method tune files")
    ap.add_argument("--out-csv", type=Path, default=Path("results/lambda_tuning_summary.csv"))
    ap.add_argument("--out-config", type=Path, default=Path("config/lambda.json"))
    args = ap.parse_args()

    strong_paths = sorted(args.results_dir.glob(args.strong_pattern))
    weak_paths = sorted(args.results_dir.glob(args.weak_pattern))
    if not strong_paths and not weak_paths:
        raise SystemExit(f"no tune files matched {args.strong_pattern} or {args.weak_pattern}")
    print(f"strong files ({len(strong_paths)}): {[p.name for p in strong_paths]}")
    print(f"weak   files ({len(weak_paths)}): {[p.name for p in weak_paths]}")
    rows = [parse_one(p, args.true_D_override) for p in (strong_paths + weak_paths)]

    # Announce the D_norm_true used (assumes all rows share one — true for Phase 8)
    distinct_true = sorted({round(r["D_norm_true"], 6) for r in rows if r["D_norm_true"]})
    if distinct_true:
        print(f"D_norm_true (derived from stack metadata, L=2): {distinct_true}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "lambda_phys", "val_mse", "D_recovered",
                  "D_norm_true", "D_phys_in_stack", "rel_D_err",
                  "median_strong_residual", "median_weak_residual",
                  "elapsed_sec", "failed", "source"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    # Pretty print per-method table
    for method in ["strong", "weak"]:
        method_rows = sorted([r for r in rows if r["method"] == method],
                             key=lambda r: r["lambda_phys"])
        if not method_rows:
            print(f"\n{method}: no runs")
            continue
        print(f"\n=== {method} ===")
        print(f"  {'lambda':>10s}  {'val_mse':>10s}  {'D':>8s}  "
              f"{'rel_D_err':>10s}  {'med_str_r':>10s}  {'med_wk_r':>10s}  "
              f"{'sec':>5s}  {'flag':>6s}")
        for r in method_rows:
            flag = "FAIL" if r["failed"] else "ok"
            print(f"  {r['lambda_phys']:>10.4g}  {r['val_mse']:>10.3e}  "
                  f"{r['D_recovered']:>8.4f}  {r['rel_D_err']:>10.3f}  "
                  f"{r['median_strong_residual']:>10.3e}  "
                  f"{r['median_weak_residual']:>10.3e}  "
                  f"{r['elapsed_sec']:>5.1f}  {flag:>6s}")

    # Pick best per method
    best = {}
    for method in ["strong", "weak"]:
        method_rows = [r for r in rows if r["method"] == method]
        chosen = select_for_method(method_rows, args.val_mse_tol)
        if chosen is None:
            print(f"\n!! {method}: no stable runs - cannot select lambda")
            best[method] = None
            continue
        best[method] = chosen["lambda_phys"]
        print(f"\n>> {method} selected: lambda={chosen['lambda_phys']}  "
              f"val_mse={chosen['val_mse']:.3e}  "
              f"D={chosen['D_recovered']:.4f}  "
              f"rel_D_err={chosen['rel_D_err']:.3f}")

    if all(v is not None for v in best.values()):
        args.out_config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_config, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nwrote {args.out_config}")
    else:
        print(f"\n!! NOT writing {args.out_config} - one or more methods had no stable run")

    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
