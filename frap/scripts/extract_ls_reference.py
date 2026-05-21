"""Extract LS-fit reference parameters from DeepFRAP's results_{32,56}ww.mat.

Per DeepFRAP MATLAB convention (validated against
data/deepfrap/validation_exp/analyze_results_{32,56}ww.m and run_{32,56}ww.m):

  sys_param_hat_rc[:, 0] = D in pixel^2 / s
  sys_param_hat_rc[:, 2] = C0 (initial concentration)
  sys_param_hat_rc[:, 3] = alpha (mobile fraction-like parameter)
  remaining columns are fixed-bound nuisance parameters

We convert column 0 to physical D (m^2 / s and um^2 / s) by multiplying by
pixel_size^2 where pixel_size = 7.598e-7 m (consistent across all 40 stacks).

USE STRICTLY AS A SANITY-CHECK ANCHOR. We are NOT claiming our PINN
recovers the same number as the published LS fit; we report D in
normalized units (D_norm) for the weak-vs-strong comparison and only
cross-walk to physical D for reader orientation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io


PIXEL_SIZE_M = 7.598e-7
D_COL = 0
C0_COL = 2
ALPHA_COL = 3


def _load(path: Path) -> dict:
    m = scipy.io.loadmat(path, squeeze_me=False)
    arr = m["sys_param_hat_rc"]
    ss = m["ss_rc"].ravel()
    D_pixel2_per_s = arr[:, D_COL].astype(np.float64)
    D_m2_per_s = D_pixel2_per_s * PIXEL_SIZE_M ** 2
    D_um2_per_s = D_m2_per_s * 1e12
    return {
        "n_stacks": int(arr.shape[0]),
        "ss_rc_median": float(np.median(ss)),
        "D_pixel2_per_s": D_pixel2_per_s.tolist(),
        "D_m2_per_s": D_m2_per_s.tolist(),
        "D_um2_per_s": D_um2_per_s.tolist(),
        "C0": arr[:, C0_COL].astype(float).tolist(),
        "alpha": arr[:, ALPHA_COL].astype(float).tolist(),
        "summary": {
            "D_um2_per_s_min": float(D_um2_per_s.min()),
            "D_um2_per_s_max": float(D_um2_per_s.max()),
            "D_um2_per_s_median": float(np.median(D_um2_per_s)),
            "D_um2_per_s_mean": float(D_um2_per_s.mean()),
            "D_um2_per_s_std": float(D_um2_per_s.std()),
            "alpha_median": float(np.median(arr[:, ALPHA_COL])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/deepfrap/validation_exp"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ls_reference_D.json"),
    )
    args = parser.parse_args()

    out: dict = {
        "pixel_size_m": PIXEL_SIZE_M,
        "convention_source": "data/deepfrap/validation_exp/analyze_results_56ww.m line 12: D_hat_LS = sys_param_hat_rc(:, 1) * pixel_size^2",
        "column_index_used": {"D": D_COL, "C0": C0_COL, "alpha": ALPHA_COL},
        "note": "Use only as sanity-check anchor. Not a target for PINN to match.",
        "conditions": {},
    }

    for cond in ("32ww", "56ww"):
        path = args.results_dir / f"results_{cond}.mat"
        if not path.exists():
            print(f"!! missing: {path}")
            continue
        out["conditions"][cond] = _load(path)
        s = out["conditions"][cond]["summary"]
        print(
            f"{cond}  n={out['conditions'][cond]['n_stacks']}  "
            f"D = {s['D_um2_per_s_median']:.4f} um^2/s (median, IQR-ish range "
            f"{s['D_um2_per_s_min']:.4f} - {s['D_um2_per_s_max']:.4f})  "
            f"alpha_med={s['alpha_median']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f">> wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
