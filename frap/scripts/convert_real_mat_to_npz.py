"""Convert the two selected real FRAP .mat stacks to .npz that train_frap_pinn.py can consume.

Inputs (from results/selected_stacks.json):
  data/deepfrap/validation_exp/data/frap_32ww_010.mat
  data/deepfrap/validation_exp/data/frap_56ww_005.mat

Outputs (consumed identically to synthetic_*.npz):
  data/real_32ww.npz   (postbleach stack, shape (100, 256, 256), float32)
  data/real_56ww.npz

Each .npz has:
  stack            : (T=100, H=256, W=256) float32 in roughly [0, 1]
                     (raw uint16 / 65535; preprocess.normalize_stack rescales further)
  dt_s             : 0.265
  pixel_size_m     : 7.598e-7
  T, H, W          : 100, 256, 256
  condition        : "32ww" or "56ww"
  source_mat       : filename
  D_ls_pixel2_per_s, D_ls_m2_per_s  : LS-fit reference D (sanity-check anchor only)
  true_D           : nan (real data has no ground truth)

Notes:
  - We use ONLY the postbleach phase (T=100 frames over 26.5 s). The prebleach
    and bleach phases include the bleach pulse itself, which is a photochemical
    discontinuity our diffusion-only PDE doesn't model.
  - No background subtraction here. preprocess.normalize_stack uses min + p99.5
    which handles DC offset robustly. The DeepFRAP MATLAB pipeline does a
    smoothed-frame background subtraction; reproducing that here would add
    coupling without changing matched comparison.
  - Stack layout: MATLAB stores as (H, W, T); we transpose to (T, H, W) so it
    matches synthetic and make_coordinates() expects (T, H, W).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio


def convert_one(mat_path: Path, out_path: Path, condition: str, ls_ref: dict) -> dict:
    m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    exp = m["experiment"]
    post = exp.postbleach.image_data
    bit_max = float((1 << int(exp.postbleach.bit_depth)) - 1)
    pixel_size = float(exp.postbleach.pixel_size_x)
    dt_s = float(exp.postbleach.time_frame)
    assert post.dtype == np.uint16, f"expected uint16, got {post.dtype}"
    assert post.shape == (256, 256, 100), f"unexpected shape {post.shape}"

    # (H, W, T) -> (T, H, W)
    stack = np.transpose(post, (2, 0, 1)).astype(np.float32) / bit_max
    T, H, W = stack.shape

    meta = {
        "stack": stack,
        "dt_s": dt_s,
        "pixel_size_m": pixel_size,
        "T": T,
        "H": H,
        "W": W,
        "condition": condition,
        "source_mat": mat_path.name,
        "D_ls_pixel2_per_s": ls_ref["D_ls_pixel2_per_s"],
        "D_ls_m2_per_s": ls_ref["D_ls_m2_per_s"],
        "true_D": np.nan,
    }
    np.savez(out_path, **meta)
    return {
        "out": str(out_path),
        "stack_shape": stack.shape,
        "stack_range": [float(stack.min()), float(stack.max())],
        "stack_mean": float(stack.mean()),
        "pixel_size_m": pixel_size,
        "dt_s": dt_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("results/selected_stacks.json"),
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("data/deepfrap/validation_exp/data"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    with open(args.selected) as f:
        sel = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cond in ("32ww", "56ww"):
        info = sel[cond]
        mat_path = args.exp_dir / info["file"]
        out_path = args.out_dir / f"real_{cond}.npz"
        ls_ref = {
            "D_ls_pixel2_per_s": info["D_ls_pixel2_per_s"],
            "D_ls_m2_per_s": info["D_ls_m2_per_s"],
        }
        summary = convert_one(mat_path, out_path, cond, ls_ref)
        print(
            f"{cond}: {info['file']}  ->  {summary['out']}  "
            f"shape={summary['stack_shape']}  "
            f"range=[{summary['stack_range'][0]:.4f}, {summary['stack_range'][1]:.4f}]  "
            f"mean={summary['stack_mean']:.4f}  "
            f"dt={summary['dt_s']}s  px={summary['pixel_size_m']*1e6:.4f}µm"
        )


if __name__ == "__main__":
    main()
