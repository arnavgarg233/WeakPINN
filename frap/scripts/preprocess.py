"""Phase 4 - preprocessing helpers for FRAP stacks.

Per PLAN.md Phase 4. Three pure functions used by every training run:

  normalize_stack    : robust [0, 1] normalization via 99.5th percentile.
                       Tolerant to NaN/Inf, handles degenerate stacks.

  make_coordinates   : (T, H, W) stack -> (N=T*H*W, 3) coordinates in [-1, 1]^3
                       and (N, 1) values, in canonical (x, y, t) order.

  split_chronological: HELD-OUT BY TIME, not by random pixels. Matches
                       Flare-PINN discipline that the validation set must
                       be a contiguous temporal block the model has not
                       seen, so the recovered diffusion physics generalizes
                       forward rather than fitting noise pixel-wise.

This module is importable from training scripts. The CLI block at the bottom
runs a deterministic smoke test on data/synthetic_clean.npz so we can
confirm shapes and dtypes before Phase 7.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def normalize_stack(stack: NDArray) -> NDArray:
    """Map stack to roughly [0, 1] using 99.5th-percentile denominator.

    The percentile (vs max) protects against single-pixel cosmic-ray outliers
    and Poisson-tail blowups in low-photon regimes. NaNs/Infs are scrubbed at
    the end. Output dtype is always float32.
    """
    stack = stack.astype(np.float32)
    stack = stack - np.nanmin(stack)
    denom = np.nanpercentile(stack, 99.5)
    if denom <= 0:
        denom = np.nanmax(stack) + 1e-8
    stack = stack / (denom + 1e-8)
    return np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0)


def make_coordinates(stack: NDArray) -> tuple[NDArray, NDArray]:
    """Return (coords, values) flattened over (T, H, W).

    coords:  (N, 3) float32 with columns (x, y, t) in [-1, 1]
    values:  (N, 1) float32 stack flattened to a column vector

    Convention: x is the WIDTH axis (last), y is HEIGHT (middle), t is TIME (first).
    This matches every downstream call in PLAN.md (Phase 5/6/7).
    """
    T, H, W = stack.shape
    t = np.linspace(-1, 1, T, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    x = np.linspace(-1, 1, W, dtype=np.float32)
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    coords = np.stack([xx, yy, tt], axis=-1).reshape(-1, 3)
    values = stack.reshape(-1, 1)
    return coords.astype(np.float32), values.astype(np.float32)


def split_chronological(
    coords: NDArray,
    values: NDArray,
    T: int,
    H: int,
    W: int,
    train_frac: float = 0.8,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Held-out split by TIME (not random pixels). Matches Flare-PINN discipline."""
    n_train_t = int(T * train_frac)
    coords_grid = coords.reshape(T, H, W, 3)
    values_grid = values.reshape(T, H, W, 1)
    train_coords = coords_grid[:n_train_t].reshape(-1, 3)
    train_values = values_grid[:n_train_t].reshape(-1, 1)
    val_coords = coords_grid[n_train_t:].reshape(-1, 3)
    val_values = values_grid[n_train_t:].reshape(-1, 1)
    return train_coords, train_values, val_coords, val_values


def _smoke_test(stack_path: Path) -> int:
    print(f">> loading {stack_path}")
    data = np.load(stack_path)
    stack = data["stack"]
    T, H, W = stack.shape
    print(f"   stack shape={stack.shape}  dtype={stack.dtype}  "
          f"range=[{stack.min():.4f}, {stack.max():.4f}]")

    stack_n = normalize_stack(stack)
    assert stack_n.dtype == np.float32, stack_n.dtype
    assert stack_n.shape == stack.shape
    print(f"   normalized range=[{stack_n.min():.4f}, {stack_n.max():.4f}]")

    coords, values = make_coordinates(stack_n)
    assert coords.shape == (T * H * W, 3), coords.shape
    assert values.shape == (T * H * W, 1), values.shape
    assert coords.dtype == np.float32 and values.dtype == np.float32
    assert -1.0 - 1e-6 <= coords.min() and coords.max() <= 1.0 + 1e-6
    print(f"   coords={coords.shape}  values={values.shape}  "
          f"x:[{coords[:, 0].min():.3f},{coords[:, 0].max():.3f}]  "
          f"y:[{coords[:, 1].min():.3f},{coords[:, 1].max():.3f}]  "
          f"t:[{coords[:, 2].min():.3f},{coords[:, 2].max():.3f}]")

    tr_c, tr_v, va_c, va_v = split_chronological(coords, values, T, H, W, train_frac=0.8)
    n_train_t = int(T * 0.8)
    n_val_t = T - n_train_t
    assert tr_c.shape == (n_train_t * H * W, 3), tr_c.shape
    assert va_c.shape == (n_val_t * H * W, 3), va_c.shape
    assert tr_c[:, 2].max() < va_c[:, 2].min(), "TEMPORAL LEAK between train and val"
    print(f"   split: train {tr_c.shape}  val {va_c.shape}")
    print(f"   train t in [{tr_c[:, 2].min():.3f}, {tr_c[:, 2].max():.3f}]")
    print(f"   val   t in [{va_c[:, 2].min():.3f}, {va_c[:, 2].max():.3f}]  <- no leak")
    print(">> preprocess smoke test PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke test for preprocess helpers.")
    parser.add_argument(
        "--stack",
        type=Path,
        default=Path("data/synthetic_clean.npz"),
        help="Path to an .npz with key 'stack' (T, H, W).",
    )
    args = parser.parse_args()
    return _smoke_test(args.stack)


if __name__ == "__main__":
    raise SystemExit(main())
