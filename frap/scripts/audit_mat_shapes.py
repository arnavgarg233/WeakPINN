"""Audit .mat file shapes to decide FRAP experiment branch.

Phase 2 of PLAN.md. Inspects every .mat file passed on the command line
and classifies its top-level variables into one of:

  - IMAGE_STACK_3D    : looks like a (T, H, W) or (H, W, T) movie
  - IMAGE_STACK_4D    : 4D - e.g. (H, W, T, N_samples) sweep
  - RECOVERY_CURVE    : 1D (T,) or 2D (T, n_curves) recovery trace
  - PARAMETER_VECTOR  : small 1D (n_params,) or (1, n_params)
  - SCALAR            : single number
  - METADATA          : strings, cells, structs (h5py: groups / object refs)
  - UNKNOWN

Handles both legacy (<v7.3, scipy.io.loadmat) and v7.3 HDF5 .mat files
(loaded via h5py). The branch decision lives in PLAN.md Phase 2:

  Branch A - real experimental data has image stacks - full real + synthetic
  Branch B - only simulated has image stacks                - synthetic only
  Branch C - no image stacks anywhere                       - abort FRAP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import scipy.io


_IMAGE_DIMS_MIN = 16
_CURVE_DIMS_MAX_LEN = 8192
_CURVE_LIKELY_T_RANGE = (16, 4096)


def _classify_shape(shape: tuple[int, ...]) -> str:
    """Return one of IMAGE_STACK_3D/IMAGE_STACK_4D/RECOVERY_CURVE/PARAMETER_VECTOR/SCALAR/UNKNOWN."""
    if len(shape) == 0:
        return "SCALAR"
    if len(shape) == 1:
        n = shape[0]
        if n <= 1:
            return "SCALAR"
        if n <= 32:
            return "PARAMETER_VECTOR"
        if n <= _CURVE_DIMS_MAX_LEN:
            return "RECOVERY_CURVE"
        return "PARAMETER_VECTOR"
    if len(shape) == 2:
        rows, cols = shape
        if min(rows, cols) <= 8:
            return "RECOVERY_CURVE" if max(rows, cols) >= _CURVE_LIKELY_T_RANGE[0] else "PARAMETER_VECTOR"
        if min(rows, cols) >= _IMAGE_DIMS_MIN and max(rows, cols) >= _IMAGE_DIMS_MIN:
            return "UNKNOWN"
        return "UNKNOWN"
    if len(shape) == 3:
        if all(d >= _IMAGE_DIMS_MIN for d in shape):
            return "IMAGE_STACK_3D"
        spatial_like = sum(1 for d in shape if d >= _IMAGE_DIMS_MIN)
        time_like = sum(1 for d in shape if _CURVE_LIKELY_T_RANGE[0] <= d <= _CURVE_LIKELY_T_RANGE[1])
        if spatial_like >= 2 and time_like >= 1:
            return "IMAGE_STACK_3D"
        return "UNKNOWN"
    if len(shape) == 4:
        return "IMAGE_STACK_4D"
    return "UNKNOWN"


def _walk_struct(prefix: str, obj, out: list[tuple[str, tuple[int, ...], str, str]]) -> None:
    """Recurse into MATLAB structs (loaded with struct_as_record=False) and
    emit (dotted_name, shape, dtype, class) rows for any contained ndarrays."""
    if hasattr(obj, "_fieldnames"):
        for fn in obj._fieldnames:
            _walk_struct(f"{prefix}.{fn}", getattr(obj, fn), out)
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.shape == (1, 1):
                _walk_struct(prefix, obj[0, 0], out)
                return
            for idx, item in np.ndenumerate(obj):
                _walk_struct(f"{prefix}[{','.join(str(i) for i in idx)}]", item, out)
            return
        shape = tuple(int(s) for s in obj.shape)
        klass = _classify_shape(shape)
        out.append((prefix, shape, str(obj.dtype), klass))
        return
    out.append((prefix, (), type(obj).__name__, "METADATA"))


def _audit_legacy(path: Path) -> list[tuple[str, tuple[int, ...], str, str]]:
    """Audit a non-HDF5 .mat (loadable by scipy.io.loadmat).

    Uses struct_as_record=False so MATLAB structs become Python objects with
    `_fieldnames`, allowing recursive descent into nested experiment.prebleach
    / bleach / postbleach hierarchies.
    """
    rows: list[tuple[str, tuple[int, ...], str, str]] = []
    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        _walk_struct(key, val, rows)
    return rows


def _audit_hdf5(path: Path) -> list[tuple[str, tuple[int, ...], str, str]]:
    """Audit a v7.3 (HDF5) .mat file."""
    rows: list[tuple[str, tuple[int, ...], str, str]] = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                shape = tuple(int(s) for s in obj.shape)
                dtype = str(obj.dtype)
                klass = _classify_shape(shape)
                rows.append((key, shape, dtype, klass))
            elif isinstance(obj, h5py.Group):
                rows.append((key, (), "Group", "METADATA"))
    return rows


def audit_file(path: Path) -> list[tuple[str, tuple[int, ...], str, str]]:
    """Return list of (key, shape, dtype, classification) rows for one file."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic[:2] == b"\x89H":
            return _audit_hdf5(path)
        return _audit_legacy(path)
    except NotImplementedError:
        return _audit_hdf5(path)
    except Exception as e:
        return [("<ERROR>", (), type(e).__name__, f"FAILED: {e}")]


def summarize(by_file: dict[Path, list[tuple[str, tuple[int, ...], str, str]]]) -> tuple[bool, bool]:
    """Return (has_real_image_stacks, has_sim_image_stacks)."""
    has_real, has_sim = False, False
    for path, rows in by_file.items():
        is_exp = "validation_exp" in str(path) or "frap_matlab" in str(path)
        is_sim = ("validation_sim" in str(path)) or ("benchmark_ls_loss" in str(path))
        for _, _, _, klass in rows:
            if klass in ("IMAGE_STACK_3D", "IMAGE_STACK_4D"):
                if is_exp:
                    has_real = True
                if is_sim:
                    has_sim = True
    return has_real, has_sim


def recommend_branch(has_real: bool, has_sim: bool) -> str:
    if has_real:
        return "A"
    if has_sim:
        return "B"
    return "C"


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help=".mat files to audit")
    args = parser.parse_args(list(argv))

    by_file: dict[Path, list[tuple[str, tuple[int, ...], str, str]]] = {}
    for p in args.paths:
        if not p.exists():
            print(f"!! MISSING: {p}", file=sys.stderr)
            continue
        by_file[p] = audit_file(p)

    print("=" * 88)
    print(f"FRAP .mat audit  ({len(by_file)} files)")
    print("=" * 88)

    for path in sorted(by_file):
        rows = by_file[path]
        print(f"\n>> {path}")
        if not rows:
            print("   (no top-level variables)")
            continue
        for key, shape, dtype, klass in rows:
            shape_str = "x".join(str(s) for s in shape) if shape else "scalar"
            print(f"   {key:24s}  shape={shape_str:32s}  dtype={dtype:14s}  class={klass}")

    has_real, has_sim = summarize(by_file)
    branch = recommend_branch(has_real, has_sim)

    print("\n" + "=" * 88)
    print(f"Real (validation_exp/frap_matlab) image stacks present: {has_real}")
    print(f"Simulated (validation_sim/data/) image stacks present : {has_sim}")
    print(f"Recommended branch                                    : {branch}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
