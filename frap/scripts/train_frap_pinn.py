"""Phase 7 - Train a FRAP PINN (data / strong / weak) on one stack and emit a result JSON.

Imports Agent B's modules:
  src.models.PINN_FRAP
  src.losses.strong_form_residual, weak_form_residuals, compute_residuals_fair

Train/val split is chronological (last 20% of timesteps held out) - see preprocess.split_chronological.

Output JSON schema (consumed by scripts/analyze.py):
  method, seed, lambda_phys, init_D, stack
  D_recovered, k_recovered, true_D, D_ls_pixel2_per_s (if present), D_ls_m2_per_s
  val_mse                           : MSE on the held-out 20% of timesteps
  median_strong_residual            : median |strong-form residual| on validation pts
                                       (computed on THIS model regardless of training method)
  median_weak_residual              : median |weak-form residual| on validation pts
                                       (computed on THIS model regardless of training method)
  elapsed_sec, losses (per-log-step list)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

# Add project root to sys.path so we can import src/ and scripts/preprocess
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

from src.models import PINN_FRAP                                # noqa: E402
from src.losses import strong_form_residual                    # noqa: E402
from src.losses import weak_form_residuals                     # noqa: E402
from src.losses import compute_residuals_fair                  # noqa: E402
from scripts.preprocess import normalize_stack                  # noqa: E402
from scripts.preprocess import make_coordinates                 # noqa: E402
from scripts.preprocess import split_chronological              # noqa: E402


L_NORM = 2.0  # PINN normalized spatial extent is [-1, 1]


def compute_D_norm_true(true_D_phys, dt, T):
    """For synthetic stacks: convert simulator D to PINN-coord D.

    D_norm = D_phys * 2 * T_phys / L^2, where T_phys = (T-1)*dt and L=2.
    Matches scripts/select_lambda.py. Returns None when input is missing/invalid.
    """
    if true_D_phys is None or dt is None or T is None:
        return None
    try:
        T_phys = (int(T) - 1) * float(dt)
        return float(true_D_phys) * 2.0 * T_phys / (L_NORM * L_NORM)
    except (TypeError, ValueError):
        return None


def sample_batch(coords: np.ndarray, values: np.ndarray, batch_size: int,
                 device: torch.device, rng: np.random.Generator
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(coords.shape[0], size=batch_size, replace=False)
    return (torch.tensor(coords[idx], device=device),
            torch.tensor(values[idx], device=device))


def load_stack_npz(path: Path) -> tuple[np.ndarray, dict]:
    """Load a .npz produced by generate_synthetic_frap or convert_real_mat_to_npz."""
    d = np.load(path)
    stack = d["stack"]
    meta = {}
    for k in d.files:
        if k == "stack":
            continue
        v = d[k]
        if v.ndim == 0:
            try:
                meta[k] = float(v) if v.dtype.kind in "fui" else str(v)
            except (TypeError, ValueError):
                meta[k] = str(v)
        else:
            meta[k] = v.tolist()
    return stack, meta


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    stack, meta = load_stack_npz(Path(args.stack))
    true_D = meta.get("D", None) if isinstance(meta.get("D", None), float) else None
    D_ls_pixel2 = meta.get("D_ls_pixel2_per_s", None)
    D_ls_m2 = meta.get("D_ls_m2_per_s", None)
    dt_meta = meta.get("dt", None)
    T_meta = meta.get("T", None)  # synthetic stacks store T explicitly

    stack = normalize_stack(stack)
    T, H, W = stack.shape
    if T_meta is None:
        T_meta = T
    D_norm_true = compute_D_norm_true(true_D, dt_meta, T_meta)
    coords, values = make_coordinates(stack)
    train_c, train_v, val_c, val_v = split_chronological(
        coords, values, T, H, W, train_frac=args.train_frac
    )
    print(f"loaded {args.stack}  T={T} H={H} W={W}  "
          f"train pts={len(train_c)}  val pts={len(val_c)}")

    model = PINN_FRAP(
        hidden=args.hidden, depth=args.depth,
        init_D=args.init_D, learn_k=args.learn_k,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_start = time.time()
    losses: list[dict] = []
    for step in trange(args.steps, desc=f"{args.method}|seed{args.seed}|lam{args.lambda_phys}"):
        xb, yb = sample_batch(train_c, train_v, args.batch_size, device, rng)
        pred = model(xb)
        data_loss = F.mse_loss(pred, yb)
        if args.method == "data":
            phys_loss = torch.tensor(0.0, device=device)
        elif args.method == "strong":
            r = strong_form_residual(model, xb)
            phys_loss = torch.mean(r ** 2)
        elif args.method == "weak":
            wr = weak_form_residuals(model, xb, n_tests=args.n_tests, sigma=args.sigma)
            phys_loss = torch.mean(wr ** 2)
        else:
            raise ValueError(args.method)
        loss = data_loss + args.lambda_phys * phys_loss
        opt.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            losses.append({
                "step": step,
                "loss": float(loss.item()),
                "data": float(data_loss.item()),
                "phys": float(phys_loss.item()),
                "D": float(model.D().item()),
                "k": float(model.k().item()),
            })
    elapsed = time.time() - t_start

    # Held-out validation MSE
    model.eval()
    with torch.no_grad():
        chunks_pred, chunks_true = [], []
        for i in range(0, len(val_c), 100_000):
            vx = torch.tensor(val_c[i:i + 100_000], device=device)
            vy = torch.tensor(val_v[i:i + 100_000], device=device)
            chunks_pred.append(model(vx).cpu().numpy())
            chunks_true.append(vy.cpu().numpy())
        val_pred = np.concatenate(chunks_pred)
        val_true = np.concatenate(chunks_true)
        val_mse = float(np.mean((val_pred - val_true) ** 2))

    # Fair residual computation: both metrics on this model
    eval_n = min(8192, len(val_c))
    eval_idx = rng.choice(len(val_c), size=eval_n, replace=False)
    eval_xyt = torch.tensor(val_c[eval_idx], device=device)
    strong_r, weak_r = compute_residuals_fair(
        model, eval_xyt, n_tests=args.n_tests, sigma=args.sigma
    )

    result = {
        "method": args.method,
        "seed": args.seed,
        "lambda_phys": args.lambda_phys,
        "init_D": args.init_D,
        "learn_k": args.learn_k,
        "stack": args.stack,
        "T": T, "H": H, "W": W,
        "D_recovered": float(model.D().item()),
        "k_recovered": float(model.k().item()),
        "true_D": true_D,
        "D_norm_true": D_norm_true,
        "D_ls_pixel2_per_s": D_ls_pixel2,
        "D_ls_m2_per_s": D_ls_m2,
        "val_mse": val_mse,
        "median_strong_residual": float(strong_r.abs().median().item()),
        "median_weak_residual": float(weak_r.abs().median().item()),
        "elapsed_sec": elapsed,
        "n_tests": args.n_tests,
        "sigma": args.sigma,
        "losses": losses,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    summary = {k: v for k, v in result.items() if k != "losses"}
    print("RESULT", json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stack", required=True)
    p.add_argument("--method", choices=["data", "strong", "weak"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_phys", type=float, default=1.0)
    p.add_argument("--n_tests", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--init_D", type=float, default=0.05)
    p.add_argument("--learn_k", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    train(args)
