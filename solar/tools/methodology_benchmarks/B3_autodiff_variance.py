"""
B3 autodiff-mechanism evidence plot for WeakPINN §2.2.

Controlled demonstration that Var[(Laplacian c)^2] grows much faster than
Var[|grad c|^2] as observation noise sigma rises, when both are computed by
autodiff through a fixed neural-network fit to noisy data.

Self-contained: no dependency on FRAP or solar code.

Outputs:
  /tmp/cmame_b16_runs/B3_autodiff_variance.png
  /tmp/cmame_b16_runs/B3_autodiff_variance.csv
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------- reproducibility ----------
GLOBAL_SEED = 1234
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

OUT_DIR = Path("/tmp/cmame_b16_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUT_DIR / "B3_autodiff_variance.png"
CSV_PATH = OUT_DIR / "B3_autodiff_variance.csv"

# Use CPU for full determinism (MPS has nondeterministic kernels for some ops).
DEVICE = torch.device("cpu")

# ---------- protocol parameters ----------
GRID_N_TRAIN = 64
GRID_N_EVAL = 128
N_STEPS = 5000
LR = 1e-3
SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


# ---------- ground truth ----------
def c_true(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x) * torch.exp(-(y ** 2))


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy).squeeze(-1)


def make_grid(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    lin = torch.linspace(-1.0, 1.0, n)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    return X, Y


def fit_one(sigma: float, seed: int) -> tuple[float, float]:
    """Fit an MLP to c_true + sigma*noise and return Var[|grad|^2], Var[(lap)^2]."""

    # Per-sigma seeding so weight init, noise, optimizer are reproducible.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training grid + noisy observations.
    Xt, Yt = make_grid(GRID_N_TRAIN)
    c_clean = c_true(Xt, Yt)
    noise = torch.randn_like(c_clean)
    c_obs = c_clean + sigma * noise

    train_xy = torch.stack([Xt.reshape(-1), Yt.reshape(-1)], dim=1).to(DEVICE)
    train_c = c_obs.reshape(-1).to(DEVICE)

    model = MLP(hidden=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(N_STEPS):
        opt.zero_grad()
        pred = model(train_xy)
        loss = torch.mean((pred - train_c) ** 2)
        loss.backward()
        opt.step()

    # Eval on dense grid via autodiff.
    Xe, Ye = make_grid(GRID_N_EVAL)
    xy = torch.stack([Xe.reshape(-1), Ye.reshape(-1)], dim=1).to(DEVICE)
    xy.requires_grad_(True)

    c = model(xy)
    grads = torch.autograd.grad(c.sum(), xy, create_graph=True)[0]  # (N,2)
    c_x = grads[:, 0]
    c_y = grads[:, 1]

    c_xx = torch.autograd.grad(c_x.sum(), xy, create_graph=False, retain_graph=True)[0][:, 0]
    c_yy = torch.autograd.grad(c_y.sum(), xy, create_graph=False, retain_graph=True)[0][:, 1]

    grad_sq = (c_x ** 2 + c_y ** 2).detach().cpu().numpy()
    lap_sq = ((c_xx + c_yy) ** 2).detach().cpu().numpy()

    var_grad = float(np.var(grad_sq))
    var_lap = float(np.var(lap_sq))
    return var_grad, var_lap


def main() -> None:
    rows = []
    print(f"{'sigma':>8} {'Var[|grad|^2]':>16} {'Var[(lap)^2]':>16}  ratio")
    for i, sigma in enumerate(SIGMAS):
        seed = GLOBAL_SEED + i  # reproducible, sigma-specific
        vg, vl = fit_one(sigma, seed)
        ratio = vl / vg if vg > 0 else float("inf")
        rows.append((sigma, vg, vl))
        print(f"{sigma:>8.4f} {vg:>16.6e} {vl:>16.6e}  {ratio:.3e}")

    # CSV
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "var_grad", "var_lap"])
        for sigma, vg, vl in rows:
            w.writerow([sigma, vg, vl])

    # Plot
    sig_arr = np.array([r[0] for r in rows])
    vg_arr = np.array([r[1] for r in rows])
    vl_arr = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(sig_arr, vg_arr, "o-", color="#1f77b4", lw=2, ms=7,
                label=r"$\mathrm{Var}[|\nabla c|^{2}]$  (gradient field)")
    ax.semilogy(sig_arr, vl_arr, "s-", color="#d62728", lw=2, ms=7,
                label=r"$\mathrm{Var}[(\nabla^{2} c)^{2}]$  (Laplacian field)")
    ax.set_xlabel(r"observation noise amplitude  $\sigma$")
    ax.set_ylabel("variance over evaluation grid (log scale)")
    ax.set_title("Autodiff noise amplification: Laplacian vs gradient")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=180)
    plt.close(fig)

    # Headline summary
    last_sigma = rows[-1][0]
    last_vg = rows[-1][1]
    last_vl = rows[-1][2]
    if last_vg > 0:
        ratio = last_vl / last_vg
        print(
            f"\nHEADLINE: at sigma={last_sigma}, Var[(lap c)^2] / Var[|grad c|^2] "
            f"= {ratio:.2e}"
        )
    print(f"Saved PNG -> {PNG_PATH}")
    print(f"Saved CSV -> {CSV_PATH}")


if __name__ == "__main__":
    main()
