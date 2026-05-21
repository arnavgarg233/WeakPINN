"""
B2-v2 convergence benchmark with per-method lambda.

Modified from b2_workdir/b2_convergence.py to accept --lam_strong and --lam_weak
separately so each method uses its tuned regularization weight.
"""

import os
import csv
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

D_TRUE = 0.1
D_INIT = 0.05
PI = math.pi

DEVICE = torch.device("cpu")


# ----------------------------- model -----------------------------
class MLP(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        mods = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


def exact_u(x, t, D=D_TRUE):
    return np.exp(-PI**2 * D * t) * np.sin(PI * x)


# ----------------------------- data -----------------------------
def make_grid(N, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, N)
    t = np.linspace(0.0, 1.0, N)
    X, T = np.meshgrid(x, t, indexing="ij")
    U_clean = exact_u(X, T)
    return X, T, U_clean, rng


# ----------------------------- weak-form test functions -----------------------------
def phi_and_grads_gauss(x, t, x0=0.0, t0=0.5, sx=0.5, st=0.5):
    bx = 1.0 - x * x
    bt = 1.0 - (2.0 * t - 1.0) ** 2
    G = torch.exp(-(((x - x0) ** 2) / sx + ((t - t0) ** 2) / st))
    phi = bx * bt * G

    dbx = -2.0 * x
    dbt = -4.0 * (2.0 * t - 1.0)
    dG_dx = G * (-2.0 * (x - x0) / sx)
    dG_dt = G * (-2.0 * (t - t0) / st)
    dphi_dx = dbx * bt * G + bx * bt * dG_dx
    dphi_dt = bx * dbt * G + bx * bt * dG_dt
    return phi, dphi_dx, dphi_dt


def build_test_functions(x_grid, t_grid):
    centers = [(0.0, 0.5), (-0.4, 0.3), (0.4, 0.7), (-0.3, 0.7), (0.3, 0.3)]
    tests = []
    for (x0, t0) in centers:
        tests.append(phi_and_grads_gauss(x_grid, t_grid, x0=x0, t0=t0, sx=0.4, st=0.4))
    return tests


def trapz2d(F, x_axis, t_axis):
    Ix = torch.trapz(F, x_axis, dim=0)
    I = torch.trapz(Ix, t_axis, dim=0)
    return I


# ----------------------------- training -----------------------------
def train_one(method, N, sigma, seed, n_steps=4000, lam=1.0, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_np, T_np, U_clean, rng = make_grid(N, seed)
    noise = rng.normal(0.0, sigma, size=U_clean.shape) if sigma > 0 else np.zeros_like(U_clean)
    U_obs = U_clean + noise

    Nf = 64
    xf = np.linspace(-1.0, 1.0, Nf)
    tf = np.linspace(0.0, 1.0, Nf)
    Xf, Tf = np.meshgrid(xf, tf, indexing="ij")
    Uf = exact_u(Xf, Tf)

    x_axis = torch.tensor(X_np[:, 0], dtype=torch.float32, device=DEVICE)
    t_axis = torch.tensor(T_np[0, :], dtype=torch.float32, device=DEVICE)

    x_flat = torch.tensor(X_np.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    t_flat = torch.tensor(T_np.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    u_obs_flat = torch.tensor(U_obs.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

    xf_flat = torch.tensor(Xf.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    tf_flat = torch.tensor(Tf.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    uf_flat = torch.tensor(Uf.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

    if method == "weak":
        with torch.no_grad():
            x_grid_t = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
            t_grid_t = torch.tensor(T_np, dtype=torch.float32, device=DEVICE)
            tests = build_test_functions(x_grid_t, t_grid_t)

    model = MLP().to(DEVICE)
    D_param = nn.Parameter(torch.tensor(D_INIT, device=DEVICE))
    params = list(model.parameters()) + [D_param]
    opt = torch.optim.Adam(params, lr=lr)

    t_start = time.time()
    for step in range(n_steps):
        opt.zero_grad()

        u_pred = model(x_flat, t_flat)
        data_loss = ((u_pred - u_obs_flat) ** 2).mean()

        if method == "strong":
            x_req = x_flat.detach().clone().requires_grad_(True)
            t_req = t_flat.detach().clone().requires_grad_(True)
            up = model(x_req, t_req)
            grads = torch.autograd.grad(up, (x_req, t_req), grad_outputs=torch.ones_like(up),
                                        create_graph=True)
            u_x, u_t = grads[0], grads[1]
            u_xx = torch.autograd.grad(u_x, x_req, grad_outputs=torch.ones_like(u_x),
                                       create_graph=True)[0]
            phys_loss = ((u_t - D_param * u_xx) ** 2).mean()
        else:
            x_req = x_flat.detach().clone().requires_grad_(True)
            t_req = t_flat.detach().clone().requires_grad_(True)
            up = model(x_req, t_req)
            grads = torch.autograd.grad(up, (x_req, t_req), grad_outputs=torch.ones_like(up),
                                        create_graph=True)
            u_x = grads[0].reshape(N, N)
            u_t = grads[1].reshape(N, N)
            u_grid = up.reshape(N, N)

            phys_loss = 0.0
            for (phi_m, dphi_m_dx, _dphi_m_dt) in tests:
                term1 = trapz2d(phi_m * u_t, x_axis, t_axis)
                term2 = D_param * trapz2d(dphi_m_dx * u_x, x_axis, t_axis)
                residual = term1 + term2
                norm = trapz2d(phi_m * phi_m, x_axis, t_axis).clamp_min(1e-8)
                phys_loss = phys_loss + (residual ** 2) / norm
            phys_loss = phys_loss / len(tests)

        loss = data_loss + lam * phys_loss
        loss.backward()
        opt.step()

    train_time = time.time() - t_start

    with torch.no_grad():
        uf_pred = model(xf_flat, tf_flat)
        val_mse = ((uf_pred - uf_flat) ** 2).mean().item()
        D_rec = D_param.item()
        D_err = abs(D_rec - D_TRUE)
    return D_rec, D_err, val_mse, train_time


# ----------------------------- driver -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="/tmp/cmame_b16_runs/b2v2_workdir/B2v2_convergence_sweep.csv")
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_steps", type=int, default=4000)
    ap.add_argument("--lam_strong", type=float, default=1.0)
    ap.add_argument("--lam_weak", type=float, default=0.1)
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.out_csv):
        with open(args.out_csv, "r") as f:
            r = csv.DictReader(f)
            rows = list(r)
    done = {(int(r["grid_N"]), float(r["noise_sigma"]), r["method"], int(r["seed"])) for r in rows}

    methods = ["strong", "weak"]
    method_lam = {"strong": args.lam_strong, "weak": args.lam_weak}
    total = len(args.grids) * len(args.sigmas) * len(methods) * len(args.seeds)
    i = 0
    for N in args.grids:
        for sigma in args.sigmas:
            for method in methods:
                for seed in args.seeds:
                    i += 1
                    key = (N, sigma, method, seed)
                    if key in done:
                        print(f"[{i}/{total}] skip {key} (cached)")
                        continue
                    lam_use = method_lam[method]
                    print(f"[{i}/{total}] N={N} sigma={sigma} method={method} seed={seed} lam={lam_use}", flush=True)
                    D_rec, D_err, val_mse, tt = train_one(
                        method, N, sigma, seed, n_steps=args.n_steps, lam=lam_use
                    )
                    row = dict(
                        grid_N=N, noise_sigma=sigma, method=method, lam=lam_use, seed=seed,
                        D_recovered=D_rec, D_err=D_err, val_mse=val_mse, train_time_s=tt,
                    )
                    rows.append(row)
                    print(f"   -> D_rec={D_rec:.4f}  D_err={D_err:.4f}  val_mse={val_mse:.2e}  t={tt:.1f}s",
                          flush=True)
                    with open(args.out_csv, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["grid_N", "noise_sigma", "method", "lam", "seed",
                                                          "D_recovered", "D_err", "val_mse", "train_time_s"])
                        w.writeheader()
                        for rr in rows:
                            w.writerow(rr)

    print("DONE")


if __name__ == "__main__":
    main()
