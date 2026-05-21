"""Plot B2 convergence sweep: D_err vs noise_sigma, panel per grid_N."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_PATH = "/tmp/cmame_b16_runs/B2_convergence_sweep.csv"
OUT_PNG = "/tmp/cmame_b16_runs/B2_convergence_plot.png"

COLOR = {"weak": "#1f9d9b", "strong": "#e07b1c"}  # teal, orange


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                grid_N=int(r["grid_N"]),
                noise_sigma=float(r["noise_sigma"]),
                method=r["method"],
                seed=int(r["seed"]),
                D_recovered=float(r["D_recovered"]),
                D_err=float(r["D_err"]),
                val_mse=float(r["val_mse"]),
            ))
    return rows


def main():
    rows = load()
    grids = sorted({r["grid_N"] for r in rows})
    sigmas = sorted({r["noise_sigma"] for r in rows})

    fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 3.6), sharey=True)
    if len(grids) == 1:
        axes = [axes]

    for ax, N in zip(axes, grids):
        for method in ["strong", "weak"]:
            means, stds = [], []
            for s in sigmas:
                vals = [r["D_err"] for r in rows
                        if r["grid_N"] == N and r["noise_sigma"] == s and r["method"] == method]
                if not vals:
                    means.append(np.nan); stds.append(np.nan); continue
                means.append(np.mean(vals)); stds.append(np.std(vals))
            means = np.array(means); stds = np.array(stds)
            ax.plot(sigmas, means, "o-", color=COLOR[method], lw=2, ms=6,
                    label=f"{method}-form")
            ax.fill_between(sigmas, np.clip(means - stds, 1e-6, None), means + stds,
                            color=COLOR[method], alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel(r"observation noise $\sigma$")
        ax.set_title(f"grid N = {N}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
    axes[0].set_ylabel(r"$|D_\mathrm{recovered} - D_\mathrm{true}|$")
    axes[-1].legend(loc="best", frameon=True)
    fig.suptitle("B2 canonical benchmark: 1D heat equation, $D$ inversion, weak vs strong PINN")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()
