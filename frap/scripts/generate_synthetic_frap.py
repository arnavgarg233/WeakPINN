"""Phase 3 - synthetic FRAP ground-truth generator.

Generates a clean stack (D=0.05) and three noisy stacks with optical-mapping
style imperfections at three photon-count levels. All four use the SAME
underlying diffusion physics (so the PINN's enforced PDE matches the data)
but differ only in noise realism. This is the controlled axis for the
strong-vs-weak comparison.

Per PLAN.md Phase 3:

  fd_diffusion_step: explicit centered-difference 2D Laplacian step with
    Neumann (zero-flux) boundary conditions. Implemented by np.pad with
    mode='edge' so the ghost cell at the boundary equals the adjacent
    interior cell, giving ∂c/∂n = 0. CFL check enforced per substep.

Outputs into data/:
  synthetic_clean.npz       (D=0.05, photon_count=0, psf=0, no bleach decay)
  synthetic_noise_low.npz   (photon_count=10000, psf=1.5, bleach=0.005)
  synthetic_noise_med.npz   (photon_count=1000,  psf=1.5, bleach=0.005)
  synthetic_noise_high.npz  (photon_count=100,   psf=1.5, bleach=0.005)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


def fd_diffusion_step(c: NDArray, D: float, dt: float, dx: float) -> NDArray:
    """One explicit Euler step of 2D diffusion with Neumann (zero-flux) BCs.

    Neumann BC implemented by padding with mode='edge': the ghost cell at
    the boundary takes the value of the adjacent interior cell, so
    ∂c/∂n = 0 at the boundary. The PINN's weak form integrates over the
    same bounded Ω, so data and constraint match.
    """
    c_pad = np.pad(c, 1, mode="edge")
    lap = (
        c_pad[:-2, 1:-1]
        + c_pad[2:, 1:-1]
        + c_pad[1:-1, :-2]
        + c_pad[1:-1, 2:]
        - 4.0 * c_pad[1:-1, 1:-1]
    ) / (dx * dx)
    return c + dt * D * lap


def simulate_frap(
    H: int = 128,
    W: int = 128,
    T: int = 80,
    D: float = 0.05,
    dt: float = 0.01,
    dx: float = 1.0 / 64,
    bleach_radius: float = 0.2,
    bleach_depth: float = 0.8,
    psf_sigma: float = 1.5,
    photon_count: int = 1000,
    bleach_during_imaging: float = 0.0,
    seed: int = 0,
) -> tuple[NDArray, dict]:
    rng = np.random.default_rng(seed)
    n_substeps = 10
    cfl = D * (dt / n_substeps) / (dx * dx)
    assert cfl < 0.25, f"CFL violated: D*dt_sub/dx^2 = {cfl:.4f} >= 0.25"
    c = np.ones((H, W), dtype=np.float32)
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2)
    bleach_mask = (r < bleach_radius).astype(np.float32)
    c = c * (1.0 - bleach_depth * bleach_mask)

    frames: list[NDArray] = []
    for _ in range(T):
        if bleach_during_imaging > 0:
            c = c * np.exp(-bleach_during_imaging * dt)
        if psf_sigma > 0:
            observed_clean = gaussian_filter(c, sigma=psf_sigma)
        else:
            observed_clean = c.copy()
        if photon_count > 0:
            photons = rng.poisson(observed_clean * photon_count)
            observed = photons.astype(np.float32) / photon_count
        else:
            observed = observed_clean
        frames.append(observed)

        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            c = fd_diffusion_step(c, D, sub_dt, dx)

    meta = {
        "D": D,
        "dt": dt,
        "dx": dx,
        "photon_count": photon_count,
        "psf_sigma": psf_sigma,
        "bleach_during_imaging": bleach_during_imaging,
    }
    return np.stack(frames, axis=0), meta


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(">> simulating clean (D=0.05, photons=infinity, no psf, no bleach decay)")
    stack_clean, meta_clean = simulate_frap(
        D=0.05, photon_count=0, psf_sigma=0, bleach_during_imaging=0, seed=0,
    )
    np.savez(data_dir / "synthetic_clean.npz", stack=stack_clean, **meta_clean)
    print(f"   shape={stack_clean.shape}, range=[{stack_clean.min():.4f}, {stack_clean.max():.4f}]")

    for photon, name in [(10000, "low"), (1000, "med"), (100, "high")]:
        print(f">> simulating noise_{name} (photons={photon}, psf=1.5, bleach=0.005)")
        stack, meta = simulate_frap(
            D=0.05, photon_count=photon, psf_sigma=1.5, bleach_during_imaging=0.005, seed=0,
        )
        np.savez(data_dir / f"synthetic_noise_{name}.npz", stack=stack, **meta)
        print(f"   shape={stack.shape}, range=[{stack.min():.4f}, {stack.max():.4f}]")

    print(">> done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
