"""Unit tests for src/losses.py - CPU only.

THE critical test: both strong-form and weak-form residuals must be
near zero on a known analytic solution of the PDE c_t - D ∇²c = 0:

    c(x, y, t) = exp(-2 π² D t) · sin(π x) · sin(π y)

Verification of correctness:
  c_t  = -2π² D · c
  c_xx = -π²    · c
  c_yy = -π²    · c
  c_t - D (c_xx + c_yy) = -2π² D c - D (-2π² c) = 0   ✓

This catches sign errors, swapped axes, and Laplacian wiring bugs that
would otherwise silently waste hours of GPU time. Per PLAN.md Phase 6
note + user brief, this is the single most important test in the suite.

The weak-form test uses ORIGIN-CENTERED Gaussian test functions because
the analytic solution does NOT satisfy zero-flux Neumann BCs (∂c/∂n ≠ 0
on x, y = ±1). However, on each ∂Ω edge the boundary integrand
  φ(boundary) · ∂c/∂n(boundary)
is odd in the perpendicular axis when φ is symmetric about the origin
(e.g. centered Gaussian), so the dropped IBP boundary term is exactly
zero by symmetry. This makes the weak-form library output ≈ 0 modulo
Monte Carlo noise on the interior integral.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import (  # noqa: E402
    compute_residuals_fair,
    eval_gaussian_phi,
    gradients,
    make_gaussian_tests,
    strong_form_residual,
    weak_form_residuals,
)
from src.models import PINN_FRAP  # noqa: E402


PI = math.pi


class _AnalyticFRAP(nn.Module):
    """Test-only model: forward returns c(x,y,t) = exp(-2π²Dt) sin(πx) sin(πy)."""

    def __init__(self, D_value: float, k_value: float = 0.0) -> None:
        super().__init__()
        self._D = float(D_value)
        self._k = float(k_value)

    def D(self) -> torch.Tensor:
        return torch.tensor(self._D, dtype=torch.float32)

    def k(self) -> torch.Tensor:
        return torch.tensor(self._k, dtype=torch.float32)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        return torch.exp(-2.0 * (PI ** 2) * self._D * t) * torch.sin(PI * x) * torch.sin(PI * y)


def _interior_grid(n: int, eps: float = 0.02, seed: int = 0) -> torch.Tensor:
    """Sample n points with x, y in [-1+eps, 1-eps] and t in [eps, 1-eps].

    Forward-time only: the analytic solution `exp(-2π²Dt)` decays for t > 0
    and BLOWS UP for t < 0. Sampling t in [eps, 1-eps] keeps `c` bounded
    (worst case at t = eps with c ≈ 1), so the float32 cancellation error
    in `c_t - D(c_xx + c_yy)` stays at machine precision for any D in the
    physical range we care about.
    """
    g = torch.Generator().manual_seed(seed)
    u_xy = torch.rand((n, 2), generator=g)
    u_t = torch.rand((n, 1), generator=g)
    xy = (1.0 - eps) * (2.0 * u_xy - 1.0)
    t = eps + u_t * (1.0 - 2.0 * eps)
    return torch.cat([xy, t], dim=1)


def test_gradients_known_quadratic():
    """gradients(y, x) for y = sum_i x_i^2 should be 2x."""
    x = torch.randn(20, 3, requires_grad=True)
    y = (x ** 2).sum(dim=-1, keepdim=True)
    g = gradients(y, x)
    assert g.shape == x.shape
    assert torch.allclose(g, 2.0 * x, atol=1e-5)


def test_eval_gaussian_phi_shapes():
    """phi: (M, N), grad_phi: (M, N, 2).

    With the boundary cutoff W = (1 - x^2)^2 (1 - y^2)^2, phi at the
    Gaussian center equals W(center) (not 1 in general).
    """
    centers = torch.tensor([[0.0, 0.0], [0.3, -0.2]])
    sigmas = torch.tensor([[0.1], [0.2]])
    xy = torch.tensor([[0.0, 0.0], [0.3, -0.2], [0.5, 0.5]])
    phi, grad_phi = eval_gaussian_phi(xy, centers, sigmas)
    assert phi.shape == (2, 3), phi.shape
    assert grad_phi.shape == (2, 3, 2), grad_phi.shape
    # center 0 at (0, 0), eval at (0, 0): G = 1, W = 1
    assert torch.isclose(phi[0, 0], torch.tensor(1.0), atol=1e-6)
    # center 1 at (0.3, -0.2), eval at (0.3, -0.2): G = 1, W = (1-0.09)^2 (1-0.04)^2
    expected_W = (1.0 - 0.3 ** 2) ** 2 * (1.0 - (-0.2) ** 2) ** 2
    assert torch.isclose(phi[1, 1], torch.tensor(expected_W), atol=1e-6)


def test_eval_gaussian_phi_zero_on_boundary():
    """phi must vanish at any boundary point of [-1, 1]^2, regardless of
    where the Gaussian center is. This is the property that justifies
    dropping the IBP boundary integral in the weak-form residual."""
    centers = torch.tensor([
        [0.0, 0.0],
        [0.5, -0.3],
        [-0.7, 0.4],
        [0.6, 0.6],
    ])
    sigmas = torch.full((4, 1), 0.25)
    boundary = torch.tensor([
        [1.0, 0.5],
        [-1.0, -0.3],
        [0.4, 1.0],
        [-0.2, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ])
    phi, grad_phi = eval_gaussian_phi(boundary, centers, sigmas)
    assert phi.shape == (4, 8)
    assert phi.abs().max().item() < 1e-12, (
        f"phi must vanish on boundary, got max |phi| = {phi.abs().max():.4e}"
    )


def test_eval_gaussian_phi_grad_matches_autograd():
    """Hand-derived grad_phi must agree with autograd of phi, across centers
    that are both at origin and off-origin. Catches a product-rule typo."""
    centers = torch.tensor([[0.0, 0.0], [0.3, -0.2], [-0.5, 0.4]])
    sigmas = torch.tensor([[0.2], [0.15], [0.3]])
    xy = (2.0 * torch.rand(50, 2, generator=torch.Generator().manual_seed(7)) - 1.0) * 0.9
    xy = xy.detach().requires_grad_(True)
    phi, grad_phi = eval_gaussian_phi(xy, centers, sigmas)
    # Sum phi over collocation points per test function so we can take a
    # single backward pass and recover per-point gradients via the linear
    # structure (grad_xy of sum_n phi[m, n] gives, per-row, ∂phi[m, n]/∂xy[n]).
    grad_phi_ag = torch.zeros_like(grad_phi)
    for m in range(phi.shape[0]):
        g_m = torch.autograd.grad(
            phi[m].sum(), xy, retain_graph=True, create_graph=False
        )[0]
        grad_phi_ag[m] = g_m
    assert torch.allclose(grad_phi, grad_phi_ag, atol=1e-5), (
        f"max diff {(grad_phi - grad_phi_ag).abs().max():.4e}"
    )


def test_make_gaussian_tests_in_range():
    centers, sigmas = make_gaussian_tests(32, device="cpu", sigma=0.2)
    assert centers.shape == (32, 2)
    assert sigmas.shape == (32, 1)
    assert centers.min().item() >= -0.7 - 1e-6
    assert centers.max().item() <= 0.7 + 1e-6
    assert torch.all(sigmas == 0.2)


def test_strong_form_residual_zero_on_analytic_solution():
    """THE Laplacian-wiring check. Strong residual on c=exp(-2π²Dt)sin(πx)sin(πy) ≈ 0."""
    D_val = 0.05
    model = _AnalyticFRAP(D_val, k_value=0.0)
    xyt = _interior_grid(800, eps=0.05, seed=0)
    r = strong_form_residual(model, xyt)
    max_abs = r.abs().max().item()
    assert math.isfinite(max_abs)
    assert max_abs < 1e-4, f"max strong residual {max_abs:.4e} (sign/Laplacian wiring bug?)"


def test_strong_form_residual_includes_k_term():
    """With k ≠ 0 and analytic c (which satisfies only the diffusion part), residual ≈ k·c."""
    D_val = 0.05
    k_val = 0.7
    model = _AnalyticFRAP(D_val, k_value=k_val)
    xyt = _interior_grid(800, eps=0.05, seed=1)
    r = strong_form_residual(model, xyt).detach()
    c = model(xyt).detach()
    err = (r - k_val * c).abs().max().item()
    assert err < 1e-4, f"strong residual minus k*c = {err:.4e}; k term wiring wrong"


def test_weak_form_residual_near_zero_origin_centered_tests():
    """With the boundary-cutoff W = (1 - x^2)^2 (1 - y^2)^2 baked into
    eval_gaussian_phi, the IBP boundary integral vanishes identically (phi = 0
    on partial Omega), so weak_form_residuals on the sin analytic solution
    must be 0 modulo Monte Carlo noise."""
    D_val = 0.05
    model = _AnalyticFRAP(D_val, k_value=0.0)
    torch.manual_seed(0)
    xyt = (2.0 * torch.rand(20000, 3) - 1.0).requires_grad_(True)

    c = model(xyt)
    grad_c = gradients(c, xyt)
    c_x, c_y, c_t = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]

    n_tests = 16
    centers = torch.zeros(n_tests, 2)
    sigmas = torch.linspace(0.1, 0.35, n_tests).unsqueeze(-1)

    xy = xyt[:, 0:2]
    phi, grad_phi = eval_gaussian_phi(xy, centers, sigmas)
    grad_c_xy = torch.cat([c_x, c_y], dim=1)
    term_t = (phi * c_t.squeeze(-1).unsqueeze(0)).mean(dim=1)
    dot = (grad_phi * grad_c_xy.unsqueeze(0)).sum(dim=-1)
    term_diff = D_val * dot.mean(dim=1)
    residual = term_t + term_diff

    max_abs = residual.abs().max().item()
    assert math.isfinite(max_abs)
    assert max_abs < 5e-3, (
        f"max weak residual {max_abs:.4e} (sign/IBP wiring bug?). "
        f"This test uses origin-centered Gaussians so the boundary integral "
        f"vanishes by symmetry; nonzero output here implies a real bug."
    )


def test_weak_form_residual_includes_k_term_origin_centered():
    """With analytic c and k>0, weak residual against centered Gaussian
    should equal ∫φ·k·c (the only surviving term beyond the cancelling pair)."""
    D_val = 0.05
    k_val = 0.7
    model = _AnalyticFRAP(D_val, k_value=k_val)
    torch.manual_seed(1)
    xyt = (2.0 * torch.rand(20000, 3) - 1.0).requires_grad_(True)

    c = model(xyt)
    grad_c = gradients(c, xyt)
    c_x, c_y, c_t = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]

    n_tests = 8
    centers = torch.zeros(n_tests, 2)
    sigmas = torch.linspace(0.1, 0.3, n_tests).unsqueeze(-1)

    xy = xyt[:, 0:2]
    phi, grad_phi = eval_gaussian_phi(xy, centers, sigmas)
    grad_c_xy = torch.cat([c_x, c_y], dim=1)
    term_t = (phi * c_t.squeeze(-1).unsqueeze(0)).mean(dim=1)
    dot = (grad_phi * grad_c_xy.unsqueeze(0)).sum(dim=-1)
    term_diff = D_val * dot.mean(dim=1)
    term_decay = k_val * (phi * c.squeeze(-1).unsqueeze(0)).mean(dim=1)
    residual = term_t + term_diff + term_decay
    expected = term_decay

    err = (residual - expected).abs().max().item()
    assert err < 5e-3, f"weak residual minus k-term = {err:.4e}"


def test_weak_form_residual_near_zero_off_center_with_cutoff():
    """The whole POINT of the boundary cutoff: residual ≈ 0 even when test
    function centers are NOT at the origin (symmetry argument no longer
    applies). Without the cutoff, this test would fail by an amount equal to
    D * boundary_integral_of_phi_grad_c.
    """
    D_val = 0.05
    model = _AnalyticFRAP(D_val, k_value=0.0)
    torch.manual_seed(2)
    xyt = (2.0 * torch.rand(40000, 3) - 1.0).requires_grad_(True)

    c = model(xyt)
    grad_c = gradients(c, xyt)
    c_x, c_y, c_t = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]

    # Off-center, asymmetric placements - exactly what make_gaussian_tests produces.
    centers = torch.tensor([
        [0.5, 0.3],
        [-0.4, 0.6],
        [0.6, -0.5],
        [-0.3, -0.4],
    ])
    sigmas = torch.full((4, 1), 0.25)

    xy = xyt[:, 0:2]
    phi, grad_phi = eval_gaussian_phi(xy, centers, sigmas)
    grad_c_xy = torch.cat([c_x, c_y], dim=1)
    term_t = (phi * c_t.squeeze(-1).unsqueeze(0)).mean(dim=1)
    dot = (grad_phi * grad_c_xy.unsqueeze(0)).sum(dim=-1)
    term_diff = D_val * dot.mean(dim=1)
    residual = term_t + term_diff

    max_abs = residual.abs().max().item()
    assert math.isfinite(max_abs)
    assert max_abs < 5e-3, (
        f"max weak residual {max_abs:.4e} (off-center, cutoff should make this ~ 0)"
    )


def test_weak_form_residuals_library_runs_end_to_end():
    """End-to-end smoke: the public weak_form_residuals returns a finite
    (n_tests,) tensor of the right dtype on a real PINN_FRAP model."""
    model = PINN_FRAP(hidden=16, depth=2, init_D=0.05)
    torch.manual_seed(0)
    xyt = 2.0 * torch.rand(500, 3) - 1.0
    r = weak_form_residuals(model, xyt, n_tests=8, sigma=0.25)
    assert r.shape == (8,), r.shape
    assert r.dtype == torch.float32
    assert torch.isfinite(r).all()


def test_compute_residuals_fair_returns_detached_both():
    """compute_residuals_fair returns (strong, weak) detached, on a vanilla model."""
    model = PINN_FRAP(hidden=16, depth=2, init_D=0.05)
    torch.manual_seed(0)
    xyt = 2.0 * torch.rand(200, 3) - 1.0
    strong_r, weak_r = compute_residuals_fair(model, xyt, n_tests=8, sigma=0.25)
    assert strong_r.shape == (200, 1)
    assert weak_r.shape == (8,)
    assert not strong_r.requires_grad
    assert not weak_r.requires_grad


@pytest.mark.parametrize("D_val", [0.005, 0.05, 0.5])
def test_strong_form_residual_zero_across_D_scales(D_val: float):
    """Analytic-solution check should hold for any D > 0 (we span the LS-fit
    32ww/56ww normalized range ~ 0.013 - 0.13 plus extremes)."""
    model = _AnalyticFRAP(D_val, k_value=0.0)
    xyt = _interior_grid(400, eps=0.05, seed=int(D_val * 1000))
    r = strong_form_residual(model, xyt)
    max_abs = r.abs().max().item()
    assert max_abs < 5e-4, f"D={D_val}: max residual {max_abs:.4e}"
