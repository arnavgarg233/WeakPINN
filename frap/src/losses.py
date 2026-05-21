"""Strong-form and weak-form FRAP residual losses (PLAN.md Phase 6 verbatim).

PDE: ∂ₜc - D(∂ₓₓc + ∂ᵧᵧc) + k·c = 0   on (x, y, t) ∈ [-1, 1]³.

Two ways to enforce it in PINN training:

  strong_form_residual(model, xyt) -> (N, 1) pointwise residual r(x,y,t).
      Requires SECOND derivatives of c w.r.t. spatial coords.
      Cost: 4 autograd passes (c, c_x, c_y, then c_xx/c_yy). Noise on c
      gets differentiated twice and is amplified.

  weak_form_residuals(model, xyt, n_tests=64, sigma=0.25, generator=None)
      Returns (n_tests,) variational residual, one entry per test function.
      Requires only FIRST derivatives of c (IBP transfers one derivative
      onto the smooth Gaussian test functions). Cost: 1 autograd pass.
      Noise on c is integrated against ∇φ (smooth and bounded), not
      differentiated. This is the Flare-PINN advantage we are reproducing
      in a second domain.

Both share an identical signature: take (model, xyt, ...) and return a
residual tensor. compute_residuals_fair returns both so we never report
only the matched residual for a matched model (reviewer fairness).
"""
from __future__ import annotations

import torch


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]


def strong_form_residual(model: torch.nn.Module, xyt: torch.Tensor) -> torch.Tensor:
    """r = c_t - D(c_xx + c_yy) + k*c"""
    xyt = xyt.clone().detach().requires_grad_(True)
    c = model(xyt)
    grad_c = gradients(c, xyt)
    c_x, c_y, c_t = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]
    grad_cx = gradients(c_x, xyt)
    grad_cy = gradients(c_y, xyt)
    c_xx, c_yy = grad_cx[:, 0:1], grad_cy[:, 1:2]
    return c_t - model.D() * (c_xx + c_yy) + model.k() * c


def make_gaussian_tests(
    n_tests: int,
    device: torch.device | str,
    sigma: float = 0.25,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if generator is None:
        centers = torch.empty(n_tests, 2, device=device).uniform_(-0.7, 0.7)
    else:
        centers = (torch.rand(n_tests, 2, generator=generator, device=device) * 1.4 - 0.7)
    sigmas = torch.full((n_tests, 1), sigma, device=device)
    return centers, sigmas


def eval_gaussian_phi(
    xy: torch.Tensor,
    centers: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """xy: (N,2). Returns phi: (M,N), grad_phi: (M,N,2).

    phi(x, y) = G(x, y; c, sigma) * W(x, y)

    where G is the Gaussian bump and W(x, y) = (1 - x^2)^2 (1 - y^2)^2 is a
    polynomial boundary cutoff on [-1, 1]^2. W (and its normal derivative)
    vanishes on the four edges of the unit square, so phi = 0 on the full
    boundary regardless of where the Gaussian center is placed. This makes
    the IBP boundary integral dropped by `weak_form_residuals` exactly zero:

        integral over partial Omega of (phi * grad c . n) dS = 0,

    which is what the weak-form derivation needs. Centers from
    `make_gaussian_tests` can sit anywhere in [-0.7, 0.7]^2 without spoiling
    that identity.

    Gradient via the product rule:
        grad phi = (grad G) * W + G * (grad W)
        d W / d x = -4 x (1 - x^2) (1 - y^2)^2
        d W / d y = -4 y (1 - x^2)^2 (1 - y^2)
    """
    diff = xy.unsqueeze(0) - centers.unsqueeze(1)            # (M, N, 2)
    r2 = (diff ** 2).sum(dim=-1)                              # (M, N)
    sigma2 = sigmas ** 2                                      # (M, 1)
    g = torch.exp(-0.5 * r2 / sigma2)                         # (M, N)

    x = xy[:, 0:1]                                            # (N, 1)
    y = xy[:, 1:2]                                            # (N, 1)
    one_minus_x2 = 1.0 - x * x                                # (N, 1)
    one_minus_y2 = 1.0 - y * y                                # (N, 1)
    wx = one_minus_x2 ** 2                                    # (N, 1)
    wy = one_minus_y2 ** 2                                    # (N, 1)
    w = (wx * wy).squeeze(-1)                                 # (N,)

    phi = g * w.unsqueeze(0)                                  # (M, N)

    grad_g = g.unsqueeze(-1) * (-diff / sigma2.unsqueeze(-1))  # (M, N, 2)

    dwdx = -4.0 * x * one_minus_x2 * wy                       # (N, 1)
    dwdy = -4.0 * y * wx * one_minus_y2                       # (N, 1)
    grad_w = torch.cat([dwdx, dwdy], dim=-1).unsqueeze(0)     # (1, N, 2)

    grad_phi = grad_g * w.unsqueeze(0).unsqueeze(-1) + g.unsqueeze(-1) * grad_w
    return phi, grad_phi


def weak_form_residuals(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    n_tests: int = 64,
    sigma: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Weak form of c_t - D∇²c + kc = 0: ∫φ c_t dΩ + D ∫∇φ·∇c dΩ + ∫φ k c dΩ = 0"""
    device = xyt.device
    xyt = xyt.clone().detach().requires_grad_(True)
    c = model(xyt)
    grad_c = gradients(c, xyt)
    c_x, c_y, c_t = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]
    xy = xyt[:, 0:2]
    centers, sigmas = make_gaussian_tests(n_tests, device, sigma, generator)
    phi, grad_phi = eval_gaussian_phi(xy, centers, sigmas)
    grad_c_xy = torch.cat([c_x, c_y], dim=1)
    term_t = (phi * c_t.squeeze(-1).unsqueeze(0)).mean(dim=1)
    dot = (grad_phi * grad_c_xy.unsqueeze(0)).sum(dim=-1)
    term_diff = model.D() * dot.mean(dim=1)
    term_decay = model.k() * (phi * c.squeeze(-1).unsqueeze(0)).mean(dim=1)
    return term_t + term_diff + term_decay


def compute_residuals_fair(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    n_tests: int = 64,
    sigma: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both residuals on a trained model. Never report only matched residual for matched model."""
    with torch.enable_grad():
        strong_r = strong_form_residual(model, xyt)
        weak_r = weak_form_residuals(model, xyt, n_tests, sigma)
    return strong_r.detach(), weak_r.detach()
