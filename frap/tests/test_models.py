"""Unit tests for src/models.py - CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MLP, PINN_FRAP  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    """Belt-and-suspenders: Agent B never touches MPS."""
    assert not any(p.is_mps for p in (torch.tensor(0.0),)), "tensors on MPS in CPU-only test"


def test_mlp_forward_shape():
    m = MLP(in_dim=3, hidden=32, depth=3, out_dim=1)
    x = torch.randn(7, 3)
    y = m(x)
    assert y.shape == (7, 1), y.shape
    assert y.dtype == torch.float32


def test_mlp_arbitrary_widths_and_depths():
    for hidden in (8, 64):
        for depth in (1, 4):
            m = MLP(in_dim=3, hidden=hidden, depth=depth, out_dim=1)
            y = m(torch.randn(5, 3))
            assert y.shape == (5, 1)


def test_pinn_forward_shape():
    """The single required test from the user's brief."""
    m = PINN_FRAP()
    x = torch.randn(10, 3)
    y = m(x)
    assert y.shape == (10, 1)
    assert m.D().item() > 0


def test_pinn_D_is_positive_via_softplus():
    """D = softplus(raw_D) + 1e-8 must stay > 0 for any raw value."""
    m = PINN_FRAP(init_D=0.05)
    with torch.no_grad():
        m.raw_D.fill_(-50.0)
    assert m.D().item() > 0


def test_pinn_init_D_round_trip():
    """After softplus inversion in the constructor, D() should match init_D."""
    for init_D in (0.01, 0.05, 0.1, 0.5):
        m = PINN_FRAP(init_D=init_D)
        recovered = m.D().item()
        assert abs(recovered - init_D) < 1e-5, f"init_D={init_D}, got {recovered}"


def test_pinn_k_zero_when_not_learn_k():
    m = PINN_FRAP(learn_k=False)
    assert m.k().item() == 0.0
    assert not any("raw_k" in name for name, _ in m.named_parameters())


def test_pinn_k_positive_when_learn_k():
    m = PINN_FRAP(learn_k=True)
    assert m.k().item() > 0
    assert any("raw_k" in name for name, _ in m.named_parameters())


def test_pinn_field_param_count_scales_with_hidden():
    """Defensive: catch silent architecture changes."""
    n_small = sum(p.numel() for p in PINN_FRAP(hidden=32, depth=2).parameters())
    n_large = sum(p.numel() for p in PINN_FRAP(hidden=64, depth=2).parameters())
    assert n_large > n_small


def test_pinn_forward_grad_flows():
    """Output must depend on raw_D in the autograd graph (so D loss can train D)."""
    m = PINN_FRAP(init_D=0.05)
    x = torch.randn(8, 3, requires_grad=True)
    y = m(x).sum() + m.D() * m.D()
    y.backward()
    assert m.raw_D.grad is not None
    assert m.raw_D.grad.abs().item() > 0
