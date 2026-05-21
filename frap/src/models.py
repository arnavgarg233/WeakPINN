"""FRAP PINN model definitions.

Implements PLAN.md Phase 5 verbatim:

  MLP        - tanh-activated fully-connected coordinate network.
  PINN_FRAP  - wraps the MLP with learnable D (and optionally learnable k).

Training stays CPU-or-MPS agnostic; this module never assumes a device.
The constructor takes init_D in physical (positive) units; we store its
softplus inverse so that the learnable raw parameter can range over R while
exposed D = softplus(raw_D) + 1e-8 stays strictly positive.

When learn_k=False (default), k is held at zero via a registered buffer.
When learn_k=True, k = softplus(raw_k) is also positive. Phase 8.5 of
PLAN.md prescribes learn_k=True for the real-data runs because imaging-
bleach during the 26.5 s recovery window is non-negligible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 128, depth: int = 4, out_dim: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)


class PINN_FRAP(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        depth: int = 4,
        init_D: float = 0.1,
        learn_k: bool = False,
    ) -> None:
        """init_D=0.1 deliberately above typical true ~0.05 so optimizer descends.

        Phase 8.5 override for real runs: call with init_D=0.05 (matched for
        strong and weak so neither method has a head start).
        """
        super().__init__()
        self.field = MLP(3, hidden, depth, 1)
        init_raw = torch.log(torch.exp(torch.tensor(init_D)) - 1.0)
        self.raw_D = nn.Parameter(init_raw.clone().float())
        self.learn_k = learn_k
        if learn_k:
            self.raw_k = nn.Parameter(torch.tensor(-8.0))
        else:
            self.register_buffer("_zero_k", torch.tensor(0.0))

    def D(self) -> torch.Tensor:
        return F.softplus(self.raw_D) + 1e-8

    def k(self) -> torch.Tensor:
        if self.learn_k:
            return F.softplus(self.raw_k)
        return self._zero_k

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.field(xyt)
