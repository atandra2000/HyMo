"""Mixture of Experts and DenseFFN (Phase 1 placeholders).

The real implementation (architecture doc §2.5 and §2.6, roadmap B3)
implements the DeepSeekMoE with:

- FP32 router cast in ``gate_forward``.
- 16 routed experts + 1 shared expert + top-2 routing.
- Aux-loss-free routing (bias updated from EMA-smoothed expert counts,
  threshold 1.05×).
- Per-expert :class:`SwiGLUExpert` modules (each its own FSDP instance
  in Phase 3 for sort-by-size sharding).

:class:`DenseFFN` is the dense SwiGLU used on GDN blocks (design §2.6).

This placeholder defines the parameter shapes so the FSDP wrapper in
Phase 3 can correctly wrap each expert; forward passes raise
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = ["SwiGLUExpert", "DenseFFN", "DeepSeekMoE"]


class SwiGLUExpert(nn.Module):
    """A single SwiGLU expert — the workhorse of the MoE.

    Three linear projections: ``w1`` (gate), ``w2`` (down), ``w3`` (up).
    Each expert is wrapped as its own FSDP instance in Phase 3
    (architecture doc §13.2).
    """

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "SwiGLUExpert.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.5)."
        )


class DenseFFN(nn.Module):
    """Dense SwiGLU FFN (used on GDN blocks).

    Architecture doc §2.6. Three projections: ``w1`` (gate), ``w2``
    (down), ``w3`` (up). Inter dim is :attr:`ModelConfig.inter_dim`
    (default 2560).
    """

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "DenseFFN.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.6)."
        )


class DeepSeekMoE(nn.Module):
    """DeepSeek-style MoE with aux-loss-free routing.

    Architecture doc §2.5. Phase 1 placeholder.

    The v1.0 spec is:

    - :attr:`n_routed_experts` (16) routed + :attr:`n_shared_experts` (1)
      shared + :attr:`n_activated_experts` (2) top-k.
    - FP32 router (the gate forward casts input + weight to FP32).
    - Aux-loss-free: :attr:`balance_loss_alpha = 0`; gate bias is
      updated from EMA-smoothed expert counts, threshold 1.05×.
    - Capacity factor 1.5 (overflow tokens are dropped).

    In Phase 3 the MoE expert weights are partitioned to AdamW (not
    NorMuon) per the optimizer-partition spec (design §5.1, claim 2).
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._config = config
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.n_activated = config.n_activated_experts
        self.moe_inter_dim = config.moe_inter_dim
        self.ema_alpha = config.moe_ema_alpha
        self.capacity_factor = config.moe_capacity_factor

        # Gate (router).
        self.gate = nn.Linear(config.dim, self.n_routed, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.006)

        # Routed experts.
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config.dim, self.moe_inter_dim) for _ in range(self.n_routed)]
        )

        # Shared expert.
        self.shared_expert: SwiGLUExpert | None
        if self.n_shared > 0:
            self.shared_expert = SwiGLUExpert(config.dim, self.moe_inter_dim)
        else:
            self.shared_expert = None

        # EMA-tracked expert counts (non-persistent buffer).
        self.register_buffer(
            "ema_expert_counts",
            torch.zeros(self.n_routed),
            persistent=False,
        )

    def gate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 router cast (Phase 2 real implementation).

        Phase 1 placeholder: just runs the linear in the input dtype
        so the module is constructable; the real FP32 cast lands in
        Phase 2 (design §2.5).
        """
        return cast(torch.Tensor, self.gate(x))

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """EMA-smoothed expert-load bias update (Phase 2)."""
        raise NotImplementedError_(
            "DeepSeekMoE.update_gate_bias is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.5)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "DeepSeekMoE.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.5, "
            "roadmap B3)."
        )
