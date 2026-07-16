"""Mixture of Experts and DenseFFN (Phase 2).

Implements the DeepSeekMoE (architecture doc §2.5 and §2.6, roadmap B3):

- FP32 router cast in ``gate_forward``.
- 16 routed experts + 1 shared expert + top-2 routing.
- Aux-loss-free routing (bias updated from EMA-smoothed expert counts,
  threshold 1.05×).
- Per-expert :class:`SwiGLUExpert` modules (each its own FSDP instance
  in Phase 3 for sort-by-size sharding).

:class:`DenseFFN` is the dense SwiGLU used on GDN blocks (design §2.6).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig

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
        """SwiGLU: ``w2(silu(w1(x)) * w3(x))``."""
        return cast(
            torch.Tensor, self.w2(F.silu(self.w1(x)) * self.w3(x))
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
        """Dense SwiGLU: ``w2(silu(w1(x)) * w3(x))``."""
        return cast(
            torch.Tensor, self.w2(F.silu(self.w1(x)) * self.w3(x))
        )


class DeepSeekMoE(nn.Module):
    """DeepSeek-style MoE with aux-loss-free routing.

    Architecture doc §2.5. Phase 2 implementation.

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
        """FP32 router cast (design §2.5).

        The gate matmul is computed in float32 (input + weight + bias)
        to avoid sigmoid rounding at BF16, then cast back to the input
        dtype for downstream routing. Stored as ``_last_indices`` so
        :meth:`update_gate_bias` can consume the most recent routing.
        """
        x_fp32 = x.float()
        w_fp32 = self.gate.weight.float()
        b_fp32 = self.gate.bias.float()
        logits = F.linear(x_fp32, w_fp32, b_fp32)            # (..., n_routed)
        return logits.to(x.dtype)

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """EMA-smoothed expert-load bias update (design §2.5).

        Updates ``ema_expert_counts`` with the most recent routing
        counts, then nudges the gate bias toward load balance:
        over-loaded experts (``> avg * 1.05``) are penalized, under-
        loaded experts (``< avg * 0.95``) are rewarded. The threshold
        tightens the prior 1.10× to 1.05×.
        """
        if getattr(self, "_last_indices", None) is None:
            return
        counts = torch.bincount(
            self._last_indices.flatten(), minlength=self.n_routed
        ).float()
        ema = cast(torch.Tensor, self.ema_expert_counts)
        ema.mul_(1.0 - self.ema_alpha).add_(counts, alpha=self.ema_alpha)
        avg = ema.mean()
        over = ema > avg * 1.05
        under = ema < avg * 0.95
        with torch.no_grad():
            new_bias = self.gate.bias.clone()
            new_bias[over] -= speed
            new_bias[under] += speed
            self.gate.bias.copy_(new_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE forward: top-k routing over activated tokens (design §2.5).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, dim)`` — the weighted sum of the
            top-k routed experts plus the (always-on) shared expert.
        """
        B, T, D = x.shape
        # Router logits in FP32 (design §2.5).
        logits = self.gate_forward(x)                        # (B, T, n_routed)
        probs = F.softmax(logits.float(), dim=-1)           # (B, T, n_routed)
        # Top-k expert selection (capacity-aware).
        k = min(self.n_activated, self.n_routed)
        top_weights, top_indices = torch.topk(probs, k, dim=-1)  # (B, T, k)
        self._last_indices = top_indices                   # for EMA bias update

        # Flatten the token dimension for expert dispatch.
        x_flat = x.view(B * T, D)
        out = x_flat.new_zeros(B * T, D)

        # Capacity cap: max tokens dispatched to any single expert.
        capacity = int(self.capacity_factor * (B * T * k) / self.n_routed)
        capacity = max(capacity, 1)

        # Dispatch each activated slot to its chosen expert. Outer loop is
        # over experts (n_routed=16) so the per-expert matmul is batched;
        # capacity capping keeps the first `capacity` tokens per expert.
        for e in range(self.n_routed):
            # Tokens routed to expert e in any of the k slots.
            e_mask = (top_indices == e)                     # (B, T, k) bool
            flat_mask = e_mask.any(dim=-1).reshape(-1)      # (B*T,)
            sel = flat_mask.nonzero(as_tuple=False).reshape(-1)
            if sel.numel() == 0:
                continue
            # Apply capacity cap: keep only the first `capacity` tokens.
            if sel.numel() > capacity:
                sel = sel[:capacity]
            # Routing weight = sum of weights across the slots that chose e.
            w_e = probs.gather(-1, top_indices).masked_fill(
                ~e_mask, 0.0
            ).sum(dim=-1).reshape(-1)                       # (B*T,)
            w_e = w_e[sel].unsqueeze(-1)                    # (n_e, 1)
            y_e = self.experts[e](x_flat[sel])              # (n_e, D)
            out.index_add_(0, sel, y_e * w_e)
        # Shared expert is always active (added to every token).
        if self.shared_expert is not None:
            out = out + self.shared_expert(x_flat)
        return out.view(B, T, D)
