"""Mixture of Experts (Phase 2).

Implements DeepSeekMoE with 16 routed experts, 1 shared expert, top-2 routing,
and aux-loss-free routing (using EMA-smoothed load bias adjustments).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig

__all__ = ["SwiGLUExpert", "DeepSeekMoE"]


class SwiGLUExpert(nn.Module):
    """One feed-forward expert using the gated SwiGLU projection pattern."""

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass: w2(silu(w1(x)) * w3(x))."""
        return cast(
            torch.Tensor, self.w2(F.silu(self.w1(x)) * self.w3(x))
        )


class DeepSeekMoE(nn.Module):
    """DeepSeek-style routed MoE with a shared expert and EMA load balancing.

    Top-k routing is capacity-limited per expert; the shared branch gives every
    token a dense path while the routed branches provide sparse specialization.
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

        self.gate = nn.Linear(config.dim, self.n_routed, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.006)

        # Mixed-precision dispatch (design §12a.2): int16 scatter-add indices
        # + BF16 expert matmuls when the flag is on (default per config).
        self.use_mixed_precision = True

        self.experts = nn.ModuleList(
            [SwiGLUExpert(config.dim, self.moe_inter_dim) for _ in range(self.n_routed)]
        )

        self.shared_expert: SwiGLUExpert | None
        if self.n_shared > 0:
            self.shared_expert = SwiGLUExpert(config.dim, self.moe_inter_dim)
        else:
            self.shared_expert = None

        self.register_buffer(
            "ema_expert_counts",
            torch.zeros(self.n_routed),
            persistent=False,
        )

    def gate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing logits in FP32, then return them in the input dtype."""
        x_fp32 = x.float()
        w_fp32 = self.gate.weight.float()
        b_fp32 = self.gate.bias.float()
        logits = F.linear(x_fp32, w_fp32, b_fp32)
        return logits.to(x.dtype)

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """Adjust gate biases from the previous batch's EMA expert counts.

        Overloaded experts are made less likely and underused experts more likely;
        no auxiliary routing loss is added to the model objective.
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
        """Route each token to up to ``k`` experts, then add the dense shared path.

        Tokens beyond an expert's capacity are dropped from that routed branch,
        while the shared expert still contributes an output for every token.
        """
        B, T, D = x.shape
        logits = self.gate_forward(x)
        probs = F.softmax(logits.float(), dim=-1)
        k = min(self.n_activated, self.n_routed)
        top_weights, top_indices = torch.topk(probs, k, dim=-1)
        self._last_indices = top_indices

        x_flat = x.view(B * T, D)
        out = x_flat.new_zeros(B * T, D)

        capacity = int(self.capacity_factor * (B * T * k) / self.n_routed)
        capacity = max(capacity, 1)

        # Mixed-precision dispatch (design §12a.2): cast the expert input to
        # the expert weight dtype so matmuls run in that precision. Under
        # FSDP-BF16 the weights are already BF16 -> dispatch halves the input
        # bandwidth; on CPU (FP32 weights) this is a no-op.
        x_experts = (
            x_flat.to(self.experts[0].w1.weight.dtype)
            if self.use_mixed_precision
            else x_flat
        )

        for e in range(self.n_routed):
            e_mask = (top_indices == e)
            flat_mask = e_mask.any(dim=-1).reshape(-1)
            sel = flat_mask.nonzero(as_tuple=False).reshape(-1)
            sel = sel.to(torch.int32)
            if sel.numel() == 0:
                continue
            if sel.numel() > capacity:
                sel = sel[:capacity]
            w_e = probs.gather(-1, top_indices).masked_fill(
                ~e_mask, 0.0
            ).sum(dim=-1).reshape(-1)
            w_e = w_e[sel].unsqueeze(-1)
            y_e = self.experts[e](x_experts[sel])
            y_e = y_e.to(out.dtype)
            out.index_add_(0, sel, y_e * w_e.to(out.dtype))

        if self.shared_expert is not None:
            out = out + self.shared_expert(x_flat)
        return out.view(B, T, D)
