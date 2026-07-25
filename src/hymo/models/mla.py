"""Multi-Head Latent Attention (MLA) block (architecture doc §2.4).

Implements DeepSeek-V2/V3 MLA with MQA-4 grouping and partial-RoPE on 25% of head_dim.
Uses single SDPA call with GQA broadcast to avoid explicit broadcast materialization.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig
from hymo.models.rope import RotaryEmbedding

__all__ = ["MultiHeadLatentAttention", "MLABlock"]


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) with MQA-4 grouping.

    Features decoupled RoPE on 25% of the head dimension and low-rank compression:
    - Query: x -> wq_a -> RMSNorm -> wq_b -> q (split into q_nope and q_pe).
    - KV: x -> wkv_a -> (kv_latent, k_pe), and kv_latent -> RMSNorm -> wkv_b -> (k_nope, v).
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._config = config
        d = config.dim
        n_heads = config.n_heads
        n_kv_groups = config.n_kv_groups
        qk_rope_head_dim = config.qk_rope_head_dim
        qk_nope_head_dim = config.qk_nope_head_dim
        kv_lora_rank = config.kv_lora_rank
        q_lora_rank = config.q_lora_rank
        v_head_dim = config.v_head_dim

        self.wq_a = nn.Linear(d, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(
            q_lora_rank, n_heads * (qk_rope_head_dim + qk_nope_head_dim), bias=False
        )

        self.wkv_a = nn.Linear(d, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank,
            n_kv_groups * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(n_heads * v_head_dim, d, bias=False)

        self.q_norm_qk = nn.Parameter(
            torch.ones(n_heads * (qk_rope_head_dim + qk_nope_head_dim))
        )
        self.k_norm_qk = nn.Parameter(
            torch.ones(n_kv_groups * qk_rope_head_dim)
        )

        self.rope = RotaryEmbedding(
            head_dim=qk_rope_head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    @property
    def n_heads(self) -> int:
        return self._config.n_heads

    @property
    def n_kv_groups(self) -> int:
        return self._config.n_kv_groups

    @property
    def qk_rope_head_dim(self) -> int:
        return self._config.qk_rope_head_dim

    @property
    def qk_nope_head_dim(self) -> int:
        return self._config.qk_nope_head_dim

    @property
    def v_head_dim(self) -> int:
        return self._config.v_head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLA mapping input (B, T, d) -> (B, T, d)."""
        B, T, _ = x.shape
        H, G = self.n_heads, self.n_kv_groups
        D_pe, D_nope, D_v = (
            self.qk_rope_head_dim, self.qk_nope_head_dim, self.v_head_dim
        )

        q_latent = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_latent)
        head_dim_q = D_pe + D_nope
        q = q.view(B, T, H, head_dim_q) * self.q_norm_qk.view(1, 1, H, head_dim_q)
        q_pe = q[..., :D_pe]
        q_nope = q[..., D_pe:]
        q_pe_rot = self.rope.apply_rope(
            q_pe.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        kv = self.wkv_a(x)
        kv_latent, k_pe = kv.split(
            [self._config.kv_lora_rank, D_pe], dim=-1
        )
        kv_latent = self.kv_norm(kv_latent)
        kv_out = self.wkv_b(kv_latent).view(B, T, G, D_nope + D_v)
        k_nope = kv_out[..., :D_nope]
        v = kv_out[..., D_nope:]
        k_pe_normed = k_pe.unsqueeze(2) * self.k_norm_qk.view(1, 1, G, D_pe)
        k_pe_rot = self.rope.apply_rope(
            k_pe_normed.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        q_assembled = torch.cat([q_pe_rot, q_nope], dim=-1)
        k_assembled = torch.cat([k_pe_rot, k_nope], dim=-1)

        q_sdpa = q_assembled.permute(0, 2, 1, 3)
        k_sdpa = k_assembled.permute(0, 2, 1, 3)
        v_sdpa = v.permute(0, 2, 1, 3)
        heads_per_group = H // G
        try:
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                enable_gqa=True,
            )
        except (TypeError, RuntimeError):
            k_sdpa = k_sdpa.repeat_interleave(heads_per_group, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(heads_per_group, dim=1)
            out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_v)
        y = self.wo(out)
        return cast(torch.Tensor, y)


class MLABlock(nn.Module):
    """MLA Block combining latent attention, MoE feed-forward, pre-norms, and residuals."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._config = config
        from hymo.models.moe import DeepSeekMoE

        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
        self.moe_norm = nn.RMSNorm(config.dim)
        self.moe = DeepSeekMoE(config, layer_idx=layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying pre-norm Attention followed by pre-norm MoE."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x
