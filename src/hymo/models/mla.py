"""Multi-Head Latent Attention block (architecture doc §2.4).

Implements the DeepSeek-V2/V3 MLA with MQA-4 grouping (4 KV heads
serving 16 query heads) and partial-RoPE on the first 25 % of the
head_dim (32 of 128 dim). The full attention path is a single SDPA
call (PyTorch's scaled-dot-product attention with GQA broadcast, so
the 4 KV heads are repeated to serve the 16 query heads without an
explicit broadcast materialization).
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

    Architecture doc §2.4. Two low-rank compression paths:

    - **Query:** ``x → wq_a → q_norm → wq_b → q`` of shape
      ``(B, T, n_heads * head_dim)``. ``q`` is split per-head into
      ``q_nope`` (no position) and ``q_pe`` (position-encoded via
      RoPE on the first 25 % of the head_dim).
    - **KV:** ``x → wkv_a → (kv_latent, k_pe)`` where ``kv_latent`` is
      further expanded via ``kv_norm → wkv_b`` to per-group
      ``(k_nope, v)`` and ``k_pe`` is RoPE-encoded on the rope-split.
      The 4 KV groups are broadcast to 16 query heads via SDPA's
      ``enable_gqa``.

    Parameters
    ----------
    config : ModelConfig
    layer_idx : int
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

        # Query: low-rank projection (q_lora) then full projection (wq_b).
        self.wq_a = nn.Linear(d, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(
            q_lora_rank, n_heads * (qk_rope_head_dim + qk_nope_head_dim), bias=False
        )

        # KV: low-rank projection (wkv_a) then full projection (wkv_b).
        # wkv_a produces both the kv_latent (kv_lora_rank) and the
        # k_pe (qk_rope_head_dim) so the k_pe stays position-aware.
        self.wkv_a = nn.Linear(d, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank,
            n_kv_groups * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection: from per-head v_head_dim back to d.
        self.wo = nn.Linear(n_heads * v_head_dim, d, bias=False)

        # Per-head query/key norms. ``q_norm_qk`` is applied after
        # the per-head split, so it's per-head-per-element.
        self.q_norm_qk = nn.Parameter(
            torch.ones(n_heads * (qk_rope_head_dim + qk_nope_head_dim))
        )
        self.k_norm_qk = nn.Parameter(
            torch.ones(n_kv_groups * qk_rope_head_dim)
        )

        # RoPE for the rope-split of q and k.
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
        """Apply MLA to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, d)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, d)``.
        """
        B, T, _ = x.shape
        H, G = self.n_heads, self.n_kv_groups
        D_pe, D_nope, D_v = (
            self.qk_rope_head_dim, self.qk_nope_head_dim, self.v_head_dim
        )

        # ---- Query path ----
        # q_latent: (B, T, q_lora_rank)
        q_latent = self.q_norm(self.wq_a(x))
        # q: (B, T, n_heads * (D_pe + D_nope))
        q = self.wq_b(q_latent)
        head_dim_q = D_pe + D_nope
        # Reshape to (B, T, n_heads, head_dim_q) for per-head RMSNorm gain.
        q = q.view(B, T, H, head_dim_q) * self.q_norm_qk.view(1, 1, H, head_dim_q)
        # Split: q_pe is the rope-split (first D_pe); q_nope is the
        # last D_nope dims.
        q_pe = q[..., :D_pe]                                        # (B, T, H, D_pe)
        q_nope = q[..., D_pe:]                                      # (B, T, H, D_nope)
        # RoPE on q_pe: (B, T, H, D_pe) → (B, H, T, D_pe) → rotate → back
        q_pe_rot = self.rope.apply_rope(
            q_pe.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        # ---- KV path ----
        # wkv_a: (B, T, kv_lora_rank + D_pe)
        kv = self.wkv_a(x)
        kv_latent, k_pe = kv.split(
            [self._config.kv_lora_rank, D_pe], dim=-1
        )
        # kv_latent: (B, T, kv_lora_rank) → norm → wkv_b
        kv_latent = self.kv_norm(kv_latent)
        kv_out = self.wkv_b(kv_latent)                             # (B, T, G * (D_nope + D_v))
        kv_out = kv_out.view(B, T, G, D_nope + D_v)
        k_nope = kv_out[..., :D_nope]                               # (B, T, G, D_nope)
        v = kv_out[..., D_nope:]                                    # (B, T, G, D_v)
        # k_pe is a single shared (B, T, D_pe) vector across all G
        # KV groups (per the DeepSeek-V3 design: the rope-encoded
        # portion of the key is a single projection, not per-group).
        # Apply the per-group norm gain by broadcasting.
        k_pe_normed = k_pe.unsqueeze(2) * self.k_norm_qk.view(1, 1, G, D_pe)
        # k_pe_normed: (B, T, G, D_pe). Now RoPE on the last axis.
        k_pe_rot = self.rope.apply_rope(
            k_pe_normed.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        # ---- Assemble q, k, v with head_dim (D_pe + D_nope) ----
        # The total per-head dim is D_pe + D_nope. We concatenate
        # (q_pe_rot, q_nope) so the rope contribution is in the
        # first D_pe dims and the nope contribution in the last
        # D_nope dims. Same for k.
        q_assembled = torch.cat([q_pe_rot, q_nope], dim=-1)         # (B, T, H, head_dim_q)
        k_assembled = torch.cat([k_pe_rot, k_nope], dim=-1)         # (B, T, G, head_dim_q)

        # ---- SDPA with MQA-4 broadcast ----
        # SDPA expects (B, n_heads, T, head_dim). Use enable_gqa
        # (PyTorch ≥ 2.5) so the G=4 KV heads are broadcast to
        # H=16 query heads without explicit materialization. If
        # enable_gqa isn't supported (older torch or CPU-only path
        # that doesn't handle it), fall back to explicit broadcast.
        q_sdpa = q_assembled.permute(0, 2, 1, 3)                    # (B, H, T, head_dim_q)
        k_sdpa = k_assembled.permute(0, 2, 1, 3)                    # (B, G, T, head_dim_q)
        v_sdpa = v.permute(0, 2, 1, 3)                              # (B, G, T, D_v)
        heads_per_group = H // G
        try:
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                enable_gqa=True,
            )
        except (TypeError, RuntimeError):
            # PyTorch < 2.5 OR backend without enable_gqa support:
            # explicit broadcast of KV heads.
            k_sdpa = k_sdpa.repeat_interleave(heads_per_group, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(heads_per_group, dim=1)
            out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        # out: (B, H, T, D_v) → (B, T, H * D_v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_v)
        y = self.wo(out)
        return cast(torch.Tensor, y)


class MLABlock(nn.Module):
    """MLA + MoE + residual + norms wrapper.

    Architecture doc §2.4. The MLA block is the *full attention*
    primitive. The :class:`hymo.models.moe.DeepSeekMoE` below it is
    the asymmetric feed-forward block (MoE is restricted to MLA blocks
    per design §2.3 / §2.5).
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._config = config
        # Lazy import to avoid circular dependency at module load time.
        from hymo.models.moe import DeepSeekMoE

        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = MultiHeadLatentAttention(config, layer_idx=layer_idx)
        self.moe_norm = nn.RMSNorm(config.dim)
        # MoE-on-MLA-only is the v1.0 primary spec; see CR-12 mitigation.
        self.moe = DeepSeekMoE(config, layer_idx=layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLA block: pre-norm → attention → residual →
        pre-norm → MoE → residual."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x
