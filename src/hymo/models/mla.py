"""Multi-Head Latent Attention block (Phase 1 placeholder).

The real implementation (architecture doc §2.4, roadmap B2) wires the
MLA projection pair ``wq_a → wq_b`` and ``wkv_a → wkv_b`` and applies
partial-RoPE to the first 25% of head_dim (32 of 128 dim).

The MLA block is the *full attention* primitive. The MQA-4 grouping
(4 KV groups serving 16 query heads) is implemented in Phase 2.

This placeholder:

- Subclasses :class:`torch.nn.Module` with the right parameter names
  and shapes.
- Forward raises :class:`NotImplementedError_`.
- The :class:`MLABlock` wrapper holds the MLA attention + the
  :class:`hymo.models.moe.DeepSeekMoE` (or :class:`DenseFFN`) below it.
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.exceptions import NotImplementedError_
from hymo.models.rope import RotaryEmbedding

__all__ = ["MultiHeadLatentAttention", "MLABlock"]


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) with MQA-4 grouping.

    Architecture doc §2.4. Phase 1 placeholder.

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
        self.wkv_a = nn.Linear(d, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(
            kv_lora_rank,
            n_kv_groups * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection.
        self.wo = nn.Linear(n_heads * v_head_dim, d, bias=False)

        # Per-head query/key RMSNorms.
        self.q_norm_qk = nn.Parameter(torch.ones(qk_rope_head_dim + qk_nope_head_dim))
        self.k_norm_qk = nn.Parameter(torch.ones(qk_rope_head_dim))

        # RoPE.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "MultiHeadLatentAttention.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.4, "
            "roadmap B2)."
        )


class MLABlock(nn.Module):
    """MLA + MoE + residual + norms wrapper.

    Architecture doc §2.4. The MLA block is the *full attention*
    primitive. The :class:`hymo.models.moe.DeepSeekMoE` below it is
    the asymmetric feed-forward block (MoE is restricted to MLA blocks
    per design §2.3 / §2.5).

    Phase 1 placeholder. The MoE submodule is also a placeholder.
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
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "MLABlock.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.4, "
            "roadmap B2)."
        )
