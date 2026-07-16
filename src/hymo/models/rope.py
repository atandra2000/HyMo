"""Rotary Position Embedding (RoPE) and NoPE helpers.

This module is a Phase 1 placeholder. The real implementation
(wire :class:`RotaryEmbedding` to the per-layer ``use_rope`` flag in
:class:`hymo.models.gdn.GatedDeltaNetBlock` and to the
``q_pe`` split in :class:`hymo.models.mla.MultiHeadLatentAttention`)
lands in Phase 2.

The signatures are stable; the bodies raise
:class:`hymo.core.exceptions.NotImplementedError_`.
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.exceptions import NotImplementedError_
from hymo.core.types import DType

__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) cache.

    Computes ``cos`` and ``sin`` tables of shape
    ``(max_seq_len, head_dim)`` once at construction time and applies
    them to the input via ``apply(x, start_pos)``.

    Phase 1 placeholder: this class constructs and has the right
    signature, but :meth:`apply` raises ``NotImplementedError_``.

    Parameters
    ----------
    head_dim : int
        The dimension of the head to apply RoPE to. Typically
        ``qk_rope_head_dim`` (32 for HyMo, the 25% partial-RoPE).
    max_seq_len : int
        The maximum sequence length. Tables are precomputed for this
        length.
    theta : float
        The base of the RoPE frequencies (default 10,000).
    dtype : torch.dtype or None
        The dtype of the cached tables. Defaults to the model's
        compute dtype.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        theta: float = 10_000.0,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._dtype = dtype or torch.float32
        # Precompute the cos/sin tables on construction (real impl in Phase 2).
        self._tables_computed = False

    def apply_rope(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE to ``x`` at the given ``start_pos``.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        """
        raise NotImplementedError_(
            "RotaryEmbedding.apply_rope is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §3.1, "
            "roadmap B1)."
        )

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}"
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> RotaryEmbedding:
        """Build a :class:`RotaryEmbedding` from a :class:`ModelConfig`."""
        return cls(
            head_dim=config.qk_rope_head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
