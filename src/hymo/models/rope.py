"""Rotary Position Embedding (RoPE) and NoPE helpers.

Precomputes cos/sin tables once and applies standard per-pair complex rotation
on the specified partial head dimension slice (NoPE-hybrid pattern).
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.types import DType

__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) cache (Phase 2)."""

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

        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)
        cos_tab = angles.cos().repeat_interleave(2, dim=-1)
        sin_tab = angles.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos_tab.to(self._dtype), persistent=False)
        self.register_buffer("sin_cached", sin_tab.to(self._dtype), persistent=False)

    def apply_rope(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE rotation to input tensor x of shape (..., T, head_dim)."""
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"x.shape[-1] ({x.shape[-1]}) must equal "
                f"self.head_dim ({self.head_dim})"
            )
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")
        seq_len = x.shape[-2]
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"start_pos + seq_len ({start_pos + seq_len}) exceeds "
                f"max_seq_len ({self.max_seq_len})"
            )

        cos = self.cos_cached[start_pos:start_pos + seq_len].to(x.dtype).view(1, seq_len, self.head_dim)
        sin = self.sin_cached[start_pos:start_pos + seq_len].to(x.dtype).view(1, seq_len, self.head_dim)

        cos_even, cos_odd = cos[..., 0::2], cos[..., 1::2]
        sin_even, sin_odd = sin[..., 0::2], sin[..., 1::2]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rot_even = x_even * cos_even - x_odd * sin_even
        rot_odd = x_even * sin_odd + x_odd * cos_odd

        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}"
        )
