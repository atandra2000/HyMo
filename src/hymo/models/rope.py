"""Rotary Position Embedding (RoPE) and NoPE helpers.

This module implements :class:`RotaryEmbedding` (Phase 2). The cos/sin
tables are precomputed once at construction time and cached as
non-persistent buffers; :meth:`RotaryEmbedding.apply_rope` looks up the
tables by position and applies the per-pair rotation to the rope-split
slice of the head.

Wire-up (Phase 2, downstream):

- :class:`hymo.models.gdn.GatedDeltaNetBlock` instantiates
  ``self.rope = RotaryEmbedding.from_config(config)`` when ``use_rope``
  is ``True`` and applies it to the first 25 % of ``v``'s head_dim
  (design §3.1).
- :class:`hymo.models.mla.MultiHeadLatentAttention` instantiates
  ``self.rope = RotaryEmbedding.from_config(config)`` and applies it to
  the ``q_pe`` and ``k_pe`` splits (the rope-split of the head).
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.types import DType

__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) cache.

    Computes ``cos`` and ``sin`` tables of shape
    ``(max_seq_len, head_dim)`` once at construction time and applies
    them to the input via :meth:`apply_rope`.

    The rotation pairs adjacent elements ``(x[..., 0::2], x[..., 1::2])``
    as a 2D complex plane and applies the standard per-pair rotation
    (Su et al. 2021):

        x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos

    For HyMo, ``head_dim`` is the rope-split of the head (32 for the
    25 % partial-RoPE); the remainder of the head is *not* passed to
    this module (NoPE).

    Parameters
    ----------
    head_dim : int
        The dimension of the head to apply RoPE to. Must be even.
        Typically ``qk_rope_head_dim`` (32 for HyMo, the 25 % partial-RoPE).
    max_seq_len : int
        The maximum sequence length. Tables are precomputed for this
        length.
    theta : float
        The base of the RoPE frequencies (default 10,000).
    dtype : torch.dtype or None
        The dtype of the cached cos/sin tables. Defaults to ``float32``
        so the rotation is stable under BF16 inputs. The output dtype
        matches the input dtype.
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

        # Precompute the cos/sin tables once. Frequencies are
        # inv_theta^(2i / head_dim) for i in [0, head_dim/2); the
        # tables tile the half-frequencies across adjacent pairs so
        # the rotation is a single fused op (no per-element indexing).
        # Non-persistent: derived from theta, doesn't belong in
        # state_dict (would 2× checkpoint size for nothing).
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                                dtype=torch.float32)
                                  / head_dim))                  # (head_dim/2,)
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        # Outer product: angle[p, i] = p * freqs[i].
        angles = torch.outer(positions, freqs)                  # (max_seq_len, head_dim/2)
        # Tile to (max_seq_len, head_dim) by repeating the half-freqs.
        cos_tab = angles.cos().repeat_interleave(2, dim=-1)     # (max_seq_len, head_dim)
        sin_tab = angles.sin().repeat_interleave(2, dim=-1)     # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", cos_tab.to(self._dtype),
                              persistent=False)
        self.register_buffer("sin_cached", sin_tab.to(self._dtype),
                              persistent=False)
        # Typed local references so mypy can see these as Tensor
        # (not the ``Tensor | Module`` union from
        # ``nn.Module.__getattr__``).
        self._cos: torch.Tensor = self.cos_cached  # type: ignore[assignment]
        self._sin: torch.Tensor = self.sin_cached  # type: ignore[assignment]

    def apply_rope(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE to ``x`` at the given ``start_pos``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., T, head_dim)`` — the leading
            dimensions are treated as a batch and may be any size
            (typically ``(B, n_heads, T, head_dim)`` or
            ``(B, T, n_heads, head_dim)``). The last dim must equal
            ``self.head_dim``.
        start_pos : int
            The position of the first token in ``x`` along the
            sequence axis. Used during incremental decoding where the
            cache is a moving window. Defaults to 0.

        Returns
        -------
        torch.Tensor
            Same shape and dtype as ``x``; the last dim is rotated.
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"x.shape[-1] ({x.shape[-1]}) must equal "
                f"self.head_dim ({self.head_dim})"
            )
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}")
        # The sequence axis is always the second-to-last.
        seq_len = x.shape[-2]
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"start_pos + seq_len ({start_pos + seq_len}) exceeds "
                f"max_seq_len ({self.max_seq_len})"
            )

        # Lookup the (seq_len, head_dim) slice for this call. The
        # buffers follow .to(device) automatically; the input may be
        # in a different dtype, so cast on the fly. ``self._cos`` and
        # ``self._sin`` are typed references set in ``__init__`` so
        # mypy can see them as ``Tensor``.
        cos = self._cos[start_pos:start_pos + seq_len].to(x.dtype)
        sin = self._sin[start_pos:start_pos + seq_len].to(x.dtype)
        # Broadcast across all leading dims: tables are (T, head_dim),
        # x is (..., T, head_dim). Insert a leading singleton dim and
        # rely on PyTorch's left-aligned broadcasting.
        cos = cos.view(1, seq_len, self.head_dim)
        sin = sin.view(1, seq_len, self.head_dim)

        # Per-pair rotation. The cos/sin tables are already tiled
        # (head_dim) so the even and odd halves carry their own
        # values. Read them directly via strided indexing.
        cos_even = cos[..., 0::2]    # (1, T, head_dim/2)
        cos_odd = cos[..., 1::2]
        sin_even = sin[..., 0::2]
        sin_odd = sin[..., 1::2]
        x_even = x[..., 0::2]        # (..., T, head_dim/2)
        x_odd = x[..., 1::2]
        rot_even = x_even * cos_even - x_odd * sin_even
        rot_odd = x_even * sin_odd + x_odd * cos_odd
        # Interleave back to (..., T, head_dim).
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

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
