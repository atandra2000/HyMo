"""Gated Delta Net block (Phase 1 placeholder).

The real implementation (architecture doc §2.3, roadmap B1) wires
:func:`fla.layers.gated_delta_net.chunk_gated_delta_rule` for the
recurrence and supports the per-layer ``use_rope`` flag for the
NoPE-hybrid pattern.

This placeholder:

- Subclasses :class:`torch.nn.Module` with the right parameter names
  and shapes (so the FSDP wrapper in Phase 3 can wrap it correctly).
- Forwards raise :class:`hymo.core.exceptions.NotImplementedError_`.
- The :class:`hymo.core.exceptions.HyMoError` inheritance lets callers
  catch the placeholder uniformly.
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.exceptions import NotImplementedError_
from hymo.models.rope import RotaryEmbedding

__all__ = ["GatedDeltaNetBlock"]


class GatedDeltaNetBlock(nn.Module):
    """Gated Delta Net (linear attention) block.

    Architecture doc §2.3. Phase 1 placeholder.

    Parameters
    ----------
    config : ModelConfig
        The model config.
    layer_idx : int
        The layer index in the 32-block stack (0..31).
    use_rope : bool
        Whether to apply RoPE. Defaults to ``True``. For the NoPE-hybrid
        pattern (CR-12, default OFF for v1.0), set to ``False`` on the
        7 GDN positions {3, 7, 11, 15, 19, 23, 27}.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, use_rope: bool = True) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        self._config = config

        d_model = config.dim
        d_inner = config.gdn_d_inner
        d_state = config.gdn_d_state
        d_conv = config.gdn_d_conv
        headdim = config.gdn_headdim
        n_heads = d_inner // headdim

        # Linear projections (the real weights live in Phase 2).
        self.in_proj = nn.Linear(d_model, 6 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1, bias=False
        )
        self.b_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.c_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.dt_proj = nn.Linear(d_inner, n_heads, bias=False)
        self.g_proj = nn.Linear(d_inner, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Scalar parameters (no weight decay).
        self.A_log = nn.Parameter(torch.zeros(n_heads * d_state))
        self.D = nn.Parameter(torch.ones(n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))

        # Optional RoPE.
        self.rope: RotaryEmbedding | None
        if use_rope:
            self.rope = RotaryEmbedding(
                head_dim=config.qk_rope_head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        else:
            self.rope = None

    @property
    def n_heads(self) -> int:
        return self._config.gdn_d_inner // self._config.gdn_headdim

    @property
    def d_inner(self) -> int:
        return self._config.gdn_d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "GatedDeltaNetBlock.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.3, "
            "roadmap B1)."
        )

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, use_rope={self.use_rope}, "
            f"d_inner={self.d_inner}, n_heads={self.n_heads}"
        )
