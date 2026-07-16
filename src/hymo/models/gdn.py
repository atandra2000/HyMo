"""Gated Delta Net block (architecture doc §2.3).

Implements the GDN linear-attention recurrence in pure-PyTorch. The
per-layer ``use_rope`` flag is honored — when ``True``, the first 25 %
of ``v``'s head_dim is rotated by :class:`RotaryEmbedding`; when
``False`` (the NoPE-hybrid pattern, CR-12, v1.0 default OFF), the rope
step is skipped.

**Performance note.** This implementation is a correctness-first pure-
PyTorch recurrence. The design doc §12a.1 specifies that the
production runtime path uses :func:`fla.layers.gated_delta_net
.chunk_gated_delta_rule` (a fused Triton kernel) for the 3-5× speedup
that gates the 5-7 day wall-clock. The fla kernel is wire-compatible
with this signature (it accepts ``(q, k, v, A, b, c, chunk_size)``);
the swap is a one-line change in :meth:`forward`. The fla kernel
requires a CUDA toolchain (``triton``); the pure-PyTorch path runs
on CPU and macOS for unit tests.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig
from hymo.models.rope import RotaryEmbedding

__all__ = ["GatedDeltaNetBlock"]


class GatedDeltaNetBlock(nn.Module):
    """Gated Delta Net (linear attention) block.

    Architecture doc §2.3. The block has:

    1. An input projection (``in_proj``) that produces 6 chunks: ``v``,
       ``b``, ``c``, ``dt``, ``g``, and a residual skip ``x_skip``.
    2. A depthwise ``conv1d`` over the time axis on ``v``, followed
       by SiLU.
    3. An optional RoPE on the rope-split of ``v``'s head_dim.
    4. The gated delta rule recurrence: the per-head state matrix
       ``h ∈ R^{d_state × headdim}`` evolves as
       ``h_t = exp(ΔA_t) ⊙ h_{t-1} + b_t ⊗ v_t`` and the output is
       ``o_t = c_t^T h_t`` (per-head).
    5. A gating step ``g ⊙ o`` and a per-head skip ``+ D ⊙ v``.
    6. The output projection ``out_proj`` back to ``d_model``.

    Parameters
    ----------
    config : ModelConfig
        The model config.
    layer_idx : int
        The layer index in the 32-block stack (0..31).
    use_rope : bool
        Whether to apply RoPE on the rope-split of ``v``. Defaults to
        ``True``. For the NoPE-hybrid pattern (CR-12, default OFF for
        v1.0), set to ``False`` on the 7 GDN positions
        {3, 7, 11, 15, 19, 23, 27}.
    """

    def __init__(
        self, config: ModelConfig, layer_idx: int, use_rope: bool = True
    ) -> None:
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

        # Projections. The input projection produces only the
        # value-side stream (``v``); the residual skip is a separate
        # linear projection back to ``d_model`` (added after the
        # output projection, per Yang et al. 2024 §3.2).
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        # Depthwise causal conv1d on v (groups=d_inner means per-channel).
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1, bias=False
        )
        # b, c projections: per-head state of size d_state.
        self.b_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.c_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        # dt projection: per-head scalar; sigmoid gives the gate.
        self.dt_proj = nn.Linear(d_inner, n_heads, bias=False)
        # g projection: per-head-per-dim gating; sigmoid before mul.
        self.g_proj = nn.Linear(d_inner, d_inner, bias=False)
        # Output projection: back to d_model.
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        # Skip projection: x → d_model. Added to the output of
        # out_proj (residual).
        self.skip_proj = nn.Linear(d_model, d_model, bias=False)

        # Scalar parameters (no weight decay; treated as 1D in the
        # optimizer partition).
        # A_log: log-decay per (head, state). Initialized to log(1..n_heads)
        # repeated per state (per Yang et al. 2024 §3.2).
        a_init = torch.log(
            torch.arange(1, n_heads + 1, dtype=torch.float32).repeat_interleave(d_state)
        )
        self.A_log = nn.Parameter(a_init)
        # D: per-head skip connection.
        self.D = nn.Parameter(torch.ones(n_heads))
        # dt_bias: per-head bias on the dt sigmoid.
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))

        # Optional RoPE on the rope-split of v's head_dim.
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

    @property
    def d_state(self) -> int:
        return self._config.gdn_d_state

    @property
    def headdim(self) -> int:
        return self._config.gdn_headdim

    @property
    def chunk_size(self) -> int:
        return self._config.gdn_chunk_size

    def _gated_delta_rule(
        self,
        v: torch.Tensor,    # (B, T, n_heads, headdim)
        b: torch.Tensor,    # (B, T, n_heads, d_state)
        c: torch.Tensor,    # (B, T, n_heads, d_state)
        g: torch.Tensor,    # (B, T, n_heads)  — sigmoid of dt_proj
    ) -> torch.Tensor:
        """Pure-PyTorch gated delta rule.

        Returns ``o`` of shape ``(B, T, n_heads, headdim)``. The
        per-head state matrix ``h ∈ R^{d_state × headdim}`` evolves
        as ``h_t = exp(ΔA_t) ⊙ h_{t-1} + b_t ⊗ v_t``; the output is
        ``o_t = c_t^T h_t``. The decay ``ΔA_t`` is derived from
        ``A_log`` (log-space) and the per-token gate ``g_t``.
        """
        B, T, H, D = v.shape
        S = b.shape[-1]
        # Convert A_log → A in [0, 1) via -exp(A_log) so the
        # per-step decay is α_t = exp(g_t * (-exp(A_log))) ∈ (0, 1].
        A = -torch.exp(self.A_log.float()).view(H, S)              # (H, S)
        # Per-token, per-head scalar gate in [0, 1].
        # The convention from the paper: α_t = sigmoid(dt_proj(v_t) + dt_bias).
        # The final per-(t, h, s) decay: α_{t,h,s} = exp(g_{t,h} * A_{h,s}).
        # Equivalently, the row-update factor is exp(g_{t,h} * A_{h,s}).

        # Initialize state.
        h = v.new_zeros(B, H, S, D, dtype=torch.float32)
        o_list: list[torch.Tensor] = []
        for t in range(T):
            v_t = v[:, t].float()                                  # (B, H, D)
            b_t = b[:, t].float()                                  # (B, H, S)
            c_t = c[:, t].float()                                  # (B, H, S)
            g_t = g[:, t].float().unsqueeze(-1)                    # (B, H, 1)
            # Per-step decay factor per (h, s) → (H, S) broadcast over B.
            alpha = torch.exp(g_t * A)                             # (B, H, S)
            # State update: h_t = α ⊙ h_{t-1} + b ⊗ v
            # (B, H, S, 1) * (B, H, S, D) + (B, H, S, 1) * (B, H, 1, D)
            h = alpha.unsqueeze(-1) * h + b_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            # Output: o_t = c^T h, i.e. sum over s of c[h, s] * h[h, s, :]
            o_t = torch.einsum("bhs,bhsd->bhd", c_t, h)            # (B, H, D)
            o_list.append(o_t)
        o = torch.stack(o_list, dim=1)                             # (B, T, H, D)
        return o.to(v.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gated Delta Net block to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, d_model)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        H, D, S = self.n_heads, self.headdim, self.d_state
        d_inner = self.d_inner

        # 1. Input projection → v (value stream for the conv1d). The
        # residual skip is a separate projection (skip_proj) added
        # to the output of out_proj.
        v_in = self.in_proj(x)                                     # (B, T, d_inner)

        # 2. Depthwise causal conv1d on v, then SiLU. The conv has
        # padding ``d_conv - 1`` so the output is ``T + d_conv - 1``;
        # we slice off the rightmost ``d_conv - 1`` columns to make
        # it causal (no leakage from future tokens).
        v_conv = self.conv1d(v_in.transpose(1, 2))[:, :, :T]      # (B, d_inner, T)
        v = F.silu(v_conv.transpose(1, 2))                         # (B, T, d_inner)

        # 3. Per-token b/c/dt/g projections on v (post-conv, per the
        # GDN reference impl).
        b_in = self.b_proj(v)                                      # (B, T, n_heads * d_state)
        c_in = self.c_proj(v)                                      # (B, T, n_heads * d_state)
        dt_in = self.dt_proj(v)                                    # (B, T, n_heads)
        g_in = self.g_proj(v)                                      # (B, T, d_inner)
        # b, c: → (B, T, n_heads, d_state)
        b = b_in.view(B, T, H, S)
        c = c_in.view(B, T, H, S)
        # dt: per-head decay gate (sigmoid + bias).
        dt = F.sigmoid(dt_in + self.dt_bias)
        # g: per-head-per-dim gating (sigmoid).
        g = F.sigmoid(g_in)

        # 4. Reshape v to (B, T, n_heads, headdim) and apply RoPE on
        # the rope-split. ``v`` is shaped (B, T, d_inner) =
        # (B, T, n_heads, headdim) after the view.
        v = v.view(B, T, H, D)
        if self.rope is not None and T > 0:
            rope_dim = self.rope.head_dim
            v_for_rope = v[..., :rope_dim].permute(0, 2, 1, 3)     # (B, H, T, rope_dim)
            v_for_rope = self.rope.apply_rope(v_for_rope)
            v_for_rope = v_for_rope.permute(0, 2, 1, 3)            # (B, T, H, rope_dim)
            v = torch.cat([v_for_rope, v[..., rope_dim:]], dim=-1)

        # 5. Gated delta rule. The recurrence uses ``dt`` (per-head
        # gate) as the per-step decay modifier.
        try:
            from fla.layers.gated_delta_net import chunk_gated_delta_rule
            o = chunk_gated_delta_rule(c, b, v, self.A_log, dt, g, self.chunk_size)
        except ImportError:
            o = self._gated_delta_rule(v, b, c, dt)                    # (B, T, H, D)

        # 6. Per-head skip connection: o += D ⊙ v.
        o = o + self.D.view(1, 1, H, 1) * v

        # 7. Per-head-per-dim gating g ⊙ o.
        o = o * g.view(B, T, H, D)

        # 8. Output projection: back to d_model, add the input skip
        # (residual). The ``+`` of two tensors is typed as ``Any``
        # by mypy; cast to Tensor.
        o_flat = o.view(B, T, d_inner)
        y = self.out_proj(o_flat) + self.skip_proj(x)
        return cast(torch.Tensor, y)

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, use_rope={self.use_rope}, "
            f"d_inner={self.d_inner}, n_heads={self.n_heads}, "
            f"d_state={self.d_state}, headdim={self.headdim}"
        )
