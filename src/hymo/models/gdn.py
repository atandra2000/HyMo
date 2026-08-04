"""Gated Delta Net block (architecture doc §2.3).

Implements the GDN linear-attention recurrence in pure-PyTorch.
The fused Triton kernel (``hymo.models.gdn_triton``) is the sanctioned
fast path on CUDA; ``fused_gdn`` in the training config selects it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig
from hymo.models.rope import RotaryEmbedding

__all__ = ["GatedDeltaNetBlock"]


class GatedDeltaNetBlock(nn.Module):
    """Gated Delta Net (linear attention) block (architecture doc §2.3)."""

    def __init__(
        self, config: ModelConfig, layer_idx: int, use_rope: bool = True
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        self._config = config

        # Optimization flags, defaulting to the design intent (config defaults).
        # The Trainer threads ``training.fused_gdn`` / ``torch_compile_gdn``
        # here; blocks built standalone keep the on-by-default behaviour.
        self.use_triton = True
        self.use_compile = True

        d_model = config.dim
        d_inner = config.gdn_d_inner
        d_state = config.gdn_d_state
        d_conv = config.gdn_d_conv
        headdim = config.gdn_headdim
        n_heads = d_inner // headdim

        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1, bias=False
        )
        self.b_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.c_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.dt_proj = nn.Linear(d_inner, n_heads, bias=False)
        self.g_proj = nn.Linear(d_inner, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.skip_proj = nn.Linear(d_model, d_model, bias=False)

        a_init = torch.log(
            torch.arange(1, n_heads + 1, dtype=torch.float32).repeat_interleave(d_state)
        )
        self.A_log = nn.Parameter(a_init)
        self.D = nn.Parameter(torch.ones(n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))

        self.rope: RotaryEmbedding | None
        if use_rope:
            self.rope = RotaryEmbedding(
                head_dim=config.qk_rope_head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        else:
            self.rope = None

        self._forward_compiled = self._build_compiled_forward()

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

    def _gated_delta_rule(
        self,
        v: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch gated delta rule recurrence (h_t = exp(ΔA_t) * h_{t-1} + b_t * v_t)."""
        B, T, H, D = v.shape
        S = b.shape[-1]
        A = -torch.exp(self.A_log.float()).view(H, S)

        h = v.new_zeros(B, H, S, D, dtype=torch.float32)
        o_list: list[torch.Tensor] = []
        for t in range(T):
            v_t = v[:, t].float()
            b_t = b[:, t].float()
            c_t = c[:, t].float()
            g_t = g[:, t].float().unsqueeze(-1)
            alpha = torch.exp(g_t * A)
            h = alpha.unsqueeze(-1) * h + b_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            o_t = torch.einsum("bhs,bhsd->bhd", c_t, h)
            o_list.append(o_t)
        o = torch.stack(o_list, dim=1)
        return o.to(v.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GDN block mapping (B, T, d_model) -> (B, T, d_model)."""
        if x.is_cuda and self.use_compile:
            return self._forward_compiled(self, x)
        return self._forward_eager(x)

    @classmethod
    def _build_compiled_forward(cls) -> Callable[..., torch.Tensor]:
        """Return a torch.compile-wrapped _forward_eager that takes (self, x).

        Skipped on CPU: torch.compile requires the inductor triton backend,
        which isn't usable in this environment. The CPU path runs the eager
        implementation (default test suite). The CUDA path uses the compiled
        version (GPU test suite under --run-heavy).
        """
        if not torch.cuda.is_available():
            return cls._forward_eager
        return torch.compile(cls._forward_eager)

    def _kernel_out(
        self,
        v: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused GDN recurrence via the sanctioned Triton kernel.

        Fail-fast: on CUDA with fused_gdn enabled, a kernel failure must
        surface as an error rather than silently degrading to the eager
        recurrence (AGENTS.md hard don't). The eager path is a CPU/reference
        fallback only.
        """
        if self.use_triton and v.is_cuda:
            from hymo.models.gdn_triton import triton_gated_delta_rule

            return triton_gated_delta_rule(v, b, c, decay, self.A_log)
        return self._gated_delta_rule(v, b, c, decay)

    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, S = self.n_heads, self.headdim, self.d_state
        d_inner = self.d_inner

        v_in = self.in_proj(x)

        v_conv = self.conv1d(v_in.transpose(1, 2))[:, :, :T]
        v = F.silu(v_conv.transpose(1, 2))

        b_in = self.b_proj(v)
        c_in = self.c_proj(v)
        dt_in = self.dt_proj(v)
        g_in = self.g_proj(v)
        b = b_in.view(B, T, H, S)
        c = c_in.view(B, T, H, S)
        dt = F.softplus(dt_in + self.dt_bias)
        g_gate = F.sigmoid(g_in).view(B, T, H, D).mean(dim=-1)
        decay = dt * g_gate

        v = v.view(B, T, H, D)
        if self.rope is not None and T > 0:
            rope_dim = self.rope.head_dim
            v_for_rope = v[..., :rope_dim].permute(0, 2, 1, 3)
            v_for_rope = self.rope.apply_rope(v_for_rope)
            v_for_rope = v_for_rope.permute(0, 2, 1, 3)
            v = torch.cat([v_for_rope, v[..., rope_dim:]], dim=-1)

        o = self._kernel_out(v, b, c, decay)

        o = o + self.D.view(1, 1, H, 1) * v
        o = o * g_gate.unsqueeze(-1)

        o_flat = o.view(B, T, d_inner)
        y = self.out_proj(o_flat) + self.skip_proj(x)
        return cast(torch.Tensor, y)

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, use_rope={self.use_rope}, "
            f"d_inner={self.d_inner}, n_heads={self.n_heads}, "
            f"d_state={self.d_state}, headdim={self.headdim}"
        )
