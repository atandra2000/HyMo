# mypy: disable-error-code="misc"
"""Custom Triton kernel for the Gated Delta Net recurrence (Phase 3 Optimization).

Architecture:
    h_t = exp(g_t * A) * h_{t-1} + b_t ⊗ v_t
    o_t = c_t · h_t

where:
    v, o : (B, T, H, D)   — value / output
    b, c : (B, T, H, S)   — write / read keys
    g    : (B, T, H)       — per-head scalar gate (sigmoid of dt_proj output)
    A_log: (H, S)          — log of decay eigenvalues (learnable, negative)

S and D must be powers of 2 at most 256 for Triton; the Python wrapper pads
if needed and strips the padding before returning.
"""

import typing

import torch
from typing import Any

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


if HAS_TRITON:
    @triton.jit
    def gdn_fwd_kernel(  # type: ignore[no-untyped-def]
        v_ptr: Any, b_ptr: Any, c_ptr: Any, g_ptr: Any, A_log_ptr: Any, o_ptr: Any, h_out_ptr: Any,
        stride_vb: int, stride_vt: int, stride_vh: int, stride_vd: int,
        stride_bb: int, stride_bt: int, stride_bh: int, stride_bs: int,
        stride_gb: int, stride_gt: int, stride_gh: int,
        stride_ob: int, stride_ot: int, stride_oh: int, stride_od: int,
        stride_hb: int, stride_ht: int, stride_hh: int, stride_hs: int, stride_hd: int,
        B: int, T: int, H: int,
        S: tl.constexpr, D: tl.constexpr,
    ) -> None:
        """Forward kernel: one program per (batch, head)."""
        batch_idx = tl.program_id(0)
        head_idx  = tl.program_id(1)

        v_base = batch_idx * stride_vb + head_idx * stride_vh
        b_base = batch_idx * stride_bb + head_idx * stride_bh
        c_base = batch_idx * stride_bb + head_idx * stride_bh  # c same shape as b
        g_base = batch_idx * stride_gb + head_idx * stride_gh
        o_base = batch_idx * stride_ob + head_idx * stride_oh  # @torch.compile(mode="reduce-overhead")  # type: ignore[misc]
        h_base = batch_idx * stride_hb + head_idx * stride_hh

        A_offset = head_idx * S
        s_idx = tl.arange(0, S)
        d_idx = tl.arange(0, D)

        # Load learnable log-decay, compute A = -exp(A_log)  [S]
        A_log = tl.load(A_log_ptr + A_offset + s_idx)
        A = -tl.exp(A_log)  # negative: ensures stability

        # Running state  [S, D]
        h = tl.zeros((S, D), dtype=tl.float32)

        for t in range(T):
            v_t = tl.load(v_ptr + v_base + t * stride_vt + d_idx)          # [D]
            b_t = tl.load(b_ptr + b_base + t * stride_bt + s_idx)          # [S]
            c_t = tl.load(c_ptr + c_base + t * stride_bt + s_idx)          # [S]  ← uses same strides as b
            g_t = tl.load(g_ptr + g_base + t * stride_gt)                  # scalar

            alpha = tl.exp(g_t * A)                                          # [S]

            # h_t = exp(g*A) * h_{t-1}  +  b_t ⊗ v_t
            h = h * alpha[:, None] + b_t[:, None] * v_t[None, :]           # [S, D]

            # o_t = Σ_s  c_t[s] * h_t[s, :]                                # [D]
            o_t = tl.sum(c_t[:, None] * h, axis=0)

            tl.store(o_ptr + o_base + t * stride_ot + d_idx, o_t)

            # Save full h_t for backward
            tl.store(
                h_out_ptr + h_base + t * stride_ht
                + s_idx[:, None] * stride_hs + d_idx[None, :] * stride_hd,
                h,
            )

    @triton.jit  # type: ignore[misc]
    def gdn_bwd_kernel(  # type: ignore[no-untyped-def]
        v_ptr, b_ptr, c_ptr, g_ptr, A_log_ptr, h_out_ptr,
        do_ptr,
        dv_ptr, db_ptr, dc_ptr, dg_ptr, dA_log_ptr,
        # strides for v/dv/do  (B, T, H, D)
        stride_vb, stride_vt, stride_vh, stride_vd,
        # strides for b/db/c/dc  (B, T, H, S)
        stride_bb, stride_bt, stride_bh, stride_bs,
        # strides for g/dg  (B, T, H)
        stride_gb, stride_gt, stride_gh,
        # strides for h_out  (B, T, H, S, D)
        stride_hb, stride_ht, stride_hh, stride_hs, stride_hd,
        B, T, H,
        S: tl.constexpr,
        D: tl.constexpr,
    ) -> None:
        """Backward kernel: one program per (batch, head), reverse time loop."""
        batch_idx = tl.program_id(0)
        head_idx  = tl.program_id(1)

        v_base  = batch_idx * stride_vb + head_idx * stride_vh
        b_base  = batch_idx * stride_bb + head_idx * stride_bh
        g_base  = batch_idx * stride_gb + head_idx * stride_gh
        h_base  = batch_idx * stride_hb + head_idx * stride_hh
        do_base = batch_idx * stride_vb + head_idx * stride_vh   # do has same layout as v/o
        dv_base = batch_idx * stride_vb + head_idx * stride_vh
        db_base = batch_idx * stride_bb + head_idx * stride_bh
        dc_base = batch_idx * stride_bb + head_idx * stride_bh
        dg_base = batch_idx * stride_gb + head_idx * stride_gh

        A_offset = head_idx * S
        s_idx = tl.arange(0, S)
        d_idx = tl.arange(0, D)

        A_log = tl.load(A_log_ptr + A_offset + s_idx)
        A = -tl.exp(A_log)           # [S]

        # Accumulated state gradient and A_log gradient
        dh     = tl.zeros((S, D), dtype=tl.float32)
        dA_log = tl.zeros((S,),   dtype=tl.float32)

        for t in range(T - 1, -1, -1):
            do_t = tl.load(do_ptr + do_base + t * stride_vt + d_idx)       # [D]
            c_t  = tl.load(c_ptr  + b_base  + t * stride_bt + s_idx)       # [S]  c has same strides as b
            v_t  = tl.load(v_ptr  + v_base  + t * stride_vt + d_idx)       # [D]
            b_t  = tl.load(b_ptr  + b_base  + t * stride_bt + s_idx)       # [S]
            g_t  = tl.load(g_ptr  + g_base  + t * stride_gt)               # scalar

            # h_curr = h_out[t],  h_prev = h_out[t-1] or 0
            h_curr = tl.load(
                h_out_ptr + h_base + t * stride_ht
                + s_idx[:, None] * stride_hs + d_idx[None, :] * stride_hd
            )
            if t > 0:
                h_prev = tl.load(
                    h_out_ptr + h_base + (t - 1) * stride_ht
                    + s_idx[:, None] * stride_hs + d_idx[None, :] * stride_hd
                )
            else:
                h_prev = tl.zeros((S, D), dtype=tl.float32)

            # Gradient from output: dL/dh_curr += c_t ⊗ do_t
            dh = dh + c_t[:, None] * do_t[None, :]         # [S, D]

            # dc_t = Σ_d  do_t[d] * h_curr[:, d]           # [S]
            dc_t = tl.sum(h_curr * do_t[None, :], axis=1)

            # db_t = Σ_d  dh[:, d] * v_t[d]               # [S]
            db_t = tl.sum(dh * v_t[None, :], axis=1)

            # dv_t = Σ_s  dh[s, :] * b_t[s]               # [D]
            dv_t = tl.sum(dh * b_t[:, None], axis=0)

            alpha = tl.exp(g_t * A)                          # [S]

            # dg_t = Σ_{s,d}  dh[s,d] * h_prev[s,d] * A[s] * alpha[s]   → scalar
            dg_t = tl.sum(dh * h_prev * A[:, None] * alpha[:, None])

            # dA_log[s] += dh[s,d] * h_prev[s,d] * g_t * A[s] * alpha[s]  (chain: d(-exp)/dA_log = -A)
            dA_log_t = tl.sum(dh * h_prev * (g_t * A[:, None] * alpha[:, None]), axis=1)
            dA_log = dA_log + dA_log_t

            # Backprop through h_t = alpha * h_prev + …: dL/dh_prev += alpha * dh
            dh = dh * alpha[:, None]

            tl.store(dc_ptr + dc_base + t * stride_bt + s_idx, dc_t)
            tl.store(db_ptr + db_base + t * stride_bt + s_idx, db_t)
            tl.store(dv_ptr + dv_base + t * stride_vt + d_idx, dv_t)
            tl.store(dg_ptr + dg_base + t * stride_gt,         dg_t)

        # dA_log is accumulated per (batch, head) — reduction over batch done in Python
        dA_log_base = batch_idx * (H * S) + head_idx * S
        tl.store(dA_log_ptr + dA_log_base + s_idx, dA_log)

    class TritonGDNFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: typing.Any,
            v: torch.Tensor,        # (B, T, H, D)
            b: torch.Tensor,        # (B, T, H, S)
            c: torch.Tensor,        # (B, T, H, S)
            g: torch.Tensor,        # (B, T, H)  — per-head scalar gate
            A_log: torch.Tensor,    # (H, S)     — learnable log-decay
        ) -> torch.Tensor:
            B, T, H, D = v.shape
            S = b.shape[-1]

            o     = torch.empty_like(v)
            h_out = torch.empty((B, T, H, S, D), device=v.device, dtype=torch.float32)

            grid = (B, H)  # type: ignore[misc]  # type: ignore[misc]
            gdn_fwd_kernel[grid](
                v, b, c, g, A_log, o, h_out,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                g.stride(0), g.stride(1), g.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                h_out.stride(0), h_out.stride(1), h_out.stride(2), h_out.stride(3), h_out.stride(4),
                B, T, H, S=S, D=D,
            )

            ctx.save_for_backward(v, b, c, g, A_log, h_out)
            return o

        @staticmethod
        def backward(
            ctx: typing.Any, *grad_output: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            do = grad_output[0]
            v, b, c, g, A_log, h_out = ctx.saved_tensors
            B, T, H, D = v.shape
            S = b.shape[-1]

            dv      = torch.empty_like(v)
            db      = torch.empty_like(b)
            dc      = torch.empty_like(c)
            dg      = torch.empty_like(g)
            dA_log_out = torch.zeros((B, H, S), device=v.device, dtype=torch.float32)

            do_c = do.contiguous()

            grid = (B, H)
            gdn_bwd_kernel[grid](
                v, b, c, g, A_log, h_out,
                do_c,
                dv, db, dc, dg, dA_log_out,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                g.stride(0), g.stride(1), g.stride(2),
                h_out.stride(0), h_out.stride(1), h_out.stride(2), h_out.stride(3), h_out.stride(4),
                B, T, H, S=S, D=D,
            )

            # Sum per-batch contributions → gradient shape (H, S)
            dA_log = dA_log_out.sum(dim=0)
            return dv, db, dc, dg, dA_log


def triton_gated_delta_rule(
    v: torch.Tensor,        # (B, T, H, D)
    b: torch.Tensor,        # (B, T, H, S)
    c: torch.Tensor,        # (B, T, H, S)
    g: torch.Tensor,        # (B, T, H)   — per-head gate (NOT d_inner-wide)
    A_log: torch.Tensor,    # (H, S)
) -> torch.Tensor:
    """Triton-accelerated Gated Delta Rule forward+backward.

    S and D must be powers of 2. If not, tensors are zero-padded along those
    dimensions before the kernel and un-padded on return.
    """
    if not HAS_TRITON:
        raise ImportError(
            "Triton is required for triton_gated_delta_rule. "
            "Install it with: pip install triton"
        )

    B, T, H, D = v.shape
    S = b.shape[-1]

    # Pad S and D up to the next power of 2 so Triton constexpr dims are valid
    D_pad = _next_power_of_2(D)
    S_pad = _next_power_of_2(S)

    def _pad(t: torch.Tensor, target_last: int) -> torch.Tensor:
        if t.shape[-1] == target_last:
            return t
        pad = [0] * (2 * t.ndim)
        pad[1] = target_last - t.shape[-1]   # pad last dim on the right
        return torch.nn.functional.pad(t, pad)

    v_p     = _pad(v.float().contiguous(),     D_pad)
    b_p     = _pad(b.float().contiguous(),     S_pad)
    c_p     = _pad(c.float().contiguous(),     S_pad)
    A_log_p = _pad(A_log.float().contiguous(), S_pad)

    # g must be (B, T, H) — strictly per-head scalar
    g_p = g.float().contiguous()
    if g_p.ndim == 4:
        # Caller passed (B, T, H, 1) — squeeze last dim
        g_p = g_p.squeeze(-1)
    assert g_p.shape == (B, T, H), (
        f"triton_gated_delta_rule: g must be (B,T,H), got {g_p.shape}"
    )

    out_p = TritonGDNFunction.apply(v_p, b_p, c_p, g_p, A_log_p)

    # Strip padding and cast back to original dtype
    out = out_p[..., :D]
    return out.to(v.dtype)
