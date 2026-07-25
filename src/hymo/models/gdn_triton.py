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

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


# Triton 3.7+ rejects non-constexpr globals (including typing.Any) referenced
# from inside @triton.jit functions. The Python type annotations on the kernel
# signature are stripped before JIT, so we annotate them with `object` only in
# the .pyi hint file — but at runtime the kernel sees no annotation at all.
# Marking them as `object` here is purely a mypy hint and is the only way to
# avoid the "Cannot access global variable Any" NameError on first launch.
if HAS_TRITON:
    @triton.jit
    def gdn_fwd_kernel(  # type: ignore[no-untyped-def]
        v_ptr, b_ptr, c_ptr, g_ptr, A_log_ptr, o_ptr, h_out_ptr,
        stride_vb, stride_vt, stride_vh, stride_vd,
        stride_bb, stride_bt, stride_bh, stride_bs,
        stride_gb, stride_gt, stride_gh,
        stride_ob, stride_ot, stride_oh, stride_od,
        stride_hb, stride_ht, stride_hh, stride_hs, stride_hd,
        B, T, H,
        S: tl.constexpr, D: tl.constexpr,
    ) -> None:
        """Forward kernel: one program per (batch, head)."""
        batch_idx = tl.program_id(0)
        head_idx  = tl.program_id(1)

        v_base = batch_idx * stride_vb + head_idx * stride_vh
        b_base = batch_idx * stride_bb + head_idx * stride_bh
        c_base = batch_idx * stride_bb + head_idx * stride_bh
        g_base = batch_idx * stride_gb + head_idx * stride_gh
        o_base = batch_idx * stride_ob + head_idx * stride_oh
        h_base = batch_idx * stride_hb + head_idx * stride_hh

        A_offset = head_idx * S
        s_idx = tl.arange(0, S)
        d_idx = tl.arange(0, D)

        A_log = tl.load(A_log_ptr + A_offset + s_idx)
        A = -tl.exp(A_log)

        h = tl.zeros((S, D), dtype=tl.float32)

        for t in range(T):
            v_t = tl.load(v_ptr + v_base + t * stride_vt + d_idx)
            b_t = tl.load(b_ptr + b_base + t * stride_bt + s_idx)
            c_t = tl.load(c_ptr + c_base + t * stride_bt + s_idx)
            decay_t = tl.load(g_ptr + g_base + t * stride_gt)

            alpha = tl.exp(decay_t * A)

            h = h * alpha[:, None] + b_t[:, None] * v_t[None, :]

            o_t = tl.sum(c_t[:, None] * h, axis=0)

            tl.store(o_ptr + o_base + t * stride_ot + d_idx, o_t)

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
        stride_vb, stride_vt, stride_vh, stride_vd,
        stride_bb, stride_bt, stride_bh, stride_bs,
        stride_gb, stride_gt, stride_gh,
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
        do_base = batch_idx * stride_vb + head_idx * stride_vh
        dv_base = batch_idx * stride_vb + head_idx * stride_vh
        db_base = batch_idx * stride_bb + head_idx * stride_bh
        dc_base = batch_idx * stride_bb + head_idx * stride_bh
        dg_base = batch_idx * stride_gb + head_idx * stride_gh

        A_offset = head_idx * S
        s_idx = tl.arange(0, S)
        d_idx = tl.arange(0, D)

        A_log = tl.load(A_log_ptr + A_offset + s_idx)
        A = -tl.exp(A_log)

        dh     = tl.zeros((S, D), dtype=tl.float32)
        dA_log = tl.zeros((S,),   dtype=tl.float32)

        for t in range(T - 1, -1, -1):
            do_t = tl.load(do_ptr + do_base + t * stride_vt + d_idx)
            c_t  = tl.load(c_ptr  + b_base  + t * stride_bt + s_idx)
            v_t  = tl.load(v_ptr  + v_base  + t * stride_vt + d_idx)
            b_t  = tl.load(b_ptr  + b_base  + t * stride_bt + s_idx)
            g_t  = tl.load(g_ptr  + g_base  + t * stride_gt)

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

            dh = dh + c_t[:, None] * do_t[None, :]

            dc_t = tl.sum(h_curr * do_t[None, :], axis=1)

            db_t = tl.sum(dh * v_t[None, :], axis=1)

            dv_t = tl.sum(dh * b_t[:, None], axis=0)

            alpha = tl.exp(g_t * A)

            dg_t = tl.sum(dh * h_prev * A[:, None] * alpha[:, None])

            dA_log_t = tl.sum(dh * h_prev * (g_t * A[:, None] * alpha[:, None]), axis=1)
            dA_log = dA_log + dA_log_t

            dh = dh * alpha[:, None]

            tl.store(dc_ptr + dc_base + t * stride_bt + s_idx, dc_t)
            tl.store(db_ptr + db_base + t * stride_bt + s_idx, db_t)
            tl.store(dv_ptr + dv_base + t * stride_vt + d_idx, dv_t)
            tl.store(dg_ptr + dg_base + t * stride_gt,         dg_t)

        dA_log_base = batch_idx * (H * S) + head_idx * S
        tl.store(dA_log_ptr + dA_log_base + s_idx, dA_log)

    class TritonGDNFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: typing.Any,
            v: torch.Tensor,
            b: torch.Tensor,
            c: torch.Tensor,
            g: torch.Tensor,
            A_log: torch.Tensor,
        ) -> torch.Tensor:
            B, T, H, D = v.shape
            S = b.shape[-1]

            o     = torch.empty_like(v)
            h_out = torch.empty((B, T, H, S, D), device=v.device, dtype=torch.float32)

            grid = (B, H)
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

            dA_log = dA_log_out.sum(dim=0)
            return dv, db, dc, dg, dA_log


def triton_gated_delta_rule(
    v: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,
) -> torch.Tensor:
    """Triton-accelerated Gated Delta Rule forward+backward.

    Recurrence:  h_t = exp(decay_t * A) * h_{t-1} + b_t ⊗ v_t
                 o_t = c_t · h_t
    where A = -exp(A_log) is negative and decay >= 0, so the decay factor
    alpha = exp(decay * A) stays in (0, 1).

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

    D_pad = _next_power_of_2(D)
    S_pad = _next_power_of_2(S)

    def _pad(t: torch.Tensor, target_last: int) -> torch.Tensor:
        if t.shape[-1] == target_last:
            return t
        pad = [0] * (2 * t.ndim)
        pad[1] = target_last - t.shape[-1]
        return torch.nn.functional.pad(t, pad)

    v_p     = _pad(v.float().contiguous(),     D_pad)
    b_p     = _pad(b.float().contiguous(),     S_pad)
    c_p     = _pad(c.float().contiguous(),     S_pad)
    A_log_p = _pad(A_log.float().contiguous(), S_pad)

    g_p = g.float().contiguous()
    if g_p.ndim == 4:
        g_p = g_p.squeeze(-1)
    assert g_p.shape == (B, T, H), (
        f"triton_gated_delta_rule: g must be (B,T,H), got {g_p.shape}"
    )

    out_p = TritonGDNFunction.apply(v_p, b_p, c_p, g_p, A_log_p)

    out = out_p[..., :D]
    return out.to(v.dtype)
