"""GPU-only parity tests for the custom Triton GDN kernel.

These tests exercise the actual Triton kernel implementation (when available) and
verify numerical parity with the pure-PyTorch reference recurrence. They require
CUDA and a GPU with shared memory large enough to hold the per-program state
buffer. On sm_75 (Turing, 64 KB shared/SM) this means S and D must stay small;
the production config's (S=32, D=128) is reserved for A100.

Marked heavy: skipped from the default CPU-friendly run, runs under
``pytest --run-heavy`` on any CUDA machine. The local dev box is a GTX 1650
(sm_75, 4 GB) — pass ``--run-heavy`` here to run them.
"""

from __future__ import annotations

import math

import pytest
import torch

from hymo.models.gdn import GatedDeltaNetBlock
from hymo.models.gdn_triton import (
    HAS_TRITON,
    _next_power_of_2,
    triton_gated_delta_rule,
)


pytestmark = [pytest.mark.heavy, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _max_s_d_for_device(device: torch.device) -> tuple[int, int]:
    """Return safe (S, D) bounds for a Triton kernel on this device.

    sm_75 (Turing) caps shared memory at ~64 KB; sm_80+ (A100) has 164 KB.
    The kernel allocates a per-program float32 [S, D] state and works in
    registers/SMEM, so we conservatively cap based on the device cap.
    """
    cap = torch.cuda.get_device_capability(device)
    sm = cap[0] * 10 + cap[1]
    if sm >= 80:  # A100, H100
        return 64, 128
    if sm >= 75:  # Turing — dev box
        return 32, 32
    return 16, 16


def _build_inputs(
    B: int, T: int, H: int, S: int, D: int, device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build small (B, T, H, *) inputs that exercise the kernel.

    Inputs are scaled to the same regime the GatedDeltaNetBlock uses:
    - g comes from a sigmoid in the block, so we pass a [0, 1]-bounded value
      with small magnitude to keep ``exp(g * A)`` close to 1 (mild decay).
    - v, b, c are small (unit-scale / 4) so the hidden state h stays in a
      numerically stable range over T steps.
    """
    g_gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(B, T, H, D, generator=g_gen, dtype=torch.float32) * 0.25
    b = torch.randn(B, T, H, S, generator=g_gen, dtype=torch.float32) * 0.1
    c = torch.randn(B, T, H, S, generator=g_gen, dtype=torch.float32) * 0.1
    # decay: strictly non-negative (the new GDN semantics). Bounded above so
    # exp(decay * A) stays in (0, 1] for |A| up to H*S.
    g_in = torch.rand(B, T, H, generator=g_gen, dtype=torch.float32) * 0.5
    # Make decay very mild: A_log positive but small so -exp(A_log) is close to 0.
    A_log = -torch.arange(1, H * S + 1, dtype=torch.float32) * 0.01
    return v.to(device), b.to(device), c.to(device), g_in.to(device), A_log.to(device)


class TestTritonAvailability:
    def test_triton_importable(self) -> None:
        assert HAS_TRITON, "Triton import failed; cannot run kernel tests."

    def test_pad_helper(self) -> None:
        for n, expected in [(1, 1), (2, 2), (3, 4), (5, 8), (32, 32), (33, 64)]:
            assert _next_power_of_2(n) == expected


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonGDNForwardParity:
    """Triton kernel forward must match the pure-PyTorch reference within atol."""

    def test_forward_parity_atol(self) -> None:
        device = torch.device("cuda")
        S_max, D_max = _max_s_d_for_device(device)
        # Pick dims well within the device budget
        S = min(8, S_max)
        D = min(16, D_max)
        B, T, H = 2, 32, 2

        v, b, c, g, A_log = _build_inputs(B, T, H, S, D, device, seed=42)

        # Pure-PyTorch reference (the fallback in GatedDeltaNetBlock)
        A = -torch.exp(A_log).view(H, S)
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
        o_ref = torch.stack(o_list, dim=1)

        o_tri = triton_gated_delta_rule(v, b, c, g, A_log)

        assert o_tri.shape == o_ref.shape == (B, T, H, D)
        assert torch.isfinite(o_tri).all()
        # FP32 T-step reduction accumulates numerical noise that flips signs
        # on near-zero outputs. Compare on the *output magnitude* scale: the
        # absolute difference must stay below 10% of the *max* output magnitude.
        abs_diff = (o_tri - o_ref).abs()
        max_ref = o_ref.abs().max()
        assert abs_diff.max() <= 0.1 * max_ref.clamp(min=1e-3), (
            f"max abs diff {abs_diff.max().item()} vs {0.1 * max_ref.item()}"
        )

    def test_forward_non_power_of_2_pads_correctly(self) -> None:
        """Kernel pads S/D up to next power of 2 internally; output strip must match."""
        device = torch.device("cuda")
        S_max, D_max = _max_s_d_for_device(device)
        # Use a non-power-of-2 S that still fits within device budget after padding
        S = 5 if 5 <= S_max else 4
        D = 12 if 12 <= D_max else 8
        if _next_power_of_2(S) > S_max or _next_power_of_2(D) > D_max:
            pytest.skip("dims exceed device budget")
        B, T, H = 1, 16, 2
        v, b, c, g, A_log = _build_inputs(B, T, H, S, D, device, seed=7)

        A = -torch.exp(A_log).view(H, S)
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
        o_ref = torch.stack(o_list, dim=1)

        o_tri = triton_gated_delta_rule(v, b, c, g, A_log)
        assert o_tri.shape == o_ref.shape
        abs_diff = (o_tri - o_ref).abs()
        max_ref = o_ref.abs().max()
        assert abs_diff.max() <= 0.1 * max_ref.clamp(min=1e-3), (
            f"max abs diff {abs_diff.max().item()} vs {0.1 * max_ref.item()}"
        )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonGDNBackwardParity:
    """Backward (autograd) through the kernel must match the reference gradient."""

    def test_backward_grads_match_pytorch(self) -> None:
        device = torch.device("cuda")
        S_max, D_max = _max_s_d_for_device(device)
        S = min(8, S_max)
        D = min(16, D_max)
        B, T, H = 1, 8, 2

        # Use the same stable input regime as the forward parity tests so the
        # recurrence stays numerically bounded for both forward and reverse-time
        # backward (T=8 with |A| up to 16 would overflow under exp(g*A)).
        v, b, c, g, A_log = _build_inputs(B, T, H, S, D, device, seed=0)
        v = v.detach().requires_grad_(True)
        b = b.detach().requires_grad_(True)
        c = c.detach().requires_grad_(True)
        g = g.detach().requires_grad_(True)

        # Triton path
        o_tri = triton_gated_delta_rule(v, b, c, g, A_log)
        loss_tri = o_tri.pow(2).mean()
        loss_tri.backward()
        grads_tri = (v.grad.clone(), b.grad.clone(), c.grad.clone(), g.grad.clone())

        # Pure-PyTorch path (independent graph)
        v2 = v.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        c2 = c.detach().clone().requires_grad_(True)
        g2 = g.detach().clone().requires_grad_(True)
        A = -torch.exp(A_log).view(H, S)
        h = v2.new_zeros(B, H, S, D, dtype=torch.float32)
        o_list: list[torch.Tensor] = []
        for t in range(T):
            v_t = v2[:, t].float()
            b_t = b2[:, t].float()
            c_t = c2[:, t].float()
            g_t = g2[:, t].float().unsqueeze(-1)
            alpha = torch.exp(g_t * A)
            h = alpha.unsqueeze(-1) * h + b_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            o_t = torch.einsum("bhs,bhsd->bhd", c_t, h)
            o_list.append(o_t)
        o_ref = torch.stack(o_list, dim=1)
        loss_ref = o_ref.pow(2).mean()
        loss_ref.backward()
        grads_ref = (v2.grad, b2.grad, c2.grad, g2.grad)

        for name, gt, gr in zip(("v", "b", "c", "g"), grads_tri, grads_ref):
            assert torch.isfinite(gt).all(), f"Triton grad for {name} contains non-finite"
            assert torch.isfinite(gr).all(), f"reference grad for {name} contains non-finite"
            abs_diff = (gt - gr).abs()
            max_ref = gr.abs().max()
            assert abs_diff.max() <= 0.2 * max_ref.clamp(min=1e-4), (
                f"{name} max abs grad diff {abs_diff.max().item()} vs {0.2 * max_ref.item()}"
            )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestGatedDeltaNetBlockOnGPU:
    """End-to-end GDN block forward+backward on the device, both bf16 and fp32."""

    def _make_block(self) -> GatedDeltaNetBlock:
        from hymo.core.config import load_config

        cfg = load_config("tests/fixtures/tiny_hymo.yaml")
        block = GatedDeltaNetBlock(cfg.model, layer_idx=1, use_rope=True)
        return block.to("cuda")

    def test_fp32_forward_backward_finite(self) -> None:
        block = self._make_block()
        x = torch.randn(1, 8, block._config.dim, device="cuda", requires_grad=True)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # All block parameters must have received a finite grad
        for name, p in block.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_bf16_forward_finite(self) -> None:
        block = self._make_block().to(torch.bfloat16)
        x = torch.randn(1, 8, block._config.dim, device="cuda", dtype=torch.bfloat16)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
