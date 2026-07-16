"""Tests for the HyMo model implementations (Phase 2).

Each module is tested for:
- It can be constructed with the v1.0 default config.
- Its forward returns a same-shape tensor of finite values.
- The optimizer partition (test_optimizer_partition) routes the right
  parameters to NorMuon vs AdamW.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# CPU-friendly config policy
# ----------------------------------------------------------------------
# Every submodule test in this file builds a model piece (GDN/MLA/MoE/...)
# from a ``ModelConfig``. The production :class:`ModelConfig` default is
# the 1.86 B-parameter v1.0 spec — instantiating it on every test
# heats the machine and takes hours. So, at module scope, ``ModelConfig()``
# (called bare) now resolves to the *tiny* test config (~1 M params),
# which preserves every architectural feature (3:1 GDN:MLA, MQA-4,
# partial-RoPE, MoE, MTP depth=2) at a fraction of the size. Tests that
# genuinely need the production scale are marked ``@pytest.mark.heavy``
# and call ``PRODUCTION_MODEL_CONFIG()`` explicitly instead.
from pathlib import Path as _Path

import pytest
import torch

from hymo.core.config import ModelConfig, load_config
from hymo.models import (
    DeepSeekMoE,
    DenseFFN,
    GatedDeltaNetBlock,
    HyMo,
    MLABlock,
    MTPOutput,
    MultiHeadLatentAttention,
    MultiTokenPrediction,
    SwiGLUExpert,
    build_hymo,
    mup_init,
)
from hymo.models.rope import RotaryEmbedding

_TINY_CONFIG_PATH = (
    _Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "fixtures"
    / "tiny_hymo.yaml"
)
_PRODUCTION_MODEL_CONFIG = ModelConfig  # the real v1.0 default (heavy tests only)
_TINY_MODEL_CONFIG_CACHE: ModelConfig | None = None  # type: ignore[name-defined]


def ModelConfig(*args: object, **kwargs: object) -> ModelConfig:  # type: ignore[valid-type]
    """Tiny ``ModelConfig`` for submodule tests (CPU-friendly).

    Calling with arguments forwards to the real production
    :class:`ModelConfig` (preserves any test that passes explicit
    fields). Called bare, returns the (cached) tiny test config's model.
    """
    if args or kwargs:
        return _PRODUCTION_MODEL_CONFIG(*args, **kwargs)  # type: ignore[call-arg]
    global _TINY_MODEL_CONFIG_CACHE
    if _TINY_MODEL_CONFIG_CACHE is None:
        _TINY_MODEL_CONFIG_CACHE = load_config(str(_TINY_CONFIG_PATH)).model
    return _TINY_MODEL_CONFIG_CACHE

# ----------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------


class TestRotaryEmbedding:
    def test_construct(self) -> None:
        r = RotaryEmbedding(head_dim=32, max_seq_len=4096, theta=10_000.0)
        assert r.head_dim == 32
        assert r.max_seq_len == 4096
        assert r.theta == 10_000.0

    def test_odd_head_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            RotaryEmbedding(head_dim=33)

    def test_zero_max_seq_len_raises(self) -> None:
        with pytest.raises(ValueError):
            RotaryEmbedding(head_dim=32, max_seq_len=0)

    def test_apply_returns_finite_tensor(self) -> None:
        """apply_rope returns a same-shape tensor of finite values."""
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(2, 4, 8, 32)
        y = r.apply_rope(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    def test_apply_is_invertible_by_position(self) -> None:
        """Rotating the same vector at position p, then at position -p
        (i.e., the cos/sin of the inverse angle) returns x. RoPE is
        a per-position rotation; R(p) is invertible by R(-p)."""
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(1, 1, 32)
        # At a single position, apply_rope with all-zero cos/sin (p=0)
        # is the identity; that's a degenerate test. Instead, rotate
        # by a non-trivial angle and verify the rotation is
        # orthogonal (preserves the input vector when undone via
        # the transpose).
        # The closed-form check: applying apply_rope(x) at position
        # p and then at position q gives R(q)·R(p)·x. R(q)·R(p) =
        # R(p+q) only if the per-pair frequencies are the same
        # (which they are, since RoPE uses p·freq per pair).
        # So apply_rope(apply_rope(x, p), q) should equal
        # apply_rope(x, p+q).
        y = r.apply_rope(x, start_pos=3)
        y2 = r.apply_rope(y, start_pos=4)
        y3 = r.apply_rope(x, start_pos=7)
        assert torch.allclose(y2, y3, atol=1e-5)

    def test_apply_orthogonal(self) -> None:
        """RoPE preserves the L2 norm of each pair (it's a rotation)."""
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(1, 4, 32)
        y = r.apply_rope(x, start_pos=5)
        # ||x||_2 == ||y||_2 per token.
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
        # Per-pair norm is also preserved.
        x_pair_norm = (x[..., 0::2] ** 2 + x[..., 1::2] ** 2).sqrt()
        y_pair_norm = (y[..., 0::2] ** 2 + y[..., 1::2] ** 2).sqrt()
        assert torch.allclose(x_pair_norm, y_pair_norm, atol=1e-5)

    def test_apply_position_dependent(self) -> None:
        """Rotating the same vector at different positions gives
        different outputs (the position encoding is doing something)."""
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(1, 1, 1, 32)
        y0 = r.apply_rope(x, start_pos=0)
        y1 = r.apply_rope(x, start_pos=1)
        y10 = r.apply_rope(x, start_pos=10)
        assert not torch.allclose(y0, y1)
        assert not torch.allclose(y0, y10)
        assert not torch.allclose(y1, y10)

    def test_apply_start_pos_offset(self) -> None:
        """Rotating the same vector at the same position gives the
        same result whether the rotation comes from a length-T tensor
        or a length-1 tensor with start_pos."""
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x_full = torch.randn(1, 8, 32)
        # Path A: rotate the whole length-8 sequence, take position 7.
        y_shift = r.apply_rope(x_full, start_pos=0)[0, -1]
        # Path B: rotate just position 7 (slice then offset).
        y_offset = r.apply_rope(x_full[:, -1:, :], start_pos=7)[0, 0]
        assert torch.allclose(y_shift, y_offset, atol=1e-5)

    def test_apply_preserves_norm(self) -> None:
        """RoPE is a rotation, so ||x||_2 is preserved per token."""
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=64)
        x = torch.randn(2, 16, 32)
        y = r.apply_rope(x)
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        assert torch.allclose(x_norm, y_norm, atol=1e-5)

    def test_apply_wrong_head_dim_raises(self) -> None:
        r = RotaryEmbedding(head_dim=32)
        with pytest.raises(ValueError):
            r.apply_rope(torch.zeros(1, 4, 1, 16))  # wrong head_dim

    def test_apply_out_of_range_raises(self) -> None:
        r = RotaryEmbedding(head_dim=32, max_seq_len=8)
        with pytest.raises(ValueError):
            r.apply_rope(torch.zeros(1, 4, 1, 32), start_pos=10)

    def test_from_config(self) -> None:
        m = ModelConfig()
        r = RotaryEmbedding.from_config(m)
        assert r.head_dim == m.qk_rope_head_dim
        assert r.max_seq_len == m.max_seq_len
        assert r.theta == m.rope_theta


# ----------------------------------------------------------------------
# GDN
# ----------------------------------------------------------------------


class TestGatedDeltaNetBlock:
    def test_construct_with_rope(self) -> None:
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1, use_rope=True)
        assert block.layer_idx == 1
        assert block.use_rope is True
        assert block.rope is not None
        assert block.n_heads == m.gdn_d_inner // m.gdn_headdim
        assert block.d_inner == m.gdn_d_inner

    def test_construct_without_rope(self) -> None:
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=3, use_rope=False)
        assert block.use_rope is False
        assert block.rope is None

    def test_forward_smoke(self) -> None:
        """Forward returns a same-shape tensor of finite values."""
        torch.manual_seed(0)
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        x = torch.randn(1, 8, m.dim)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_forward_with_and_without_rope(self) -> None:
        """Both use_rope=True and use_rope=False produce finite
        outputs; the two outputs differ (RoPE is doing something)."""
        torch.manual_seed(0)
        m = ModelConfig()
        b_rope = GatedDeltaNetBlock(m, layer_idx=1, use_rope=True)
        b_nope = GatedDeltaNetBlock(m, layer_idx=3, use_rope=False)
        x = torch.randn(1, 8, m.dim)
        y_rope = b_rope(x)
        y_nope = b_nope(x)
        assert y_rope.shape == y_nope.shape == x.shape
        assert torch.isfinite(y_rope).all()
        assert torch.isfinite(y_nope).all()
        assert not torch.allclose(y_rope, y_nope, atol=1e-5)

    def test_forward_zero_input(self) -> None:
        """Zero input → zero output (no bias, all paths linear in x)."""
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        x = torch.zeros(1, 4, m.dim)
        y = block(x)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    def test_forward_batch_independence(self) -> None:
        """Two different inputs produce different outputs (no batch
        leakage through the recurrent state)."""
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        block.eval()
        x1 = torch.randn(1, 8, m.dim)
        x2 = torch.randn(1, 8, m.dim)
        y1 = block(x1)
        y2 = block(x2)
        assert not torch.allclose(y1, y2, atol=1e-5)

    def test_forward_state_evolves_across_time(self) -> None:
        """A 2-token input where only the first token is non-zero
        should produce a non-zero output at position 1 (the state
        carries information forward)."""
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        block.eval()
        x = torch.zeros(1, 2, m.dim)
        x[0, 0] = 1.0
        y = block(x)
        assert y[0, 1].abs().max() > 1e-6


# ----------------------------------------------------------------------
# MLA
# ----------------------------------------------------------------------


class TestMLA:
    def test_construct(self) -> None:
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        assert mla.layer_idx == 0
        assert mla.n_heads == m.n_heads
        assert mla.n_kv_groups == m.n_kv_groups

    def test_block_construct(self) -> None:
        m = ModelConfig()
        block = MLABlock(m, layer_idx=0)
        assert block.layer_idx == 0
        assert hasattr(block, "attn")
        assert hasattr(block, "moe")

    def test_mla_forward_smoke(self) -> None:
        """MLA forward returns a same-shape tensor of finite values."""
        torch.manual_seed(0)
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        x = torch.randn(1, 8, m.dim)
        y = mla(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_block_forward_smoke(self) -> None:
        """MLABlock forward returns a same-shape tensor of finite values."""
        torch.manual_seed(0)
        m = ModelConfig()
        block = MLABlock(m, layer_idx=0)
        x = torch.randn(1, 8, m.dim)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_mla_zero_input_nonzero_norm(self) -> None:
        """Zero input → no NaN/Inf (the per-head norm gains make
        the output nonzero, but the magnitudes should be bounded)."""
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        x = torch.zeros(1, 4, m.dim)
        y = mla(x)
        assert torch.isfinite(y).all()

    def test_mla_position_dependent(self) -> None:
        """The same logical input at different positions gives
        different outputs (RoPE on q_pe and k_pe)."""
        torch.manual_seed(0)
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        mla.eval()
        x = torch.randn(1, 4, m.dim)
        y1 = mla(x)
        # Shift by 1 (pad a zero at the front).
        x_shifted = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        y2 = mla(x_shifted)
        assert not torch.allclose(y1, y2, atol=1e-5)


# ----------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------


class TestMoE:
    def test_construct(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        assert moe.n_routed == m.n_routed_experts
        assert moe.n_shared == m.n_shared_experts
        assert moe.n_activated == m.n_activated_experts
        assert len(moe.experts) == m.n_routed_experts
        assert moe.shared_expert is not None

    def test_construct_no_shared(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), n_shared_experts=0)
        moe = DeepSeekMoE(m, layer_idx=0)
        assert moe.shared_expert is None

    def test_update_gate_bias_noop_without_routing(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        # No routing recorded yet → no-op, bias unchanged.
        bias_before = moe.gate.bias.clone()
        moe.update_gate_bias()
        assert torch.allclose(moe.gate.bias, bias_before)

    def test_update_gate_bias_balances(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        torch.manual_seed(0)
        x = torch.randn(4, 8, m.dim)
        moe(x)  # records _last_indices
        moe.update_gate_bias(speed=0.01)
        assert torch.isfinite(moe.gate.bias).all()

    def test_forward_shape_and_finite(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        x = torch.randn(2, 6, m.dim)
        y = moe(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestSwiGLUExpert:
    def test_construct(self) -> None:
        e = SwiGLUExpert(dim=64, inter_dim=128)
        assert e.dim == 64
        assert e.inter_dim == 128
        # Has w1, w2, w3
        assert e.w1.in_features == 64
        assert e.w1.out_features == 128
        assert e.w2.in_features == 128
        assert e.w2.out_features == 64

    def test_forward_shape_and_finite(self) -> None:
        e = SwiGLUExpert(dim=64, inter_dim=128)
        x = torch.randn(2, 5, 64)
        y = e(x)
        assert y.shape == (2, 5, 64)
        assert torch.isfinite(y).all()


class TestDenseFFN:
    def test_construct(self) -> None:
        ffn = DenseFFN(dim=64, inter_dim=128)
        assert ffn.w1.in_features == 64
        assert ffn.w1.out_features == 128

    def test_forward_shape_and_finite(self) -> None:
        ffn = DenseFFN(dim=64, inter_dim=128)
        x = torch.randn(2, 5, 64)
        y = ffn(x)
        assert y.shape == (2, 5, 64)
        assert torch.isfinite(y).all()


# ----------------------------------------------------------------------
# MTP
# ----------------------------------------------------------------------


class TestMTP:
    def test_construct(self) -> None:
        m = ModelConfig()
        mtp = MultiTokenPrediction(m, main_model=None)  # type: ignore[arg-type]
        assert mtp.depth == m.mtp_depth
        assert mtp.mtp_loss_weights == (0.3, 0.1)
        assert len(mtp.mtp_modules) == m.mtp_depth

    def test_construct_no_mtp(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), mtp_depth=0, mtp_loss_weights=())
        mtp = MultiTokenPrediction(m, main_model=None)  # type: ignore[arg-type]
        assert mtp.depth == 0
        assert len(mtp.mtp_modules) == 0

    def test_forward_returns_chained_heads(self) -> None:
        from hymo.models.fusionllm import HyMo

        m = ModelConfig()
        model = HyMo(m)
        mtp = MultiTokenPrediction(m, main_model=model)
        tokens = torch.randint(0, m.vocab_size, (2, 8))
        main_logits, outputs = mtp(tokens)
        assert main_logits.shape == (2, 8, m.vocab_size)
        # depth=2 → two chained heads; head d predicts tokens d+1 ahead.
        assert len(outputs) == 2
        assert outputs[0].logits.shape[1] == 7   # T - 1
        assert outputs[1].logits.shape[1] == 6   # T - 2
        assert outputs[0].loss_weight == 0.3
        assert outputs[1].loss_weight == 0.1


class TestMTPOutput:
    def test_construct(self) -> None:
        out = MTPOutput(
            logits=torch.zeros(1, 4, 1024),
            targets=torch.zeros(1, 4, dtype=torch.long),
            loss_weight=0.3,
        )
        assert out.loss_weight == 0.3


# ----------------------------------------------------------------------
# HyMo stack
# ----------------------------------------------------------------------


class TestHyMo:
    def test_construct_default(self, tiny_hymo_model: HyMo) -> None:
        # ``tiny_hymo_model`` is the canonical HyMo; assert it was
        # built from a config (i.e. config is not None).
        assert tiny_hymo_model.config is not None

    def test_n_layers_matches_config(self, tiny_hymo_model: HyMo) -> None:
        """n_layers on the tiny model matches the tiny config (4)."""
        assert len(tiny_hymo_model.layers) == 4

    @pytest.mark.heavy
    def test_32_layers_full(self) -> None:
        """v1.0 production: 32 layers. Gated by the ``heavy`` marker
        (skipped on M1 by default — the 1.86B-param model OOMs)."""
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        assert len(model.layers) == 32

    def test_3_to_1_ratio(self, tiny_hymo_model: HyMo) -> None:
        """Tiny config has 1 MLA + 3 GDN; the 3:1 ratio holds for any
        layer count that's a multiple of 4."""
        n_mla = sum(1 for layer in tiny_hymo_model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in tiny_hymo_model.layers if isinstance(layer, GatedDeltaNetBlock))
        # Tiny: 1 MLA, 3 GDN. Ratio is 3:1.
        assert n_mla == 1
        assert n_gdn == 3
        assert n_mla + n_gdn == 4

    @pytest.mark.heavy
    def test_eight_mla_twenty_four_gdn_full(self) -> None:
        """v1.0 production: 8 MLA + 24 GDN. Gated by ``heavy``."""
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        n_mla = sum(1 for layer in model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert n_mla == 8
        assert n_gdn == 24

    def test_mla_at_position_0(self, tiny_hymo_model: HyMo) -> None:
        """Tiny: MLA at position 0 (the only MLA in a 4-layer model)."""
        mla_indices = [
            i for i, layer in enumerate(tiny_hymo_model.layers) if isinstance(layer, MLABlock)
        ]
        assert mla_indices == [0]

    @pytest.mark.heavy
    def test_mla_at_positions_0_4_8_etc_full(self) -> None:
        """v1.0 production: MLA at every 4th position starting at 0."""
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        mla_indices = [
            i for i, layer in enumerate(model.layers) if isinstance(layer, MLABlock)
        ]
        assert mla_indices == [0, 4, 8, 12, 16, 20, 24, 28]

    def test_gdn_at_other_positions_tiny(self, tiny_hymo_model: HyMo) -> None:
        """Tiny: GDN at positions 1, 2, 3 (the complement of MLA at 0)."""
        gdn_indices = [
            i for i, layer in enumerate(tiny_hymo_model.layers)
            if isinstance(layer, GatedDeltaNetBlock)
        ]
        assert gdn_indices == [1, 2, 3]

    @pytest.mark.heavy
    def test_gdn_at_other_positions_full(self) -> None:
        """v1.0 production: GDN at every non-MLA position."""
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        gdn_indices = [
            i for i, layer in enumerate(model.layers) if isinstance(layer, GatedDeltaNetBlock)
        ]
        assert gdn_indices == [
            1, 2, 3, 5, 6, 7, 9, 10, 11,
            13, 14, 15, 17, 18, 19, 21, 22, 23,
            25, 26, 27, 29, 30, 31,
        ]

    def test_nope_hybrid_disabled_by_default(self, tiny_hymo_model: HyMo) -> None:
        """CR-12: when nope_hybrid_gdn_enabled is False (the v1.0
        default), every GDN layer has use_rope=True. The rule is
        independent of layer count, so this is identical for the
        tiny and full configs."""
        for layer in tiny_hymo_model.layers:
            if isinstance(layer, GatedDeltaNetBlock):
                assert layer.use_rope is True

    @pytest.mark.heavy
    def test_nope_hybrid_enabled_marks_correct_gdn_layers(self) -> None:
        """v1.0 production: with the hybrid enabled, the 7 GDN
        positions {3, 7, 11, 15, 19, 23, 27} are noPE. The tiny
        config has only 3 GDN positions, so this is a v1.0-only
        test (``heavy`` marker)."""
        from dataclasses import replace

        m = replace(_PRODUCTION_MODEL_CONFIG(), nope_hybrid_gdn_enabled=True)
        model = HyMo(m)
        nope_layers = [
            i for i, layer in enumerate(model.layers)
            if isinstance(layer, GatedDeltaNetBlock) and not layer.use_rope
        ]
        assert nope_layers == [3, 7, 11, 15, 19, 23, 27]

    def test_tied_embeddings(self, tiny_hymo_model: HyMo) -> None:
        """When tie_embeddings=True (the v1.0 default), the head's
        weight is the embed's weight (verified at the tiny scale)."""
        # tiny_hymo_config sets tie_embeddings: true
        assert tiny_hymo_model.head.weight is tiny_hymo_model.embed.weight

    @pytest.mark.heavy
    def test_untied_embeddings_full(self) -> None:
        """v1.0 production: with tie_embeddings=False, the head has
        its own weight tensor (not the embed's)."""
        from dataclasses import replace

        m = replace(_PRODUCTION_MODEL_CONFIG(), tie_embeddings=False)
        model = HyMo(m)
        assert model.head.weight is not model.embed.weight

    def test_num_parameters_tiny(self, tiny_hymo_model: HyMo) -> None:
        """Tiny model has <1M parameters (sanity: not the 1.86B v1.0)."""
        n = tiny_hymo_model.num_parameters()
        assert n < 1_000_000
        assert n > 0

    @pytest.mark.heavy
    def test_num_parameters_full(self) -> None:
        """v1.0 production: ~1.86B stored params. Gated by ``heavy``."""
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        n = model.num_parameters()
        assert n > 100_000_000
        assert n < 5_000_000_000

    def test_num_parameters_trainable_only_tiny(self, tiny_hymo_model: HyMo) -> None:
        """All params are trainable by default (rule is dtype-agnostic)."""
        n_all = tiny_hymo_model.num_parameters(only_trainable=False)
        n_train = tiny_hymo_model.num_parameters(only_trainable=True)
        assert n_train == n_all

    def test_softcap(self) -> None:
        """softcap is shape/dtype-independent: a tiny config verifies
        the same tanh-bounded behavior as v1.0."""
        from dataclasses import replace

        m = replace(ModelConfig(), logit_softcap=15.0)
        model = HyMo(m)
        x = torch.tensor([100.0, 0.0, -100.0])
        out = model.softcap(x)
        assert out.max() < 15.0 + 1e-6
        assert out.min() > -15.0 - 1e-6
        assert out[1].item() == pytest.approx(0.0)

    def test_softcap_disabled(self) -> None:
        """logit_softcap=0 disables the cap (no model build required
        — softcap is a tensor-only op)."""
        from dataclasses import replace

        m = replace(ModelConfig(), logit_softcap=0)
        model = HyMo(m)
        x = torch.tensor([1.0, 2.0, 3.0])
        out = model.softcap(x)
        assert torch.equal(out, x)

    def test_forward_shape_and_finite(self, tiny_hymo_model: HyMo) -> None:
        """Real Phase-2 forward returns (B, T, vocab) logits."""
        tokens = torch.randint(0, tiny_hymo_model.config.vocab_size, (2, 8))
        out = tiny_hymo_model(tokens)
        assert out.shape == (2, 8, tiny_hymo_model.config.vocab_size)
        assert torch.isfinite(out).all()

    def test_forward_with_hidden_returns_hidden(self, tiny_hymo_model: HyMo) -> None:
        """``forward_with_hidden`` returns (logits, hidden); hidden is
        the normalized pre-head representation of shape (B, T, dim)."""
        tokens = torch.randint(0, tiny_hymo_model.config.vocab_size, (2, 8))
        logits, hidden = tiny_hymo_model.forward_with_hidden(tokens)
        assert logits.shape == (2, 8, tiny_hymo_model.config.vocab_size)
        assert hidden.shape == (2, 8, tiny_hymo_model.config.dim)


class TestBuildHyMo:
    def test_from_tiny_hymo_config(self, tiny_hymo_config) -> None:
        """``build_hymo`` on the tiny config returns a HyMo with the
        right layer count for tiny (4)."""
        model = build_hymo(tiny_hymo_config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 4

    @pytest.mark.heavy
    def test_from_hymo_config_full(self) -> None:
        """v1.0 production: ``build_hymo`` on the default config
        returns a HyMo with 32 layers. Gated by ``heavy``."""
        from hymo.core.config import HyMoConfig

        config = HyMoConfig()
        model = build_hymo(config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 32


class TestMupInit:
    def test_predicate_zero_keywords(self) -> None:
        from hymo.models.init import zero_init_predicate

        # Should match.
        assert zero_init_predicate("layers.0.attn.gate") is True
        assert zero_init_predicate("layers.0.g_proj.weight") is True
        assert zero_init_predicate("layers.0.attn.A_log") is True
        assert zero_init_predicate("layers.0.gdn.dt_bias") is True
        assert zero_init_predicate("head.weight") is True
        assert zero_init_predicate("embed.weight") is True
        # Should NOT match (regular 2D weight).
        assert zero_init_predicate("layers.0.attn.in_proj.weight") is False

    def test_predicate_handles_embed_d_substring(self) -> None:
        """The "D" keyword should not match "embed"."""
        from hymo.models.init import zero_init_predicate

        # "D" is in the keyword list but should NOT match "embed"
        # because the predicate excludes the "embed" case.
        assert zero_init_predicate("embed.weight") is True  # matched by "embed"
        # The "D" alone is the GDN scalar (single-char param name).
        assert zero_init_predicate("layers.5.gdn.D") is True

    def test_mup_init_zeroes_scalars(self, tiny_hymo_model: HyMo) -> None:
        """mup_init zero-inits scalars/gains (gate, A_log, dt_bias, D, ...)
        and μP-scales the 2D weights."""
        m = tiny_hymo_model.config
        mup_init(tiny_hymo_model, m)
        # A_log / dt_bias / D are exactly zero.
        for name, p in tiny_hymo_model.named_parameters():
            if "A_log" in name or "dt_bias" in name or ".D" in name:
                assert torch.allclose(p, torch.zeros_like(p)), name
        # 2D weights are finite and non-trivially scaled.
        named = dict(tiny_hymo_model.named_parameters())
        in_proj_name = next(n for n in named if n.endswith("in_proj.weight"))
        in_proj = named[in_proj_name]
        assert torch.isfinite(in_proj).all()
        assert in_proj.std() > 0.0


# ----------------------------------------------------------------------
# Public API: build from production config
# ----------------------------------------------------------------------


class TestHyMoFromConfig:
    def test_production_config_loads(self, production_config_only) -> None:
        """The production YAML loads without building a model.
        The v1.0 spec values are verified here (cheap, no model build)."""
        config = production_config_only
        # Architecture numbers from the v1.0 spec.
        assert config.model.n_layers == 32
        assert config.model.vocab_size == 64_256
        assert config.model.mtp_depth == 2
        assert config.model.n_kv_groups == 4  # MQA-4
        assert config.model.qk_rope_head_dim == 32  # 25% partial-RoPE

    @pytest.mark.heavy
    def test_loads_production_config_full(self) -> None:
        """v1.0 production: build a HyMo from the production YAML
        and verify the 32-layer assembly. Gated by ``heavy``."""

        config = load_config("configs/hymo_750m.yaml")
        model = build_hymo(config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 32
        assert model.config.vocab_size == 64_256
        assert model.config.mtp_depth == 2
