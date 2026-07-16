"""Tests for the model placeholders.

Each placeholder is tested for:
- It can be constructed with the v1.0 default config.
- Its forward raises :class:`NotImplementedError_` (project-level).
- It can be constructed in the FSDP test (the parameters exist).
- The optimizer partition (test_optimizer_partition) routes the right
  parameters to NorMuon vs AdamW.
"""

from __future__ import annotations

import pytest
import torch

from hymo.core.config import ModelConfig, load_config
from hymo.core.exceptions import NotImplementedError_
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

    def test_apply_raises_not_implemented(self) -> None:
        r = RotaryEmbedding(head_dim=32)
        with pytest.raises(NotImplementedError_):
            r.apply_rope(torch.zeros(1, 4, 1, 32))

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

    def test_forward_raises_not_implemented(self) -> None:
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        with pytest.raises(NotImplementedError_):
            block(torch.zeros(1, 4, m.dim))


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

    def test_block_forward_raises(self) -> None:
        m = ModelConfig()
        block = MLABlock(m, layer_idx=0)
        with pytest.raises(NotImplementedError_):
            block(torch.zeros(1, 4, m.dim))


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

    def test_update_gate_bias_raises(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        with pytest.raises(NotImplementedError_):
            moe.update_gate_bias()

    def test_forward_raises(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        with pytest.raises(NotImplementedError_):
            moe(torch.zeros(1, 4, m.dim))


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

    def test_forward_raises(self) -> None:
        e = SwiGLUExpert(dim=64, inter_dim=128)
        with pytest.raises(NotImplementedError):
            e(torch.zeros(1, 4, 64))


class TestDenseFFN:
    def test_construct(self) -> None:
        ffn = DenseFFN(dim=64, inter_dim=128)
        assert ffn.w1.in_features == 64
        assert ffn.w1.out_features == 128

    def test_forward_raises(self) -> None:
        ffn = DenseFFN(dim=64, inter_dim=128)
        with pytest.raises(NotImplementedError):
            ffn(torch.zeros(1, 4, 64))


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

    def test_forward_raises(self) -> None:
        m = ModelConfig()
        mtp = MultiTokenPrediction(m, main_model=None)  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError_):
            mtp(torch.zeros(1, 8, dtype=torch.long))


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
    def test_construct_default(self) -> None:
        m = ModelConfig()
        model = HyMo(m)
        assert model.config is m

    def test_32_layers(self) -> None:
        model = HyMo(ModelConfig())
        assert len(model.layers) == 32

    def test_eight_mla_twenty_four_gdn(self) -> None:
        model = HyMo(ModelConfig())
        n_mla = sum(1 for layer in model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert n_mla == 8
        assert n_gdn == 24

    def test_mla_at_positions_0_4_8_etc(self) -> None:
        model = HyMo(ModelConfig())
        mla_indices = [
            i for i, layer in enumerate(model.layers) if isinstance(layer, MLABlock)
        ]
        assert mla_indices == [0, 4, 8, 12, 16, 20, 24, 28]

    def test_gdn_at_other_positions(self) -> None:
        model = HyMo(ModelConfig())
        gdn_indices = [
            i for i, layer in enumerate(model.layers) if isinstance(layer, GatedDeltaNetBlock)
        ]
        assert gdn_indices == [
            1, 2, 3, 5, 6, 7, 9, 10, 11,
            13, 14, 15, 17, 18, 19, 21, 22, 23,
            25, 26, 27, 29, 30, 31,
        ]

    def test_nope_hybrid_disabled_by_default(self) -> None:
        """CR-12: when nope_hybrid_gdn_enabled is False, all GDN layers
        have use_rope=True."""
        model = HyMo(ModelConfig())
        for layer in model.layers:
            if isinstance(layer, GatedDeltaNetBlock):
                assert layer.use_rope is True

    def test_nope_hybrid_enabled_marks_correct_gdn_layers(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), nope_hybrid_gdn_enabled=True)
        model = HyMo(m)
        nope_layers = [
            i for i, layer in enumerate(model.layers)
            if isinstance(layer, GatedDeltaNetBlock) and not layer.use_rope
        ]
        # The 7 GDN positions immediately after each MLA position
        # (excluding position 0 because no MLA is at -1).
        assert nope_layers == [3, 7, 11, 15, 19, 23, 27]

    def test_tied_embeddings(self) -> None:
        m = ModelConfig(tie_embeddings=True)
        model = HyMo(m)
        # The head's weight is the embed's weight.
        assert model.head.weight is model.embed.weight

    def test_untied_embeddings(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), tie_embeddings=False)
        model = HyMo(m)
        # The head has its own weight tensor (not the embed's).
        assert model.head.weight is not model.embed.weight

    def test_num_parameters(self) -> None:
        model = HyMo(ModelConfig())
        n = model.num_parameters()
        # The v1.0 spec says ~750M active / ~1.86B stored. The exact
        # count depends on whether MQA-4 is enabled, etc. We just check
        # it's in a sensible range.
        assert n > 100_000_000
        assert n < 5_000_000_000

    def test_num_parameters_trainable_only(self) -> None:
        model = HyMo(ModelConfig())
        n_all = model.num_parameters(only_trainable=False)
        n_train = model.num_parameters(only_trainable=True)
        # All params are trainable by default.
        assert n_train == n_all

    def test_softcap(self) -> None:
        m = ModelConfig(logit_softcap=15.0)
        model = HyMo(m)
        x = torch.tensor([100.0, 0.0, -100.0])
        out = model.softcap(x)
        # tanh is bounded; softcap = 15.0 * tanh(x / 15.0).
        assert out.max() < 15.0 + 1e-6
        assert out.min() > -15.0 - 1e-6
        assert out[1].item() == pytest.approx(0.0)

    def test_softcap_disabled(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), logit_softcap=0)
        model = HyMo(m)
        x = torch.tensor([1.0, 2.0, 3.0])
        out = model.softcap(x)
        # No cap → identity.
        assert torch.equal(out, x)

    def test_forward_raises_not_implemented(self) -> None:
        model = HyMo(ModelConfig())
        with pytest.raises(NotImplementedError_):
            model(torch.zeros(1, 4, dtype=torch.long))

    def test_forward_with_hidden_raises(self) -> None:
        model = HyMo(ModelConfig())
        with pytest.raises(NotImplementedError_):
            model.forward_with_hidden(torch.zeros(1, 4, dtype=torch.long))


class TestBuildHyMo:
    def test_from_hymo_config(self) -> None:
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

    def test_mup_init_raises_not_implemented(self) -> None:
        m = ModelConfig()
        model = HyMo(m)
        with pytest.raises(NotImplementedError_):
            mup_init(model, m)


# ----------------------------------------------------------------------
# Public API: build from production config
# ----------------------------------------------------------------------


class TestHyMoFromConfig:
    def test_loads_production_config(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = build_hymo(config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 32
        assert model.config.vocab_size == 64_256
        assert model.config.mtp_depth == 2
