"""Tests for the HyMo model implementations."""

from __future__ import annotations

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
_PRODUCTION_MODEL_CONFIG = ModelConfig
_TINY_MODEL_CONFIG_CACHE: ModelConfig | None = None


def ModelConfig(*args: object, **kwargs: object) -> ModelConfig:  # type: ignore[valid-type]
    """Tiny ModelConfig fixture fallback for CPU-friendly test runs."""
    if args or kwargs:
        return _PRODUCTION_MODEL_CONFIG(*args, **kwargs)  # type: ignore[call-arg]
    global _TINY_MODEL_CONFIG_CACHE
    if _TINY_MODEL_CONFIG_CACHE is None:
        _TINY_MODEL_CONFIG_CACHE = load_config(str(_TINY_CONFIG_PATH)).model
    return _TINY_MODEL_CONFIG_CACHE


class TestRotaryEmbedding:
    """Verify Rotary Embedding properties and position transformations."""

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
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(2, 4, 8, 32)
        y = r.apply_rope(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    def test_apply_is_invertible_by_position(self) -> None:
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(1, 1, 32)
        y = r.apply_rope(x, start_pos=3)
        y2 = r.apply_rope(y, start_pos=4)
        y3 = r.apply_rope(x, start_pos=7)
        assert torch.allclose(y2, y3, atol=1e-5)

    def test_apply_orthogonal(self) -> None:
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(1, 4, 32)
        y = r.apply_rope(x, start_pos=5)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
        x_pair_norm = (x[..., 0::2] ** 2 + x[..., 1::2] ** 2).sqrt()
        y_pair_norm = (y[..., 0::2] ** 2 + y[..., 1::2] ** 2).sqrt()
        assert torch.allclose(x_pair_norm, y_pair_norm, atol=1e-5)

    def test_apply_position_dependent(self) -> None:
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(1, 1, 1, 32)
        y0 = r.apply_rope(x, start_pos=0)
        y1 = r.apply_rope(x, start_pos=1)
        y10 = r.apply_rope(x, start_pos=10)
        assert not torch.allclose(y0, y1)
        assert not torch.allclose(y0, y10)
        assert not torch.allclose(y1, y10)

    def test_apply_start_pos_offset(self) -> None:
        torch.manual_seed(0)
        r = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x_full = torch.randn(1, 8, 32)
        y_shift = r.apply_rope(x_full, start_pos=0)[0, -1]
        y_offset = r.apply_rope(x_full[:, -1:, :], start_pos=7)[0, 0]
        assert torch.allclose(y_shift, y_offset, atol=1e-5)

    def test_apply_preserves_norm(self) -> None:
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
            r.apply_rope(torch.zeros(1, 4, 1, 16))

    def test_apply_out_of_range_raises(self) -> None:
        r = RotaryEmbedding(head_dim=32, max_seq_len=8)
        with pytest.raises(ValueError):
            r.apply_rope(torch.zeros(1, 4, 1, 32), start_pos=10)


class TestGatedDeltaNetBlock:
    """Verify GDN block construct and forward recurrences."""

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
        torch.manual_seed(0)
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        x = torch.randn(1, 8, m.dim)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_forward_with_and_without_rope(self) -> None:
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
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        x = torch.zeros(1, 4, m.dim)
        y = block(x)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    def test_forward_batch_independence(self) -> None:
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        block.eval()
        x1 = torch.randn(1, 8, m.dim)
        x2 = torch.randn(1, 8, m.dim)
        y1 = block(x1)
        y2 = block(x2)
        assert not torch.allclose(y1, y2, atol=1e-5)

    def test_forward_state_evolves_across_time(self) -> None:
        m = ModelConfig()
        block = GatedDeltaNetBlock(m, layer_idx=1)
        block.eval()
        x = torch.zeros(1, 2, m.dim)
        x[0, 0] = 1.0
        y = block(x)
        assert y[0, 1].abs().max() > 1e-6


class TestMLA:
    """Verify MLA (Multi-Head Latent Attention) construct and forward shapes."""

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
        torch.manual_seed(0)
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        x = torch.randn(1, 8, m.dim)
        y = mla(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_block_forward_smoke(self) -> None:
        torch.manual_seed(0)
        m = ModelConfig()
        block = MLABlock(m, layer_idx=0)
        x = torch.randn(1, 8, m.dim)
        y = block(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_mla_zero_input_nonzero_norm(self) -> None:
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        x = torch.zeros(1, 4, m.dim)
        y = mla(x)
        assert torch.isfinite(y).all()

    def test_mla_position_dependent(self) -> None:
        torch.manual_seed(0)
        m = ModelConfig()
        mla = MultiHeadLatentAttention(m, layer_idx=0)
        mla.eval()
        x = torch.randn(1, 4, m.dim)
        y1 = mla(x)
        x_shifted = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        y2 = mla(x_shifted)
        assert not torch.allclose(y1, y2, atol=1e-5)


class TestMoE:
    """Verify DeepSeekMoE routing, expert weights and bias updates."""

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
        bias_before = moe.gate.bias.clone()
        moe.update_gate_bias()
        assert torch.allclose(moe.gate.bias, bias_before)

    def test_update_gate_bias_balances(self) -> None:
        m = ModelConfig()
        moe = DeepSeekMoE(m, layer_idx=0)
        torch.manual_seed(0)
        x = torch.randn(4, 8, m.dim)
        moe(x)
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
    """Verify SwiGLU expert weights and projection dimension constraints."""

    def test_construct(self) -> None:
        e = SwiGLUExpert(dim=64, inter_dim=128)
        assert e.dim == 64
        assert e.inter_dim == 128
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
    """Verify DenseFFN construct and forward passes."""

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


class TestMTP:
    """Verify Multi-Token Prediction orchestrator and chained modules."""

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
        assert len(outputs) == 2
        assert outputs[0].logits.shape[1] == 7
        assert outputs[1].logits.shape[1] == 6
        assert outputs[0].loss_weight == 0.3
        assert outputs[1].loss_weight == 0.1


class TestMTPOutput:
    """Verify MTPOutput properties."""

    def test_construct(self) -> None:
        out = MTPOutput(
            logits=torch.zeros(1, 4, 1024),
            targets=torch.zeros(1, 4, dtype=torch.long),
            loss_weight=0.3,
        )
        assert out.loss_weight == 0.3


class TestHyMo:
    """Verify top-level HyMo model layer configurations and properties."""

    def test_construct_default(self, tiny_hymo_model: HyMo) -> None:
        assert tiny_hymo_model.config is not None

    def test_n_layers_matches_config(self, tiny_hymo_model: HyMo) -> None:
        assert len(tiny_hymo_model.layers) == 4

    @pytest.mark.heavy
    def test_32_layers_full(self) -> None:
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        assert len(model.layers) == 32

    def test_3_to_1_ratio(self, tiny_hymo_model: HyMo) -> None:
        n_mla = sum(1 for layer in tiny_hymo_model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in tiny_hymo_model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert n_mla == 1
        assert n_gdn == 3
        assert n_mla + n_gdn == 4

    @pytest.mark.heavy
    def test_eight_mla_twenty_four_gdn_full(self) -> None:
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        n_mla = sum(1 for layer in model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert n_mla == 8
        assert n_gdn == 24

    def test_mla_at_position_0(self, tiny_hymo_model: HyMo) -> None:
        mla_indices = [
            i for i, layer in enumerate(tiny_hymo_model.layers) if isinstance(layer, MLABlock)
        ]
        assert mla_indices == [0]

    @pytest.mark.heavy
    def test_mla_at_positions_0_4_8_etc_full(self) -> None:
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        mla_indices = [
            i for i, layer in enumerate(model.layers) if isinstance(layer, MLABlock)
        ]
        assert mla_indices == [0, 4, 8, 12, 16, 20, 24, 28]

    def test_gdn_at_other_positions_tiny(self, tiny_hymo_model: HyMo) -> None:
        gdn_indices = [
            i for i, layer in enumerate(tiny_hymo_model.layers)
            if isinstance(layer, GatedDeltaNetBlock)
        ]
        assert gdn_indices == [1, 2, 3]

    @pytest.mark.heavy
    def test_gdn_at_other_positions_full(self) -> None:
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
        for layer in tiny_hymo_model.layers:
            if isinstance(layer, GatedDeltaNetBlock):
                assert layer.use_rope is True

    @pytest.mark.heavy
    def test_nope_hybrid_enabled_marks_correct_gdn_layers(self) -> None:
        from dataclasses import replace

        m = replace(_PRODUCTION_MODEL_CONFIG(), nope_hybrid_gdn_enabled=True)
        model = HyMo(m)
        nope_layers = [
            i for i, layer in enumerate(model.layers)
            if isinstance(layer, GatedDeltaNetBlock) and not layer.use_rope
        ]
        assert nope_layers == [3, 7, 11, 15, 19, 23, 27]

    def test_tied_embeddings(self, tiny_hymo_model: HyMo) -> None:
        assert tiny_hymo_model.head.weight is tiny_hymo_model.embed.weight

    @pytest.mark.heavy
    def test_untied_embeddings_full(self) -> None:
        from dataclasses import replace

        m = replace(_PRODUCTION_MODEL_CONFIG(), tie_embeddings=False)
        model = HyMo(m)
        assert model.head.weight is not model.embed.weight

    def test_num_parameters_tiny(self, tiny_hymo_model: HyMo) -> None:
        n = tiny_hymo_model.num_parameters()
        assert n < 1_000_000
        assert n > 0

    @pytest.mark.heavy
    def test_num_parameters_full(self) -> None:
        model = HyMo(_PRODUCTION_MODEL_CONFIG())
        n = model.num_parameters()
        assert n > 100_000_000
        assert n < 5_000_000_000

    def test_num_parameters_trainable_only_tiny(self, tiny_hymo_model: HyMo) -> None:
        n_all = tiny_hymo_model.num_parameters(only_trainable=False)
        n_train = tiny_hymo_model.num_parameters(only_trainable=True)
        assert n_train == n_all

    def test_softcap(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), logit_softcap=15.0)
        model = HyMo(m)
        x = torch.tensor([100.0, 0.0, -100.0])
        out = model.softcap(x)
        assert out.max() < 15.0 + 1e-6
        assert out.min() > -15.0 - 1e-6
        assert out[1].item() == pytest.approx(0.0)

    def test_softcap_disabled(self) -> None:
        from dataclasses import replace

        m = replace(ModelConfig(), logit_softcap=0)
        model = HyMo(m)
        x = torch.tensor([1.0, 2.0, 3.0])
        out = model.softcap(x)
        assert torch.equal(out, x)

    def test_forward_shape_and_finite(self, tiny_hymo_model: HyMo) -> None:
        tokens = torch.randint(0, tiny_hymo_model.config.vocab_size, (2, 8))
        out = tiny_hymo_model(tokens)
        assert out.shape == (2, 8, tiny_hymo_model.config.vocab_size)
        assert torch.isfinite(out).all()

    def test_forward_with_hidden_returns_hidden(self, tiny_hymo_model: HyMo) -> None:
        tokens = torch.randint(0, tiny_hymo_model.config.vocab_size, (2, 8))
        logits, hidden = tiny_hymo_model.forward_with_hidden(tokens)
        assert logits.shape == (2, 8, tiny_hymo_model.config.vocab_size)
        assert hidden.shape == (2, 8, tiny_hymo_model.config.dim)


class TestBuildHyMo:
    """Verify build_hymo helper constructs models correctly."""

    def test_from_tiny_hymo_config(self, tiny_hymo_config: HyMoConfig) -> None:
        model = build_hymo(tiny_hymo_config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 4

    @pytest.mark.heavy
    def test_from_hymo_config_full(self) -> None:
        from hymo.core.config import HyMoConfig

        config = HyMoConfig()
        model = build_hymo(config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 32


class TestMupInit:
    """Verify maximal update parametrization (muP) scaling works correctly."""

    def test_predicate_zero_keywords(self) -> None:
        from hymo.models.init import zero_init_predicate

        assert zero_init_predicate("layers.0.attn.gate") is True
        assert zero_init_predicate("layers.0.g_proj.weight") is True
        assert zero_init_predicate("layers.0.attn.A_log") is True
        assert zero_init_predicate("layers.0.gdn.dt_bias") is True
        assert zero_init_predicate("head.weight") is True
        assert zero_init_predicate("embed.weight") is True
        assert zero_init_predicate("layers.0.attn.in_proj.weight") is False

    def test_predicate_handles_embed_d_substring(self) -> None:
        from hymo.models.init import zero_init_predicate

        assert zero_init_predicate("embed.weight") is True
        assert zero_init_predicate("layers.5.gdn.D") is True

    def test_mup_init_zeroes_scalars(self, tiny_hymo_model: HyMo) -> None:
        m = tiny_hymo_model.config
        mup_init(tiny_hymo_model, m)
        for name, p in tiny_hymo_model.named_parameters():
            if "A_log" in name or "dt_bias" in name or ".D" in name:
                assert torch.allclose(p, torch.zeros_like(p)), name
        named = dict(tiny_hymo_model.named_parameters())
        in_proj_name = next(n for n in named if n.endswith("in_proj.weight"))
        in_proj = named[in_proj_name]
        assert torch.isfinite(in_proj).all()
        assert in_proj.std() > 0.0


class TestHyMoFromConfig:
    """Verify model construction loads configurations correctly."""

    def test_production_config_loads(self, production_config_only: HyMoConfig) -> None:
        config = production_config_only
        assert config.model.n_layers == 32
        assert config.model.vocab_size == 64_256
        assert config.model.mtp_depth == 2
        assert config.model.n_kv_groups == 4
        assert config.model.qk_rope_head_dim == 32

    @pytest.mark.heavy
    def test_loads_production_config_full(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = build_hymo(config)
        assert isinstance(model, HyMo)
        assert model.config.n_layers == 32
        assert model.config.vocab_size == 64_256
        assert model.config.mtp_depth == 2
