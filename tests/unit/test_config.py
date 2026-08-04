"""Tests for the :mod:`hymo.core.config` module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from hymo.core.config import (
    HyMoConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
    derive_config,
    load_config,
    load_config_from_dict,
    save_config,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestModelConfigDefaults:
    """Verify default ModelConfig matches configuration specifications."""

    def test_vocab_size_is_64256(self) -> None:
        assert ModelConfig().vocab_size == 64_256

    def test_32_layers(self) -> None:
        assert ModelConfig().n_layers == 32

    def test_dim_896(self) -> None:
        assert ModelConfig().dim == 896

    def test_mqa4(self) -> None:
        m = ModelConfig()
        assert m.n_kv_groups == 4
        assert m.n_heads == 16
        assert m.n_heads % m.n_kv_groups == 0

    def test_partial_rope_25_percent(self) -> None:
        m = ModelConfig()
        assert m.qk_rope_head_dim == 32
        assert m.qk_nope_head_dim == 96
        assert m.qk_rope_head_dim + m.qk_nope_head_dim == m.head_dim
        assert m.qk_rope_head_dim / m.head_dim == pytest.approx(0.25)

    def test_16_routed_1_shared_top2(self) -> None:
        m = ModelConfig()
        assert m.n_routed_experts == 16
        assert m.n_shared_experts == 1
        assert m.n_activated_experts == 2

    def test_mtp_depth_2_weights_0_3_0_1(self) -> None:
        m = ModelConfig()
        assert m.mtp_depth == 2
        assert tuple(m.mtp_loss_weights) == (0.3, 0.1)

    def test_nope_hybrid_disabled_by_default(self) -> None:
        m = ModelConfig()
        assert m.nope_hybrid_gdn_enabled is False
        assert m.nope_hybrid_gdn_positions == frozenset()

    def test_layer_distribution_3_to_1(self) -> None:
        m = ModelConfig()
        assert m.n_mla_layers == 8
        assert m.n_gdn_layers == 24

    def test_mla_positions_0_4_8_etc(self) -> None:
        m = ModelConfig()
        assert m.mla_positions == frozenset({0, 4, 8, 12, 16, 20, 24, 28})

    def test_gdn_positions_complement(self) -> None:
        m = ModelConfig()
        assert m.gdn_positions == frozenset(
            i for i in range(32) if i not in m.mla_positions
        )


class TestOptimizerConfigDefaults:
    """Verify default OptimizerConfig matches configuration specifications."""

    def test_muon_lr_002(self) -> None:
        assert OptimizerConfig().muon_lr == 0.02

    def test_adamw_lr_3e_minus_4(self) -> None:
        assert OptimizerConfig().adamw_lr == 3e-4

    def test_lr_ratio_preserved(self) -> None:
        o = OptimizerConfig()
        assert o.muon_lr / o.adamw_lr == pytest.approx(66.67, abs=0.01)

    def test_adamw_betas(self) -> None:
        assert OptimizerConfig().adamw_betas == (0.9, 0.95)

    def test_cautious_wd_default(self) -> None:
        assert OptimizerConfig().cautious_wd is True


class TestSchedulerConfigDefaults:
    """Verify default SchedulerConfig settings."""

    def test_57220_steps(self) -> None:
        assert SchedulerConfig().total_steps == 57_220

    def test_2_percent_warmup(self) -> None:
        s = SchedulerConfig()
        assert s.warmup_frac == 0.02
        assert s.warmup_steps == int(57_220 * 0.02)

    def test_83_percent_stable(self) -> None:
        s = SchedulerConfig()
        assert s.stable_frac == 0.83

    def test_15_percent_decay_to_5_percent(self) -> None:
        s = SchedulerConfig()
        assert s.decay_frac == 0.15
        assert s.min_lr_ratio == 0.05

    def test_fractions_sum_to_one(self) -> None:
        s = SchedulerConfig()
        assert s.warmup_frac + s.stable_frac + s.decay_frac == pytest.approx(1.0)


class TestTrainingConfigDefaults:
    """Verify default TrainingConfig settings."""

    def test_micro_batch_4(self) -> None:
        assert TrainingConfig().micro_batch_size == 4

    def test_grad_accum_8(self) -> None:
        assert TrainingConfig().gradient_accumulation_steps == 8

    def test_world_size_4(self) -> None:
        assert TrainingConfig().world_size == 4

    def test_per_step_tokens_524288(self) -> None:
        t = TrainingConfig()
        assert t.per_step_tokens == 524_288

    def test_save_every_4000(self) -> None:
        assert TrainingConfig().save_interval == 4_000

    def test_eval_every_2000(self) -> None:
        assert TrainingConfig().eval_interval == 2_000

    def test_optimizations_on_by_default(self) -> None:
        t = TrainingConfig()
        assert t.fused_gdn is True
        assert t.moe_mixed_precision is True
        assert t.torch_compile_gdn is True


class TestFrozenDataclasses:
    """Verify dataclasses are frozen and prevent mutation."""

    def test_model_config_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            ModelConfig().dim = 1024  # type: ignore[misc]

    def test_optimizer_config_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            OptimizerConfig().muon_lr = 0.04  # type: ignore[misc]

    def test_hymo_config_is_frozen(self) -> None:
        c = HyMoConfig()
        with pytest.raises(FrozenInstanceError):
            c.model = ModelConfig()  # type: ignore[misc]


class TestValidation:
    """Verify validator checking on input fields."""

    @pytest.mark.parametrize("v", [0, -1, -100])
    def test_vocab_size_must_be_positive(self, v: int) -> None:
        with pytest.raises(ValueError):
            ModelConfig(vocab_size=v)

    def test_n_layers_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(n_layers=0)

    def test_n_kv_groups_must_divide_n_heads(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(n_heads=15, n_kv_groups=4)

    def test_qk_rope_plus_nope_equals_head_dim(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(qk_rope_head_dim=10, qk_nope_head_dim=10, head_dim=128)

    def test_mtp_weights_length_must_equal_depth(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(mtp_depth=2, mtp_loss_weights=(0.3,))

    def test_mtp_weights_must_be_nonneg(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(mtp_depth=1, mtp_loss_weights=(-0.1,))

    def test_logit_softcap_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(logit_softcap=-1)

    def test_lr_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            OptimizerConfig(muon_lr=0)

    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            SchedulerConfig(warmup_frac=0.01, stable_frac=0.83, decay_frac=0.15)

    def test_decay_must_be_known(self) -> None:
        with pytest.raises(ValueError):
            SchedulerConfig(decay="exponential")

    def test_grad_clip_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(grad_clip=0)

    def test_run_name_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError):
            RunConfig(name="")


class TestYamlRoundTrip:
    """Verify loading from and writing to YAML files."""

    def test_loads_default_yaml(self) -> None:
        config = load_config(FIXTURES / "tiny_hymo.yaml")
        assert config.model.n_layers == 4

    def test_production_yaml_loads(self) -> None:
        config = load_config(Path("configs/hymo_750m.yaml"))
        assert config.model.n_layers == 32
        assert config.scheduler.total_steps == 57_220

    def test_save_and_reload(self, tmp_path: Path) -> None:
        original = load_config(FIXTURES / "tiny_hymo.yaml")
        out = tmp_path / "saved.yaml"
        save_config(original, out)
        reloaded = load_config(out)
        assert reloaded == original

    def test_load_from_dict(self) -> None:
        raw = {
            "model": {"n_layers": 8, "dim": 128},
            "scheduler": {"total_steps": 1000},
        }
        c = load_config_from_dict(raw)
        assert c.model.n_layers == 8
        assert c.scheduler.total_steps == 1000

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")


class TestDerivation:
    """Verify configuration derivation behaves correctly."""

    def test_derive_with_replace(self) -> None:
        c = HyMoConfig()
        c2 = replace(c, run=replace(c.run, name="hymo-debug"))
        assert c.run.name == "hymo-v1.0"
        assert c2.run.name == "hymo-debug"
        assert c2.model.n_layers == c.model.n_layers

    def test_derive_config_helper(self) -> None:
        c = HyMoConfig()
        c2 = derive_config(
            c,
            run=RunConfig(name="hymo-ablation-A"),
        )
        assert c2.run.name == "hymo-ablation-A"
        assert c2.model.n_layers == 32

    def test_derive_returns_new_instance(self) -> None:
        c = HyMoConfig()
        c2 = derive_config(c, run=replace(c.run, name="x"))
        assert c is not c2
        assert c.run is not c2.run
        assert c.model is c2.model
