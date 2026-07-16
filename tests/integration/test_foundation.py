"""Integration tests: the foundation pieces fit together.

These tests don't exercise any algorithmic logic (Phase 2+ work) but
they verify that the public API surface is internally consistent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hymo import (
    HyMoConfig,
    ModelConfig,
    build_hymo,
    load_config,
)
from hymo.models import HyMo
from hymo.registry import MODELS, Registry
from hymo.training import (
    CautiousAdamW,
    NorMuon,
    build_optimizers,
    partition_parameters,
)
from hymo.utils import (
    MetricsLogger,
)


class TestPublicApi:
    """The top-level package re-exports the public API."""

    def test_config_classes_exported(self) -> None:
        assert HyMoConfig is not None
        assert ModelConfig is not None

    def test_model_classes_exported(self) -> None:
        assert HyMo is not None
        assert callable(build_hymo)

    def test_training_classes_exported(self) -> None:
        assert NorMuon is not None
        assert CautiousAdamW is not None

    def test_utils_classes_exported(self) -> None:
        assert MetricsLogger is not None

    def test_registry_exported(self) -> None:
        from hymo.registry import DATA_SOURCES, MODELS, TOKENIZERS

        assert isinstance(MODELS, Registry)
        assert isinstance(TOKENIZERS, Registry)
        assert isinstance(DATA_SOURCES, Registry)


class TestEndToEndConfig:
    """Load the production config, build a model, build optimizers."""

    def test_load_and_build_tiny(self, tiny_hymo_config) -> None:
        """Build a HyMo + optimizers from the tiny config (M1-friendly)."""
        model = build_hymo(tiny_hymo_config)

        # Tiny: 4 layers, 1 MLA + 3 GDN.
        assert model.config.n_layers == 4
        from hymo.models import GatedDeltaNetBlock, MLABlock

        n_mla = sum(1 for layer in model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert (n_mla, n_gdn) == (1, 3)

        # Build optimizers.
        opts = build_optimizers(model, tiny_hymo_config.optimizer)
        assert opts.nor_muon is not None
        assert opts.adamw is not None

    @pytest.mark.heavy
    def test_load_and_build_full(self) -> None:
        """v1.0 production: load + build the 1.86B-param HyMo.
        Gated by ``heavy`` (M1 default skips)."""
        config = load_config("configs/hymo_750m.yaml")
        model = build_hymo(config)

        assert model.config.n_layers == 32
        from hymo.models import GatedDeltaNetBlock, MLABlock

        n_mla = sum(1 for layer in model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert (n_mla, n_gdn) == (8, 24)

        opts = build_optimizers(model, config.optimizer)
        assert opts.nor_muon is not None
        assert opts.adamw is not None

    def test_lr_ratio_in_production_config(self, production_config_only) -> None:
        """The 66.7× lr ratio is a config property (no model build)."""
        config = production_config_only
        assert config.optimizer.muon_lr / config.optimizer.adamw_lr == pytest.approx(
            66.67, abs=0.01
        )

    def test_30b_tokens_57220_steps(self, production_config_only) -> None:
        """The 30B / 57,220-step arithmetic is a config property."""
        config = production_config_only
        per_step = config.training.per_step_tokens
        steps = config.scheduler.total_steps
        # 57,220 * 524,288 = 29,999,759,360 ≈ 30B.
        assert per_step * steps == pytest.approx(30e9, rel=1e-3)

    def test_wsd_fractions_sum_to_one(self, production_config_only) -> None:
        """WSD fractions sum to 1.0 (a config property)."""
        config = production_config_only
        s = config.scheduler
        assert s.warmup_frac + s.stable_frac + s.decay_frac == pytest.approx(1.0)


class TestPartitioningEndToEnd:
    """The partition routes the right parameters to the right optimizer.

    M1-friendly: runs on the tiny model. The v1.0 production
    numbers (384 routed expert weights, etc.) are verified in the
    ``heavy`` test below.
    """

    def test_routed_expert_weights_on_adamw_tiny(self, tiny_hymo_model) -> None:
        """Tiny: n MLA × n_routed_experts × 3 routed expert weights on
        AdamW. Counts derived from the tiny config (no hardcoding)."""
        m = tiny_hymo_model.config
        n_mla = sum(1 for layer in tiny_hymo_model.layers if hasattr(layer, "moe"))
        partition = partition_parameters(tiny_hymo_model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_expert = 0
        for name, p in tiny_hymo_model.named_parameters():
            if (
                ".experts." in name
                and (
                    name.endswith(".w1.weight")
                    or name.endswith(".w2.weight")
                    or name.endswith(".w3.weight")
                )
                and id(p) in adamw_ids
            ):
                n_expert += 1
        assert n_expert == n_mla * m.n_routed_experts * 3

    @pytest.mark.heavy
    def test_384_routed_expert_weights_on_adamw_full(self) -> None:
        """v1.0 production: 8 MLA × 16 routed × 3 = 384 expert
        weights on AdamW. Gated by ``heavy``."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_expert = 0
        for name, p in model.named_parameters():
            if (
                ".experts." in name
                and (
                    name.endswith(".w1.weight")
                    or name.endswith(".w2.weight")
                    or name.endswith(".w3.weight")
                )
                and id(p) in adamw_ids
            ):
                n_expert += 1
        assert n_expert == 384

    def test_shared_expert_weights_on_adamw_tiny(self, tiny_hymo_model) -> None:
        """Tiny: 1 MLA layer × 1 shared × 3 matrices = 3."""
        partition = partition_parameters(tiny_hymo_model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_shared = 0
        for name, p in tiny_hymo_model.named_parameters():
            if ".shared_expert." in name and id(p) in adamw_ids:
                n_shared += 1
        assert n_shared == 3

    @pytest.mark.heavy
    def test_24_shared_expert_weights_on_adamw_full(self) -> None:
        """v1.0 production: 8 MLA × 1 shared × 3 = 24. Gated by ``heavy``."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_shared = 0
        for name, p in model.named_parameters():
            if ".shared_expert." in name and id(p) in adamw_ids:
                n_shared += 1
        assert n_shared == 24

    def test_gdn_a_log_on_adamw_tiny(self, tiny_hymo_model) -> None:
        """Tiny: 3 GDN layers × 1 A_log = 3 scalars on AdamW."""
        partition = partition_parameters(tiny_hymo_model)
        adamw_ids = {id(p) for p in partition.adamw}
        n = sum(
            1
            for name, p in tiny_hymo_model.named_parameters()
            if name.endswith(".A_log") and id(p) in adamw_ids
        )
        assert n == 3

    @pytest.mark.heavy
    def test_24_gdn_a_log_on_adamw_full(self) -> None:
        """v1.0 production: 24 GDN × 1 A_log = 24. Gated by ``heavy``."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}
        n = sum(
            1
            for name, p in model.named_parameters()
            if name.endswith(".A_log") and id(p) in adamw_ids
        )
        assert n == 24

    def test_no_expert_on_nor_muon_tiny(self, tiny_hymo_model) -> None:
        """No MoE expert weight should be on NorMuon (claim 2).
        Verified on the tiny model — the rule is size-independent."""
        partition = partition_parameters(tiny_hymo_model)
        for p in partition.nor_muon:
            for name, q in tiny_hymo_model.named_parameters():
                if q is p and ".experts." in name:
                    pytest.fail(f"Expert weight {name} on NorMuon")
                if q is p and ".shared_expert." in name:
                    pytest.fail(f"Shared expert weight {name} on NorMuon")


class TestDerivedConfig:
    """Derive a v1.1 ablation-style config from the v1.0 base."""

    def test_derive_with_smaller_layer_count(self) -> None:
        base = load_config("configs/hymo_750m.yaml")
        derived = replace(
            base,
            model=replace(base.model, n_layers=4),  # hypothetical
            run=replace(base.run, name="hymo-test"),
        )
        assert derived.model.n_layers == 4
        assert derived.run.name == "hymo-test"
        # Other fields unchanged.
        assert derived.scheduler.total_steps == base.scheduler.total_steps

    def test_derive_to_disable_mtp(self) -> None:
        base = load_config("configs/hymo_750m.yaml")
        derived = replace(
            base,
            model=replace(base.model, mtp_depth=0, mtp_loss_weights=()),
            run=replace(base.run, name="hymo-no-mtp"),
        )
        assert derived.model.mtp_depth == 0
        assert derived.model.mtp_loss_weights == ()


class TestMetricsLoggerRoundTrip:
    """Trainer writes metrics, eval reads them."""

    def test_log_and_replay(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as m:
            for step in range(5):
                m.log(step=step, loss=11.0 - step * 0.1, lr=step * 1e-5)

        recs = list(MetricsLogger(path).iter_records())
        assert len(recs) == 5
        assert [r.step for r in recs] == [0, 1, 2, 3, 4]
        assert recs[0].metrics["loss"] == pytest.approx(11.0)
        assert recs[4].metrics["loss"] == pytest.approx(10.6)


class TestModelRegistry:
    """The HyMo class is registered with MODELS."""

    def test_hymo_registered(self) -> None:
        assert MODELS.has("hymo")
        cls = MODELS.get("hymo")
        assert cls is HyMo

    def test_build_via_registry(self) -> None:
        from hymo.core.config import HyMoConfig

        config = HyMoConfig()
        cls = MODELS.get("hymo")
        model = cls(config.model)
        assert isinstance(model, HyMo)
