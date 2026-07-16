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
    Callback,
    CallbackList,
    MetricsLogger,
    ProjectPaths,
    TrainerState,
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
        assert Callback is not None
        assert ProjectPaths is not None

    def test_registry_exported(self) -> None:
        from hymo.registry import DATA_SOURCES, MODELS, TOKENIZERS

        assert isinstance(MODELS, Registry)
        assert isinstance(TOKENIZERS, Registry)
        assert isinstance(DATA_SOURCES, Registry)


class TestEndToEndConfig:
    """Load the production config, build a model, build optimizers."""

    def test_load_and_build(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = build_hymo(config)

        # Model has 32 layers, 8 MLA + 24 GDN.
        assert model.config.n_layers == 32
        from hymo.models import GatedDeltaNetBlock, MLABlock

        n_mla = sum(1 for layer in model.layers if isinstance(layer, MLABlock))
        n_gdn = sum(1 for layer in model.layers if isinstance(layer, GatedDeltaNetBlock))
        assert (n_mla, n_gdn) == (8, 24)

        # Build optimizers.
        opts = build_optimizers(model, config.optimizer)
        assert opts.nor_muon is not None
        assert opts.adamw is not None

    def test_lr_ratio_in_production_config(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        # lr_muon / lr_adamw ≈ 66.7
        assert config.optimizer.muon_lr / config.optimizer.adamw_lr == pytest.approx(
            66.67, abs=0.01
        )

    def test_30b_tokens_57220_steps(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        per_step = config.training.per_step_tokens
        steps = config.scheduler.total_steps
        # 57,220 * 524,288 = 29,999,759,360 ≈ 30B.
        assert per_step * steps == pytest.approx(30e9, rel=1e-3)

    def test_wsd_fractions_sum_to_one(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        s = config.scheduler
        assert s.warmup_frac + s.stable_frac + s.decay_frac == pytest.approx(1.0)


class TestPartitioningEndToEnd:
    """The partition routes the right parameters to the right optimizer."""

    def test_384_routed_expert_weights_on_adamw(self) -> None:
        """8 MLA layers × 16 experts × 3 matrices = 384."""
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

    def test_24_shared_expert_weights_on_adamw(self) -> None:
        """8 MLA layers × 1 shared × 3 matrices = 24."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_shared = 0
        for name, p in model.named_parameters():
            if ".shared_expert." in name and id(p) in adamw_ids:
                n_shared += 1
        assert n_shared == 24

    def test_24_gdn_a_log_on_adamw(self) -> None:
        """24 GDN layers × 1 A_log = 24."""
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

    def test_no_expert_on_nor_muon(self) -> None:
        """No MoE expert weight should be on NorMuon (claim 2)."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        for p in partition.nor_muon:
            for name, q in model.named_parameters():
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


class TestCallbackListWithTrainer:
    """The CallbackList integrates with the trainer (Phase 1 surface)."""

    def test_callback_runs_during_event_dispatch(self) -> None:
        seen: list[str] = []

        class C:
            def on_train_begin(self, state: TrainerState) -> None:
                seen.append("begin")
            def on_step_end(self, state: TrainerState) -> None:
                seen.append("step")

        cl = CallbackList([C()])
        cl.dispatch("on_train_begin", TrainerState())
        cl.dispatch("on_step_end", TrainerState())
        assert seen == ["begin", "step"]


class TestProjectPathsFromConfig:
    """Paths can be derived from a :class:`RunConfig`."""

    def test_paths_from_production_config(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        paths = ProjectPaths.from_config(config.run)
        # ``output_dir`` is ``<root> / config.output_dir``; default root
        # is the current working directory.
        assert paths.output_dir == Path.cwd() / "checkpoints/pretrain"
        assert paths.log_dir == Path.cwd() / "logs"
        assert paths.eval_dir == Path.cwd() / "checkpoints/pretrain/eval"
        assert paths.metrics_path == Path.cwd() / "logs" / "metrics.jsonl"


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
