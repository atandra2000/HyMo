"""Tests for the FSDP and trainer placeholders."""

from __future__ import annotations

import pytest

from hymo.core.config import HyMoConfig, load_config
from hymo.core.exceptions import NotImplementedError_
from hymo.models import HyMo
from hymo.training import (
    CheckpointState,
    Trainer,
    fsdp_auto_wrap_policy,
    shard_nor_muon_params,
    train_step_result,
    wrap_model_with_fsdp,
)
from hymo.utils.callbacks import CallbackList


class TestFSDPPlaceholders:
    def test_fsdp_auto_wrap_policy_raises(self) -> None:
        with pytest.raises(NotImplementedError_):
            fsdp_auto_wrap_policy(None, recurse=True, non_blocking=True)

    def test_shard_nor_muon_params_raises(self) -> None:
        with pytest.raises(NotImplementedError_):
            shard_nor_muon_params(None, world_size=4)  # type: ignore[arg-type]

    def test_wrap_model_with_fsdp_raises(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        with pytest.raises(NotImplementedError_):
            wrap_model_with_fsdp(model, config.training)


class TestCheckpointState:
    def test_defaults(self) -> None:
        s = CheckpointState()
        assert s.step == 0
        assert s.token_count == 0
        assert s.best_loss == float("inf")
        assert s.rng_state is None
        assert s.metrics_extra is None

    def test_construct(self) -> None:
        s = CheckpointState(
            step=1000,
            token_count=524_288_000,
            best_loss=2.45,
            rng_state={"python": "..."},
        )
        assert s.step == 1000
        assert s.best_loss == 2.45


class TestTrainerPlaceholder:
    def test_construct(self) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        assert trainer.step == 0
        assert trainer.token_count == 0
        assert trainer.best_loss == float("inf")
        assert isinstance(trainer.callbacks, CallbackList)

    def test_construct_with_callbacks(self) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        cb = CallbackList()
        trainer = Trainer(config, model, callbacks=cb)
        assert trainer.callbacks is cb

    def test_train_step_raises(self) -> None:
        import torch

        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        with pytest.raises(NotImplementedError_):
            trainer.train_step(
                tokens=torch.zeros(1, 4, dtype=torch.long),
                targets=torch.zeros(1, 4, dtype=torch.long),
            )

    def test_save_raises(self) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        with pytest.raises(NotImplementedError):
            trainer.save()

    def test_load_raises(self, tmp_path) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        with pytest.raises(NotImplementedError):
            trainer.load(tmp_path / "ckpt")

    def test_train_raises(self) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        with pytest.raises(NotImplementedError):
            trainer.train()

    def test_evaluate_raises(self) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        with pytest.raises(NotImplementedError):
            trainer.evaluate()

    def test_make_state(self) -> None:
        config = HyMoConfig()
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        trainer.step = 100
        trainer.token_count = 1_000_000
        state = trainer._make_state()
        assert state.step == 100
        assert state.token_count == 1_000_000


class TestTrainStepResult:
    def test_defaults(self) -> None:
        r = train_step_result(
            loss=11.06,
            grad_norm=5.0,
            lr_muon=0.02,
            lr_adamw=3e-4,
        )
        assert r.loss == 11.06
        assert r.grad_norm == 5.0
        assert r.lr_muon == 0.02
        assert r.lr_adamw == 3e-4
        assert r.skipped is False
        assert r.metrics == {}

    def test_skipped(self) -> None:
        r = train_step_result(
            loss=0.0,
            grad_norm=0.0,
            lr_muon=0.0,
            lr_adamw=0.0,
            skipped=True,
        )
        assert r.skipped is True
