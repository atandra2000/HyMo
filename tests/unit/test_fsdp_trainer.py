"""Tests for the FSDP and trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hymo.core.config import load_config
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

# ---------------------------------------------------------------------------
# Helper: create a small val.bin for trainer.evaluate tests
# ---------------------------------------------------------------------------


def _make_val_bin(tmp_path: Path, num_tokens: int = 4096) -> Path:
    """Write a small validation binary and return its path."""
    path = tmp_path / "val.bin"
    tokens = np.random.randint(0, 1024, size=num_tokens, dtype=np.uint32)
    tokens.tofile(path)
    return path


class TestFSDPPlaceholders:
    def test_fsdp_auto_wrap_policy_raises(self) -> None:
        with pytest.raises(NotImplementedError_):
            fsdp_auto_wrap_policy(None, recurse=True, non_blocking=True)

    @pytest.mark.heavy
    def test_shard_nor_muon_params_raises(self) -> None:
        with pytest.raises(NotImplementedError_):
            shard_nor_muon_params(None, world_size=4)  # type: ignore[arg-type]

    @pytest.mark.heavy
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


class TestTrainer:
    def test_construct(self, tiny_hymo_config) -> None:
        model = HyMo(tiny_hymo_config.model)
        trainer = Trainer(tiny_hymo_config, model)
        assert trainer.step == 0
        assert trainer.token_count == 0
        assert trainer.best_loss == float("inf")
        assert trainer.optimizers is not None
        assert trainer.scheduler is not None

    def test_train_step(self, tiny_hymo_config) -> None:
        model = HyMo(tiny_hymo_config.model)
        trainer = Trainer(tiny_hymo_config, model)
        B, T = 1, tiny_hymo_config.model.max_seq_len
        tokens = torch.randint(0, tiny_hymo_config.model.vocab_size, (B, T))
        targets = torch.randint(0, tiny_hymo_config.model.vocab_size, (B, T))
        result = trainer.train_step(tokens, targets)
        assert isinstance(result, train_step_result)
        assert result.skipped is False
        assert torch.isfinite(torch.tensor(result.loss))
        assert result.lr_adamw > 0
        assert trainer.step == 1
        assert trainer.token_count == B * T

    def test_save_and_load(self, tiny_hymo_config, tmp_path) -> None:
        from dataclasses import replace
        config = replace(tiny_hymo_config, run=replace(tiny_hymo_config.run, output_dir=str(tmp_path)))
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        ckpt_path = trainer.save(tag="test_save")
        assert ckpt_path.exists()
        assert ckpt_path.suffix == ".pt"

        # Load into a fresh trainer.
        model2 = HyMo(config.model)
        trainer2 = Trainer(config, model2)
        step = trainer2.load(ckpt_path)
        assert step == 0
        assert trainer2.step == 0

    def test_load_after_training(self, tiny_hymo_config, tmp_path) -> None:
        from dataclasses import replace
        config = replace(tiny_hymo_config, run=replace(tiny_hymo_config.run, output_dir=str(tmp_path)))
        model = HyMo(config.model)
        trainer = Trainer(config, model)
        B, T = 1, config.model.max_seq_len
        tokens = torch.randint(0, config.model.vocab_size, (B, T))
        targets = torch.randint(0, config.model.vocab_size, (B, T))
        trainer.train_step(tokens, targets)
        assert trainer.step == 1

        ckpt_path = trainer.save(tag="step_1")

        model2 = HyMo(config.model)
        trainer2 = Trainer(config, model2)
        trainer2.load(ckpt_path)
        assert trainer2.step == 1

    def test_train_basic(self, tiny_hymo_config) -> None:
        model = HyMo(tiny_hymo_config.model)
        trainer = Trainer(tiny_hymo_config, model)
        B, T = 1, tiny_hymo_config.model.max_seq_len
        vocab = tiny_hymo_config.model.vocab_size

        def data_iter():
            while True:
                yield (
                    torch.randint(0, vocab, (B, T)),
                    torch.randint(0, vocab, (B, T)),
                )

        trainer.train(data_iter(), max_steps=3)
        assert trainer.step == 3

    def test_evaluate(self, tiny_hymo_config, tmp_path) -> None:
        model = HyMo(tiny_hymo_config.model)
        trainer = Trainer(tiny_hymo_config, model)
        val_path = _make_val_bin(tmp_path)
        metrics = trainer.evaluate(val_bin_path=val_path)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "val_ppl" in metrics
        assert torch.isfinite(torch.tensor(metrics["val_loss"]))


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
