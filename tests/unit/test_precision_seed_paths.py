"""Tests for the :mod:`hymo.utils.precision`, :mod:`hymo.utils.seed`,
and :mod:`hymo.utils.paths` modules.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hymo.core.config import RunConfig
from hymo.utils.paths import ProjectPaths
from hymo.utils.precision import (
    autocast_disabled,
    bf16_forward,
    fp32_master_weights,
    resolve_dtype,
)
from hymo.utils.seed import seed_for_rank, set_seed


class TestResolveDtype:
    def test_bf16_aliases(self) -> None:
        assert resolve_dtype("bfloat16") is torch.bfloat16
        assert resolve_dtype("bf16") is torch.bfloat16

    def test_fp32_aliases(self) -> None:
        assert resolve_dtype("float32") is torch.float32
        assert resolve_dtype("fp32") is torch.float32

    def test_fp16_aliases(self) -> None:
        assert resolve_dtype("float16") is torch.float16
        assert resolve_dtype("fp16") is torch.float16
        assert resolve_dtype("half") is torch.float16

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_dtype("float128")


class TestPrecisionContexts:
    def test_autocast_disabled_is_context_manager(self) -> None:
        with autocast_disabled():
            pass  # no error

    def test_bf16_forward_is_context_manager(self) -> None:
        with bf16_forward():
            pass

    def test_fp32_master_weights_is_context_manager(self) -> None:
        with fp32_master_weights():
            pass


class TestSetSeed:
    def test_set_seed_is_deterministic(self) -> None:
        set_seed(42)
        a1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        set_seed(42)
        a2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert a1 == a2
        assert n1 == n2
        assert t1 == t2

    def test_set_seed_different_seeds(self) -> None:
        set_seed(42)
        a1 = random.random()
        set_seed(99)
        a2 = random.random()
        assert a1 != a2

    def test_seed_for_rank(self) -> None:
        assert seed_for_rank(42, 0) == 42
        assert seed_for_rank(42, 1) == 43
        assert seed_for_rank(42, 3) == 45

    def test_set_seed_sets_python_hash_seed(self) -> None:
        set_seed(42)
        assert os.environ.get("PYTHONHASHSEED") == "42"


class TestProjectPaths:
    def test_from_config(self, tmp_path: Path) -> None:
        config = RunConfig(
            output_dir="checkpoints/pretrain",
            log_dir="logs",
            eval_dir="checkpoints/pretrain/eval",
        )
        paths = ProjectPaths.from_config(config, root=tmp_path)
        assert paths.root == tmp_path
        assert paths.output_dir == tmp_path / "checkpoints/pretrain"
        assert paths.log_dir == tmp_path / "logs"
        assert paths.eval_dir == tmp_path / "checkpoints/pretrain/eval"
        assert paths.data_dir == tmp_path / "data"

    def test_subpaths(self, tmp_path: Path) -> None:
        config = RunConfig()
        paths = ProjectPaths.from_config(config, root=tmp_path)
        assert paths.metrics_path == paths.log_dir / "metrics.jsonl"
        assert paths.val_bin_path == paths.data_dir / "tokens" / "val.bin"

    def test_ensure_creates_dirs(self, tmp_path: Path) -> None:
        config = RunConfig(
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
            eval_dir=str(tmp_path / "eval"),
        )
        paths = ProjectPaths.from_config(config)
        paths.ensure()
        assert paths.output_dir.exists()
        assert paths.log_dir.exists()
        assert paths.eval_dir.exists()

    def test_ensure_is_idempotent(self, tmp_path: Path) -> None:
        config = RunConfig(
            output_dir=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
            eval_dir=str(tmp_path / "eval"),
        )
        paths = ProjectPaths.from_config(config)
        paths.ensure()
        paths.ensure()  # no error
