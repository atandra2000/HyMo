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
from hymo.utils import (
    autocast_disabled,
    bf16_forward,
    fp32_master_weights,
    resolve_dtype,
)
from hymo.utils import seed_for_rank, set_seed


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


