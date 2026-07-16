"""Pytest configuration and shared fixtures for the HyMo test suite.

The fixtures here are M1-friendly: they default to the tiny model
config (4 layers, dim=48, vocab=1024) at ~1/1800th the param count
of the production v1.0 model (~760 K params vs 1.86 B). A developer
laptop can run the *entire* default suite in under a minute and
without heating up.

**Cool + fast by default.** Any test that would build the
1.86 B-parameter production model is marked ``@pytest.mark.heavy``
and is **skipped automatically** in the default run (see
``pytest_collection_modifyitems``). Pass ``--run-heavy`` to opt in
(use on CI / the GPU pod). Submodule tests in ``test_models.py``
shadow ``ModelConfig()`` to return the tiny config, so a bare
``ModelConfig()`` never instantiates the full model.

Tests that need the production config *values* (e.g. the
30B-tokens/57,220-steps arithmetic) use the
``production_config_only`` fixture, which loads
``configs/hymo_750m.yaml`` *without* building a model.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from hymo.core.config import HyMoConfig, ModelConfig, load_config
from hymo.models import HyMo

if TYPE_CHECKING:
    pass

# ----------------------------------------------------------------------
# Project layout helpers
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CONFIG_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_hymo.yaml"
PRODUCTION_CONFIG_PATH = REPO_ROOT / "configs" / "hymo_750m.yaml"


# ----------------------------------------------------------------------
# M1 / Apple Silicon tuning
# ----------------------------------------------------------------------


def _is_apple_silicon() -> bool:
    """True when running on an Apple M-series chip."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.fixture(autouse=True, scope="session")
def _cap_cpu_threads() -> None:
    """Cap CPU threads on Apple Silicon to avoid thermal throttle.

    The M1 MacBook Air is fanless; running all 4 cores at full
    utilization for >2 minutes throttles the CPU and slows every
    subsequent test. Capping at 2 cores keeps the tests fast *and*
    keeps the chassis at a sane temperature. No effect on Intel /
    AMD machines (we only touch threads on Apple Silicon).
    """
    if _is_apple_silicon():
        # 2 of 4 cores: tests run ~1.4× slower per test but the
        # aggregate wall-clock is shorter because we never throttle.
        torch.set_num_threads(2)


# ----------------------------------------------------------------------
# Keep the default test run cool + fast
# ----------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``heavy`` tests in the default run.

    ``heavy`` tests build the 1.86 B-parameter production model and are
    meant for CI / the GPU pod, not a developer laptop. They run only
    when explicitly requested (e.g. ``pytest -m heavy`` or by disabling
    the skip via ``--run-heavy``). This keeps every default ``pytest``
    run on CPU-only hardware fast and cool.
    """
    if config.getoption("--run-heavy", default=False):
        return
    skip_heavy = pytest.mark.skip(reason="needs --run-heavy (builds the 1.86B model)")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-heavy`` opt-in flag."""
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="run tests marked heavy (they build the production 1.86B model)",
    )


# ----------------------------------------------------------------------
# Tiny config + model fixtures (the M1-friendly default)
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_hymo_config() -> HyMoConfig:
    """The tiny HyMo config (4 layers, dim=64, vocab=1024).

    Use this when a test needs a complete ``HyMoConfig`` (model +
    optimizer + scheduler + training + run) but not a built model.
    """
    return load_config(str(TINY_CONFIG_PATH))


@pytest.fixture(scope="session")
def tiny_hymo_model(tiny_hymo_config: HyMoConfig) -> HyMo:
    """The tiny HyMo model. Constructed once per test session.

    ~1 M parameters; runs in <50 ms on M1.
    """
    return HyMo(tiny_hymo_config.model)


@pytest.fixture(scope="session")
def tiny_model_config(tiny_hymo_config: HyMoConfig) -> ModelConfig:
    """The tiny model config (``ModelConfig`` view of ``tiny_hymo_config``).

    Use this for any test that needs a ``ModelConfig`` to build a single
    submodule (GDN/MLA/MoE/...) instead of the production
    :class:`ModelConfig()` default. Building a submodule from the
    production config instantiates 1.86 B-parameter-scale tensors on
    every test — this keeps each submodule at ~1 M params so the full
    suite runs in seconds, not hours, and without heating the machine.
    """
    return tiny_hymo_config.model


@pytest.fixture(scope="session")
def tiny_hymo_model_bf16(tiny_hymo_config: HyMoConfig) -> HyMo:
    """The tiny HyMo model in bfloat16.

    Halves memory; recommended for forward tests where FP32 precision
    isn't required.
    """
    model = HyMo(tiny_hymo_config.model)
    return model.to(torch.bfloat16)


# ----------------------------------------------------------------------
# Production config fixture (no model build — for math + arithmetic)
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def production_config_only() -> HyMoConfig:
    """The production v1.0 config *without* building a model.

    Use this for tests that verify the 30B-tokens/57,220-steps
    arithmetic, the 66.7× lr ratio, etc. Loading the YAML is cheap
    (~1 ms); building the model is what's expensive (~7.4 GB at
    FP32, ~3.7 GB at BF16).

    Tests that build a model from the production config should be
    marked ``@pytest.mark.heavy`` so the default M1 run skips them.
    """
    return load_config(str(PRODUCTION_CONFIG_PATH))


# ----------------------------------------------------------------------
# Convenience: a configured, ready-to-call model
# ----------------------------------------------------------------------


@pytest.fixture
def tiny_hymo_in_eval(tiny_hymo_model: HyMo) -> HyMo:
    """The tiny model in ``eval()`` mode (disables dropout, etc.).

    Per-test scope so each test gets a clean eval-mode model.
    """
    tiny_hymo_model.eval()
    return tiny_hymo_model
