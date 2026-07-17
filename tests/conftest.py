"""Pytest configuration and shared fixtures for the HyMo test suite."""

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

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CONFIG_PATH = REPO_ROOT / "tests" / "fixtures" / "tiny_hymo.yaml"
PRODUCTION_CONFIG_PATH = REPO_ROOT / "configs" / "hymo_750m.yaml"


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.fixture(autouse=True, scope="session")
def _cap_cpu_threads() -> None:
    """Cap CPU threads on Apple Silicon to prevent thermal throttling."""
    if _is_apple_silicon():
        torch.set_num_threads(2)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip heavy tests unless --run-heavy is specified."""
    if config.getoption("--run-heavy", default=False):
        return
    skip_heavy = pytest.mark.skip(reason="needs --run-heavy (builds the 1.86B model)")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --run-heavy command line option."""
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="run tests marked heavy (they build the production 1.86B model)",
    )


@pytest.fixture(scope="session")
def tiny_hymo_config() -> HyMoConfig:
    """Return the loaded tiny model configuration."""
    return load_config(str(TINY_CONFIG_PATH))


@pytest.fixture(scope="session")
def tiny_hymo_model(tiny_hymo_config: HyMoConfig) -> HyMo:
    """Construct and return the tiny model module."""
    return HyMo(tiny_hymo_config.model)


@pytest.fixture(scope="session")
def tiny_model_config(tiny_hymo_config: HyMoConfig) -> ModelConfig:
    """Return the tiny model sub-config."""
    return tiny_hymo_config.model


@pytest.fixture(scope="session")
def tiny_hymo_model_bf16(tiny_hymo_config: HyMoConfig) -> HyMo:
    """Return a bfloat16 version of the tiny model."""
    model = HyMo(tiny_hymo_config.model)
    return model.to(torch.bfloat16)


@pytest.fixture(scope="session")
def production_config_only() -> HyMoConfig:
    """Load and return the production configuration without constructing the model."""
    return load_config(str(PRODUCTION_CONFIG_PATH))


@pytest.fixture
def tiny_hymo_in_eval(tiny_hymo_model: HyMo) -> HyMo:
    """Place the tiny model in eval mode and return it."""
    tiny_hymo_model.eval()
    return tiny_hymo_model
