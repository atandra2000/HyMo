"""Stable public API for configuration, validation, and shared types.

The core package owns dependency-light configuration objects; model and training
packages consume these objects without importing one another.
"""

from __future__ import annotations

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
from hymo.core.config_validation import validate_full_config
from hymo.core.types import (
    Device,
    DType,
    ExpertIndex,
    LayerIndex,
    MicroStep,
    Path,
    Shape,
    Step,
    TokenId,
)

__all__ = [
    # Config
    "HyMoConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RunConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "derive_config",
    "load_config",
    "load_config_from_dict",
    "save_config",
    "validate_full_config",
    # Types
    "DType",
    "Device",
    "ExpertIndex",
    "LayerIndex",
    "MicroStep",
    "Path",
    "Shape",
    "Step",
    "TokenId",
]

