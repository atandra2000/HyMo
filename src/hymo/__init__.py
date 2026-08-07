"""Public entry point for the HyMo hybrid language model.

The package exposes configuration loading and model construction directly;
implementation modules remain organized under ``core``, ``models``, and
``training`` so callers need not depend on internal paths.

Conventions
-----------
- All public modules re-export their public API from ``__init__.py``.
- Internal modules are prefixed with ``_`` (e.g. ``_config_impl.py``).
- All config classes are immutable ``@dataclass(frozen=True)`` to make
  accidental mutation in the training loop a hard error.
- All public functions and methods are fully type-annotated.
"""

from __future__ import annotations

from hymo.core import (
    HyMoConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
    load_config,
)
from hymo.models import build_hymo

__version__ = "0.1.0"
__all__ = [
    "__version__",
    # Config
    "HyMoConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RunConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "load_config",
    # Models
    "build_hymo",
]
