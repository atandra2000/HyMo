"""Exception hierarchy rooted at :class:`HyMoError`.

Every HyMo-specific exception inherits from HyMoError.
"""

from __future__ import annotations

__all__ = [
    "HyMoError",
    "ConfigError",
    "ConfigValidationError",
    "ConfigNotFoundError",
    "ShapeError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointCorruptError",
    "DistributedError",
    "DataError",
    "TokenizerError",
    "PathsError",
    "AblationConfigError",
    "NotImplementedError_",
]


class HyMoError(Exception):
    """Base class for all HyMo-specific exceptions."""


class ConfigError(HyMoError):
    """Invalid or inconsistent configuration."""


class ConfigValidationError(ConfigError):
    """A config value failed validation check."""


class ConfigNotFoundError(ConfigError):
    """A config file does not exist on disk."""


class ShapeError(HyMoError):
    """A tensor shape is invalid for the operation."""


class CheckpointError(HyMoError):
    """Checkpoint save/load failure."""


class CheckpointNotFoundError(CheckpointError):
    """A checkpoint file does not exist on disk."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint file is unreadable or corrupted."""


class DistributedError(HyMoError):
    """Distributed-training setup or collective failure."""


class DataError(HyMoError):
    """Data pipeline failure (download, tokenization, sharding)."""


class TokenizerError(HyMoError):
    """Tokenizer load/encode/decode failure."""


class PathsError(HyMoError):
    """A filesystem / project-paths operation failed."""


class AblationConfigError(HyMoError):
    """Unknown or invalid ablation configuration."""


class NotImplementedError_(HyMoError, NotImplementedError):
    """Project-level NotImplementedError for placeholders."""
