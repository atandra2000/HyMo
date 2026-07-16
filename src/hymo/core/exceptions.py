"""Exception hierarchy rooted at :class:`HyMoError`.

Every HyMo-specific exception inherits from :class:`HyMoError`. Catch
``HyMoError`` to handle every project-raised error. Catch a subclass to
handle a specific failure mode.

Hierarchy
---------
- :class:`HyMoError` — root.
  - :class:`ConfigError` — invalid or inconsistent config.
    - :class:`ConfigValidationError` — schema check failed.
    - :class:`ConfigNotFoundError` — referenced config file missing.
  - :class:`ShapeError` — tensor shape mismatch.
  - :class:`CheckpointError` — save/load failure.
    - :class:`CheckpointNotFoundError` — checkpoint file missing.
    - :class:`CheckpointCorruptError` — checkpoint file is unreadable.
  - :class:`DistributedError` — distributed-training setup failure.
  - :class:`DataError` — data pipeline failure.
  - :class:`TokenizerError` — tokenizer load/encode/decode failure.
  - :class:`PathsError` — filesystem / project-paths operation failure.
  - :class:`NotImplementedError_` — placeholder method not yet implemented.

The trailing underscore on :class:`NotImplementedError_` is to avoid
shadowing the Python built-in :class:`NotImplementedError`; the
built-in is the right thing to *raise* (we re-export it via the
alias), and :class:`NotImplementedError_` is the project-level
exception that callers can catch.
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
    """Base class for every HyMo-raised exception.

    Subclass this for new failure modes; do not raise plain
    :class:`Exception`.
    """


class ConfigError(HyMoError):
    """Invalid or inconsistent configuration."""


class ConfigValidationError(ConfigError):
    """A config value failed its schema check."""


class ConfigNotFoundError(ConfigError):
    """A referenced config file does not exist on disk."""


class ShapeError(HyMoError):
    """A tensor shape is wrong for the operation.

    Always includes the expected and actual shapes in the message.
    """


class CheckpointError(HyMoError):
    """Checkpoint save/load failure."""


class CheckpointNotFoundError(CheckpointError):
    """A referenced checkpoint file does not exist on disk."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint file is unreadable (truncated, wrong magic, etc.)."""


class DistributedError(HyMoError):
    """Distributed-training setup or collective failure."""


class DataError(HyMoError):
    """Data pipeline failure (download, tokenization, sharding)."""


class TokenizerError(HyMoError):
    """Tokenizer load/encode/decode failure."""


class PathsError(HyMoError):
    """A filesystem / project-paths operation failed.

    Raised when a directory cannot be created (e.g. permission denied).
    """


class AblationConfigError(HyMoError):
    """Unknown or invalid ablation configuration."""


class NotImplementedError_(HyMoError, NotImplementedError):
    """Project-level ``NotImplementedError`` for placeholder methods.

    The class inherits from both :class:`HyMoError` and the Python
    built-in :class:`NotImplementedError`, so:

    - ``raise NotImplementedError_("...")`` is the natural way to
      signal that a Phase 1 placeholder is awaiting Phase 2 work.
    - ``except NotImplementedError_`` catches only HyMo placeholders,
      not built-in ``NotImplementedError``s raised elsewhere.
    - ``except HyMoError`` catches both.
    """
