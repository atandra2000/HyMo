"""Public registry: named-constructor pattern for models, optimizers, etc.

The :class:`hymo.utils.registry.Registry` is the general-purpose
implementation; this subpackage re-exports the *typed* registries used
across HyMo so import sites are clean.
"""

from __future__ import annotations

from hymo.utils.registry import Registry

# Typed, module-level registries. Constructors register themselves at
# import time (see e.g. ``hymo.models.fusionllm``).
MODELS: Registry = Registry("model")
OPTIMIZERS: Registry = Registry("optimizer")
SCHEDULERS: Registry = Registry("scheduler")
TOKENIZERS: Registry = Registry("tokenizer")
DATA_SOURCES: Registry = Registry("data_source")
CALLBACKS: Registry = Registry("callback")

__all__ = [
    "MODELS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "TOKENIZERS",
    "DATA_SOURCES",
    "CALLBACKS",
    "Registry",
]
