"""Real held-out validation (Phase 1 placeholder).

The real implementation (architecture doc §6.3, roadmap C6) reads
batches from ``data/tokens/val.bin`` (450M held-out FineWeb-Edu
tokens), runs the model in eval mode, and returns the cross-entropy
loss and perplexity.

This placeholder defines the public surface; the body raises
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from hymo.core.exceptions import NotImplementedError_

__all__ = ["get_val_batch", "compute_validation_loss", "ValMetrics"]


@dataclass
class ValMetrics:
    """The result of a validation pass.

    Attributes
    ----------
    loss : float
        Mean cross-entropy loss (per token).
    ppl : float
        ``exp(loss)`` perplexity.
    num_batches : int
        Number of batches evaluated.
    num_tokens : int
        Number of tokens evaluated.
    """

    loss: float
    ppl: float
    num_batches: int
    num_tokens: int


# Path to the held-out validation binary (architecture doc §6.3).
# This is the canonical location; the data pipeline (Phase 4) writes
# the file here.
DEFAULT_VAL_BIN = Path("data/tokens/val.bin")

# Module-level cache: the val.bin is ~1.8 GB, so we mmap it once.
_val_cache: npt.NDArray[np.uint32] | None = None
_val_cache_path: Path | None = None


def _load_val_tokens(path: Path = DEFAULT_VAL_BIN) -> npt.NDArray[np.uint32]:
    """Load (or return cached) validation tokens from ``path``."""
    global _val_cache, _val_cache_path
    if _val_cache is None or _val_cache_path != path:
        if not path.exists():
            from hymo.core.exceptions import DataError

            raise DataError(
                f"Validation binary not found: {path}. "
                f"Run the data prep pipeline first (Phase 4, roadmap A7)."
            )
        _val_cache = np.fromfile(path, dtype=np.uint32)
        _val_cache_path = path
    return _val_cache


def get_val_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    seed: int = 42,
    path: Path = DEFAULT_VAL_BIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a deterministic (tokens, targets) window from ``val.bin``.

    Phase 1 placeholder — returns empty tensors; the real
    implementation lands in Phase 3 (design §6.3, roadmap C6).
    """
    raise NotImplementedError_(
        "get_val_batch is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §6.3, roadmap C6)."
    )


def compute_validation_loss(
    model: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_batches: int = 32,
    device: torch.device | str = "cpu",
    seed: int = 42,
    val_bin_path: Path = DEFAULT_VAL_BIN,
) -> ValMetrics:
    """Run ``num_batches`` validation batches and return the metrics.

    Phase 1 placeholder — raises :class:`NotImplementedError_`.
    """
    raise NotImplementedError_(
        "compute_validation_loss is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §6.3, roadmap C6)."
    )
