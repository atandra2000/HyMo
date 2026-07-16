"""Real held-out validation (Phase 3 implementation).

Reads batches from ``data/tokens/val.bin``, runs the model in eval
mode, and returns the cross-entropy loss and perplexity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.exceptions import DataError

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

    Deterministic across runs: uses ``seed`` to compute a fixed offset
    into the token array, then slices ``B * (T + 1)`` tokens to produce
    the ``(B, T)`` input and ``(B, T)`` target (shifted by 1).
    """
    tokens_np = _load_val_tokens(path)
    total_tokens = batch_size * (seq_len + 1)

    offset = (seed * 7919) % (len(tokens_np) - total_tokens - 1)
    chunk_np = tokens_np[offset: offset + total_tokens].astype(np.int64)

    chunk_t = torch.from_numpy(chunk_np).to(device=device)
    chunk_t = chunk_t.view(batch_size, seq_len + 1)

    x = chunk_t[:, :seq_len].contiguous()
    y = chunk_t[:, 1:seq_len + 1].contiguous()
    return x, y


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

    Iterates ``num_batches`` deterministic windows from ``val.bin``,
    runs the model in ``eval()`` mode, computes CE loss, and returns
    aggregated :class:`ValMetrics`.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(num_batches):
            batch_seed = seed + i * 131
            tokens, targets = get_val_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                seed=batch_seed,
                path=val_bin_path,
            )
            logits = model(tokens)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
            )
            n = targets.numel()
            total_loss += loss.item() * n
            total_tokens += n

    if was_training:
        model.train()

    mean_loss = total_loss / max(total_tokens, 1)
    return ValMetrics(
        loss=mean_loss,
        ppl=math.exp(mean_loss),
        num_batches=num_batches,
        num_tokens=total_tokens,
    )
