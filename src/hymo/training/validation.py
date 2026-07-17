"""Real held-out validation loss and perplexity evaluation."""

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
    """The result metrics computed over validation evaluation data."""

    loss: float
    ppl: float
    num_batches: int
    num_tokens: int


DEFAULT_VAL_BIN = Path("data/tokens/val.bin")

_val_cache: npt.NDArray[np.uint32] | None = None
_val_cache_path: Path | None = None


def _load_val_tokens(path: Path = DEFAULT_VAL_BIN) -> npt.NDArray[np.uint32]:
    """Memory-map or load validation tokens from path."""
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
    """Slice a deterministic validation batch from val.bin."""
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
    """Compute average validation cross-entropy loss and perplexity."""
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
