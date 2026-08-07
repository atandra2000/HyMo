"""Multi-Token Prediction (Phase 2).

Implements Multi-Token Prediction (MTP) with depth=2, chaining on previous MTP heads'
hidden states and sharing the main model's head representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig

if TYPE_CHECKING:
    from hymo.models.model import HyMo

__all__ = ["MultiTokenPrediction", "MTPOutput"]


class MTPOutput:
    """Container for one auxiliary prediction head's logits, targets, and weight."""

    __slots__ = ("logits", "targets", "loss_weight")

    def __init__(
        self, logits: torch.Tensor, targets: torch.Tensor, loss_weight: float
    ) -> None:
        self.logits = logits
        self.targets = targets
        self.loss_weight = loss_weight

    def __repr__(self) -> str:
        return (
            f"MTPOutput(logits.shape={tuple(self.logits.shape)}, "
            f"targets.shape={tuple(self.targets.shape)}, "
            f"loss_weight={self.loss_weight})"
        )


class MTPBlock(nn.Module):
    """SwiGLU projection that fuses a hidden state with a future-token embedding."""

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)


class MultiTokenPrediction(nn.Module):
    """Chain auxiliary heads so each predicts a progressively later token.

    Every head consumes the prior head's hidden state plus the embedding of its
    target position, while the main model's output projection is shared.
    """

    def __init__(self, config: ModelConfig, main_model: HyMo) -> None:
        super().__init__()
        self.depth = config.mtp_depth
        self.mtp_inter_dim = config.mtp_inter_dim
        self.mtp_loss_weights = tuple(config.mtp_loss_weights)
        self._config = config

        object.__setattr__(self, "_main_model", main_model)

        self.mtp_modules = nn.ModuleList(
            [MTPBlock(config.dim, self.mtp_inter_dim) for _ in range(self.depth)]
        )

    def _mtp_head(
        self, head_idx: int, hidden: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one auxiliary projection and return its logits and next hidden state."""
        block = cast(MTPBlock, self.mtp_modules[head_idx])
        fused = hidden + emb
        h = F.silu(block.w1(fused)) * block.w3(fused)
        out = block.w2(h)
        main = cast(Any, self._main_model)
        logits = main.head(out)
        return logits, out

    def forward(
        self, tokens: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, list[MTPOutput]]:
        """Produce main logits and aligned auxiliary targets for each usable depth.

        The sequence shortens by one position per head so every auxiliary logit
        has a target exactly one or more tokens ahead in the original sequence.
        """
        B, T = tokens.shape
        main = cast(Any, self._main_model)
        main_logits, main_hidden = main.forward_with_hidden(
            tokens, start_pos
        )
        embed = main.embed

        outputs: list[MTPOutput] = []
        prev_hidden = main_hidden
        for d in range(self.depth):
            usable = T - d - 1
            if usable <= 0:
                break
            target_ids = tokens[:, d + 1: d + 1 + usable]
            emb = embed(target_ids)
            h_in = prev_hidden[:, :usable]
            logits, new_hidden = self._mtp_head(d, h_in, emb)
            weight = self.mtp_loss_weights[d]
            outputs.append(
                MTPOutput(logits=logits, targets=target_ids, loss_weight=weight)
            )
            prev_hidden = new_hidden
        return main_logits, outputs
