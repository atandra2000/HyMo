"""Multi-Token Prediction (Phase 1 placeholder).

The real implementation (architecture doc §2.8, roadmap B4) is
``mtp_depth=2`` with weights ``[0.3, 0.1]``. The 2nd MTP head chains
on the 1st MTP's output hidden state (not the main model's hidden
state). Both MTP heads share the main model's head (no extra storage).

This placeholder defines the parameter shapes and the loss-weight
tuple; the forward pass raises :class:`NotImplementedError_`.
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = ["MultiTokenPrediction", "MTPOutput"]


class MTPOutput:
    """A single MTP head's output: (logits, targets, loss_weight).

    Returned by :meth:`MultiTokenPrediction.forward` as a list of
    length :attr:`MultiTokenPrediction.depth`.
    """

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


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction head, depth=2 with chained hidden.

    Architecture doc §2.8. Phase 1 placeholder.

    Parameters
    ----------
    config : ModelConfig
    main_model : nn.Module
        The main :class:`hymo.models.fusionllm.HyMo` model. The MTP
        head shares ``main_model.embed`` and ``main_model.head``.
    """

    def __init__(self, config: ModelConfig, main_model: nn.Module) -> None:
        super().__init__()
        self.depth = config.mtp_depth
        self.mtp_inter_dim = config.mtp_inter_dim
        self.mtp_loss_weights = tuple(config.mtp_loss_weights)
        self._config = config
        self._main_model = main_model

        # One MTP module per head (the chained hidden architecture).
        # Phase 2: each is a small SwiGLU that takes the previous
        # hidden state + the embed-token shifted by (depth+1) and
        # produces logits via the main head.
        self.mtp_modules = nn.ModuleList()
        for _ in range(self.depth):
            # Placeholder: a single Linear that maps from dim to dim.
            # Real impl in Phase 2 will be a SwiGLU + projection.
            self.mtp_modules.append(nn.Linear(config.dim, config.dim, bias=False))

    def forward(
        self, tokens: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, list[MTPOutput]]:
        """Phase 1 placeholder — raises :class:`NotImplementedError_`."""
        raise NotImplementedError_(
            "MultiTokenPrediction.forward is a Phase 1 placeholder; "
            "the real implementation lands in Phase 2 (design §2.8, "
            "roadmap B4)."
        )
