"""Multi-Token Prediction (Phase 2).

The real implementation (architecture doc §2.8, roadmap B4) is
``mtp_depth=2`` with weights ``[0.3, 0.1]``. The 2nd MTP head chains
on the 1st MTP's output hidden state (not the main model's hidden
state). Both MTP heads share the main model's head (no extra storage).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from hymo.core.config import ModelConfig

if TYPE_CHECKING:
    from hymo.models.fusionllm import HyMo

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


class MTPBlock(nn.Module):
    """A single MTP head: a small SwiGLU that fuses the previous hidden
    state with the shifted embedding and emits logits via the shared
    main head."""

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction head, depth=2 with chained hidden.

    Architecture doc §2.8. Phase 2 implementation.

    Parameters
    ----------
    config : ModelConfig
    main_model : nn.Module
        The main :class:`hymo.models.fusionllm.HyMo` model. The MTP
        head shares ``main_model.embed`` and ``main_model.head``.
    """

    def __init__(self, config: ModelConfig, main_model: HyMo) -> None:
        super().__init__()
        self.depth = config.mtp_depth
        self.mtp_inter_dim = config.mtp_inter_dim
        self.mtp_loss_weights = tuple(config.mtp_loss_weights)
        self._config = config
        # Store via object.__setattr__ to avoid PyTorch registering
        # _main_model as a child module (which would create a circular
        # reference and infinite recursion in .train() / .eval()).
        object.__setattr__(self, "_main_model", main_model)

        # One MTP module per head (the chained hidden architecture).
        # Each head is a small SwiGLU that fuses the previous hidden
        # state with the shifted embedding and produces logits via the
        # main model's shared head.
        self.mtp_modules = nn.ModuleList(
            [MTPBlock(config.dim, self.mtp_inter_dim) for _ in range(self.depth)]
        )

    def _mtp_head(
        self, head_idx: int, hidden: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MTP head: fuse ``hidden`` and ``emb``, return
        ``(logits, new_hidden)``. The new hidden is the head's output
        representation, chained into the next depth."""
        block = cast(MTPBlock, self.mtp_modules[head_idx])
        fused = hidden + emb
        h = F.silu(block.w1(fused)) * block.w3(fused)  # (B, T, mtp_inter_dim)
        out = block.w2(h)                              # (B, T, dim)
        main = cast(Any, self._main_model)
        logits = main.head(out)           # (B, T, vocab)
        return logits, out

    def forward(
        self, tokens: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, list[MTPOutput]]:
        """Multi-Token Prediction forward (design §2.8).

        The first MTP head consumes the main model's hidden state; the
        second head chains on the first head's output hidden state. Each
        head predicts the token ``depth+1`` steps ahead, with logits
        produced through the main model's shared head.

        Parameters
        ----------
        tokens : torch.Tensor
            Input token ids of shape ``(B, T)``.
        start_pos : int
            Offset for the main model's ``forward_with_hidden`` (unused
            for the MTP fusion; accepted for API symmetry).

        Returns
        -------
        tuple[torch.Tensor, list[MTPOutput]]
            ``(main_logits, mtp_outputs)`` where ``main_logits`` is the
            next-token logits and ``mtp_outputs`` is a list of length
            ``depth`` of :class:`MTPOutput`.
        """
        B, T = tokens.shape
        main = cast(Any, self._main_model)
        main_logits, main_hidden = main.forward_with_hidden(
            tokens, start_pos
        )
        embed = main.embed  # weight-sharing lookup

        outputs: list[MTPOutput] = []
        prev_hidden = main_hidden
        for d in range(self.depth):
            usable = T - d - 1
            if usable <= 0:
                break
            # Targets: tokens shifted by (d+1); inputs: the first `usable`
            # positions of the hidden state.
            target_ids = tokens[:, d + 1: d + 1 + usable]       # (B, usable)
            emb = embed(target_ids)                             # (B, usable, dim)
            h_in = prev_hidden[:, :usable]                      # (B, usable, dim)
            logits, new_hidden = self._mtp_head(d, h_in, emb)
            weight = self.mtp_loss_weights[d]
            outputs.append(
                MTPOutput(logits=logits, targets=target_ids, loss_weight=weight)
            )
            prev_hidden = new_hidden
        return main_logits, outputs
