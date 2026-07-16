"""The HyMo 32-layer 3:1 GDN:MLA stack (Phase 2).

The real implementation (architecture doc §2, roadmap B5) builds the
32-block stack with 8 MLA layers at positions {0, 4, 8, ..., 28} and
24 GDN layers at the complement, with the per-layer ``use_rope`` flag
honoring the NoPE-hybrid (CR-12; default OFF for v1.0).

Public API
----------
- :class:`HyMo` — the assembled model.
- :func:`build_hymo` — factory function that takes a
  :class:`hymo.core.config.HyMoConfig` and returns a :class:`HyMo`.
"""

from __future__ import annotations

import torch
from torch import nn

from hymo.core.config import HyMoConfig, ModelConfig
from hymo.models.gdn import GatedDeltaNetBlock
from hymo.models.mla import MLABlock
from hymo.models.mtp import MultiTokenPrediction
from hymo.registry import MODELS

__all__ = ["HyMo", "build_hymo"]


@MODELS.register("hymo")
class HyMo(nn.Module):
    """The 32-layer 3:1 GDN:MLA hybrid model.

    Architecture doc §2. Phase 2 implementation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

        # Token embedding.
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        # Tied output head: a thin reference to the embed weight.
        if config.tie_embeddings:
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
            self.head.weight = self.embed.weight
        else:
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Final RMSNorm + logit softcap (PaLM-style).
        self.norm = nn.RMSNorm(config.dim)
        self.logit_softcap = config.logit_softcap

        # 32-layer 3:1 GDN:MLA stack.
        mla_positions = config.mla_positions
        nope_hybrid = config.nope_hybrid_gdn_positions

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if i in mla_positions:
                self.layers.append(MLABlock(config, layer_idx=i))
            else:
                # GDN block with NoPE if i is in the NoPE-hybrid set
                # (empty for v1.0 by default; CR-12 mitigation).
                use_rope = i not in nope_hybrid
                self.layers.append(
                    GatedDeltaNetBlock(config, layer_idx=i, use_rope=use_rope)
                )

        # MTP head (depth=2; shares main head).
        self._mtp: MultiTokenPrediction | None
        if config.mtp_depth > 0:
            self._mtp = MultiTokenPrediction(config, main_model=self)
        else:
            self._mtp = None

    # ---- Public API -----------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        return self._config

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return the parameter count (trainable only if ``only_trainable``)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    def _run_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Run the 32-layer stack, honoring per-layer ``use_checkpoint``."""
        for layer in self.layers:
            use_cp = getattr(layer, "use_checkpoint", False)
            if use_cp and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the full forward and return next-token logits (design §2).

        Parameters
        ----------
        tokens : torch.Tensor
            Token ids of shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, T, vocab_size)``.
        """
        hidden = self.forward_with_hidden(tokens)[1]
        logits = self.head(hidden)
        return self.softcap(logits)

    def forward_with_hidden(
        self, tokens: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward that also returns the last hidden state (for MTP).

        Returns ``(logits, hidden)`` where ``hidden`` is the normalized
        pre-head representation that the MTP heads chain on.
        """
        x = self.embed(tokens)                       # (B, T, dim)
        x = self._run_layers(x)                      # (B, T, dim)
        hidden = self.norm(x)                        # (B, T, dim)
        logits = self.head(hidden)                   # (B, T, vocab)
        return self.softcap(logits), hidden

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the logit softcap (PaLM-style).

        ``logits = softcap * tanh(logits / softcap)``. The default
        softcap is :attr:`ModelConfig.logit_softcap` (15.0).

        This is a public utility so the trainer can use it
        independently of the forward pass (e.g. for distillation).
        """
        if self.logit_softcap <= 0:
            return logits
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


def build_hymo(config: HyMoConfig) -> HyMo:
    """Factory: build a :class:`HyMo` from a top-level :class:`HyMoConfig`.

    This is the canonical way to instantiate the model. The trainer
    uses this; tests use this; ablation configs use this.
    """
    return HyMo(config.model)
