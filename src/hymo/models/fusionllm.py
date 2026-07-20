"""The HyMo 32-layer 3:1 GDN:MLA stack (Phase 2).

Builds a 32-block stack with 8 MLA layers and 24 GDN layers,
honoring the NoPE-hybrid configuration.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hymo.core.config import HyMoConfig, ModelConfig
from hymo.models.gdn import GatedDeltaNetBlock
from hymo.models.mla import MLABlock
from hymo.models.mtp import MultiTokenPrediction

__all__ = ["HyMo", "build_hymo"]


class HyMo(nn.Module):
    """The 32-layer 3:1 GDN:MLA hybrid model (architecture doc §2)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        if config.tie_embeddings:
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
            self.head.weight = self.embed.weight
        else:
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.norm = nn.RMSNorm(config.dim)
        self.logit_softcap = config.logit_softcap

        mla_positions = config.mla_positions
        nope_hybrid = config.nope_hybrid_gdn_positions

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if i in mla_positions:
                self.layers.append(MLABlock(config, layer_idx=i))
            else:
                use_rope = i not in nope_hybrid
                self.layers.append(
                    GatedDeltaNetBlock(config, layer_idx=i, use_rope=use_rope)
                )

        self._mtp: MultiTokenPrediction | None
        if config.mtp_depth > 0:
            self._mtp = MultiTokenPrediction(config, main_model=self)
        else:
            self._mtp = None

    @property
    def config(self) -> ModelConfig:
        return self._config

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return the parameter count (trainable only if only_trainable is True)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    def _run_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Run the 32-layer stack, honoring per-layer gradient checkpointing."""
        for layer in self.layers:
            use_cp = getattr(layer, "use_checkpoint", False)
            if use_cp and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass and return next-token logits (B, T, vocab_size)."""
        hidden = self.forward_with_hidden(tokens)[1]
        logits = self.head(hidden)
        return self.softcap(logits)

    def forward_with_hidden(
        self, tokens: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both the logits and final pre-head hidden states (for MTP)."""
        x = self.embed(tokens)
        x = self._run_layers(x)
        hidden = self.norm(x)
        logits = self.head(hidden)
        return self.softcap(logits), hidden

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logit softcapping (PaLM-style) using tanh scaling."""
        if self.logit_softcap <= 0:
            return logits
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


def build_hymo(config: HyMoConfig) -> HyMo:
    """Factory function to build a HyMo instance from a HyMoConfig."""
    return HyMo(config.model)
