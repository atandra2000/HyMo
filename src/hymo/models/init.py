"""μP initialization (Phase 2).

Applies zero-init to specific modules (gates, biases, norms, routing, decay log parameters)
and scales 2D weights according to the maximal update parametrization (μP).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from hymo.core.config import ModelConfig

__all__ = ["mup_init", "MUP_ZERO_KEYWORDS"]

# Keywords indicating parameters to be zero-initialized
MUP_ZERO_KEYWORDS: frozenset[str] = frozenset(
    {
        "gate",
        "g_proj",
        "a_log",
        "dt_bias",
        "router",
        "output_head",
        "bias",
        "q_norm",
        "kv_norm",
        "q_norm_qk",
        "k_norm_qk",
        "mtp",
        "embed",
        "d",
    }
)


def mup_init(model: nn.Module, config: ModelConfig) -> None:
    """Apply μP initialization in place (architecture doc §4)."""
    dim = config.dim
    attn_std = 1.0 / dim
    embed_std = 1.0 / math.sqrt(dim)
    for name, p in model.named_parameters():
        if zero_init_predicate(name):
            with torch.no_grad():
                p.data.zero_()
            continue
        if p.dim() < 2:
            continue
        with torch.no_grad():
            std = embed_std if "embed" in name else attn_std
            p.data.normal_(mean=0.0, std=std)


def zero_init_predicate(param_name: str) -> bool:
    """Return True if the parameter should be zero-initialized under μP."""
    lowered = param_name.lower()
    for kw in MUP_ZERO_KEYWORDS:
        if kw in lowered:
            if kw == "d" and "embed" in lowered:
                continue
            return True
    return False
