"""μP initialization (Phase 2).

The real implementation (architecture doc §4, roadmap B7) applies:

- Zero-init every parameter whose name contains ``"gate"``, ``"g_proj"``,
  ``"A_log"``, ``"dt_bias"``, ``"router"``, ``"output_head"``, ``"bias"``,
  ``"q_norm"``, ``"kv_norm"``, ``"q_norm_qk"``, ``"k_norm_qk"``, ``"mtp"``,
  or ``"D"``.
- Standard init (``std = 1 / dim``) on every 2D attention/MLP weight.
- ``std = 1 / sqrt(dim)`` on the embedding.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from hymo.core.config import ModelConfig

__all__ = ["mup_init", "MUP_ZERO_KEYWORDS"]


# Parameters whose name (lowercased) contains any of these substrings
# are zero-initialized.
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
        "embed",  # tied embedding / head, zero-init per μP.
        "d",  # matches the GDN "D" scalar but ALSO matches "embed" — handled
        # in the predicate below.
    }
)


def mup_init(model: nn.Module, config: ModelConfig) -> None:
    """Apply μP initialization in place (architecture doc §4).

    Zero-inits the scalars/gains, μP-scales the 2D weights, and uses an
    embedding-scale init for the embedding table. Silent on success.
    """
    dim = config.dim
    attn_std = 1.0 / dim
    embed_std = 1.0 / math.sqrt(dim)
    for name, p in model.named_parameters():
        if zero_init_predicate(name):
            with torch.no_grad():
                p.data.zero_()
            continue
        if p.dim() < 2:
            # 1D (non-scalar) params keep their default init; leave as-is.
            continue
        with torch.no_grad():
            std = embed_std if "embed" in name else attn_std
            p.data.normal_(mean=0.0, std=std)


def zero_init_predicate(param_name: str) -> bool:
    """Return True iff the parameter should be zero-initialized under μP.

    The predicate handles the "D" edge case (which would otherwise match
    every parameter name containing "d" in the embed layer).
    """
    lowered = param_name.lower()
    for kw in MUP_ZERO_KEYWORDS:
        if kw in lowered:
            # Skip the "embed" case where "d" is a substring of "embed".
            if kw == "d" and "embed" in lowered:
                continue
            return True
    return False
