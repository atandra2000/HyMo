"""μP initialization (Phase 1 placeholder).

The real implementation (architecture doc §4, roadmap B7) applies:

- Zero-init every parameter whose name contains ``"gate"``, ``"g_proj"``,
  ``"A_log"``, ``"dt_bias"``, ``"router"``, ``"output_head"``, ``"bias"``,
  ``"q_norm"``, ``"kv_norm"``, ``"q_norm_qk"``, ``"k_norm_qk"``, ``"mtp"``,
  or ``"D"``.
- Standard init (``std = 0.02``) on every 1D parameter.
- μP-scaled init (``std = 1 / dim``) on every 2D attention/MLP weight.
- ``std = 1 / sqrt(dim)`` on the embedding.

This placeholder defines the function signature; the body raises
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from torch import nn

from hymo.core.config import ModelConfig
from hymo.core.exceptions import NotImplementedError_

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
    """Apply μP initialization in place.

    Architecture doc §4. Phase 1 placeholder.

    The function is silent on success. On failure, raise the underlying
    error.
    """
    raise NotImplementedError_(
        "mup_init is a Phase 1 placeholder; the real implementation "
        "lands in Phase 2 (design §4, roadmap B7)."
    )


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
