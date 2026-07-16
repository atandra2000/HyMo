"""Ablation framework (v1.1, deferred).

Architecture doc §16. The 4 ablations are NOT part of v1.0; they run
on separate pods after the v1.0 primary is complete. This module
provides the framework + config templates so the v1.1 ablations can
be launched without code changes.

Each ablation is a 7.5B-token (25% of primary) run on a separate
4× A100 80GB SXM pod. The 4 ablations can run in parallel (1.3-2 days
total) or sequentially (1.3 days each = 5.2 days total).

Ablation families (architecture doc §16):

A. MoE-on-attention-only (claim 1) — 2 variants.
B. NorMuon-with-MoE-exclusion (claim 2) — 3 variants.
C. MTP depth ablation (claim 3) — 3 variants.
D. MQA-4 vs GQA-1.75 (claim 6) — 2 variants.

This module is a Phase 1 placeholder. The factory functions raise
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hymo.core.config import HyMoConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = [
    "ABLATION_FAMILIES",
    "AblationSpec",
    "build_ablation_config",
]


@dataclass(frozen=True)
class AblationSpec:
    """A single ablation's spec.

    Attributes
    ----------
    name : str
        E.g. ``"A_moe_on_attention"``.
    description : str
        One-line description.
    variants : tuple[str, ...]
        Names of the variants (each is a config derived from the v1.0
        primary).
    tokens : int
        Tokens per variant (default 7.5B = 25% of primary).
    pod_count : int
        Number of pods (1 per variant if running in parallel).
    """

    name: str
    description: str
    variants: tuple[str, ...]
    tokens: int = 7_500_000_000
    pod_count: int = 1


# The 4 ablation families (architecture doc §16).
ABLATION_FAMILIES: dict[str, AblationSpec] = {
    "A_moe_on_attention": AblationSpec(
        name="A_moe_on_attention",
        description=(
            "MoE-on-attention-only (v1.0) vs MoE-on-every-layer."
        ),
        variants=("v1_0_mla_only", "every_layer"),
        pod_count=2,
    ),
    "B_optimizer_partition": AblationSpec(
        name="B_optimizer_partition",
        description=(
            "NorMuon-with-MoE-exclusion (v1.0) vs AdamW-only vs "
            "NorMuon-everything."
        ),
        variants=("nor_muon_excl_moe", "adamw_only", "nor_muon_all"),
        pod_count=3,
    ),
    "C_mtp_depth": AblationSpec(
        name="C_mtp_depth",
        description=(
            "No-MTP vs MTP depth=1 vs MTP depth=2 (v1.0)."
        ),
        variants=("no_mtp", "depth_1", "depth_2"),
        pod_count=3,
    ),
    "D_mqa4_vs_gqa175": AblationSpec(
        name="D_mqa4_vs_gqa175",
        description="MQA-4 (v1.0) vs GQA-1.75 (earlier draft).",
        variants=("mqa4", "gqa_1_75"),
        pod_count=2,
    ),
}


def build_ablation_config(
    family: str,
    variant: str,
    base: HyMoConfig,
    output_dir: str | Path,
) -> HyMoConfig:
    """Derive an ablation config from the v1.0 base.

    Phase 1 placeholder — raises :class:`NotImplementedError_`. The
    real implementation lands in Phase 4 (design §16, roadmap F1-F4).
    """
    raise NotImplementedError_(
        f"build_ablation_config({family!r}, {variant!r}) is a Phase 1 "
        f"placeholder; the real implementation lands in Phase 4 "
        f"(design §16, roadmap F1-F4)."
    )
