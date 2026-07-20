"""Ablation framework (v1.1, deferred — Phase 4 implementation).

Provides the framework and config derivation for running v1.1 ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path

from hymo.core.config import (
    HyMoConfig,
    SchedulerConfig,
)
from hymo.core.types import Step

__all__ = [
    "ABLATION_FAMILIES",
    "AblationSpec",
    "build_ablation_config",
]


@dataclass(frozen=True)
class AblationSpec:
    """A single ablation's specification metadata."""

    name: str
    description: str
    variants: tuple[str, ...]
    tokens: int = 7_500_000_000
    pod_count: int = 1


ABLATION_FAMILIES: dict[str, AblationSpec] = {
    "A_moe_on_attention": AblationSpec(
        name="A_moe_on_attention",
        description="MoE-on-attention-only (v1.0) vs MoE-on-every-layer.",
        variants=("v1_0_mla_only", "every_layer"),
        pod_count=2,
    ),
    "B_optimizer_partition": AblationSpec(
        name="B_optimizer_partition",
        description="NorMuon-with-MoE-exclusion (v1.0) vs AdamW-only vs NorMuon-everything.",
        variants=("nor_muon_excl_moe", "adamw_only", "nor_muon_all"),
        pod_count=3,
    ),
    "C_mtp_depth": AblationSpec(
        name="C_mtp_depth",
        description="No-MTP vs MTP depth=1 vs MTP depth=2 (v1.0).",
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


def _build_ablation_config_A(
    variant: str, base: HyMoConfig, output_dir: Path
) -> HyMoConfig:
    """MoE-on-attention-only ablation configuration."""
    if variant == "v1_0_mla_only":
        return dc_replace(
            base,
            run=dc_replace(base.run, name="A_moe_on_attention-mla_only"),
        )
    if variant == "every_layer":
        return dc_replace(
            base,
            model=dc_replace(
                base.model,
                inter_dim=base.model.moe_inter_dim,
                n_activated_experts=base.model.n_activated_experts,
            ),
            run=dc_replace(base.run, name="A_moe_on_attention-every_layer"),
        )
    raise ValueError(
        f"Unknown variant {variant!r} for family A_moe_on_attention"
    )


def _build_ablation_config_B(
    variant: str, base: HyMoConfig, output_dir: Path
) -> HyMoConfig:
    """NorMuon-with-MoE-exclusion ablation configuration."""
    if variant == "nor_muon_excl_moe":
        return dc_replace(
            base,
            run=dc_replace(base.run, name="B_optimizer-nor_muon_excl_moe"),
        )
    if variant == "adamw_only":
        return dc_replace(
            base,
            optimizer=dc_replace(
                base.optimizer,
                muon_lr=base.optimizer.adamw_lr,
                adamw_lr=base.optimizer.adamw_lr,
            ),
            run=dc_replace(base.run, name="B_optimizer-adamw_only"),
        )
    if variant == "nor_muon_all":
        return dc_replace(
            base,
            run=dc_replace(base.run, name="B_optimizer-nor_muon_all"),
        )
    raise ValueError(
        f"Unknown variant {variant!r} for family B_optimizer_partition"
    )


def _build_ablation_config_C(
    variant: str, base: HyMoConfig, output_dir: Path
) -> HyMoConfig:
    """MTP depth ablation configuration."""
    if variant == "no_mtp":
        return dc_replace(
            base,
            model=dc_replace(
                base.model,
                mtp_depth=0,
                mtp_loss_weights=(),
            ),
            run=dc_replace(base.run, name="C_mtp_depth-no_mtp"),
        )
    if variant == "depth_1":
        return dc_replace(
            base,
            model=dc_replace(
                base.model,
                mtp_depth=1,
                mtp_loss_weights=(0.3,),
            ),
            run=dc_replace(base.run, name="C_mtp_depth-depth_1"),
        )
    if variant == "depth_2":
        return dc_replace(
            base,
            run=dc_replace(base.run, name="C_mtp_depth-depth_2"),
        )
    raise ValueError(
        f"Unknown variant {variant!r} for family C_mtp_depth"
    )


def _build_ablation_config_D(
    variant: str, base: HyMoConfig, output_dir: Path
) -> HyMoConfig:
    """MQA-4 vs GQA-1.75 ablation configuration."""
    if variant == "mqa4":
        return dc_replace(
            base,
            run=dc_replace(base.run, name="D_mqa4_vs_gqa175-mqa4"),
        )
    if variant == "gqa_1_75":
        return dc_replace(
            base,
            model=dc_replace(
                base.model,
                n_kv_groups=8,
                qk_nope_head_dim=96,
                qk_rope_head_dim=32,
                head_dim=128,
            ),
            run=dc_replace(base.run, name="D_mqa4_vs_gqa175-gqa_1_75"),
        )
    raise ValueError(
        f"Unknown variant {variant!r} for family D_mqa4_vs_gqa175"
    )


_BUILDERS = {
    "A_moe_on_attention": _build_ablation_config_A,
    "B_optimizer_partition": _build_ablation_config_B,
    "C_mtp_depth": _build_ablation_config_C,
    "D_mqa4_vs_gqa175": _build_ablation_config_D,
}


def build_ablation_config(
    family: str,
    variant: str,
    base: HyMoConfig,
    output_dir: str | Path,
) -> HyMoConfig:
    """Derive an ablation configuration from the v1.0 base configuration."""
    output = Path(output_dir)
    builder = _BUILDERS.get(family)
    if builder is None:
        raise ValueError(
            f"Unknown ablation family {family!r}. "
            f"Valid: {list(_BUILDERS)}"
        )
    cfg = builder(variant, base, output)
    return dc_replace(
        cfg,
        run=dc_replace(cfg.run, output_dir=str(output)),
        scheduler=SchedulerConfig(
            total_steps=Step(int(7_500_000_000 / cfg.training.per_step_tokens)),
            warmup_frac=base.scheduler.warmup_frac,
            stable_frac=base.scheduler.stable_frac,
            decay_frac=base.scheduler.decay_frac,
            min_lr_ratio=base.scheduler.min_lr_ratio,
            decay=base.scheduler.decay,
        ),
    )
