"""Tests for the :mod:`hymo.ablations` module (v1.1, Phase 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hymo.ablations import ABLATION_FAMILIES, AblationSpec, build_ablation_config
from hymo.core.config import HyMoConfig
from hymo.core.exceptions import AblationConfigError


class TestAblationFamilies:
    def test_four_families(self) -> None:
        assert set(ABLATION_FAMILIES.keys()) == {
            "A_moe_on_attention",
            "B_optimizer_partition",
            "C_mtp_depth",
            "D_mqa4_vs_gqa175",
        }

    def test_family_A(self) -> None:
        a = ABLATION_FAMILIES["A_moe_on_attention"]
        assert a.pod_count == 2
        assert "v1_0_mla_only" in a.variants
        assert "every_layer" in a.variants
        assert a.tokens == 7_500_000_000

    def test_family_B_has_three_variants(self) -> None:
        a = ABLATION_FAMILIES["B_optimizer_partition"]
        assert a.pod_count == 3

    def test_family_C_has_three_variants(self) -> None:
        a = ABLATION_FAMILIES["C_mtp_depth"]
        assert a.pod_count == 3
        assert "no_mtp" in a.variants
        assert "depth_1" in a.variants
        assert "depth_2" in a.variants

    def test_family_D_has_two_variants(self) -> None:
        a = ABLATION_FAMILIES["D_mqa4_vs_gqa175"]
        assert a.pod_count == 2
        assert "mqa4" in a.variants
        assert "gqa_1_75" in a.variants


class TestAblationSpec:
    def test_construct(self) -> None:
        s = AblationSpec(
            name="x",
            description="y",
            variants=("a", "b"),
        )
        assert s.tokens == 7_500_000_000
        assert s.pod_count == 1


class TestBuildAblationConfig:
    def test_build_v1_0_mla_only(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        cfg = build_ablation_config("A_moe_on_attention", "v1_0_mla_only", base, tmp_path)
        assert "A_moe_on_attention" in cfg.run.name

    def test_build_every_layer(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        cfg = build_ablation_config("A_moe_on_attention", "every_layer", base, tmp_path)
        assert "every_layer" in cfg.run.name

    def test_build_adamw_only(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        cfg = build_ablation_config("B_optimizer_partition", "adamw_only", base, tmp_path)
        assert "adamw_only" in cfg.run.name

    def test_build_no_mtp(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        cfg = build_ablation_config("C_mtp_depth", "no_mtp", base, tmp_path)
        assert cfg.model.mtp_depth == 0
        assert cfg.model.mtp_loss_weights == ()

    def test_build_gqa_1_75(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        cfg = build_ablation_config("D_mqa4_vs_gqa175", "gqa_1_75", base, tmp_path)
        assert "gqa_1_75" in cfg.run.name
        assert cfg.model.n_kv_groups == 8

    def test_unknown_family_raises(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        with pytest.raises(AblationConfigError):
            build_ablation_config("unknown_family", "variant", base, tmp_path)

    def test_unknown_variant_raises(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        with pytest.raises(AblationConfigError):
            build_ablation_config("A_moe_on_attention", "unknown_variant", base, tmp_path)

    def test_scheduler_is_shorter(self, tmp_path: Path) -> None:
        base = HyMoConfig()
        cfg = build_ablation_config("C_mtp_depth", "depth_2", base, tmp_path)
        assert cfg.scheduler.total_steps < base.scheduler.total_steps
