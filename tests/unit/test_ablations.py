"""Tests for the :mod:`hymo.ablations` module (v1.1, deferred)."""

from __future__ import annotations

import pytest

from hymo.ablations import ABLATION_FAMILIES, AblationSpec, build_ablation_config
from hymo.core.config import HyMoConfig
from hymo.core.exceptions import NotImplementedError_


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
        assert s.tokens == 7_500_000_000  # default
        assert s.pod_count == 1  # default


class TestBuildAblationConfig:
    def test_raises(self, tmp_path) -> None:
        base = HyMoConfig()
        with pytest.raises(NotImplementedError_):
            build_ablation_config("A_moe_on_attention", "v1_0_mla_only", base, tmp_path)
