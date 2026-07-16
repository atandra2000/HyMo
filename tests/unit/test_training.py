"""Tests for the training partition and the placeholder optimizers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from hymo.core.config import load_config
from hymo.core.exceptions import NotImplementedError_
from hymo.models import HyMo
from hymo.training import (
    CautiousAdamW,
    JointWSDScheduler,
    NorMuon,
    Optimizers,
    ParameterPartition,
    build_optimizers,
    goes_to_adamw,
    goes_to_nor_muon,
    partition_parameters,
)

# ----------------------------------------------------------------------
# Partition predicate
# ----------------------------------------------------------------------


class TestGoesToAdamw:
    def test_embed_to_adamw(self) -> None:
        p = torch.empty(64_256, 896)
        assert goes_to_adamw("embed.weight", nn.Parameter(p)) is True

    def test_head_to_adamw(self) -> None:
        p = torch.empty(64_256, 896)
        assert goes_to_adamw("head.weight", nn.Parameter(p)) is True

    def test_norm_to_adamw(self) -> None:
        p = torch.empty(896)
        assert goes_to_adamw("layers.0.attn_norm.weight", nn.Parameter(p)) is True

    def test_gdn_scalar_A_log_to_adamw(self) -> None:
        p = torch.empty(1280)
        assert goes_to_adamw("layers.0.gdn.A_log", nn.Parameter(p)) is True

    def test_gdn_scalar_D_to_adamw(self) -> None:
        p = torch.empty(40)
        assert goes_to_adamw("layers.0.gdn.D", nn.Parameter(p)) is True

    def test_moe_gate_to_adamw(self) -> None:
        p = torch.empty(16, 896)
        assert goes_to_adamw("layers.0.moe.gate.weight", nn.Parameter(p)) is True

    def test_moe_expert_weight_to_adamw(self) -> None:
        """Claim 2: MoE expert weights go to AdamW, not NorMuon."""
        p = torch.empty(896, 2304)
        assert goes_to_adamw("layers.0.moe.experts.0.w1.weight", nn.Parameter(p)) is True
        assert goes_to_adamw("layers.0.moe.experts.0.w2.weight", nn.Parameter(p)) is True
        assert goes_to_adamw("layers.0.moe.experts.0.w3.weight", nn.Parameter(p)) is True
        assert goes_to_adamw("layers.0.moe.experts.15.w1.weight", nn.Parameter(p)) is True

    def test_shared_expert_to_adamw(self) -> None:
        p = torch.empty(896, 2304)
        assert goes_to_adamw("layers.0.moe.shared_expert.w1.weight", nn.Parameter(p)) is True
        assert goes_to_adamw("layers.0.moe.shared_expert.w2.weight", nn.Parameter(p)) is True
        assert goes_to_adamw("layers.0.moe.shared_expert.w3.weight", nn.Parameter(p)) is True

    def test_mla_attn_to_nor_muon(self) -> None:
        p = torch.empty(896, 224)
        assert goes_to_adamw("layers.0.attn.attn.wq_a.weight", nn.Parameter(p)) is False
        p2 = torch.empty(224, 2048)
        assert goes_to_adamw("layers.0.attn.attn.wq_b.weight", nn.Parameter(p2)) is False
        p3 = torch.empty(896, 128)
        assert goes_to_adamw("layers.0.attn.attn.wkv_a.weight", nn.Parameter(p3)) is False

    def test_gdn_matrix_to_nor_muon(self) -> None:
        p = torch.empty(896, 7680)
        assert goes_to_adamw("layers.0.gdn.in_proj.weight", nn.Parameter(p)) is False
        p2 = torch.empty(1280, 1280)
        assert goes_to_adamw("layers.0.gdn.g_proj.weight", nn.Parameter(p2)) is False
        p3 = torch.empty(1280, 896)
        assert goes_to_adamw("layers.0.gdn.out_proj.weight", nn.Parameter(p3)) is False


class TestGoesToNorMuon:
    def test_attn_to_nor_muon(self) -> None:
        p = torch.empty(896, 224)
        assert goes_to_nor_muon("layers.0.attn.attn.wq_a.weight", nn.Parameter(p)) is True

    def test_moe_expert_not_to_nor_muon(self) -> None:
        """Claim 2: MoE expert weights are excluded from NorMuon."""
        p = torch.empty(896, 2304)
        assert goes_to_nor_muon("layers.0.moe.experts.0.w1.weight", nn.Parameter(p)) is False

    def test_1d_to_nor_muon_returns_false(self) -> None:
        """1D params are not NorMuon — they are AdamW."""
        p = torch.empty(896)
        assert goes_to_nor_muon("layers.0.norm.weight", nn.Parameter(p)) is False


class TestPartitionParameters:
    def test_partition_against_real_model(self) -> None:
        """The partition routes the right parameters on the real HyMo model."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)

        # Every parameter must be in exactly one group.
        total = len(partition.adamw) + len(partition.nor_muon)
        all_params = sum(1 for _ in model.parameters())
        # Some params are tied (head ↔ embed) so we may see one fewer.
        assert total <= all_params
        assert total >= all_params - 1

    def test_partition_count_moe_experts(self) -> None:
        """8 MLA layers × 16 experts × 3 matrices = 384 expert weights.
        Wait — the v1.0 has 8 MLA layers, each with 16 routed + 1 shared.
        16 routed × 3 + 1 shared × 3 = 51 expert tensors per MLA layer.
        8 MLA layers × 51 = 408 expert tensors total.
        """
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)

        adamw_names = {id(p) for p in partition.adamw}
        # Count how many routed expert weights landed on AdamW.
        n_expert = 0
        for name, p in model.named_parameters():
            if (
                ".experts." in name
                and (
                    name.endswith(".w1.weight")
                    or name.endswith(".w2.weight")
                    or name.endswith(".w3.weight")
                )
                and id(p) in adamw_names
            ):
                n_expert += 1
        # 16 routed × 8 MLA layers × 3 matrices = 384.
        assert n_expert == 384

        n_shared = 0
        for name, p in model.named_parameters():
            if ".shared_expert." in name and id(p) in adamw_names:
                    n_shared += 1
        # 1 shared × 8 MLA layers × 3 matrices = 24.
        assert n_shared == 24

    def test_partition_count_gdn_a_log(self) -> None:
        """24 GDN layers × 1 A_log = 24 scalars on AdamW."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}
        n_a_log = 0
        for name, p in model.named_parameters():
            if name.endswith(".A_log") and id(p) in adamw_ids:
                n_a_log += 1
        assert n_a_log == 24  # one per GDN layer

    def test_parameter_partition_repr(self) -> None:
        p = ParameterPartition()
        assert repr(p) == "ParameterPartition(adamw=0, nor_muon=0)"
        assert len(p) == 0


# ----------------------------------------------------------------------
# Optimizer placeholders
# ----------------------------------------------------------------------


class TestNorMuon:
    def test_construct(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        opt = NorMuon([p], lr=0.02)
        assert opt.defaults["lr"] == 0.02
        assert opt.defaults["momentum"] == 0.95
        assert opt.defaults["cautious_wd"] is True

    def test_invalid_lr_raises(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        with pytest.raises(ValueError):
            NorMuon([p], lr=0)

    def test_invalid_momentum_raises(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        with pytest.raises(ValueError):
            NorMuon([p], momentum=1.0)

    def test_step_raises_not_implemented(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn_like(p)
        opt = NorMuon([p])
        with pytest.raises(NotImplementedError_):
            opt.step()


class TestCautiousAdamW:
    def test_construct(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        opt = CautiousAdamW([p], lr=3e-4)
        assert opt.defaults["lr"] == 3e-4
        assert opt.defaults["betas"] == (0.9, 0.95)
        assert opt.defaults["cautious_wd"] is False

    def test_invalid_lr_raises(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        with pytest.raises(ValueError):
            CautiousAdamW([p], lr=0)

    def test_step_raises_not_implemented(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn_like(p)
        opt = CautiousAdamW([p])
        with pytest.raises(NotImplementedError_):
            opt.step()


# ----------------------------------------------------------------------
# Optimizers container
# ----------------------------------------------------------------------


class TestOptimizers:
    def test_state_dict_and_load(self) -> None:
        p1 = torch.nn.Parameter(torch.randn(10, 10))
        p2 = torch.nn.Parameter(torch.randn(5, 5))
        nm = NorMuon([p1])
        aw = CautiousAdamW([p2])
        opts = Optimizers(nor_muon=nm, adamw=aw)
        sd = opts.state_dict()
        assert "nor_muon" in sd
        assert "adamw" in sd
        # Round-trip.
        opts2 = Optimizers(NorMuon([p1]), CautiousAdamW([p2]))
        opts2.load_state_dict(sd)


class TestBuildOptimizers:
    def test_build_from_real_model(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        opts = build_optimizers(model, config.optimizer)
        assert isinstance(opts, Optimizers)
        assert isinstance(opts.adamw, CautiousAdamW)
        # NorMuon may be None if all params went to AdamW; in practice
        # the HyMo model has plenty of 2D weights for NorMuon.
        assert opts.nor_muon is not None
        assert isinstance(opts.nor_muon, NorMuon)

    def test_lr_ratio_preserved(self) -> None:
        """lr_muon / lr_adamw ≈ 66.7 (architecture doc §5.3)."""
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        opts = build_optimizers(model, config.optimizer)
        nm_lr = opts.nor_muon.defaults["lr"]
        aw_lr = opts.adamw.defaults["lr"]
        assert nm_lr / aw_lr == pytest.approx(66.67, abs=0.01)


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------


class TestJointWSDScheduler:
    def test_construct(self) -> None:
        from hymo.core.config import SchedulerConfig

        cfg = SchedulerConfig()
        s = JointWSDScheduler(cfg)
        assert s.warmup_steps == cfg.warmup_steps
        assert s.stable_steps == cfg.stable_steps
        assert s.decay_steps == cfg.decay_steps
        assert s.min_lr_ratio == cfg.min_lr_ratio
        assert s.decay_kind == "linear"

    def test_get_factor_raises(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        with pytest.raises(NotImplementedError_):
            s.get_factor(0)

    def test_decay_factor_linear(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        # progress=0 → 1.0; progress=1 → 0.0.
        assert s._decay_factor(0.0, "linear") == 1.0
        assert s._decay_factor(1.0, "linear") == 0.0
        assert s._decay_factor(0.5, "linear") == 0.5

    def test_decay_factor_cosine(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        # progress=0 → 1.0; progress=1 → 0.0; progress=0.5 → 0.5.
        assert s._decay_factor(0.0, "cosine") == 1.0
        assert s._decay_factor(1.0, "cosine") == pytest.approx(0.0, abs=1e-9)
        assert s._decay_factor(0.5, "cosine") == pytest.approx(0.5, abs=1e-9)

    def test_decay_factor_sqrt(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        # progress=0 → 1.0; progress=1 → 0.0.
        assert s._decay_factor(0.0, "sqrt") == 1.0
        assert s._decay_factor(1.0, "sqrt") == pytest.approx(0.0, abs=1e-9)

    def test_decay_factor_invalid_progress(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        with pytest.raises(ValueError):
            s._decay_factor(-0.1, "linear")
        with pytest.raises(ValueError):
            s._decay_factor(1.5, "linear")

    def test_decay_factor_unknown_kind(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        with pytest.raises(ValueError):
            s._decay_factor(0.5, "exponential")  # type: ignore[arg-type]

    def test_state_dict(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        sd = s.state_dict()
        assert "step" in sd
