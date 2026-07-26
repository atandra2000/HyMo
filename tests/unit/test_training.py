"""Tests for the training partition, optimizers, and Joint WSD scheduler."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from hymo.core.config import load_config
from hymo.models import HyMo
from hymo.training import (
    CautiousAdamW,
    JointWSDScheduler,
    NorMuon,
    Optimizers,
    ParameterPartition,
    build_optimizers,
    goes_to_adamw,
    partition_parameters,
)


class TestGoesToAdamw:
    """Verify goes_to_adamw partition routes variables to AdamW optimizer."""

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


class TestPartitionParameters:
    """Verify model parameters group partition allocations."""

    def test_partition_against_tiny_model(self, tiny_hymo_model: HyMo) -> None:
        partition = partition_parameters(tiny_hymo_model)
        total = len(partition.adamw) + len(partition.nor_muon)
        all_params = sum(1 for _ in tiny_hymo_model.parameters())
        assert total <= all_params
        assert total >= all_params - 1

    @pytest.mark.heavy
    def test_partition_against_full_model(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        total = len(partition.adamw) + len(partition.nor_muon)
        all_params = sum(1 for _ in model.parameters())
        assert total <= all_params
        assert total >= all_params - 1

    def test_partition_count_moe_experts_tiny(self, tiny_hymo_model: HyMo) -> None:
        m = tiny_hymo_model.config
        n_mla = sum(
            1 for layer in tiny_hymo_model.layers if hasattr(layer, "moe")
        )
        partition = partition_parameters(tiny_hymo_model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_expert = 0
        for name, p in tiny_hymo_model.named_parameters():
            if (
                ".experts." in name
                and (
                    name.endswith(".w1.weight")
                    or name.endswith(".w2.weight")
                    or name.endswith(".w3.weight")
                )
                and id(p) in adamw_ids
            ):
                n_expert += 1
        assert n_expert == n_mla * m.n_routed_experts * 3

        n_shared = 0
        for name, p in tiny_hymo_model.named_parameters():
            if ".shared_expert." in name and id(p) in adamw_ids:
                n_shared += 1
        assert n_shared == n_mla * m.n_shared_experts * 3

    @pytest.mark.heavy
    def test_partition_count_moe_experts_full(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}

        n_expert = 0
        for name, p in model.named_parameters():
            if (
                ".experts." in name
                and (
                    name.endswith(".w1.weight")
                    or name.endswith(".w2.weight")
                    or name.endswith(".w3.weight")
                )
                and id(p) in adamw_ids
            ):
                n_expert += 1
        assert n_expert == 384

        n_shared = 0
        for name, p in model.named_parameters():
            if ".shared_expert." in name and id(p) in adamw_ids:
                n_shared += 1
        assert n_shared == 24

    def test_partition_count_gdn_a_log_tiny(self, tiny_hymo_model: HyMo) -> None:
        partition = partition_parameters(tiny_hymo_model)
        adamw_ids = {id(p) for p in partition.adamw}
        n_a_log = 0
        for name, p in tiny_hymo_model.named_parameters():
            if name.endswith(".A_log") and id(p) in adamw_ids:
                n_a_log += 1
        assert n_a_log == 3

    @pytest.mark.heavy
    def test_partition_count_gdn_a_log_full(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        partition = partition_parameters(model)
        adamw_ids = {id(p) for p in partition.adamw}
        n_a_log = 0
        for name, p in model.named_parameters():
            if name.endswith(".A_log") and id(p) in adamw_ids:
                n_a_log += 1
        assert n_a_log == 24

    def test_parameter_partition_repr(self) -> None:
        p = ParameterPartition()
        assert repr(p) == "ParameterPartition(adamw=0, nor_muon=0)"
        assert len(p) == 0


class TestNorMuon:
    """Verify NorMuon optimizer parameters updates and master weights."""

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

    def test_step_updates_params(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn_like(p)
        orig = p.data.clone()
        opt = NorMuon([p])
        opt.step()
        assert not torch.equal(p.data, orig)
        assert torch.isfinite(p.data).all()
        assert p.grad is not None

    def test_step_preserves_fp32_master(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn_like(p)
        opt = NorMuon([p])
        opt.step()
        state = opt.state[p]
        assert "master_weight" in state
        assert state["master_weight"].dtype == torch.float32


class TestCautiousAdamW:
    """Verify CautiousAdamW optimizer parameters updates and master weights."""

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

    def test_step_updates_params(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn_like(p)
        orig = p.data.clone()
        opt = CautiousAdamW([p])
        opt.step()
        assert not torch.equal(p.data, orig)
        assert torch.isfinite(p.data).all()

    def test_step_preserves_fp32_master(self) -> None:
        p = torch.nn.Parameter(torch.randn(10, 10))
        p.grad = torch.randn_like(p)
        opt = CautiousAdamW([p])
        opt.step()
        state = opt.state[p]
        assert "master_weight" in state
        assert state["master_weight"].dtype == torch.float32


class TestOptimizers:
    """Verify dual optimizers container serialization round trips."""

    def test_state_dict_and_load(self) -> None:
        p1 = torch.nn.Parameter(torch.randn(10, 10))
        p2 = torch.nn.Parameter(torch.randn(5, 5))
        nm = NorMuon([p1])
        aw = CautiousAdamW([p2])
        opts = Optimizers(nor_muon=nm, adamw=aw)
        sd = opts.state_dict()
        assert "nor_muon" in sd
        assert "adamw" in sd
        opts2 = Optimizers(NorMuon([p1]), CautiousAdamW([p2]))
        opts2.load_state_dict(sd)


class TestBuildOptimizers:
    """Verify build_optimizers constructs dual optimizer pairs."""

    def test_build_from_tiny_model(self, tiny_hymo_config: HyMoConfig) -> None:
        model = HyMo(tiny_hymo_config.model)
        opts = build_optimizers(model, tiny_hymo_config.optimizer)
        assert isinstance(opts, Optimizers)
        assert isinstance(opts.adamw, CautiousAdamW)
        assert opts.nor_muon is not None
        assert isinstance(opts.nor_muon, NorMuon)

    def test_lr_ratio_preserved(self, tiny_hymo_config: HyMoConfig) -> None:
        model = HyMo(tiny_hymo_config.model)
        opts = build_optimizers(model, tiny_hymo_config.optimizer)
        nm_lr = opts.nor_muon.defaults["lr"]
        aw_lr = opts.adamw.defaults["lr"]
        assert nm_lr / aw_lr == pytest.approx(66.67, abs=0.01)

    @pytest.mark.heavy
    def test_build_from_full_model(self) -> None:
        config = load_config("configs/hymo_750m.yaml")
        model = HyMo(config.model)
        opts = build_optimizers(model, config.optimizer)
        assert isinstance(opts, Optimizers)
        assert isinstance(opts.adamw, CautiousAdamW)
        assert opts.nor_muon is not None
        assert isinstance(opts.nor_muon, NorMuon)


class TestJointWSDScheduler:
    """Verify Joint WSD scheduler factor transitions across steps."""

    def test_construct(self) -> None:
        from hymo.core.config import SchedulerConfig

        cfg = SchedulerConfig()
        s = JointWSDScheduler(cfg)
        assert s.warmup_steps == cfg.warmup_steps
        assert s.stable_steps == cfg.stable_steps
        assert s.decay_steps == cfg.decay_steps
        assert s.min_lr_ratio == cfg.min_lr_ratio
        assert s.decay_kind == "linear"

    def test_get_factor_warmup(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        assert s.get_factor(0) == 0.0
        warmup = s.warmup_steps
        mid_warmup = warmup // 2
        f_mid = s.get_factor(mid_warmup)
        assert 0.0 < f_mid < 1.0
        assert s.get_factor(warmup) == pytest.approx(1.0, abs=1e-6)

    def test_get_factor_stable(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        stable_start = s.warmup_steps
        stable_end = stable_start + s.stable_steps
        assert s.get_factor(stable_start) == pytest.approx(1.0, abs=1e-6)
        mid_stable = (stable_start + stable_end) // 2
        assert s.get_factor(mid_stable) == pytest.approx(1.0, abs=1e-6)

    def test_get_factor_decay(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        decay_start = s.warmup_steps + s.stable_steps
        total = decay_start + s.decay_steps
        factor_end = s.get_factor(total + 100)
        assert factor_end == pytest.approx(s.min_lr_ratio, abs=1e-6)
        mid_decay = decay_start + s.decay_steps // 2
        f_mid = s.get_factor(mid_decay)
        assert s.min_lr_ratio < f_mid < 1.0

    def test_decay_factor_linear(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        assert s._decay_factor(0.0, "linear") == 1.0
        assert s._decay_factor(1.0, "linear") == 0.0
        assert s._decay_factor(0.5, "linear") == 0.5

    def test_decay_factor_cosine(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        assert s._decay_factor(0.0, "cosine") == 1.0
        assert s._decay_factor(1.0, "cosine") == pytest.approx(0.0, abs=1e-9)
        assert s._decay_factor(0.5, "cosine") == pytest.approx(0.5, abs=1e-9)

    def test_decay_factor_sqrt(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
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

    def test_state_dict_and_step(self) -> None:
        from hymo.core.config import SchedulerConfig

        s = JointWSDScheduler(SchedulerConfig())
        sd = s.state_dict()
        assert "step" in sd
        assert sd["step"] == 0
        s.step()
        sd2 = s.state_dict()
        assert sd2["step"] == 1
        s2 = JointWSDScheduler(SchedulerConfig())
        s2.load_state_dict(sd2)
        assert s2.state_dict()["step"] == 1
