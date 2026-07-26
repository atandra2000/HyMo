"""Configuration dataclasses loaded from a single YAML file.

Contains ModelConfig, OptimizerConfig, SchedulerConfig, TrainingConfig, RunConfig,
and the top-level HyMoConfig aggregating all sub-configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hymo.core.types import Step

# ----------------------------------------------------------------------
# Sub-configs
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Architectural hyperparameters for the HyMo stack (architecture doc §2.1)."""

    # Token + sequence
    vocab_size: int = 64_256
    max_seq_len: int = 4_096

    # Stack
    n_layers: int = 32
    dim: int = 896
    tie_embeddings: bool = True

    # MLA (full attention)
    n_heads: int = 16
    n_kv_groups: int = 4
    q_lora_rank: int = 224
    kv_lora_rank: int = 128
    head_dim: int = 128
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 96
    v_head_dim: int = 128
    rope_theta: float = 10_000.0

    # GDN (linear attention)
    gdn_d_state: int = 32
    gdn_d_conv: int = 4
    gdn_headdim: int = 32
    gdn_d_inner: int = 1_280
    gdn_chunk_size: int = 64

    # NoPE-hybrid
    nope_hybrid_gdn_enabled: bool = False

    # MoE
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_inter_dim: int = 2_304
    moe_ema_alpha: float = 0.02
    moe_capacity_factor: float = 1.5

    # DenseFFN on GDN blocks
    inter_dim: int = 2_560

    # MTP
    mtp_depth: int = 2
    mtp_loss_weights: tuple[float, ...] = (0.3, 0.1)
    mtp_inter_dim: int = 2_304

    # Logit softcap
    logit_softcap: float = 15.0

    # Init
    mup_init: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.n_kv_groups <= 0 or self.n_heads % self.n_kv_groups != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of n_kv_groups "
                f"({self.n_kv_groups})"
            )
        if self.qk_rope_head_dim + self.qk_nope_head_dim != self.head_dim:
            raise ValueError(
                f"qk_rope_head_dim ({self.qk_rope_head_dim}) + "
                f"qk_nope_head_dim ({self.qk_nope_head_dim}) must equal "
                f"head_dim ({self.head_dim})"
            )
        if self.gdn_d_inner % self.gdn_headdim != 0:
            raise ValueError(
                f"gdn_d_inner ({self.gdn_d_inner}) must be a multiple of "
                f"gdn_headdim ({self.gdn_headdim})"
            )
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError(
                f"n_activated_experts ({self.n_activated_experts}) cannot "
                f"exceed n_routed_experts ({self.n_routed_experts})"
            )
        if self.mtp_depth < 0:
            raise ValueError("mtp_depth must be >= 0")
        if self.mtp_depth == 0:
            if len(self.mtp_loss_weights) != 0:
                raise ValueError(
                    f"mtp_loss_weights must be empty when mtp_depth=0, "
                    f"got {len(self.mtp_loss_weights)} weights"
                )
        elif len(self.mtp_loss_weights) != self.mtp_depth:
            raise ValueError(
                f"mtp_loss_weights length ({len(self.mtp_loss_weights)}) "
                f"must equal mtp_depth ({self.mtp_depth})"
            )
        if any(w < 0 for w in self.mtp_loss_weights):
            raise ValueError("mtp_loss_weights must be non-negative")
        if self.logit_softcap < 0:
            raise ValueError("logit_softcap must be >= 0 (0 disables)")

    @property
    def n_mla_layers(self) -> int:
        """Number of MLA (full attention) layers."""
        return self.n_layers // 4

    @property
    def n_gdn_layers(self) -> int:
        """Number of GDN (linear attention) layers."""
        return self.n_layers - self.n_mla_layers

    @property
    def mla_positions(self) -> frozenset[int]:
        """Indices of MLA layers in the stack."""
        return frozenset(i * 4 for i in range(self.n_mla_layers))

    @property
    def gdn_positions(self) -> frozenset[int]:
        """Indices of GDN layers in the stack."""
        return frozenset(i for i in range(self.n_layers) if i not in self.mla_positions)

    @property
    def nope_hybrid_gdn_positions(self) -> frozenset[int]:
        """Indices of GDN layers that get NoPE when the hybrid model is enabled."""
        if not self.nope_hybrid_gdn_enabled:
            return frozenset()
        return frozenset(mla - 1 for mla in self.mla_positions if mla > 0)


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters for Muon and AdamW (architecture doc §5.2)."""

    # NorMuon (attention + GDN matrices, 2D dense)
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_betas: tuple[float, float] = (0.95, 0.95)
    muon_eps: float = 1e-8
    muon_weight_decay: float = 0.1

    # AdamW (embed/head/norm/gate/scalars + MoE experts)
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0
    adamw_embed_weight_decay: float = 0.1

    # Master weight precision
    master_weights_dtype: str = "float32"

    # Cautious weight decay (2D weights only)
    cautious_wd: bool = True

    def __post_init__(self) -> None:
        if self.muon_lr <= 0 or self.adamw_lr <= 0:
            raise ValueError("optimizer LRs must be > 0")
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError("muon_momentum must be in [0, 1)")
        for name, betas in (("muon_betas", self.muon_betas),
                            ("adamw_betas", self.adamw_betas)):
            if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
                raise ValueError(f"{name} must each be in [0, 1)")
        if self.master_weights_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"master_weights_dtype must be 'float32' or 'bfloat16', "
                f"got {self.master_weights_dtype!r}"
            )


@dataclass(frozen=True)
class SchedulerConfig:
    """Joint WSD (warmup-stable-decay) learning rate scheduler config."""

    total_steps: Step = Step(57_220)
    warmup_frac: float = 0.02
    stable_frac: float = 0.83
    decay_frac: float = 0.15
    min_lr_ratio: float = 0.05
    decay: str = "linear"

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if not 0.0 < self.warmup_frac < 1.0:
            raise ValueError("warmup_frac must be in (0, 1)")
        if not 0.0 < self.stable_frac < 1.0:
            raise ValueError("stable_frac must be in (0, 1)")
        if not 0.0 < self.decay_frac < 1.0:
            raise ValueError("decay_frac must be in (0, 1)")
        if abs(self.warmup_frac + self.stable_frac + self.decay_frac - 1.0) > 1e-6:
            raise ValueError(
                f"warmup_frac + stable_frac + decay_frac must equal 1.0, "
                f"got {self.warmup_frac + self.stable_frac + self.decay_frac}"
            )
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1)")
        if self.decay not in ("linear", "cosine", "sqrt"):
            raise ValueError(
                f"decay must be 'linear', 'cosine', or 'sqrt', got {self.decay!r}"
            )

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_frac)

    @property
    def stable_steps(self) -> int:
        return int(self.total_steps * self.stable_frac)

    @property
    def decay_steps(self) -> int:
        return int(self.total_steps * self.decay_frac)


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop hyperparameters and distributed settings."""

    # Batch
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4_096

    # Distributed
    world_size: int = 4
    fsdp: bool = True
    fsdp_mixed_precision: str = "bfloat16"

    # Gradient handling
    grad_clip: float = 1.0
    grad_norm_threshold: float = 10.0
    loss_nan_skip: bool = True
    consecutive_nan_limit: int = 5
    empty_cache_every: int = 100

    # Checkpoint
    save_dir: str = "checkpoints/pretrain"
    save_interval: int = 4_000
    log_interval: int = 50
    eval_interval: int = 2_000
    max_keep: int = 2

    # Optimizations
    fused_gdn: bool = True
    moe_mixed_precision: bool = True
    torch_compile_gdn: bool = True
    cuda_graphs_mla: bool = True

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        if self.save_interval <= 0 or self.eval_interval <= 0:
            raise ValueError("save/eval intervals must be > 0")
        if self.consecutive_nan_limit <= 0:
            raise ValueError("consecutive_nan_limit must be > 0")
        if self.fsdp_mixed_precision not in ("bfloat16", "float32", "float16"):
            raise ValueError(
                f"fsdp_mixed_precision must be 'bfloat16', 'float32', or "
                f"'float16', got {self.fsdp_mixed_precision!r}"
            )

    @property
    def per_step_tokens(self) -> int:
        """Total tokens per optimizer step across all ranks."""
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
            * self.max_seq_len
        )


@dataclass(frozen=True)
class RunConfig:
    """Run configuration, seeds, directory names, and reproducibility flags."""

    name: str = "hymo-v1.0"
    seed: int = 42
    output_dir: str = "checkpoints/pretrain"
    log_dir: str = "logs"
    eval_dir: str = "checkpoints/pretrain/eval"
    distributed: bool = True
    deterministic: bool = True
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


@dataclass(frozen=True)
class HyMoConfig:
    """Top-level aggregated HyMo configuration loaded from YAML."""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------


def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to a plain dict for YAML dumping."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def load_config(path: str | Path) -> HyMoConfig:
    """Load a HyMoConfig from a YAML file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML at {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(
            f"Top-level YAML must be a mapping, got {type(raw).__name__}"
        )

    return _build_config(raw)


def load_config_from_dict(raw: dict[str, Any]) -> HyMoConfig:
    """Build a HyMoConfig from a plain dict (useful for tests)."""
    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> HyMoConfig:
    """Construct sub-configs from a raw dict with type coercion."""
    try:
        model = ModelConfig(**_filter(raw.get("model", {}), ModelConfig))
        optimizer = OptimizerConfig(
            **_filter(raw.get("optimizer", {}), OptimizerConfig)
        )
        scheduler = SchedulerConfig(
            **_filter(raw.get("scheduler", {}), SchedulerConfig)
        )
        training = TrainingConfig(
            **_filter(raw.get("training", {}), TrainingConfig)
        )
        run = RunConfig(**_filter(raw.get("run", {}), RunConfig))
    except TypeError as e:
        raise ValueError(f"Unknown / wrong-type config field: {e}") from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to build config: {e}") from e

    return HyMoConfig(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        run=run,
    )


def _filter(raw: dict[str, Any], cls: type) -> dict[str, Any]:
    """Filters dictionary keys to fields of dataclass cls and coerces types."""
    valid = {f.name for f in fields(cls)}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in valid:
            continue
        f = next(f for f in fields(cls) if f.name == k)
        if f.type is tuple or (
            hasattr(f.type, "__origin__") and f.type.__origin__ is tuple
        ):
            v = tuple(v) if isinstance(v, (list, tuple)) else (v,)
        out[k] = v
    return out


def save_config(config: HyMoConfig, path: str | Path) -> None:
    """Dump a HyMoConfig to a YAML file path atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(config), f, sort_keys=False, default_flow_style=False)


def derive_config(
    base: HyMoConfig,
    *,
    model: ModelConfig | None = None,
    optimizer: OptimizerConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    training: TrainingConfig | None = None,
    run: RunConfig | None = None,
) -> HyMoConfig:
    """Derive a new HyMoConfig by replacing selected sub-config properties."""
    return replace(
        base,
        model=model if model is not None else base.model,
        optimizer=optimizer if optimizer is not None else base.optimizer,
        scheduler=scheduler if scheduler is not None else base.scheduler,
        training=training if training is not None else base.training,
        run=run if run is not None else base.run,
    )


# ----------------------------------------------------------------------
# Re-exports
# ----------------------------------------------------------------------

__all__ = [
    "HyMoConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "RunConfig",
    "derive_config",
    "load_config",
    "load_config_from_dict",
    "save_config",
]
