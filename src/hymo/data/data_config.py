"""Data-pipeline config (separate from the model config).

The data pipeline is a distinct concern from the model config: the
data prep can be parallelized (one pod streaming + tokenizing) while
the model + training work happens on another. The two configs are
loaded independently.

The :class:`DataConfig` is the v1.0 spec (architecture doc §6,
roadmap A1, A5):

- 10 sources per the v1.0 quality-first mixture.
- 30B total training tokens.
- 50M-token shards, ``uint32``, ~120 GB total.
- BPE-64k + 256 byte-level fallback vocab (64,256 total).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hymo.core.exceptions import (
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
)
from hymo.core.types import Path as PathType
from hymo.utils.atomic_io import atomic_write_with

__all__ = [
    "DataConfig",
    "SourceSpec",
    "ShardingConfig",
    "TokenizationConfig",
    "DedupConfig",
    "QualityConfig",
    "load_data_config",
    "save_data_config",
]


@dataclass(frozen=True)
class SourceSpec:
    """A single data source (corpus) in the mixture.

    Attributes
    ----------
    id : str
        Source identifier (e.g. ``"fineweb_edu_q3"``).
    weight : float
        Sampling weight; the sum of weights across all sources must
        equal 1.0.
    """

    id: str
    weight: float

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigValidationError("SourceSpec.id must be non-empty")
        if not 0.0 < self.weight <= 1.0:
            raise ConfigValidationError(
                f"SourceSpec.weight must be in (0, 1], got {self.weight}"
            )


@dataclass(frozen=True)
class ShardingConfig:
    """Sharding parameters for the data pipeline."""

    shard_size_tokens: int = 50_000_000
    dtype: str = "uint32"
    target_total_tokens: int = 30_000_000_000
    cross_document_boundary_ok: bool = False
    verify_after_pack: bool = True
    output_dir: str = "data/shards"

    def __post_init__(self) -> None:
        if self.shard_size_tokens <= 0:
            raise ConfigValidationError("shard_size_tokens must be > 0")
        if self.target_total_tokens <= 0:
            raise ConfigValidationError("target_total_tokens must be > 0")
        if self.dtype not in ("uint32", "uint16", "uint8", "int32"):
            raise ConfigValidationError(
                f"dtype must be 'uint32', 'uint16', 'uint8', or 'int32', "
                f"got {self.dtype!r}"
            )


@dataclass(frozen=True)
class TokenizationConfig:
    """Tokenizer configuration."""

    name: str = "hymo-bpe-64k"
    path: str = "data/tokens/byte_bpe_vocab.json"
    vocab_size: int = 64_256
    eos_token_id: int = 0
    pad_token_id: int = 2
    add_eos: bool = True
    byte_fallback: bool = True
    batch_size: int = 1024
    add_special_tokens: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ConfigValidationError("vocab_size must be > 0")


@dataclass(frozen=True)
class DedupConfig:
    """Deduplication parameters."""

    enabled: bool = True
    method: str = "sha256"
    n_hash_buckets: int = 256
    bloom_capacity_per_bucket: int = 200_000
    bloom_error_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.method not in ("sha256", "bloom", "exact"):
            raise ConfigValidationError(
                f"method must be 'sha256', 'bloom', or 'exact', got {self.method!r}"
            )
        if not 0.0 < self.bloom_error_rate < 1.0:
            raise ConfigValidationError("bloom_error_rate must be in (0, 1)")


@dataclass(frozen=True)
class QualityConfig:
    """Document-level quality filters."""

    drop_empty: bool = True
    min_unique_chars_ratio: float = 0.05
    max_digit_ratio: float = 0.5
    max_punct_ratio: float = 0.5
    max_whitespace_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_unique_chars_ratio <= 1.0:
            raise ConfigValidationError("min_unique_chars_ratio must be in [0, 1]")
        for name, v in (
            ("max_digit_ratio", self.max_digit_ratio),
            ("max_punct_ratio", self.max_punct_ratio),
            ("max_whitespace_ratio", self.max_whitespace_ratio),
        ):
            if not 0.0 <= v <= 1.0:
                raise ConfigValidationError(f"{name} must be in [0, 1], got {v}")


@dataclass(frozen=True)
class DataConfig:
    """Top-level data-pipeline config."""

    sources: tuple[SourceSpec, ...] = field(default_factory=tuple)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    train_fraction: float = 0.97
    val_fraction: float = 0.015
    test_fraction: float = 0.015
    seed: int = 42
    streaming_download: bool = True

    def __post_init__(self) -> None:
        if not self.sources:
            raise ConfigValidationError("At least one source is required")
        total = sum(s.weight for s in self.sources)
        if abs(total - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"Sum of source weights must equal 1.0, got {total}"
            )
        if abs(self.train_fraction + self.val_fraction + self.test_fraction - 1.0) > 1e-6:
            raise ConfigValidationError(
                "train + val + test fractions must equal 1.0"
            )

    # ---- Convenience ---------------------------------------------------

    def source_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.sources)

    def get_source(self, source_id: str) -> SourceSpec:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(f"Source {source_id!r} not in mixture")


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------


def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to a plain dict for YAML dumping."""
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import fields

        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _build_data_config(raw: dict[str, Any]) -> DataConfig:
    """Build a :class:`DataConfig` from a raw dict."""
    sources_raw = raw.get("sources", [])
    if not sources_raw:
        raise ConfigValidationError("At least one source is required")
    sources = tuple(
        SourceSpec(id=s["id"], weight=float(s["weight"])) for s in sources_raw
    )
    sharding = ShardingConfig(**raw.get("sharding", {}))
    tokenization = TokenizationConfig(**raw.get("tokenization", {}))
    dedup = DedupConfig(**raw.get("dedup", {}))
    quality = QualityConfig(**raw.get("quality", {}))
    return DataConfig(
        sources=sources,
        sharding=sharding,
        tokenization=tokenization,
        dedup=dedup,
        quality=quality,
        train_fraction=raw.get("train_fraction", 0.97),
        val_fraction=raw.get("val_fraction", 0.015),
        test_fraction=raw.get("test_fraction", 0.015),
        seed=raw.get("seed", 42),
        streaming_download=raw.get("streaming_download", True),
    )


def load_data_config(path: str | PathType) -> DataConfig:
    """Load a :class:`DataConfig` from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigNotFoundError(f"Data config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML at {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    return _build_data_config(raw)


def load_data_config_from_dict(raw: dict[str, Any]) -> DataConfig:
    """Build a :class:`DataConfig` from a plain dict."""
    return _build_data_config(raw)


def save_data_config(config: DataConfig, path: str | PathType) -> None:
    """Dump a :class:`DataConfig` to YAML atomically."""

    def writer(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_to_dict(config), f, sort_keys=False, default_flow_style=False)

    atomic_write_with(path, writer)
