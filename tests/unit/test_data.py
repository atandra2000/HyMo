"""Tests for the :mod:`hymo.data` module (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hymo.core.exceptions import (
    ConfigNotFoundError,
    ConfigValidationError,
    TokenizerError,
)
from hymo.data import (
    BYTE_VOCAB_SIZE,
    DataConfig,
    DataLoaderBuilder,
    DedupConfig,
    ExtendedTokenizer,
    QualityConfig,
    ShardDataset,
    ShardingConfig,
    ShardWriter,
    SourceSpec,
    TokenizationConfig,
    load_data_config,
    load_data_config_from_dict,
    save_data_config,
)
from hymo.registry import DATA_SOURCES, TOKENIZERS

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestSourceSpec:
    def test_construct(self) -> None:
        s = SourceSpec(id="fineweb_edu_q3", weight=0.5)
        assert s.id == "fineweb_edu_q3"
        assert s.weight == 0.5

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ConfigValidationError):
            SourceSpec(id="", weight=0.5)

    def test_weight_must_be_positive(self) -> None:
        with pytest.raises(ConfigValidationError):
            SourceSpec(id="x", weight=0)
        with pytest.raises(ConfigValidationError):
            SourceSpec(id="x", weight=-0.1)
        with pytest.raises(ConfigValidationError):
            SourceSpec(id="x", weight=1.5)


class TestDataConfig:
    def test_default_construct(self) -> None:
        with pytest.raises(ConfigValidationError):
            DataConfig()

    def test_single_source(self) -> None:
        c = DataConfig(sources=(SourceSpec(id="x", weight=1.0),))
        assert len(c.sources) == 1

    def test_source_ids(self) -> None:
        c = DataConfig(
            sources=(
                SourceSpec(id="a", weight=0.5),
                SourceSpec(id="b", weight=0.5),
            )
        )
        assert c.source_ids() == ("a", "b")

    def test_get_source(self) -> None:
        c = DataConfig(
            sources=(
                SourceSpec(id="a", weight=0.5),
                SourceSpec(id="b", weight=0.5),
            )
        )
        assert c.get_source("a").weight == 0.5
        with pytest.raises(KeyError):
            c.get_source("nope")

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ConfigValidationError):
            DataConfig(
                sources=(
                    SourceSpec(id="a", weight=0.5),
                    SourceSpec(id="b", weight=0.4),
                )
            )

    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ConfigValidationError):
            DataConfig(
                sources=(SourceSpec(id="a", weight=1.0),),
                train_fraction=0.5,
                val_fraction=0.3,
                test_fraction=0.0,
            )


class TestYamlRoundTrip:
    def test_load_default_yaml(self) -> None:
        c = load_data_config(FIXTURES / "tiny_mixture.yaml")
        assert len(c.sources) == 1
        assert c.sources[0].id == "fineweb_edu_q3"
        assert c.sharding.target_total_tokens == 1_000_000

    def test_load_production_yaml(self) -> None:
        c = load_data_config("configs/hymo_mixture.yaml")
        assert len(c.sources) == 10
        total = sum(s.weight for s in c.sources)
        assert total == pytest.approx(1.0)
        assert c.sharding.target_total_tokens == 30_000_000_000

    def test_save_and_reload(self, tmp_path: Path) -> None:
        c = load_data_config(FIXTURES / "tiny_mixture.yaml")
        out = tmp_path / "saved.yaml"
        save_data_config(c, out)
        reloaded = load_data_config(out)
        assert reloaded == c

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigNotFoundError):
            load_data_config(tmp_path / "does_not_exist.yaml")

    def test_load_from_dict(self) -> None:
        raw = {
            "sources": [{"id": "x", "weight": 1.0}],
            "sharding": {"target_total_tokens": 100},
        }
        c = load_data_config_from_dict(raw)
        assert c.sharding.target_total_tokens == 100


class TestShardingConfig:
    def test_defaults(self) -> None:
        s = ShardingConfig()
        assert s.shard_size_tokens == 50_000_000
        assert s.target_total_tokens == 30_000_000_000
        assert s.dtype == "uint32"
        assert s.cross_document_boundary_ok is False

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ConfigValidationError):
            ShardingConfig(dtype="float64")

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ConfigValidationError):
            ShardingConfig(shard_size_tokens=0)


class TestTokenizationConfig:
    def test_defaults(self) -> None:
        t = TokenizationConfig()
        assert t.vocab_size == 64_256
        assert t.eos_token_id == 0
        assert t.pad_token_id == 2
        assert t.byte_fallback is True

    def test_invalid_vocab_size_raises(self) -> None:
        with pytest.raises(ConfigValidationError):
            TokenizationConfig(vocab_size=0)


class TestDedupConfig:
    def test_defaults(self) -> None:
        d = DedupConfig()
        assert d.enabled is True
        assert d.method == "sha256"
        assert d.n_hash_buckets == 256

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ConfigValidationError):
            DedupConfig(method="md5")

    def test_invalid_error_rate_raises(self) -> None:
        with pytest.raises(ConfigValidationError):
            DedupConfig(bloom_error_rate=0)
        with pytest.raises(ConfigValidationError):
            DedupConfig(bloom_error_rate=1.5)


class TestQualityConfig:
    def test_defaults(self) -> None:
        q = QualityConfig()
        assert q.drop_empty is True
        assert q.min_unique_chars_ratio == 0.05


class TestExtendedTokenizer:
    def test_construct(self, tmp_path: Path) -> None:
        t = ExtendedTokenizer(tmp_path / "tok.json")
        assert t.path == tmp_path / "tok.json"
        assert t.vocab_size == 64_000 + BYTE_VOCAB_SIZE
        assert t.eos_token_id == 0
        assert t.pad_token_id == 2

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        t = ExtendedTokenizer(tmp_path / "does_not_exist.json")
        with pytest.raises(TokenizerError):
            t.load()

    def test_registered(self) -> None:
        assert TOKENIZERS.has("hymo-bpe-64k")


class TestShardWriter:
    def test_construct(self, tmp_path: Path) -> None:
        w = ShardWriter(output_dir=tmp_path, shard_size_tokens=1024)
        assert w.output_dir == tmp_path
        assert w.shard_size_tokens == 1024

    def test_write_and_read_shard(self, tmp_path: Path) -> None:
        w = ShardWriter(output_dir=tmp_path, shard_size_tokens=100)
        tokens = np.arange(100, dtype=np.uint32)
        path = w.write_shard(0, tokens)
        assert path.exists()
        assert path.name == "shard_00000.bin"
        loaded = np.fromfile(path, dtype=np.uint32)
        np.testing.assert_array_equal(loaded, tokens)

    def test_write_batched(self, tmp_path: Path) -> None:
        w = ShardWriter(output_dir=tmp_path, shard_size_tokens=50)
        tokens = np.arange(120, dtype=np.uint32)
        paths = w.write_batched(tokens)
        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        total = sum(np.fromfile(p, dtype=np.uint32).size for p in paths)
        assert total == 150  # 120 + 30 pad


class TestShardDataset:
    def test_empty_dir(self, tmp_path: Path) -> None:
        d = ShardDataset(tmp_path, max_seq_len=64)
        assert len(d) == 0

    def test_with_shards(self, tmp_path: Path) -> None:
        w = ShardWriter(output_dir=tmp_path, shard_size_tokens=200)
        w.write_shard(0, np.arange(200, dtype=np.uint32))
        d = ShardDataset(tmp_path, max_seq_len=8)
        assert len(d) > 0
        tokens, targets = d[0]
        assert tokens.shape == (8,)
        assert targets.shape == (8,)
        # tokens[1:] should equal targets[:-1]
        assert torch.equal(tokens[1:], targets[:-1])


class TestDataLoaderBuilder:
    def test_construct(self, tmp_path: Path) -> None:
        w = ShardWriter(output_dir=tmp_path, shard_size_tokens=200)
        w.write_shard(0, np.arange(200, dtype=np.uint32))
        d = ShardDataset(tmp_path, max_seq_len=8)
        from hymo.core.config import TrainingConfig

        b = DataLoaderBuilder(d, TrainingConfig())
        loader = b.build()
        batch = next(iter(loader))
        assert len(batch) == 2  # tokens, targets
        assert batch[0].shape[0] == TrainingConfig().micro_batch_size


class TestDataSourcesRegistered:
    def test_10_sources(self) -> None:
        ids = {
            "fineweb_edu_q3",
            "fineweb",
            "stack_python",
            "stack_java",
            "stack_cpp",
            "slimpajama",
            "dclm_baseline",
            "dolma_wiki",
            "dolma_books",
            "cosmopedia",
        }
        for sid in ids:
            assert DATA_SOURCES.has(sid), f"Source {sid!r} not registered"
        assert len(DATA_SOURCES) >= 10
