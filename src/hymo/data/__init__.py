"""Public API of :mod:`hymo.data`."""

from __future__ import annotations

# Side-effect import: registers the 10 data sources with DATA_SOURCES.
import hymo.data.sources  # noqa: F401
from hymo.data.data_config import (
    DataConfig,
    DedupConfig,
    QualityConfig,
    ShardingConfig,
    SourceSpec,
    TokenizationConfig,
    load_data_config,
    load_data_config_from_dict,
)
from hymo.data.sharding import DataLoaderBuilder, ShardDataset, ShardWriter
from hymo.data.sources import (
    load_cosmopedia,
    load_dclm_baseline,
    load_dolma_books,
    load_dolma_wiki,
    load_fineweb,
    load_fineweb_edu,
    load_slimpajama,
    load_stack_cpp,
    load_stack_java,
    load_stack_python,
)
from hymo.data.tokenizer import BYTE_VOCAB_SIZE, ExtendedTokenizer

__all__ = [
    # Config
    "DataConfig",
    "DedupConfig",
    "QualityConfig",
    "ShardingConfig",
    "SourceSpec",
    "TokenizationConfig",
    "load_data_config",
    "load_data_config_from_dict",
    # Sharding
    "DataLoaderBuilder",
    "ShardDataset",
    "ShardWriter",
    # Sources (10)
    "load_cosmopedia",
    "load_dclm_baseline",
    "load_dolma_books",
    "load_dolma_wiki",
    "load_fineweb",
    "load_fineweb_edu",
    "load_slimpajama",
    "load_stack_cpp",
    "load_stack_java",
    "load_stack_python",
    # Tokenizer
    "BYTE_VOCAB_SIZE",
    "ExtendedTokenizer",
]
