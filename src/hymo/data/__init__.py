"""Data-facing public API for tokenization and validation-set preparation.

The tokenizer is importable without loading the streaming dataset; the builder
is kept in its CLI module because it performs network and filesystem I/O.
"""
from __future__ import annotations

from hymo.data.tokenizer import BYTE_VOCAB_SIZE, ExtendedTokenizer

__all__ = [
    "BYTE_VOCAB_SIZE",
    "ExtendedTokenizer",
]
