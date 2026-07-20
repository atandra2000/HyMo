"""Per-source streaming loaders (Phase 4 implementation).

Each loader streams HuggingFace dataset rows, applying per-source quality filters
and field normalization.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


__all__ = [
    "load_fineweb_edu",
    "load_fineweb",
    "load_stack_python",
    "load_stack_java",
    "load_stack_cpp",
    "load_slimpajama",
    "load_dclm_baseline",
    "load_dolma_wiki",
    "load_dolma_books",
    "load_cosmopedia",
]


def load_fineweb_edu(
    *, quality_threshold: int = 3, **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Stream FineWeb-Edu rows with score >= quality_threshold."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if quality_threshold > 0:
        ds = ds.filter(lambda row: (row.get("score") or 0) >= quality_threshold)
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds


def load_fineweb(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream non-edu FineWeb rows."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds


def load_stack_python(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Stack v2 Python rows."""
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        data_dir="data/python",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["content"]}, remove_columns=[c for c in (ds.column_names or []) if c != "content"])
    yield from ds


def load_stack_java(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Stack v2 Java rows."""
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        data_dir="data/java",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["content"]}, remove_columns=[c for c in (ds.column_names or []) if c != "content"])
    yield from ds


def load_stack_cpp(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Stack v2 C++ rows."""
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        data_dir="data/cpp",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["content"]}, remove_columns=[c for c in (ds.column_names or []) if c != "content"])
    yield from ds


def load_slimpajama(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream SlimPajama rows."""
    ds = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds


def load_dclm_baseline(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream DataComp for Language Models baseline rows."""
    ds = load_dataset(
        "mlfoundations/dclm-baseline-1.0",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds


def load_dolma_wiki(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Wikipedia (Dolma subset) rows."""
    ds = load_dataset(
        "allenai/dolma",
        data_dir="data/wikipedia",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds


def load_dolma_books(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Books (Dolma) rows."""
    ds = load_dataset(
        "allenai/dolma",
        data_dir="data/books",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds


def load_cosmopedia(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Cosmopedia synthetic textbook rows."""
    ds = load_dataset(
        "HuggingFaceTB/cosmopedia",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.column_names or []) if c != "text"])
    yield from ds
