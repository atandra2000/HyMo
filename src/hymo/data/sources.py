"""Per-source streaming loaders (Phase 1 stubs).

The real implementation (architecture doc §6, roadmap A2, A3) provides
10 source loaders, each a streaming generator over a HuggingFace
dataset:

- ``fineweb_edu_q3`` — FineWeb-Edu (10BT or 100BT sample), filtered
  on ``score >= 3``.
- ``fineweb`` — non-edu FineWeb-Edu.
- ``stack_python``, ``stack_java``, ``stack_cpp`` — Stack v2.
- ``slimpajama`` — SlimPajama (RedPajama-style diversity).
- ``dclm_baseline`` — DataComp for Language Models baseline.
- ``dolma_wiki`` — Wikipedia (Dolma, multilingual subset).
- ``dolma_books`` — Books (Dolma).
- ``cosmopedia`` — HuggingFaceTB/cosmopedia (synthetic textbook-style).

Phase 1 defines the loader *function signatures* and registers them
with :data:`hymo.registry.DATA_SOURCES`. The bodies raise
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from hymo.core.exceptions import NotImplementedError_
from hymo.registry import DATA_SOURCES

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


@DATA_SOURCES.register("fineweb_edu_q3")
def load_fineweb_edu(
    *, quality_threshold: int = 3, **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Stream FineWeb-Edu rows with ``score >= quality_threshold``."""
    raise NotImplementedError_(
        "load_fineweb_edu is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A2)."
    )


@DATA_SOURCES.register("fineweb")
def load_fineweb(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream non-edu FineWeb rows."""
    raise NotImplementedError_(
        "load_fineweb is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A2)."
    )


@DATA_SOURCES.register("stack_python")
def load_stack_python(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Stack v2 Python rows (deduplicated)."""
    raise NotImplementedError_(
        "load_stack_python is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("stack_java")
def load_stack_java(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Stack v2 Java rows (deduplicated)."""
    raise NotImplementedError_(
        "load_stack_java is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("stack_cpp")
def load_stack_cpp(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Stack v2 C++ rows (deduplicated)."""
    raise NotImplementedError_(
        "load_stack_cpp is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("slimpajama")
def load_slimpajama(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream SlimPajama (RedPajama-style diversity) rows."""
    raise NotImplementedError_(
        "load_slimpajama is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("dclm_baseline")
def load_dclm_baseline(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream DataComp for Language Models baseline rows."""
    raise NotImplementedError_(
        "load_dclm_baseline is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("dolma_wiki")
def load_dolma_wiki(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Wikipedia (Dolma, multilingual subset) rows."""
    raise NotImplementedError_(
        "load_dolma_wiki is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("dolma_books")
def load_dolma_books(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Books (Dolma) rows."""
    raise NotImplementedError_(
        "load_dolma_books is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )


@DATA_SOURCES.register("cosmopedia")
def load_cosmopedia(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Stream Cosmopedia (HuggingFaceTB) rows."""
    raise NotImplementedError_(
        "load_cosmopedia is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §6, roadmap A3)."
    )
