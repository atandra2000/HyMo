"""Doc↔code alignment checker for HyMo.

HyMo's doc corpus (docs/ + learning_docs/) uses two citation styles:

1. Symbol anchors — ``src/hymo/models/mla.py:MLABlock.forward`` — resolved
   by importing the module and walking the dotted symbol.
2. Line anchors — ``src/hymo/models/mla.py:21`` / ``:72-88`` — verified
   against the working tree (file exists, line range within EOF).

Both styles are checked; a line anchor that points past EOF or at a
missing file fails the gate. Intra-repo markdown links are validated too.

Usage:
    python3 tests/test_doc_refs.py             # resolve all anchors + links
    python3 -m pytest tests/test_doc_refs.py   # same checks in pytest
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DOC_PATHS = [
    ROOT / "docs",
    ROOT / "learning_docs",
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "SKILLS.md",
]
# The design spec embeds literal `file.py:Symbol` metasyntax examples; skip it.
SKIP_DOCS = {ROOT / "docs" / "superpowers" / "specs" / "2026-08-04-docs-expansion-design.md"}

# src/hymo/...py:Symbol  or  src/hymo/...py:123  or  src/hymo/...py:72-88
SYMBOL_ANCHOR_RE = re.compile(r"(src/hymo/[A-Za-z0-9_./-]+\.py):([A-Za-z_][A-Za-z0-9_.]*)")
LINE_ANCHOR_RE = re.compile(r"(src/hymo/[A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for p in DOC_PATHS:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.exists():
            files.append(p)
    return [f for f in files if f not in SKIP_DOCS]


_MODULE_CACHE: dict[str, object | None] = {}


def _load_module(rel_path: str):
    """Import a src/hymo module by repo-relative path; returns (module, error)."""
    if rel_path in _MODULE_CACHE:
        mod = _MODULE_CACHE[rel_path]
        return (mod, None) if mod is not None else (None, f"previous import failure: {rel_path}")
    path = ROOT / rel_path
    if not path.exists():
        _MODULE_CACHE[rel_path] = None
        return None, f"unknown file: {rel_path}"
    dotted = ".".join(Path(rel_path).with_suffix("").parts)
    try:
        import importlib

        mod = importlib.import_module(dotted)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        _MODULE_CACHE[rel_path] = None
        return None, f"import failed for {rel_path}: {type(exc).__name__}: {exc}"
    _MODULE_CACHE[rel_path] = mod
    return mod, None


def _has_instance_attr(cls, name: str) -> bool:
    """True if `name` is assigned as `self.name = …` in the class source
    (instance attributes are not visible via hasattr on the class)."""
    try:
        import inspect

        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    return re.search(rf"self\.{re.escape(name)}\s*=", src) is not None


def _resolve_symbol(rel_path: str, symbol: str) -> tuple[bool, str | None]:
    mod, err = _load_module(rel_path)
    if err:
        return False, err
    obj = mod
    for part in symbol.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        if isinstance(obj, type) and _has_instance_attr(obj, part):
            continue
        return False, f"{rel_path}:{symbol} — '{part}' not found"
    return True, None


def _resolve_line(rel_path: str, start: int, end: int) -> tuple[bool, str | None]:
    path = ROOT / rel_path
    if not path.exists():
        return False, f"unknown file: {rel_path}"
    nlines = len(path.read_text(encoding="utf-8").splitlines())
    if start < 1 or end > nlines:
        return False, f"{rel_path}:{start}-{end} beyond EOF ({nlines} lines)"
    return True, None


def check_anchors() -> list[str]:
    failures: list[str] = []
    for doc in _doc_files():
        text = doc.read_text(encoding="utf-8")
        rel = doc.relative_to(ROOT)
        for m in SYMBOL_ANCHOR_RE.finditer(text):
            ok, err = _resolve_symbol(m.group(1), m.group(2))
            if not ok:
                failures.append(f"{rel}: {m.group(1)}:{m.group(2)} — {err}")
        for m in LINE_ANCHOR_RE.finditer(text):
            ok, err = _resolve_line(m.group(1), int(m.group(2)), int(m.group(3) or m.group(2)))
            if not ok:
                failures.append(f"{rel}: {err}")
    return failures


def check_links() -> list[str]:
    """Validate intra-repo markdown links; code fences are stripped so
    kernel source is not scanned. Links may be doc-relative or root-relative."""
    broken: list[str] = []
    for doc in _doc_files():
        text = FENCE_RE.sub("", doc.read_text(encoding="utf-8"))
        rel = doc.relative_to(ROOT)
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            candidates = [(doc.parent / path_part).resolve(), (ROOT / path_part).resolve()]
            if not any(c.exists() for c in candidates):
                broken.append(f"{rel}: broken link -> {target}")
    return broken


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Doc↔code alignment checker")
    ap.add_argument("--links", action="store_true", help="also validate intra-repo markdown links")
    args = ap.parse_args()

    failures = check_anchors()
    print(f"[doc-refs] scanned {len(_doc_files())} docs")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"[doc-refs] anchors: {'PASS' if not failures else f'{len(failures)} FAILURES'}")

    broken = check_links() if args.links else []
    if args.links:
        for b in broken:
            print(f"  BROKEN-LINK {b}")
        print(f"[doc-refs] links: {'PASS' if not broken else f'{len(broken)} BROKEN'}")

    return 1 if (failures or (args.links and broken)) else 0


if __name__ == "__main__":
    sys.exit(main())


def test_doc_refs_all_anchors_resolve():
    failures = check_anchors()
    assert not failures, "\n".join(failures)


def test_doc_refs_links_resolve():
    broken = check_links()
    assert not broken, "\n".join(broken)
