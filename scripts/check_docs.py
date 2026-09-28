#!/usr/bin/env python3
"""Doc↔code alignment checker for HyMo.

Port of the portfolio gate (DiffusionGemma-Lite's `scripts/check_docs.py`)
to HyMo's citation dialect. HyMo docs cite symbols as
`src/hymo/models/mla.py:MLABlock.forward` and — unlike the portfolio
template — may also cite line ranges (`src/hymo/models/mla.py:72-88`);
both styles are verified here with the same semantics as
`tests/test_doc_refs.py` (line anchors are resolved against EOF, not
banned), and this script adds the two capabilities the pytest variant
lacks: `--coverage` (every public symbol in the shipped modules must be
cited at least once) and printed `[doc-refs]` PASS/FAIL lines.

Usage:
    python3 scripts/check_docs.py                # resolve all anchors + links off
    python3 scripts/check_docs.py --coverage --links
    python3 tests/test_doc_refs.py               # anchor+link gate (no coverage)
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DOC_PATHS = [
    ROOT / "docs",
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "SKILLS.md",
]
COVERAGE_MODULES = [
    "src/hymo/core/config.py",
    "src/hymo/core/config_validation.py",
    "src/hymo/core/types.py",
    "src/hymo/data/prepare_validation.py",
    "src/hymo/data/tokenizer.py",
    "src/hymo/models/gdn.py",
    "src/hymo/models/gdn_triton.py",
    "src/hymo/models/mla.py",
    "src/hymo/models/model.py",
    "src/hymo/models/moe.py",
    "src/hymo/models/mtp.py",
    "src/hymo/models/rope.py",
    "src/hymo/training/checkpoint.py",
    "src/hymo/training/fsdp.py",
    "src/hymo/training/optimizer.py",
    "src/hymo/training/partition.py",
    "src/hymo/training/scheduler.py",
    "src/hymo/training/trainer.py",
    "src/hymo/training/validation.py",
]
# HyMo anchors always carry the src/hymo/ prefix (the import root is src/).
ANCHOR_RE = re.compile(r"(src/hymo/[A-Za-z0-9_./-]+\.py):([A-Za-z_][A-Za-z0-9_.]*)")
LINE_ANCHOR_RE = re.compile(r"(src/hymo/[A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Heading-anchor machinery: link targets may carry a `#fragment`, which must
# match a heading id in the target file. GitHub derives ids as: drop inline
# markup, lowercase, keep word chars and hyphens, spaces -> hyphens, and give
# a repeated heading the `-1`, `-2` suffix.
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^(?:=+|-{2,})\s*$")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SLUG_STRIP_RE = re.compile(r"[^\w\s-]")

# JIT kernels/classes defined under `if HAS_TRITON:` in gdn_triton.py — never
# resolvable on a triton-less box (macOS/CI). Writers must cite the
# always-defined host wrapper `triton_gated_delta_rule` for the kernel itself.
JIT_SYMBOLS = {
    "gdn_fwd_kernel",
    "gdn_bwd_kernel",
    "TritonGDNFunction",
}

_MODULE_CACHE: dict[str, object | None] = {}


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for p in DOC_PATHS:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.exists():
            files.append(p)
    return files


def collect_anchors() -> list[tuple[Path, str, str]]:
    """Return (doc_path, src/hymo file.py, symbol) triples from all docs."""
    anchors: list[tuple[Path, str, str]] = []
    for doc in _doc_files():
        for m in ANCHOR_RE.finditer(doc.read_text(encoding="utf-8")):
            anchors.append((doc, m.group(1), m.group(2)))
    return anchors


def load_module(rel_path: str):
    """Import a src/hymo module by repo-relative path; returns (module, error)."""
    if rel_path in _MODULE_CACHE:
        mod = _MODULE_CACHE[rel_path]
        return (mod, None) if mod is not None else (None, f"previous import failure: {rel_path}")
    path = ROOT / rel_path
    if not path.exists():
        _MODULE_CACHE[rel_path] = None
        return None, f"unknown file: {rel_path}"
    parts = list(Path(rel_path).with_suffix("").parts)
    # Citations use src/hymo/... but ROOT/src is the import root; strip the
    # leading "src" component so `src.hymo.models.gdn` imports as `hymo.models.gdn`.
    if parts and parts[0] == "src":
        parts = parts[1:]
    dotted = ".".join(parts)
    try:
        mod = importlib.import_module(dotted)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        _MODULE_CACHE[rel_path] = None
        return None, f"import failed for {rel_path}: {type(exc).__name__}: {exc}"
    _MODULE_CACHE[rel_path] = mod
    return mod, None


def _has_instance_attr(cls, name: str) -> bool:
    """Instance attrs assigned via self.name = … are invisible to hasattr on the class."""
    try:
        import inspect
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    return re.search(rf"self\.{re.escape(name)}\s*=", src) is not None


def resolve_anchor(rel_path: str, symbol: str) -> tuple[bool, str | None]:
    mod, err = load_module(rel_path)
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


def resolve_line(rel_path: str, start: int, end: int) -> tuple[bool, str | None]:
    path = ROOT / rel_path
    if not path.exists():
        return False, f"unknown file: {rel_path}"
    nlines = len(path.read_text(encoding="utf-8").splitlines())
    if start < 1 or end > nlines:
        return False, f"{rel_path}:{start}-{end} beyond EOF ({nlines} lines)"
    return True, None


def check_resolution() -> list[str]:
    failures = []
    for doc, rel_path, symbol in collect_anchors():
        if symbol in JIT_SYMBOLS:
            continue
        ok, err = resolve_anchor(rel_path, symbol)
        if not ok:
            failures.append(f"{doc.relative_to(ROOT)}: {err}")
    for doc in _doc_files():
        text = doc.read_text(encoding="utf-8")
        for m in LINE_ANCHOR_RE.finditer(text):
            ok, err = resolve_line(m.group(1), int(m.group(2)), int(m.group(3) or m.group(2)))
            if not ok:
                failures.append(f"{doc.relative_to(ROOT)}: {err}")
    return failures


def public_symbols(rel_path: str) -> list[str]:
    mod, err = load_module(rel_path)
    if err:
        return []
    return sorted(
        n for n in dir(mod)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n)
        and not n.startswith("_")
        and callable(getattr(mod, n))
        and getattr(mod, n).__module__ == mod.__name__
    )


def check_coverage() -> list[str]:
    """Every public module-level symbol in COVERAGE_MODULES must be cited once."""
    cited = {f"{rel}:{sym.split('.')[0]}" for _, rel, sym in collect_anchors()}
    missing = []
    for rel_path in COVERAGE_MODULES:
        for sym in public_symbols(rel_path):
            if f"{rel_path}:{sym}" not in cited:
                missing.append(f"{rel_path}:{sym}")
    return missing


_SLUG_CACHE: dict[Path, set[str]] = {}


def _slug(text: str) -> str:
    """GitHub heading id for a heading's text."""
    text = MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"[*~]", "", text).lower()
    # GitHub maps EACH space to its own hyphen — never collapse runs, or
    # "Part B — Config" would yield `part-b-config` instead of `part-b--config`.
    return SLUG_STRIP_RE.sub("", text).strip().replace(" ", "-")


def heading_slugs(path: Path) -> set[str]:
    """Every anchor id `path` exposes. Setext headings count, thematic
    breaks do not, and a repeated heading gets GitHub's `-1`, `-2` suffix.
    """
    if path in _SLUG_CACHE:
        return _SLUG_CACHE[path]
    lines = FENCE_RE.sub("", path.read_text(encoding="utf-8")).split("\n")
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            title = m.group(1)
        elif line.strip() and i + 1 < len(lines) and SETEXT_RE.match(lines[i + 1]):
            title = line.strip()
        else:
            continue
        base = _slug(title)
        if not base:
            continue
        n = counts.get(base, 0)
        counts[base] = n + 1
        slugs.add(base if n == 0 else f"{base}-{n}")
    _SLUG_CACHE[path] = slugs
    return slugs


def check_links() -> list[str]:
    """Validate intra-repo markdown links *and* their `#fragment` anchors;
    code fences are stripped first."""
    broken = []
    for doc in _doc_files():
        text = FENCE_RE.sub("", doc.read_text(encoding="utf-8"))
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path_part, _, fragment = target.partition("#")
            if not path_part:
                continue
            candidates = [(doc.parent / path_part).resolve(), (ROOT / path_part).resolve()]
            resolved = next((c for c in candidates if c.exists()), None)
            if resolved is None:
                broken.append(f"{doc.relative_to(ROOT)}: broken link -> {target}")
            elif fragment and resolved.suffix == ".md":
                if fragment.lower() not in heading_slugs(resolved):
                    broken.append(f"{doc.relative_to(ROOT)}: dead anchor -> {target}")
    return broken


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Doc↔code alignment checker")
    ap.add_argument("--coverage", action="store_true", help="also enforce public-symbol coverage")
    ap.add_argument("--links", action="store_true", help="also validate intra-repo markdown links")
    args = ap.parse_args()

    failures = check_resolution()
    print(f"[doc-refs] scanned {len(_doc_files())} docs, {len(collect_anchors())} anchors")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"[doc-refs] resolution: {'PASS' if not failures else f'{len(failures)} FAILURES'}")

    missing = check_coverage() if args.coverage else []
    if args.coverage:
        for m in missing:
            print(f"  UNCOVERED {m}")
        print(f"[doc-refs] coverage: {'PASS' if not missing else f'{len(missing)} UNCOVERED'}")

    link_broken = check_links() if args.links else []
    if args.links:
        for b in link_broken:
            print(f"  BROKEN-LINK {b}")
        print(f"[doc-refs] links: {'PASS' if not link_broken else f'{len(link_broken)} BROKEN'}")

    return 1 if (failures or (args.coverage and missing) or (args.links and link_broken)) else 0


if __name__ == "__main__":
    sys.exit(main())
