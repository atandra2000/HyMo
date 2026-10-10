#!/usr/bin/env python3
"""
HyMo Documentation Generator
Converts project markdown files into a responsive, beautifully-styled HTML documentation portal
with full LaTeX math (KaTeX) and syntax highlighting support.
Output directory: docs_html/ (ignored by git).

Design system: the "printed reader" — cool white paper, blue-black ink, one
ink-blue accent, Literata for prose and IBM Plex Mono for code and measured
values. See `assets/style.css`.

The portal is a book. `DOC_FILES` below is the single source of truth for its
chapter list and reading order; the sidebar, the index page and the
previous/next links are all derived from it.
"""

import html
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_DIR / "docs_html"

DEFAULT_PUBLISH_BRANCH = "main"


DOC_FILES = [
    # (relative_path, part, chapter_title, card_blurb)
    # Order matters: it is the reading order for the sidebar, the index page
    # and the previous/next links at the foot of every chapter.

    # Part I — Foundations
    ("README.md", "Part I: Foundations", "Overview",
     "The project in one page: a 434M-active / 1.13B-stored GDN + MLA hybrid, "
     "the asymmetric MoE, and how to run it."),
    ("docs/README.md", "Part I: Foundations", "How to read this book",
     "What each chapter covers, where to start, and the map from source file "
     "to chapter."),
    ("AGENTS.md", "Part I: Foundations", "Repository contract",
     "Codebase contracts, Triton rules, typing invariants, and architectural "
     "limits."),
    ("SKILLS.md", "Part I: Foundations", "Skills reference",
     "Agent tooling, scripts, and the competencies the repo ships with."),

    # Part II — The model
    ("docs/concepts/model-architecture.md", "Part II: The model",
     "Model architecture walkthrough",
     "Line-by-line tour of model.py: GDN blocks, MLA blocks, DeepSeekMoE, MTP "
     "heads, and partial RoPE."),
    ("docs/concepts/gdn-and-mla.md", "Part II: The model",
     "GDN and MLA mechanisms",
     "The linear-attention delta rule, latent KV compression, and the 3:1 "
     "hybrid stack thesis."),
    ("docs/concepts/optimization.md", "Part II: The model",
     "The optimization quartet",
     "NorMuon for 2D matrices, Cautious AdamW elsewhere, WSD schedule, and "
     "FSDP-2 sharding."),
    ("docs/concepts/kernels.md", "Part II: The model",
     "Operations and Triton kernels",
     "The hand-written Triton GDN kernel: fused 1D selective scan with "
     "chunked recurrence."),
    ("docs/concepts/design.md", "Part II: The model",
     "v1.0 design spec",
     "The complete architectural specification, scaling decisions, and "
     "validation criteria."),

    # Part III — The training run
    ("docs/training.md", "Part III: The training run",
     "The training pipeline",
     "FSDP-2 sharding, the NorMuon + AdamW dual optimizer, the WSD schedule, "
     "and DCP checkpointing."),
    ("docs/guides/quickstart.md", "Part III: The training run",
     "Quickstart",
     "From zero to a running forward pass: install, smoke tests, gates, and "
     "the training loop."),

    # Part IV — Reference
    ("docs/references/config.md", "Part IV: Reference", "Typed config reference",
     "ModelConfig, OptimizerConfig, SchedulerConfig, and every validation "
     "rule."),
    ("docs/references/api.md", "Part IV: Reference", "Public API reference",
     "The model interface, Trainer, optimizers, schedulers, and factory "
     "entry points."),
]

# Chapters published as portal pages. A markdown link that resolves to a file
# outside this set is rewritten to its GitHub source URL instead of a page
# that was never generated.
PUBLISHED_MD = {rel for rel, *_ in DOC_FILES}

FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?'
             'family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,500;'
             '0,7..72,600;0,7..72,700;1,7..72,400'
             '&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&display=swap" rel="stylesheet">')

BOOT_OVERLAY_HTML = (
    '<div id="boot-overlay" aria-hidden="true">'
    '<div class="boot-inner">'
    '<div class="boot-wordmark">HyMo</div>'
    '<div class="boot-line">Loading the notes'
    '<span class="boot-bar"></span>'
    '</div></div></div>'
)

BOOT_SCRIPT = """<script>
(function () {
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.documentElement.classList.add('booting');
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(function () {
            document.documentElement.classList.remove('booting');
            var ov = document.getElementById('boot-overlay');
            if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
        }, reduced ? 0 : 350);
    });
})();
</script>"""

# Shared <head> for every generated page.
HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Fonts — Literata for prose, IBM Plex Mono for code and data. -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {font_link}
    {boot_script}
{extra_head}
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
"""

DOC_EXTRA_HEAD = """    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>"""


def layer_stack_ascii() -> str:
    """Compact 32-layer hybrid stack architectural overview with exact monospace alignment."""
    return """\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
\u2502 L00-L02 (x03) \u2502 GDN (Linear Attn, recurrence-only)   \u2502
\u2502 L03     (x01) \u2502 MLA (Latent Attn) -> DeepSeekMoE 16+1 \u2502
\u2502 L04-L06 (x03) \u2502 GDN (Linear Attn, recurrence-only)   \u2502
\u2502 L07     (x01) \u2502 MLA (Latent Attn) -> DeepSeekMoE 16+1 \u2502
\u2502 ...     (x24) \u2502 3:1 Interleaved Hybrid Recurrence     \u2502
\u2502 L28-L30 (x03) \u2502 GDN (Linear Attn, recurrence-only)   \u2502
\u2502 L31     (x01) \u2502 MLA (Latent Attn) -> DeepSeekMoE 16+1 \u2502
\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
\u2502 TOTAL 32 L    \u2502 24 GDN + 8 MLA \u00b7 434M active / 1.13B  \u2502
\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518"""


LAYER_STACK_WIDGET = f"""<div class="ascii-widget" id="widget-layer-stack">
    <div class="ascii-widget-head">FIG · 01 / 32-LAYER HYBRID STACK <span>— 3:1 ratio (24 GDN + 8 MLA)</span></div>
    <div class="ascii-widget-grid">
        <pre class="ascii-stack" aria-label="32-layer hybrid stack: 24 linear GDN layers, 8 MLA full-attention layers">{layer_stack_ascii()}</pre>
        <pre class="ascii-panel" aria-live="polite">STACK RATIO : 3:1 (24 GDN Linear Attn + 8 MLA Full Attn)
GDN BLOCKS  : 24 layers · Gated Delta Net · Recurrence-only (no FFN)
MLA BLOCKS  : 8 layers · Multi-Head Latent Attn · DeepSeekMoE (16+1 top-2)
DIM / HEADS : dim 896 · 16 heads (head_dim 128) · partial RoPE (25%)
OPTIMIZER   : NorMuon (Attention/GDN 2D) + AdamW (Embed/Norm/Gate/MoE)
MTP HEADS   : Depth 2 auxiliary heads (weights: [0.3, 0.1])
SOURCE      : src/hymo/models/model.py · configs/hymo_750m.yaml</pre>
    </div>
</div>"""

MOE_WIDGET = """<div class="ascii-widget" id="widget-moe-routing">
    <div class="ascii-widget-head">FIG / ASYMMETRIC MOE ▸ TOP-2 OF 16 ROUTED + 1 SHARED (MLA ONLY) <span>— route token</span></div>
    <pre class="moe-grid" aria-label="Expert grid: 16 routed experts, 2 active per token, 1 shared always on">gate(x) ▸ top-2 of 16 routed (on 8 MLA layers only; GDN has no FFN)
░░00 ░░01 ▓▓02 ░░03
░░04 ░░05 ░░06 ▓▓07
░░08 ░░09 ░░10 ░░11
░░12 ░░13 ░░14 ░░15
shared ▓▓sh (always active, 2304d)</pre>
</div>"""

MLA_WIDGET = """<div class="ascii-widget" id="widget-mla-absorb">
    <div class="ascii-widget-head">FIG / MLA LATENT COMPRESSION — 4 KV GROUPS <span>— low-rank KV pool</span></div>
    <pre class="mla-figure" aria-label="MLA compression comparison: standard MHA versus 4 KV groups with low-rank compression">latent compressed attention (4 KV groups)

 h[896] ── W_DKV ──▸ c_KV [128] (+ rope 32d) ──▸ cached in KV pool
 c_KV ──▸ W_UK ──▸ k [16x128] & W_UV ──▸ v [16x128] ──▸ Attention

 KV cache / token / layer : MHA 4096 dims → MLA (128 + 4×32)×2 = 512 bytes
 compression              : 16× per-layer KV cut (head_dim 128 → qk_rope 32)</pre>
</div>"""

# rel path -> widget injected between article and footer nav
WIDGET_CONTAINERS = {
    "docs/concepts/model-architecture.md": LAYER_STACK_WIDGET,
    "docs/concepts/gdn-and-mla.md": MOE_WIDGET,
    "docs/concepts/kernels.md": MLA_WIDGET,
}


# Tags the portal renders from raw markdown HTML. Anything outside this set is
# dropped rather than passed through, so a doc can never inject script.
RAW_ALLOWED_TAGS = {
    'div', 'p', 'span', 'a', 'img', 'br', 'hr', 'em', 'strong', 'b', 'i',
    'code', 'pre', 'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'th',
    'td', 'details', 'summary', 'figure', 'figcaption', 'sup', 'sub', 'kbd',
    'small', 'abbr', 'del', 'ins', 'u',
}
# Attributes kept per tag. `style` is deliberately absent: raw inline styles
# would override the design system's tokens.
RAW_ALLOWED_ATTRS = {
    '*': {'class', 'id', 'title', 'align', 'width', 'height', 'colspan', 'rowspan'},
    'a': {'href', 'target', 'rel'},
    'img': {'src', 'alt', 'loading', 'decoding'},
    'td': {'align'},
    'th': {'align', 'scope'},
    'abbr': {'title'},
}
RAW_VOID_TAGS = {'br', 'hr', 'img'}
_RAW_TAG_RE = re.compile(r'<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9-]*)((?:\s+[^<>]*?)?)\s*(/?)\s*>')


def slugify(text: str) -> str:
    """Generate clean HTML id for headings matching GitHub anchor conventions."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s', '-', text)
    return text.strip('-') or "heading"


@lru_cache(maxsize=1)
def github_base_url() -> str:
    """Derive the GitHub blob base (https://github.com/<owner>/<repo>/blob/<branch>)."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        out = out.replace("git@github.com:", "https://github.com/").removesuffix(".git")
        if not out.startswith("https://github.com/"):
            return ""
    except Exception:
        return ""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
    except Exception:
        return ""
    return f"{out}/blob/{branch}" if branch else ""


@lru_cache(maxsize=1)
def github_raw_base_url() -> str:
    """Raw-content twin of `github_base_url`, for <img src> that must render
    rather than open an HTML blob page."""
    blob_base = github_base_url()
    if not blob_base:
        return ""
    return blob_base.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")


def resolve_asset_src(src: str, src_rel_path: str) -> str:
    """Point an <img src> at something that actually renders.

    The build ships no PNGs, so a repo-relative path becomes a raw GitHub URL.
    Absolute URLs and data URIs pass through untouched.
    """
    src = src.strip()
    if not src or src.startswith(("http://", "https://", "data:", "//", "#")):
        return src
    raw_base = github_raw_base_url()
    if not raw_base:
        return src
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent
    candidate = (src_dir / src).resolve()
    try:
        repo_rel = candidate.relative_to(WORKSPACE_DIR).as_posix()
    except ValueError:
        return src
    return f"{raw_base}/{repo_rel}"


def resolve_raw_href(href: str, src_rel_path: str) -> str:
    """Resolve an href inside a raw HTML island (fix_md_links never sees these)."""
    href = href.strip()
    if not href or href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    path_part, _, anchor = href.partition("#")
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent
    if path_part.endswith(".md"):
        for candidate in (src_dir / path_part, WORKSPACE_DIR / path_part):
            if candidate.exists():
                try:
                    repo_rel = candidate.resolve().relative_to(WORKSPACE_DIR)
                except ValueError:
                    continue
                rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
                return rel[:-3] + ".html" + (f"#{anchor}" if anchor else "")
    repo_base = github_base_url()
    if repo_base:
        try:
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"{repo_base}/{repo_rel}" + (f"#{anchor}" if anchor else "")
        except ValueError:
            pass
    return href


def fix_md_links(content: str, src_rel_path: str) -> str:
    """Rewrite relative markdown links for the HTML build."""
    repo_base = github_base_url()
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            if path_part.startswith("/"):
                repo_rel = Path(path_part[1:])
            else:
                cand_src = src_dir / path_part
                cand_root = WORKSPACE_DIR / path_part
                if cand_src.exists():
                    repo_rel = cand_src.resolve().relative_to(WORKSPACE_DIR)
                elif cand_root.exists():
                    repo_rel = cand_root
                else:
                    repo_rel = Path(src_rel_path).parent / path_part
            repo_rel_posix = repo_rel.as_posix()
            if repo_rel_posix not in PUBLISHED_MD:
                # Not a chapter: point at the source file rather than at a
                # page the build never generates.
                if repo_base:
                    return f"[{label}]({repo_base}/{repo_rel_posix})"
                return f"[{label}]({url})"
            rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
            target = rel[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            try:
                repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
                return f"[{label}]({repo_base}/{repo_rel})"
            except ValueError:
                pass
        return f"[{label}]({url})"

    # `(?<!!)` keeps markdown images out of the link rewriter: their src is an
    # asset path, not a page to navigate to.
    return re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """
    Statically convert markdown to rich HTML structure with full LaTeX & Math protection.
    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)

    # STEP 1: Protect Code Blocks & Inline Code
    code_blocks = []
    def store_code_block(m):
        code_blocks.append(m.group(0))
        return f"\n\n___CODEBLOCK_{len(code_blocks)-1}___\n\n"

    md_text = re.sub(r'```[\s\S]*?```', store_code_block, md_text)

    inline_codes = []
    def store_inline_code(m):
        inline_codes.append(m.group(0))
        return f"___INLINECODE_{len(inline_codes)-1}___"

    md_text = re.sub(r'`[^`\n]+`', store_inline_code, md_text)

    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"

    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)

    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"

    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # STEP 3: Parse Document Structure Line by Line
    toc = []
    lines = md_text.splitlines()
    html_lines = []

    in_table = False
    table_headers = []
    table_rows = []

    list_stack = []
    h1_seen = False

    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def close_li():
        nonlocal list_stack
        if list_stack and list_stack[-1]['li_open']:
            html_lines.append("</li>")
            list_stack[-1]['li_open'] = False

    def flush_list():
        nonlocal list_stack
        while list_stack:
            close_li()
            html_lines.append(f"</{list_stack[-1]['tag']}>")
            list_stack.pop()

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_type, blockquote_lines
        if in_blockquote:
            content = "<br>".join(blockquote_lines)
            if blockquote_type != "normal":
                title = blockquote_type.upper()
                icon = {"NOTE": "ℹ️", "TIP": "💡", "IMPORTANT": "📌", "WARNING": "⚠️", "CAUTION": "🚨"}.get(title, "ℹ️")
                html_lines.append(
                    f'<div class="callout callout-{blockquote_type.lower()}">'
                    f'<div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{title}</span></div>'
                    f'<div class="callout-body">{content}</div>'
                    f'</div>'
                )
            else:
                html_lines.append(f'<blockquote>{content}</blockquote>')
            in_blockquote = False
            blockquote_type = "normal"
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            th_html = "".join(f"<th>{h}</th>" for h in table_headers)
            tr_html = ""
            for row in table_rows:
                td_html = "".join(f"<td>{c}</td>" for c in row)
                tr_html += f"<tr>{td_html}</tr>"
            html_lines.append(
                f'<div class="table-container"><table class="doc-table">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody>{tr_html}</tbody>'
                f'</table></div>'
            )
            in_table = False
            table_headers = []
            table_rows = []

    def escape_preserving_entities(text: str) -> str:
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        text = escape_preserving_entities(text)
        # Images before links: `[![alt](src)](href)` is a linked badge, and the
        # link regex would otherwise swallow the inner image markup.
        text = re.sub(
            r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)',
            lambda m: f'<img class="md-img" src="{html.escape(resolve_asset_src(m.group(2), src_rel_path), quote=True)}"'
                      f' alt="{html.escape(m.group(1), quote=True)}" loading="lazy" decoding="async">',
            text,
        )
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("___DISPLAYMATH_") or stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        if stripped.startswith(">"):
            flush_table()
            flush_list()
            bq_content = stripped.lstrip(">").strip()

            callout_match = re.match(r'^\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', bq_content, re.IGNORECASE)
            if callout_match:
                in_blockquote = True
                blockquote_type = callout_match.group(1).upper()
                remaining = bq_content[callout_match.end():].strip()
                if remaining:
                    blockquote_lines.append(render_inline_formatting(remaining))
            else:
                if not in_blockquote:
                    in_blockquote = True
                    blockquote_type = "normal"
                if bq_content:
                    blockquote_lines.append(render_inline_formatting(bq_content))
            i += 1
            continue

        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()

            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            heading_id = slugify(clean_title)

            rendered_heading = render_inline_formatting(heading_text_raw)

            if level in (2, 3):
                toc.append({
                    'level': level,
                    'title': clean_title,
                    'id': heading_id
                })

            if level == 1 and not h1_seen:
                h1_seen = True
                html_lines.append(f'<span class="doc-anchor" id="{heading_id}"></span>')
            else:
                html_lines.append(
                    f'<h{level} id="{heading_id}" class="heading-anchor">'
                    f'{rendered_heading}'
                    f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                    f'</h{level}>'
                )
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()

            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}')
            list_stack[-1]['li_open'] = True
            i += 1
            continue

        if list_stack and list_stack[-1]['li_open'] and line[:1] in (' ', '\t'):
            html_lines.append("<br> " + render_inline_formatting(stripped))
            i += 1
            continue

        flush_table()
        flush_list()
        flush_blockquote()
        html_lines.append(f'<p>{render_inline_formatting(stripped)}</p>')
        i += 1

    flush_table()
    flush_list()
    flush_blockquote()

    full_html = "\n".join(html_lines)

    # STEP 4: Restore Protected Tokens
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    for idx, raw_code in enumerate(inline_codes):
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.removeprefix("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)

        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'

        n_lines = len(lines_b) - 2
        collapsed = n_lines > 14
        wrapper_cls = 'code-wrapper collapsed' if collapsed else 'code-wrapper'
        expand_btn = (
            f'<button class="expand-btn" data-label="expand ▾ · {n_lines} lines" '
            f'onclick="toggleCode(this)">expand ▾ · {n_lines} lines</button>'
            if collapsed else ''
        )

        block_html = (
            f'<div class="{wrapper_cls}" data-lines="{n_lines}">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<span class="code-actions">{expand_btn}'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></span>'
            f'</div>'
            f'<pre><code{lang_attr}>{escaped_content}</code></pre>'
            f'</div>'
        )
        full_html = full_html.replace(f"___CODEBLOCK_{idx}___", block_html)

    return full_html, toc


def sanitize_raw_html(raw: str, src_rel_path: str, format_text=None) -> str:
    """Filter a raw HTML island down to the portal's tag/attribute allowlist.

    Relative `src`/`href` are rewritten so they resolve from `docs_html/`,
    where the referenced images and sibling docs do not exist on disk.
    `format_text` formats the text nodes between tags, so markdown emphasis
    inside a passthrough island still renders.
    """
    out = []
    pos = 0
    for m in _RAW_TAG_RE.finditer(raw):
        chunk = raw[pos:m.start()]
        out.append(format_text(chunk) if format_text else html.escape(chunk, quote=False))
        pos = m.end()

        closing, tag, attrs_raw, self_closing = m.group(1), m.group(2).lower(), m.group(3) or '', m.group(4)
        if tag not in RAW_ALLOWED_TAGS:
            continue
        if closing:
            out.append(f'</{tag}>')
            continue

        allowed = RAW_ALLOWED_ATTRS['*'] | RAW_ALLOWED_ATTRS.get(tag, set())
        kept = []
        for name, value in re.findall(r'([a-zA-Z-]+)\s*=\s*"([^"]*)"', attrs_raw):
            name = name.lower()
            if name not in allowed:
                continue
            if tag == 'img' and name == 'src':
                value = resolve_asset_src(html.unescape(value), src_rel_path)
            elif tag == 'a' and name == 'href':
                value = resolve_raw_href(html.unescape(value), src_rel_path)
                if 'target' in attrs_raw.lower() and 'rel' not in attrs_raw.lower():
                    kept.append(('rel', 'noopener noreferrer'))
            elif tag == 'a' and name == 'target' and value not in ('_blank', '_self'):
                continue
            kept.append((name, html.escape(value, quote=True)))

        attr_str = ''.join(f' {n}="{v}"' for n, v in kept)
        # Void tags stay self-closing; a self-closing allowlisted tag is
        # normalized to an open/close pair so following siblings stay balanced.
        if tag not in RAW_VOID_TAGS and self_closing:
            out.append(f'<{tag}{attr_str}></{tag}>')
        else:
            out.append(f'<{tag}{attr_str}>')
    tail = raw[pos:]
    out.append(format_text(tail) if format_text else html.escape(tail, quote=False))
    return ''.join(out)


def looks_like_raw_html_block(text: str) -> bool:
    """True when a paragraph's opening line is a real HTML island, not prose.

    Prose like "a < b and c > d" must stay escaped text; a line whose first
    tag is allowlisted HTML is a passthrough block.
    """
    stripped = text.strip()
    if not stripped.startswith('<'):
        return False
    m = _RAW_TAG_RE.match(stripped)
    if not m:
        return False
    return m.group(2).lower() in RAW_ALLOWED_TAGS


@lru_cache(maxsize=1)
def github_base_url() -> str:
    """Derive the GitHub blob base (https://github.com/<owner>/<repo>/blob/<branch>).

    The branch is the published default, not the checked-out one: the site is
    deployed from `main`, so a build on a feature branch or worktree must still
    link to content that actually exists on GitHub. `DOCS_PUBLISH_BRANCH`
    overrides.
    """
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        out = out.replace("git@github.com:", "https://github.com/").removesuffix(".git")
        if not out.startswith("https://github.com/"):
            return ""
    except Exception:
        return ""

    branch = os.environ.get("DOCS_PUBLISH_BRANCH")
    explicit = branch is not None
    if not explicit:
        branch = DEFAULT_PUBLISH_BRANCH
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        default_branch = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True, cwd=WORKSPACE_DIR,
        ).stdout.strip().removesuffix("/HEAD").split("/")[-1]
    except Exception:
        head = default_branch = ""
    # A local build of the published branch is exact; anything else falls back
    # to the remote default so links never point at an unpushed branch. An
    # explicit override always wins.
    if not explicit and head and head != branch and default_branch:
        branch = default_branch
    return f"{out}/blob/{branch}" if branch else ""


@lru_cache(maxsize=1)
def github_raw_base_url() -> str:
    """Raw-content twin of `github_base_url`, for <img src> that must render
    rather than open an HTML blob page."""
    blob_base = github_base_url()
    if not blob_base:
        return ""
    return blob_base.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")


def resolve_asset_src(src: str, src_rel_path: str) -> str:
    """Point an <img src> at something that actually renders.

    The build ships no PNGs, so a repo-relative path becomes a raw GitHub URL.
    Absolute URLs and data URIs pass through untouched.
    """
    src = src.strip()
    if not src or src.startswith(("http://", "https://", "data:", "//", "#")):
        return src
    raw_base = github_raw_base_url()
    if not raw_base:
        return src
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent
    candidate = (src_dir / src).resolve()
    try:
        repo_rel = candidate.relative_to(WORKSPACE_DIR).as_posix()
    except ValueError:
        return src
    return f"{raw_base}/{repo_rel}"


def resolve_raw_href(href: str, src_rel_path: str) -> str:
    """Resolve an href inside a raw HTML island (fix_md_links never sees these)."""
    href = href.strip()
    if not href or href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    path_part, _, anchor = href.partition("#")
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent
    if path_part.endswith(".md"):
        for candidate in (src_dir / path_part, WORKSPACE_DIR / path_part):
            if candidate.exists():
                try:
                    repo_rel = candidate.resolve().relative_to(WORKSPACE_DIR)
                except ValueError:
                    continue
                rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
                return rel[:-3] + ".html" + (f"#{anchor}" if anchor else "")
    repo_base = github_base_url()
    if repo_base:
        try:
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"{repo_base}/{repo_rel}" + (f"#{anchor}" if anchor else "")
        except ValueError:
            pass
    return href


def fix_md_links(content: str, src_rel_path: str) -> str:
    """Rewrite relative markdown links for the HTML build."""
    repo_base = github_base_url()
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            if path_part.startswith("/"):
                repo_rel = Path(path_part[1:])
            else:
                cand_src = src_dir / path_part
                cand_root = WORKSPACE_DIR / path_part
                if cand_src.exists():
                    repo_rel = cand_src.resolve().relative_to(WORKSPACE_DIR)
                elif cand_root.exists():
                    repo_rel = cand_root
                else:
                    repo_rel = Path(src_rel_path).parent / path_part
            repo_rel_posix = repo_rel.as_posix()
            if repo_rel_posix not in PUBLISHED_MD:
                # Not a chapter: point at the source file rather than at a
                # page the build never generates.
                if repo_base:
                    return f"[{label}]({repo_base}/{repo_rel_posix})"
                return f"[{label}]({url})"
            rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
            target = rel[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            try:
                repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
                return f"[{label}]({repo_base}/{repo_rel})"
            except ValueError:
                pass
        return f"[{label}]({url})"

    # `(?<!!)` keeps markdown images out of the link rewriter: their src is an
    # asset path, not a page to navigate to.
    return re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """Statically convert markdown to rich HTML structure with full LaTeX & Math protection.

    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)

    # STEP 1: Protect Code Blocks & Inline Code
    code_blocks = []
    def store_code_block(m):
        code_blocks.append(m.group(0))
        return f"\n\n___CODEBLOCK_{len(code_blocks)-1}___\n\n"

    md_text = re.sub(r'```[\s\S]*?```', store_code_block, md_text)

    inline_codes = []
    def store_inline_code(m):
        inline_codes.append(m.group(0))
        return f"___INLINECODE_{len(inline_codes)-1}___"

    md_text = re.sub(r'`[^`\n]+`', store_inline_code, md_text)

    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"

    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)

    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"

    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # STEP 3: Parse Document Structure Line by Line
    toc = []
    seen_slugs = {}
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    table_headers = []
    table_rows = []

    list_stack = []
    h1_seen = False

    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def close_li():
        nonlocal list_stack
        if list_stack and list_stack[-1]['li_open']:
            html_lines.append("</li>")
            list_stack[-1]['li_open'] = False

    def flush_list():
        nonlocal list_stack
        while list_stack:
            close_li()
            html_lines.append(f"</{list_stack[-1]['tag']}>")
            list_stack.pop()

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_type, blockquote_lines
        if in_blockquote:
            content = "<br>".join(blockquote_lines)
            if blockquote_type != "normal":
                title = blockquote_type.upper()
                icon = {"NOTE": "ℹ️", "TIP": "💡", "IMPORTANT": "📌", "WARNING": "⚠️", "CAUTION": "🚨"}.get(title, "ℹ️")
                html_lines.append(
                    f'<div class="callout callout-{blockquote_type.lower()}">'
                    f'<div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{title}</span></div>'
                    f'<div class="callout-body">{content}</div>'
                    f'</div>'
                )
            else:
                html_lines.append(f'<blockquote>{content}</blockquote>')
            in_blockquote = False
            blockquote_type = "normal"
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            th_html = "".join(f"<th>{h}</th>" for h in table_headers)
            tr_html = ""
            for row in table_rows:
                td_html = "".join(f"<td>{c}</td>" for c in row)
                tr_html += f"<tr>{td_html}</tr>"
            html_lines.append(
                f'<div class="table-container"><table class="doc-table">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody>{tr_html}</tbody>'
                f'</table></div>'
            )
            in_table = False
            table_headers = []
            table_rows = []

    def escape_preserving_entities(text: str) -> str:
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        text = escape_preserving_entities(text)
        # Images before links: `[![alt](src)](href)` is a linked badge, and the
        # link regex would otherwise swallow the inner image markup.
        text = re.sub(
            r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)',
            lambda m: f'<img class="md-img" src="{html.escape(resolve_asset_src(m.group(2), src_rel_path), quote=True)}"'
                      f' alt="{html.escape(m.group(1), quote=True)}" loading="lazy" decoding="async">',
            text,
        )
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("___DISPLAYMATH_") or stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        # Blockquote or Callout
        if stripped.startswith(">"):
            flush_table()
            flush_list()
            bq_content = stripped.lstrip(">").strip()
            callout_match = re.match(r'^\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', bq_content, re.IGNORECASE)
            if callout_match:
                in_blockquote = True
                blockquote_type = callout_match.group(1).upper()
                remaining = bq_content[callout_match.end():].strip()
                if remaining:
                    blockquote_lines.append(render_inline_formatting(remaining))
            else:
                if not in_blockquote:
                    in_blockquote = True
                    blockquote_type = "normal"
                if bq_content:
                    blockquote_lines.append(render_inline_formatting(bq_content))
            i += 1
            continue

        # Horizontal Rule
        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()

            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            raw_slug = slugify(clean_title)
            if raw_slug in seen_slugs:
                seen_slugs[raw_slug] += 1
                heading_id = f"{raw_slug}-{seen_slugs[raw_slug]}"
            else:
                seen_slugs[raw_slug] = 0
                heading_id = raw_slug
            rendered_heading = render_inline_formatting(heading_text_raw)

            if level in (2, 3):
                toc.append({'level': level, 'title': clean_title, 'id': heading_id})

            if level == 1 and not h1_seen:
                h1_seen = True
                html_lines.append(f'<span class="doc-anchor" id="{heading_id}"></span>')
            else:
                html_lines.append(
                    f'<h{level} id="{heading_id}" class="heading-anchor">'
                    f'{rendered_heading}'
                    f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                    f'</h{level}>'
                )
            i += 1
            continue

        # Markdown Table Detection
        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        # Lists
        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()

            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}')
            list_stack[-1]['li_open'] = True
            i += 1
            continue

        # Continuation of an open list item
        if list_stack and list_stack[-1]['li_open'] and line[:1] in (' ', '\t'):
            html_lines.append("<br> " + render_inline_formatting(stripped))
            i += 1
            continue

        # Raw HTML island. CommonMark ends an HTML block at the first blank
        # line, which is what lets `<div align="center">` wrap markdown: the
        # tag is its own block and the prose inside is still parsed below.
        if looks_like_raw_html_block(stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            block_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip():
                block_lines.append(lines[j])
                j += 1
            i = j
            html_lines.append(sanitize_raw_html(
                '\n'.join(block_lines), src_rel_path, format_text=render_inline_formatting))
            continue

        # Raw HTML island. CommonMark ends an HTML block at the first blank
        # line, which is what lets `<div align="center">` wrap markdown: the
        # tag is its own block and the prose inside is still parsed below.
        if looks_like_raw_html_block(stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            block_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip():
                block_lines.append(lines[j])
                j += 1
            i = j
            html_lines.append(sanitize_raw_html(
                '\n'.join(block_lines), src_rel_path, format_text=render_inline_formatting))
            continue

        # Standard Paragraph
        flush_table()
        flush_list()
        flush_blockquote()
        html_lines.append(f'<p>{render_inline_formatting(stripped)}</p>')
        i += 1

    flush_table()
    flush_list()
    flush_blockquote()

    full_html = "\n".join(html_lines)

    # STEP 4: Restore Protected Tokens
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    for idx, raw_code in enumerate(inline_codes):
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)

        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'

        n_lines = len(lines_b) - 2
        collapsed = n_lines > 14
        wrapper_cls = 'code-wrapper collapsed' if collapsed else 'code-wrapper'
        expand_btn = (
            f'<button class="expand-btn" data-label="expand ▾ · {n_lines} lines" '
            f'onclick="toggleCode(this)">expand ▾ · {n_lines} lines</button>'
            if collapsed else ''
        )

        block_html = (
            f'<div class="{wrapper_cls}" data-lines="{n_lines}">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<span class="code-actions">{expand_btn}'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></span>'
            f'</div>'
            f'<pre><code{lang_attr}>{escaped_content}</code></pre>'
            f'</div>'
        )
        full_html = full_html.replace(f"___CODEBLOCK_{idx}___", block_html)

    return full_html, toc


def compute_rel_prefix(target_rel_path: str) -> str:
    """Calculate relative path back to root docs_html directory."""
    parts = Path(target_rel_path).parts
    if len(parts) <= 1:
        return "./"
    return "../" * (len(parts) - 1)


def build_sidebar_html(current_rel_path: str, rel_prefix: str) -> str:
    """Build the navigation sidebar HTML matching sibling layout."""
    sidebar_sections: dict[str, list[str]] = {}

    for rel_path, part, display_title, _blurb in DOC_FILES:
        target_html_rel = rel_path.replace(".md", ".html")
        href = rel_prefix + target_html_rel
        is_active = (rel_path == current_rel_path)
        active_cls = "active" if is_active else ""
        sidebar_sections.setdefault(part, []).append(
            f'<li class="nav-item"><a href="{href}" class="nav-link {active_cls}" title="{display_title}"><span class="nav-link-text">{display_title}</span></a></li>'
        )

    html_out = ['<div class="sidebar-search"><input type="text" id="navSearch" placeholder="Search docs..." onkeyup="filterNav()"></div>']

    for part, items in sidebar_sections.items():
        html_out.append('<div class="nav-group">')
        html_out.append(f'<div class="nav-group-title">{part}</div>')
        html_out.append(f'<ul class="nav-list">{"".join(items)}</ul>')
        html_out.append('</div>')

    return "\n".join(html_out)


def build_toc_html(toc_items: list[dict]) -> str:
    """Build the right sidebar table of contents."""
    if not toc_items:
        return '<div class="toc-empty">No section headings</div>'

    toc_links = []
    for item in toc_items:
        indent_cls = "toc-h3" if item['level'] == 3 else "toc-h2"
        toc_links.append(f'<li class="{indent_cls}"><a href="#{item["id"]}" class="toc-link">{item["title"]}</a></li>')

    return f'<ul class="toc-list">{"".join(toc_links)}</ul>'


def generate_html_page(rel_path: str, part: str, display_title: str):
    """Generate single HTML file for a markdown document."""
    src_file = WORKSPACE_DIR / rel_path
    if not src_file.exists():
        print(f"Warning: {src_file} does not exist, skipping.")
        return

    md_text = src_file.read_text(encoding="utf-8")

    word_count = len(re.findall(r'\b\w+\b', md_text))
    reading_time = max(1, round(word_count / 220))

    html_body, toc_items = parse_markdown_to_html(md_text, rel_path)

    rel_prefix = compute_rel_prefix(rel_path)
    sidebar_html = build_sidebar_html(rel_path, rel_prefix)
    toc_html = build_toc_html(toc_items)
    widget_html = WIDGET_CONTAINERS.get(rel_path, "")

    curr_idx = -1
    for idx, (p, _, _, _) in enumerate(DOC_FILES):
        if p == rel_path:
            curr_idx = idx
            break

    prev_doc = DOC_FILES[curr_idx - 1] if curr_idx > 0 else None
    next_doc = DOC_FILES[curr_idx + 1] if 0 <= curr_idx < len(DOC_FILES) - 1 else None

    prev_html = ""
    if prev_doc:
        prev_href = rel_prefix + prev_doc[0].replace(".md", ".html")
        prev_html = f'<a href="{prev_href}" class="nav-card prev-card"><span class="card-label">← Previous</span><span class="card-title">{prev_doc[2]}</span></a>'

    next_html = ""
    if next_doc:
        next_href = rel_prefix + next_doc[0].replace(".md", ".html")
        next_html = f'<a href="{next_href}" class="nav-card next-card"><span class="card-label">Next →</span><span class="card-title">{next_doc[2]}</span></a>'

    page_html = HEAD_TEMPLATE.format(
        rel_prefix=rel_prefix,
        title=f"{display_title} | HyMo Documentation",
        extra_head=DOC_EXTRA_HEAD,
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body>
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="brand-name">HyMo</span>
                <span class="brand-badge">Docs</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal Home</a>
            <a href="{rel_prefix}README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <div class="content-container">
                <div class="breadcrumb">
                    <a href="{rel_prefix}index.html">Docs</a> &gt; <span>{part}</span> &gt; <span class="current">{display_title}</span>
                </div>

                <div class="doc-header">
                    <h1 class="doc-title">{display_title}</h1>
                    <div class="doc-meta">
                        <span class="meta-item">{rel_path}</span>
                        <span class="meta-item">{word_count:,} words</span>
                        <span class="meta-item">~{reading_time} min read</span>
                    </div>
                </div>

                <article class="markdown-body" id="articleBody">
                    {html_body}
                </article>

                {widget_html}

                <div class="doc-footer-nav">
                    {prev_html}
                    {next_html}
                </div>
            </div>
        </main>

        <!-- Right Sidebar Table of Contents -->
        <aside class="toc-sidebar">
            <div class="toc-inner">
                <div class="toc-title">On This Page</div>
                {toc_html}
            </div>
        </aside>
    </div>

    <!-- Scripts -->
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            if (window.hljs) {{
                hljs.highlightAll();
            }}
            if (window.renderMathInElement) {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '\\\\[', right: '\\\\]', display: true}},
                        {{left: '\\\\(', right: '\\\\)', display: false}},
                        {{left: '$', right: '$', display: false}}
                    ],
                    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                    throwOnError: false
                }});
            }}
        }});
    </script>
    <script defer src="{rel_prefix}assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / rel_path.replace(".md", ".html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page_html, encoding="utf-8")


def generate_index_portal():
    """Generate interactive index.html home portal."""
    sidebar_html = build_sidebar_html("index.html", "./")

    # The index is the book's table of contents. It is derived from DOC_FILES
    # so the sidebar and the index cannot disagree about the chapter list.
    parts: dict[str, list[str]] = {}
    chapter_no = 0
    for rel_path, part, display_title, blurb in DOC_FILES:
        chapter_no += 1
        parts.setdefault(part, []).append(f"""
            <a href="{rel_path.replace('.md', '.html')}" class="portal-card">
                <span class="card-tag">Ch {chapter_no}</span>
                <div class="card-body">
                    <h3 class="card-heading">{display_title}</h3>
                    <p class="card-desc">{blurb}</p>
                </div>
            </a>
            """)

    portal_cards_html = ""
    for part, cards in parts.items():
        n = len(cards)
        portal_cards_html += f"""
        <section class="portal-section">
            <header class="portal-section-head">
                <span class="portal-section-mark">{n} {'chapter' if n == 1 else 'chapters'}</span>
                <h2 class="portal-section-title">{part}</h2>
            </header>
            <div class="portal-grid">{''.join(cards)}</div>
        </section>
        """

    index_html = HEAD_TEMPLATE.format(
        title="HyMo Documentation Portal",
        extra_head="",
        rel_prefix="./",
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body class="index-portal">
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="index.html" class="brand-logo">
                <span class="brand-name">HyMo</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Portal Content -->
        <main class="main-content">
            <div class="content-container">
                <div class="hero-banner">
                    <h1 class="hero-title">HyMo<span class="sr-only" data-title="HYMO"> — documentation portal</span></h1>
                    <p class="hero-subtitle">A 434M-active / 1.13B-stored hybrid language model: Gated Delta Networks (linear attention) interleaved 3:1 with Multi-Head Latent Attention (full attention), routed through a 16+1 Asymmetric Mixture-of-Experts. Pure PyTorch &mdash; one hand-written Triton GDN recurrence kernel, a NorMuon &plus; AdamW dual optimizer, and a 30B-token training target.</p>

                    <div class="hymo-telemetry-ribbon" aria-label="Key HyMo Architectural Metrics">
                        <div class="telemetry-card terra">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> HYBRID RATIO</div>
                            <div class="tc-val">75% <span class="unit">SUB-QUADRATIC</span></div>
                            <div class="tc-desc">3:1 stack ratio &middot; 24 GDN linear layers + 8 MLA full-attention layers</div>
                        </div>
                        <div class="telemetry-card olive">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> ASYMMETRIC FFN</div>
                            <div class="tc-val">16+1 <span class="unit">TOP-2 MoE</span></div>
                            <div class="tc-desc">Sparse MoE (2304d) on 8 MLA blocks; 24 GDN blocks are recurrence-only (no FFN)</div>
                        </div>
                        <div class="telemetry-card gold">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> DUAL OPTIMIZER</div>
                            <div class="tc-val">NorMuon <span class="unit">+ AdamW</span></div>
                            <div class="tc-desc">NorMuon for 2D matrix weights; Cautious AdamW for 1D, Embeddings &amp; MoE</div>
                        </div>
                        <div class="telemetry-card ink">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> MTP PREDICTION</div>
                            <div class="tc-val">2 HEADS <span class="unit">[0.3, 0.1]</span></div>
                            <div class="tc-desc">Multi-Token Prediction auxiliary heads accelerate representation learning</div>
                        </div>
                    </div>

                    <div class="hero-figure" id="hero-decode" data-title="HYMO" data-sub="434M Active · 1.13B Total · GDN 24 + MLA 8 (3:1) · MoE 16+1 · MTP 2" role="region" aria-label="Figure A0: Hybrid State Dynamics (GDN Linear Recurrence, MLA Latent Attention &amp; Asymmetric MoE)">
                        <div class="hero-figure-header">
                            <div class="fig-badge">
                                <span class="fig-dot" aria-hidden="true"></span>
                                <span class="fig-tag">FIG. A0</span>
                                <span class="fig-sep" aria-hidden="true">/</span>
                                <span class="fig-title">HYBRID RECURRENCE FIELD &middot; GDN &middot; MLA &middot; MoE</span>
                                <span class="fig-dim">32 LAYERS &middot; 896 DIMS &middot; 16 HEADS</span>
                            </div>
                            <div class="fig-controls">
                                <button class="fig-btn active" data-mode="gdn" type="button" title="Gated Delta Net Linear Recurrence">GDN &middot; LINEAR</button>
                                <button class="fig-btn" data-mode="mla" type="button" title="Multi-Head Latent Attention (4 KV Groups)">MLA &middot; LATENT</button>
                                <button class="fig-btn" data-mode="moe" type="button" title="Asymmetric MoE 16+1 Routing Field">MoE &middot; 16+1 ROUTER</button>
                                <button class="fig-btn" data-mode="mtp" type="button" title="Multi-Token Prediction Speculative Tree">MTP &middot; DRAFT</button>
                                <button class="fig-btn" data-mode="hybrid" type="button" title="Full 3:1 Interleaved Hybrid Stack">3:1 &middot; HYBRID</button>
                                <button class="fig-btn fig-btn-icon" id="heroSpeedBtn" type="button" title="Simulation Speed" aria-label="Speed: 1x">1&times;</button>
                                <button class="fig-btn fig-btn-icon" id="heroPauseBtn" type="button" title="Pause / Resume" aria-label="Pause simulation">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="fig-canvas-wrap">
                            <canvas id="heroStateCanvas" class="hero-canvas" aria-hidden="true"></canvas>
                            <div class="fig-overlay-hud" aria-hidden="true">
                                <div class="hud-corner top-left">
                                    <span class="hud-lbl">SIMULATION MODE</span>
                                    <span class="hud-val" id="hudModeLabel">GDN &middot; 1D SELECTIVE SCAN (O(N))</span>
                                </div>
                                <div class="hud-corner top-right">
                                    <span class="hud-lbl">STATE FOOTPRINT</span>
                                    <span class="hud-val" id="hudKVCut">O(1) STATE <span class="unit">(O(N) COMPUTE)</span></span>
                                </div>
                                <div class="hud-corner bottom-left">
                                    <span class="hud-lbl">BLOCK ALLOCATION</span>
                                    <span class="hud-val" id="hudMoEStatus">24 LAYERS <span class="num">(DENSE SwiGLU)</span></span>
                                </div>
                                <div class="hud-corner bottom-right">
                                    <span class="hud-lbl">KERNEL DISPATCH</span>
                                    <span class="hud-val" id="hudOverlap">FUSED TRITON <span class="unit">(CHUNK 64)</span></span>
                                </div>
                                <div class="hud-probe" id="heroProbeHUD" style="opacity: 0;">
                                    <div class="probe-card">
                                        <span class="probe-tag" id="probeTag">PROBE &middot; GDN HEAD #04</span>
                                        <span class="probe-val" id="probeCoords">State Decay &alpha; = 0.942 &middot; Chunk #04</span>
                                        <span class="probe-sub" id="probeDecay">Recurrence Update &Delta;h = (1 - &beta; q k&top;) h + &beta; v k&top;</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="hero-figure-footer">
                            <div class="fig-formula" id="heroFormulaBar">
                                <span class="formula-sym">h<sub class="f-sub">t</sub></span>
                                <span class="formula-op">=</span>
                                <span class="formula-term term-a">(1 - &beta;<sub class="f-sub">t</sub> q<sub class="f-sub">t</sub> k<sub class="f-sub">t</sub><sup class="f-sub">&top;</sup>) h<sub class="f-sub">t-1</sub> + &beta;<sub class="f-sub">t</sub> v<sub class="f-sub">t</sub> k<sub class="f-sub">t</sub><sup class="f-sub">&top;</sup></span>
                                <span class="formula-dot">&middot;</span>
                                <span class="formula-sym">&alpha;<sub class="f-sub">t</sub></span>
                                <span class="formula-op">=</span>
                                <span class="formula-term term-b">exp(g<sub class="f-sub">t</sub> A)</span>
                            </div>
                            <div class="fig-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">24 GDN Linear Layers</span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">8 MLA Full Attention Layers</span></span>
                                <span class="legend-item"><span class="legend-swatch gold"></span> <span class="legend-label">16+1 Asymmetric MoE / MTP</span></span>
                                <span class="legend-hint">CLICK: SHOCKWAVE &middot; HOVER: PROBE HUD</span>
                            </div>
                        </div>
                    </div>

                    <div class="spec-sheet">
                        <div class="spec-sheet-rule" aria-hidden="true">
                            <span class="spec-rule-key">DATASHEET</span>
                            <span class="spec-rule-meta">rev 1.0 &middot; chk bf16 &middot; 30B tok target &middot; 4&times; A100-80</span>
                        </div>
                        <dl class="spec-grid">
                            <div class="spec-cell"><dt class="spec-key">01 &middot; params</dt><dd class="spec-val">~750<span class="unit">M</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">02 &middot; layers</dt><dd class="spec-val">32<span class="unit"> 24+8 (3:1)</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">03 &middot; attention</dt><dd class="spec-val">MLA<span class="unit"> 4 KV grp</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">04 &middot; recurrence</dt><dd class="spec-val">GDN<span class="unit"> chunk 64</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">05 &middot; ffn</dt><dd class="spec-val">16+1<span class="unit"> MoE / SwiGLU</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">06 &middot; vocab</dt><dd class="spec-val">64,256<span class="unit"> BPE-64k</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">07 &middot; ctx &amp; rope</dt><dd class="spec-val">4096<span class="unit"> 25% decoupled</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">08 &middot; optimizer</dt><dd class="spec-val">Dual<span class="unit"> NorMuon+AdamW</span></dd></div>
                        </dl>
                    </div>

                    <div class="mechanism-section" aria-label="Interactive HyMo Core Mechanisms">
                        <div class="mechanism-section-head">
                            <span class="mechanism-section-title">Core mechanisms, with live controls</span>
                        </div>
                        <div class="mechanism-grid">
                            <!-- Card 1: GDN Recurrence & Chunkwise Scan -->
                            <div class="mechanism-card card-gdn" id="mchCardGdn">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">01 &middot; GDN RECURRENCE</span>
                                    <span class="mch-title">Fused 1D Selective Scan</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Gated Delta Net processes linear recurrence with sub-quadratic O(N) complexity. Learned decay &alpha;_t = exp(g_t A) and write/read state decomposition.</p>
                                    <div class="mch-control-row">
                                        <span>Recurrence Chunk Size:</span>
                                        <strong id="gdnChunkLabel">64 tokens / chunk</strong>
                                    </div>
                                    <input type="range" id="gdnChunkSlider" class="mch-slider" min="32" max="128" step="32" value="64" aria-label="Chunk size in tokens">
                                    <div class="chunk-pipeline-box" id="gdnPipelineDisplay">
                                        <div class="chunk-stage">
                                            <span class="chunk-badge scan">DELTA RULE</span>
                                            <span>h_t = (1 - &beta;_t q_t k_t&top;) h_&#123;t-1&#125; + &beta;_t v_t k_t&top;</span>
                                        </div>
                                        <div class="chunk-stage">
                                            <span class="chunk-badge chunk">TRITON KERNEL</span>
                                            <span>Fused 1D selective scan with chunked recurrence (Q=64)</span>
                                        </div>
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Sub-Quadratic Ratio:</span>
                                            <span>24 of 32 Layers (75% Stack)</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>Numerical Parity:</span>
                                            <span id="gdnMatchText">&lt; 1e-5 max |&Delta;| vs Eager</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="gdnVerifyBtn">▸ Verify Chunkwise Recurrence Parity</button>
                                </div>
                            </div>

                            <!-- Card 2: Asymmetric MoE Router -->
                            <div class="mechanism-card card-moe" id="mchCardMoe">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">02 &middot; ASYMMETRIC MoE</span>
                                    <span class="mch-title">16+1 Router on MLA Layers</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">DeepSeekMoE with 16 routed experts (2304d) + 1 shared expert (2304d). Top-2 routed per token with aux-loss-free dynamic bias leveling &Delta;b.</p>
                                    <div class="moe-mini-grid" id="moeMiniGrid">
                                        <!-- 16 mini cells + 1 shared expert generated via JS -->
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Active Experts:</span>
                                            <span id="moeActiveList">#02, #07 + shared (2304d)</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>FFN Placement:</span>
                                            <span>MLA Only (GDN blocks are recurrence-only, no FFN)</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="moeRouteBatchBtn">▸ Route Token Batch &amp; Update &Delta;b</button>
                                </div>
                            </div>

                            <!-- Card 3: MLA Latent KV Compression -->
                            <div class="mechanism-card card-mla" id="mchCardMla">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">03 &middot; MLA COMPRESSION</span>
                                    <span class="mch-title">4 KV Groups Footprint Analyzer</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Multi-Head Latent Attention compresses KV cache to 4 groups (128 latent + 32 rope dims). Reduces KV cache footprint by 16&times; per layer vs standard MHA.</p>
                                    <div class="mch-control-row">
                                        <span>Context Length:</span>
                                        <strong id="mlaContextLabel">32,768 tokens</strong>
                                    </div>
                                    <input type="range" id="mlaContextSlider" class="mch-slider" min="4096" max="131072" step="4096" value="32768" aria-label="Context length in tokens">
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Standard MHA (16 heads &times; 128d &times; 32L):</span>
                                            <span id="statMhaVram">8.00 GB</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>HyMo Hybrid (24 GDN + 8 MLA):</span>
                                            <span id="statMlaVram">0.13 GB</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>KV Cache Memory Saved:</span>
                                            <span id="statMlaSaved">7.87 GB (16&times; cut per layer)</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="mlaAbsorbAnimBtn">▸ Observe MQA-4 Latent Projection</button>
                                </div>
                            </div>

                            <!-- Card 4: MTP Speculative Tree -->
                            <div class="mechanism-card card-mtp" id="mchCardMtp">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">04 &middot; MTP SPECULATION</span>
                                    <span class="mch-title">Depth-2 Speculative Tree</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Multi-token prediction. Main 32-layer backbone predicts token t while depth-1 and depth-2 modules draft candidate tokens t+1 and t+2 with shared embeddings.</p>
                                    <div class="mtp-tree-box" id="mtpTreeDisplay">
                                        <div class="mtp-branch">
                                            <span class="mtp-badge draft">MAIN HEAD</span>
                                            <span>Token t &rarr; "architectures" (p = 0.99)</span>
                                        </div>
                                        <div class="mtp-branch">
                                            <span class="mtp-badge acc">MTP DRAFT 1</span>
                                            <span>Token t+1 &rarr; "converge" (p = 0.88 &ge; 0.80)</span>
                                        </div>
                                        <div class="mtp-branch">
                                            <span class="mtp-badge acc">MTP DRAFT 2</span>
                                            <span>Token t+2 &rarr; "faster" (p = 0.82 &ge; 0.75)</span>
                                        </div>
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Speculative Verification:</span>
                                            <span id="mtpStatusText">ACCEPTED (3 Tokens / Step)</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>Throughput Acceleration:</span>
                                            <span id="mtpSpeedText">2.00&times; effective speedup</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="mtpVerifyStepBtn">▸ Step Speculative Verification</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="pass-widget" id="passWidget" role="region" aria-label="Figure A1: Complete 32-Layer Training Step Pipeline — Forward Activations, Autograd Backward Pass, Gradient Checkpointing Re-computation, and NorMuon/AdamW Step">
                        <div class="pass-widget-header">
                            <div class="pass-badge">
                                <span class="pass-dot" aria-hidden="true"></span>
                                <span class="pass-tag">FIG. A1</span>
                                <span class="pass-sep" aria-hidden="true">/</span>
                                <span class="pass-title">FULL TRAINING STEP PIPELINE</span>
                                <span class="pass-dim">32 LAYERS &middot; FORWARD &middot; AUTOGRAD &middot; DUAL OPTIMIZER</span>
                            </div>
                            <div class="pass-controls">
                                <button class="pass-btn active" data-phase="cycle" type="button" title="Full Forward / Backward / Dual Optimizer Cycle">AUTO CYCLE</button>
                                <button class="pass-btn" data-phase="forward" type="button" title="Inspect Forward Activation Flow &amp; FSDP-2 Prefetch">FORWARD</button>
                                <button class="pass-btn" data-phase="backward" type="button" title="Inspect Backward Autograd Flow &amp; Recompute">BACKWARD</button>
                                <button class="pass-btn" data-phase="opt" type="button" title="Inspect NorMuon &amp; Cautious AdamW Optimizer Partition">DUAL OPT</button>
                                <button class="pass-btn pass-btn-icon" id="passSpeedBtn" type="button" title="Simulation Speed" aria-label="Speed: 1x">1&times;</button>
                                <button class="pass-btn pass-btn-icon" id="passPauseBtn" type="button" title="Pause / Resume Pipeline" aria-label="Pause pipeline">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="pass-canvas-wrap">
                            <canvas id="passDiagramCanvas" class="pass-canvas" aria-hidden="true"></canvas>
                            <div class="pass-overlay-hud" aria-hidden="true">
                                <div class="pass-hud-chip top-left">
                                    <span class="ph-lbl">CURRENT PHASE</span>
                                    <span class="ph-val" id="phCurrentPhase">AUTO CYCLE (FWD &rarr; BWD &rarr; DUAL OPT)</span>
                                </div>
                                <div class="pass-hud-chip top-right">
                                    <span class="ph-lbl">MEMORY CONTRACT</span>
                                    <span class="ph-val" id="phStepMetrics">FSDP-2 BF16 &middot; 75% SUB-QUADRATIC &middot; FP32 MASTER</span>
                                </div>
                                <div class="pass-stage-tooltip" id="passStageTooltip" style="opacity: 0;">
                                    <div class="st-card">
                                        <span class="st-tag" id="stTag">STAGE 02: GDN BLOCKS</span>
                                        <span class="st-op" id="stOp">24× GDN Linear Scan + SwiGLU</span>
                                        <span class="st-shape" id="stShape">Tensor: [B, 4096, 896] bf16</span>
                                        <span class="st-desc" id="stDesc">Gated Delta Net sub-quadratic 1D recurrence (chunk=64)</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pass-widget-footer">
                            <div class="pass-status-ticker">
                                <span class="ticker-beacon" id="passTickerBeacon">&bull;</span>
                                <span class="ticker-text" id="passTickerText">FORWARD &middot; Tokens x<sub class="f-sub">t</sub> &rarr; Embed &rarr; 24&times; GDN (Linear) ⇄ 8&times; MLA (MoE 16+1) &rarr; MTP Heads &rarr; CE Loss &rarr; NorMuon/AdamW</span>
                            </div>
                            <div class="pass-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">Forward Activations</span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">Backward Gradients (&part;&ell;/&part;&theta;)</span></span>
                                <span class="legend-item"><span class="legend-swatch gold"></span> <span class="legend-label">NorMuon + AdamW Step</span></span>
                                <span class="legend-hint">CLICK STAGE: INSPECT &middot; HOVER: TENSOR SHAPES</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    <script defer src="assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_assets():
    """Copy assets/style.css and assets/portal.js into docs_html/assets/."""
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
    src_js = WORKSPACE_DIR / "assets" / "portal.js"
    if src_js.exists():
        shutil.copyfile(src_js, assets_dir / "portal.js")


def main():
    print("Building HyMo HTML Documentation...")
    # docs_html/ is a build artifact, so wipe it first. Otherwise a chapter
    # dropped from DOC_FILES survives as a stale page and gets published.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_assets()

    for rel_path, part, display_title, _blurb in DOC_FILES:
        print(f"Generating: {rel_path} -> docs_html/{rel_path.replace('.md', '.html')}")
        generate_html_page(rel_path, part, display_title)

    generate_index_portal()
    print("\nDocumentation build complete!")
    print(f"HTML Portal location: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
