# HyMo — AGENTS.md

Project-scoped rules. Wins over the root `CoreProjects/AGENTS.md` for this
project.

## Engineering rules (HyMo-specific)

- **Raw PyTorch by default; custom Triton kernels are first-party for
  sanctioned hot paths.** Bulk of the codebase (FSDP-2, training
  loop, data pipeline, MTP) stays raw PyTorch. No HF
  Trainer, no Lightning. The sanctioned Triton paths are:
  - **Gated Delta Net recurrence** — `src/hymo/models/gdn_triton.py`
    (fused GDN forward + autograd Function with FA2-style
    re-compute backward). This is the only currently-sanctioned
    kernel; it predates the carve-out rule and is the project
    precedent.
  No other component gets a custom kernel without updating this file
  and the kernel plan.
- **Strong typing.** Every public function and method is fully
  annotated. `mypy --strict` is the gate.
- **No hardcoded numbers.** All hyperparameters live in
  `configs/hymo_750m.yaml` (or per-ablation config). The only numeric
  literals allowed in code are architectural *constants* (e.g.
  `vocab_size = 64_256` for the BPE-64k + 256-byte tokenizer) — and
  even those should be defined in `hymo.core.config` and referenced.
- **Frozen dataclasses for config.** All config classes are
  ``@dataclass(frozen=True)`` so accidental mutation in the training
  loop is a hard error. Use ``dataclasses.replace`` to derive variants.
- **Algorithmic Logic is Active (Phase 3/4).** Models are fully
  implemented. Do not use `NotImplementedError` placeholders unless
  stubbing out future experiments.
- **No circular dependencies.** The dependency graph is strictly:
  ``core ← {models, training, data}``.
  ``models`` and ``training`` do not import each other directly; they
  share state through the config.
- **Triton kernel contract (mirror of `DeepSeek-v3-Lite/AGENTS.md`):**
  the optional-import / `HAS_TRITON` / autograd `Function` pattern
  used in `gdn_triton.py` is the template. Any new kernel must add a
  `tests/unit/test_<kernel>.py` with a CPU-runnable pure-PyTorch
  reference and a `@pytest.mark.heavy` gate for GPU behaviour.
  `triton>=3.0.0; sys_platform=='linux'` is already declared in
  `pyproject.toml` — do not duplicate that dependency declaration
  elsewhere.

## Subagent routing

- "Build the GDN kernel" → see design §2.3 / roadmap B1
- "Wire the FSDP-2 wrapper" → see design §13 / roadmap D1-D7
- "Add a new data source" → see roadmap A3 + the workspace ``LLM/shared_data`` pipeline
- "Write a config for an ablation" → see design §16 / roadmap F1-F4

## Testing rules (MANDATORY — hardcoded, do not deviate)

The production v1.0 model is 1.86 B params (full model construct in
float32 ≈ 7.4 GB, more than a dev laptop has). A test must **never**
build it by default. Follow this exact style for every test:

1. **Default tests must be CPU-friendly and cool.** Each test runs on
   CPU, in well under a minute, and never heats the machine. The
   entire default ``pytest`` run finishes in ~1 minute on an M1 Air
   with zero heavy tests executed.
2. **Never call ``ModelConfig()`` expecting the full model.** In
   ``tests/unit/test_models.py`` a bare ``ModelConfig()`` is shadowed
   at module scope to return the *tiny* config (~760 K params).
   Everywhere else, build models from the tiny fixture
   (``tiny_hymo_model`` / ``tiny_hymo_config`` in
   ``tests/conftest.py``), NOT from ``configs/hymo_750m.yaml``.
3. **Heavy tests are opt-in only.** Any test that builds the
   production model (or the full 1.86B graph) MUST be marked
   ``@pytest.mark.heavy``. ``tests/conftest.py::pytest_collection_modifyitems``
   auto-skips these in the default run; they run only with
   ``pytest --run-heavy`` (CI / GPU pod). Do not remove the marker,
   and do not add assertions that require the full model outside a
   heavy test.
4. **Verify behavior, not scale.** Tests check stability, shapes,
   numerical finiteness, pipeline wiring, optimizer partition, and
   architectural invariants — using the tiny config. Production-scale
   arithmetic (e.g. exact expert counts = 384, param totals) lives
   behind ``heavy`` and/or reads config values via
   ``production_config_only`` (which loads the YAML *without* building
   a model).
5. **Derive expected counts from config, don't hardcode.** When a
   test asserts a count that depends on a tiny-config knob (routed
   experts, layers, etc.), compute it from ``model.config`` so the
   test survives tiny-config tweaks — never hardcode the production
   number.
6. **Gates stay green.** ``mypy --strict src/hymo`` and ``ruff``
   must pass. The default ``pytest`` (no ``--run-heavy``) must pass
   with all heavy tests skipped.
7. **New Triton kernel tests** follow `DeepSeek-v3-Lite/tests/test_moe_triton.py`
   as the template: a pure-PyTorch reference class (always
   importable), `@pytest.mark.skipif(not HAS_TRITON)` for the
   GPU-only behaviour, no silent fallback in production code paths.

## Concise-comments rule

Docstrings and inline comments must justify non-obvious code, not
restate it. Verifiable targets per file:
- **Public function docstring:** ≤ 3 lines, or one short paragraph.
- **Module docstring:** ≤ 6 lines.
- **Inline comment density:** ≤ 1 comment per ~10 lines of code on
  average; comments that say what the next line does (`# compute x`,
  `# loop over rows`) are forbidden.
- **Section banners** (`# ---- ... ----`) are reserved for the top
  level of a file (≤ 3 per file) and inside kernels to delimit
  named algorithm phases.

Violations are reviewable on `wc -l <file>` and
`grep -c '^[[:space:]]*#' <file>`.

## Hard don'ts

- Don't add hyperparameters to code — extend ``hymo.core.config``.
- Don't add new dependencies without updating ``pyproject.toml`` and
  the README.
- Don't import torch from ``hymo.core`` — that subpackage must stay
  PyTorch-free.
- **Don't use NotImplementedError** for core model logic — it should
  be fully implemented and compiled.
- Don't build the full 1.86B model in a default (non-``heavy``) test.
- Don't hardcode production-scale numbers into tiny-config tests.
- Don't let a Triton kernel silently fall back to raw PyTorch during
  a default-config training run. Opt-in is explicit; failures must
  surface a clear error.
