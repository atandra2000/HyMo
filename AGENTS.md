# HyMo — AGENTS.md

Project-scoped rules. Wins over the root `CoreProjects/AGENTS.md` for this
project.

## Engineering rules (HyMo-specific)

- **Raw PyTorch first.** No HF Trainer, no Lightning. Phase 1 introduces
  only the structural types; the loop, kernels, and distributed training
  stay raw.
- **Strong typing.** Every public function and method is fully annotated.
  `mypy --strict` is the gate.
- **No hardcoded numbers.** All hyperparameters live in
  `configs/hymo_750m.yaml` (or per-ablation config). The only numeric
  literals allowed in code are architectural *constants* (e.g.
  `vocab_size = 64_256` for the BPE-64k + 256-byte tokenizer) — and
  even those should be defined in `hymo.core.config` and referenced.
- **Frozen dataclasses for config.** All config classes are
  ``@dataclass(frozen=True)`` so accidental mutation in the training
  loop is a hard error. Use ``dataclasses.replace`` to derive variants.
- **Phase 1 has no algorithmic logic.** Models are placeholders that
  raise :class:`hymo.core.exceptions.NotImplementedError`. The point of
  Phase 1 is the *interfaces* — no premature implementation.
- **No circular dependencies.** The dependency graph is strictly:
  ``core ← registry ← utils ← {models, training, data, eval}``.
  ``models`` and ``training`` do not import each other directly; they
  share state through the config and the registry.

## Subagent routing

- "Build the GDN kernel" → see design §2.3 / roadmap B1
- "Wire the FSDP-2 wrapper" → see design §13 / roadmap D1-D7
- "Add a new data source" → see roadmap A3 + ``hymo.data.sources``
- "Write a config for an ablation" → see design §16 / roadmap F1-F4

## Testing rules (MANDATORY — hardcoded, do not deviate)

The production v1.0 model is 1.86 B params (full model construct in
float32 ≈ 7.4 GB, more than a dev laptop has). A test must **never**
build it by default. Follow this exact style for every test:

1. **Default tests must be CPU-friendly and cool.** Each test runs on
   CPU, in well under a minute, and never heats the machine. The entire
   default ``pytest`` run finishes in ~1 minute on an M1 Air with zero
   heavy tests executed.
2. **Never call ``ModelConfig()`` expecting the full model.** In
   ``tests/unit/test_models.py`` a bare ``ModelConfig()`` is shadowed at
   module scope to return the *tiny* config (~760 K params). Everywhere
   else, build models from the tiny fixture
   (``tiny_hymo_model`` / ``tiny_hymo_config`` in ``tests/conftest.py``),
   NOT from ``configs/hymo_750m.yaml``.
3. **Heavy tests are opt-in only.** Any test that builds the production
   model (or the full 1.86B graph) MUST be marked ``@pytest.mark.heavy``.
   ``tests/conftest.py::pytest_collection_modifyitems`` auto-skips these
   in the default run; they run only with ``pytest --run-heavy`` (CI /
   GPU pod). Do not remove the marker, and do not add assertions that
   require the full model outside a heavy test.
4. **Verify behavior, not scale.** Tests check stability, shapes,
   numerical finiteness, pipeline wiring, optimizer partition, and
   architectural invariants — using the tiny config. Production-scale
   arithmetic (e.g. exact expert counts = 384, param totals) lives
   behind ``heavy`` and/or reads config values via
   ``production_config_only`` (which loads the YAML *without* building a
   model).
5. **Derive expected counts from config, don't hardcode.** When a test
   asserts a count that depends on a tiny-config knob (routed experts,
   layers, etc.), compute it from ``model.config`` so the test survives
   tiny-config tweaks — never hardcode the production number.
6. **Gates stay green.** ``mypy --strict src/hymo`` and ``ruff`` must
   pass. The default ``pytest`` (no ``--run-heavy``) must pass with all
   heavy tests skipped.

## Hard don'ts

- Don't add hyperparameters to code — extend ``hymo.core.config``.
- Don't add new dependencies without updating ``pyproject.toml`` and
  the README.
- Don't import torch from ``hymo.core`` — that subpackage must stay
  PyTorch-free.
- Don't implement model logic in Phase 1 — keep placeholders.
- Don't build the full 1.86B model in a default (non-``heavy``) test.
- Don't hardcode production-scale numbers into tiny-config tests.
