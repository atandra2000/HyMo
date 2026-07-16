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

## Hard don'ts

- Don't add hyperparameters to code — extend ``hymo.core.config``.
- Don't add new dependencies without updating ``pyproject.toml`` and
  the README.
- Don't import torch from ``hymo.core`` — that subpackage must stay
  PyTorch-free.
- Don't implement model logic in Phase 1 — keep placeholders.
