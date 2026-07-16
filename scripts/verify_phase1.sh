#!/usr/bin/env bash
# verify_phase1.sh — Verify the HyMo Phase 1 foundation.
#
# This script runs:
#   1. Install the package in editable mode (with dev extras).
#   2. Run the full test suite (~280 tests).
#   3. Run mypy in strict mode.
#   4. Run ruff lint.
#   5. Run a smoke import that exercises the public API end-to-end.
#
# Exit 0 = Phase 1 complete. Any non-zero exit = a failure listed below.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=================================================================="
echo "HyMo Phase 1 — Foundation verification"
echo "=================================================================="
echo

# 1. Install.
echo "[1/5] pip install -e .[dev]"
pip install -e ".[dev]"
echo

# 2. Smoke import (before tests, so any import error surfaces first).
echo "[2/5] Smoke import — public API end-to-end"
PYTHONPATH=src python3 -c "
import hymo
from hymo.core import HyMoConfig, ModelConfig, validate_full_config
from hymo.models import HyMo, build_hymo
from hymo.training import build_optimizers, partition_parameters
from hymo.utils import MetricsLogger, ProjectPaths
from hymo.eval import format_comparison_table, BASELINES

c = HyMoConfig()
validate_full_config(c)
model = build_hymo(c)
opts = build_optimizers(model, c.optimizer)
partition = partition_parameters(model)
print(f'  Model: {len(model.layers)} layers')
print(f'  Optimizer: muon={opts.nor_muon is not None} adamw={opts.adamw is not None}')
print(f'  Partition: {len(partition.adamw)} AdamW, {len(partition.nor_muon)} NorMuon')
print(f'  Baselines: {list(BASELINES)}')
print('  Foundation OK')
"
echo

# 3. Tests.
echo "[3/5] pytest tests/ -v"
pytest tests/ -v --tb=short
echo

# 4. Mypy.
echo "[4/5] mypy src/hymo (strict mode)"
mypy src/hymo
echo

# 5. Ruff.
echo "[5/5] ruff check src/hymo tests"
ruff check src/hymo tests
echo

echo "=================================================================="
echo "Phase 1 verification complete — all checks passed."
echo "=================================================================="
