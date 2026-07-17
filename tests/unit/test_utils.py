"""Tests for the :mod:`hymo.utils.logging` and :mod:`hymo.utils.metrics` modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hymo.utils import MetricsLogger, get_logger


class TestGetLogger:
    def test_returns_logger(self) -> None:
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)
        assert log.name == "hymo.test_module"

    def test_root_logger(self) -> None:
        log = get_logger()
        assert log.name == "hymo"


class TestMetricsLogger:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as m:
            m.log(step=0, loss=11.06, lr=0.0)
            m.log(step=1, loss=10.92, lr=1e-5)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert rec0["step"] == 0
        assert rec0["metrics"]["loss"] == pytest.approx(11.06)
        assert rec0["metrics"]["lr"] == 0.0

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as m:
            m.log(step=0, loss=11.06)
        with MetricsLogger(path) as m:
            m.log(step=1, loss=10.92)

        recs = list(MetricsLogger(path).iter_records())
        # The first MetricsLogger instance appended; the second opened for "a"ppend.
        # iter_records reads from start.
        assert len(recs) >= 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "metrics.jsonl"
        with MetricsLogger(path) as m:
            m.log(step=0, loss=11.06)
        assert path.exists()

    def test_iter_records(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as m:
            m.log(step=0, loss=11.06)
            m.log(step=1, loss=10.92)
            m.log(step=2, loss=10.81)

        recs = list(MetricsLogger(path).iter_records())
        assert [r.step for r in recs] == [0, 1, 2]
        assert [r.metrics["loss"] for r in recs] == pytest.approx([11.06, 10.92, 10.81])

    def test_last_step(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as m:
            m.log(step=0, loss=11.06)
            m.log(step=5, loss=10.0)
        assert MetricsLogger(path).last_step() == 5
