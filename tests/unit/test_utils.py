"""Tests for the :mod:`hymo.utils.logging` and :mod:`hymo.utils.metrics` modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hymo.utils.logging import MetricsLogger, get_logger
from hymo.utils.metrics import Metric, MetricCollection


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


class TestMetric:
    def test_mean_reduce(self) -> None:
        m = Metric("loss", reduce="mean")
        m.update(10.0)
        m.update(20.0)
        assert m.value() == 15.0

    def test_sum_reduce(self) -> None:
        m = Metric("n", reduce="sum")
        m.update(1)
        m.update(2)
        m.update(3)
        assert m.value() == 6

    def test_last_reduce(self) -> None:
        m = Metric("lr", reduce="last")
        m.update(0.001)
        m.update(0.002)
        m.update(0.003)
        assert m.value() == 0.003

    def test_max_reduce(self) -> None:
        m = Metric("x", reduce="max")
        m.update(1)
        m.update(5)
        m.update(3)
        assert m.value() == 5

    def test_min_reduce(self) -> None:
        m = Metric("x", reduce="min")
        m.update(5)
        m.update(1)
        m.update(3)
        assert m.value() == 1

    def test_empty_metric_returns_zero(self) -> None:
        assert Metric("x").value() == 0.0

    def test_reset(self) -> None:
        m = Metric("loss", reduce="mean")
        m.update(10.0)
        m.update(20.0)
        m.reset()
        assert m.value() == 0.0

    def test_unknown_reduce_raises(self) -> None:
        m = Metric("x", reduce="bogus")
        m.update(1.0)
        with pytest.raises(ValueError):
            m.value()


class TestMetricCollection:
    def test_add_and_update(self) -> None:
        c = MetricCollection()
        c.add("loss", reduce="mean")
        c.add("grad_norm", reduce="last")
        c.update("loss", 11.06)
        c.update("grad_norm", 5.2)
        assert c.value("loss") == 11.06
        assert c.value("grad_norm") == 5.2

    def test_add_duplicate_raises(self) -> None:
        c = MetricCollection()
        c.add("loss")
        with pytest.raises(ValueError):
            c.add("loss")

    def test_update_unknown_raises(self) -> None:
        c = MetricCollection()
        with pytest.raises(KeyError):
            c.update("nope", 1.0)

    def test_as_dict(self) -> None:
        c = MetricCollection()
        c.add("a", reduce="last")
        c.add("b", reduce="last")
        c.update("a", 1.0)
        c.update("b", 2.0)
        d = c.as_dict()
        assert d == {"a": 1.0, "b": 2.0}

    def test_reset_all(self) -> None:
        c = MetricCollection()
        c.add("a")
        c.add("b")
        c.update("a", 1.0)
        c.update("b", 2.0)
        c.reset()
        assert c.value("a") == 0.0
        assert c.value("b") == 0.0

    def test_reset_subset(self) -> None:
        c = MetricCollection()
        c.add("a")
        c.add("b")
        c.update("a", 1.0)
        c.update("b", 2.0)
        c.reset(names=["a"])
        assert c.value("a") == 0.0
        assert c.value("b") == 2.0

    def test_contains_and_len(self) -> None:
        c = MetricCollection()
        c.add("a")
        c.add("b")
        assert "a" in c
        assert "nope" not in c
        assert len(c) == 2
