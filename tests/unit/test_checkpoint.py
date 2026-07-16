"""Tests for the :mod:`hymo.utils.atomic_io` module (atomic file write)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hymo.utils.atomic_io import (
    atomic_write_bytes,
    atomic_write_with,
)


class TestAtomicWriteBytes:
    def test_writes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.bin"
        atomic_write_bytes(path, b"hello world")
        assert path.read_bytes() == b"hello world"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "out.bin"
        atomic_write_bytes(path, b"x")
        assert path.exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "out.bin"
        atomic_write_bytes(path, b"first")
        atomic_write_bytes(path, b"second")
        assert path.read_bytes() == b"second"

    def test_no_tmp_file_left_on_success(self, tmp_path: Path) -> None:
        path = tmp_path / "out.bin"
        atomic_write_bytes(path, b"data")
        # The .tmp file should not be left behind.
        assert not (tmp_path / "out.bin.tmp").exists()


class TestAtomicWriteWith:
    def test_writes_via_writer(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"

        def writer(tmp: Path) -> None:
            tmp.write_text("written by writer")

        atomic_write_with(path, writer)
        assert path.read_text() == "written by writer"

    def test_writer_exception_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"

        def bad_writer(tmp: Path) -> None:
            tmp.write_text("partial")
            raise RuntimeError("writer failed")

        with pytest.raises(RuntimeError):
            atomic_write_with(path, bad_writer)
        # Path should not exist; tmp should be cleaned up.
        assert not path.exists()
