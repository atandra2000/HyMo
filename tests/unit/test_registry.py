"""Tests for the :mod:`hymo.utils.registry` module."""

from __future__ import annotations

import pytest

from hymo.utils.registry import Registry, RegistryError


def make_registry() -> Registry:
    return Registry("test")


class TestRegistry:
    def test_empty_registry(self) -> None:
        r = make_registry()
        assert len(r) == 0
        assert "foo" not in r

    def test_register_decorator(self) -> None:
        r = make_registry()

        @r.register("foo")
        def foo_fn() -> str:
            return "foo"

        assert r.has("foo")
        assert r.get("foo") is foo_fn
        assert r.build("foo") == "foo"

    def test_register_fn_imperative(self) -> None:
        r = make_registry()

        def bar_fn() -> str:
            return "bar"

        r.register_fn("bar", bar_fn)
        assert r.has("bar")
        assert r.build("bar") == "bar"

    def test_duplicate_registration_raises(self) -> None:
        r = make_registry()

        @r.register("dup")
        def a() -> None:
            pass

        with pytest.raises(RegistryError):

            @r.register("dup")
            def b() -> None:
                pass

    def test_empty_name_raises(self) -> None:
        r = make_registry()
        with pytest.raises(RegistryError):

            @r.register("")
            def f() -> None:
                pass

    def test_missing_name_raises(self) -> None:
        r = make_registry()
        with pytest.raises(RegistryError):
            r.get("nope")

    def test_build_with_args(self) -> None:
        r = make_registry()

        @r.register("add")
        def add(a: int, b: int) -> int:
            return a + b

        assert r.build("add", 2, 3) == 5

    def test_iteration(self) -> None:
        r = make_registry()

        @r.register("a")
        def a() -> None:
            pass

        @r.register("b")
        def b() -> None:
            pass

        assert set(r) == {"a", "b"}
        assert len(r) == 2

    def test_keys_sorted(self) -> None:
        r = make_registry()

        for name in ("z", "a", "m"):
            r.register_fn(name, lambda: None)

        assert r.keys() == ["a", "m", "z"]

    def test_repr(self) -> None:
        r = make_registry()

        @r.register("x")
        def x() -> None:
            pass

        assert "test" in repr(r)
        assert "1 entries" in repr(r)
