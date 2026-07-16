"""Generic named-constructor registry.

Usage
-----
.. code-block:: python

    from hymo.utils.registry import Registry

    MODELS = Registry("model")

    @MODELS.register("hymo")
    class HyMo(nn.Module):
        ...

    # Later:
    model = MODELS.build("hymo", config=config)

A registry is a thin wrapper over a dict. The :meth:`register` method
is the primary entry point; :meth:`build` is the primary reader. Both
support namespacing with ``/`` (e.g. ``"optimizer/nor_muon"``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hymo.core.exceptions import HyMoError

T = TypeVar("T")


class RegistryError(HyMoError):
    """A registry operation failed (duplicate name, missing name, etc.)."""


class Registry:
    """A name → constructor mapping.

    Constructors are typically classes that take a config as their first
    argument, but any callable works.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._constructors: dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers a constructor under ``name``.

        Example::

            @MODELS.register("hymo")
            class HyMo(nn.Module):
                ...
        """
        if not name:
            raise RegistryError(f"{self._name}: cannot register empty name")
        if name in self._constructors:
            raise RegistryError(
                f"{self._name}: '{name}' is already registered "
                f"with {self._constructors[name].__qualname__}"
            )

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            self._constructors[name] = fn
            return fn

        return decorator

    def register_fn(self, name: str, fn: Callable[..., T]) -> Callable[..., T]:
        """Imperative registration (use when a decorator is awkward)."""
        if not name:
            raise RegistryError(f"{self._name}: cannot register empty name")
        if name in self._constructors:
            raise RegistryError(
                f"{self._name}: '{name}' is already registered"
            )
        self._constructors[name] = fn
        return fn

    def get(self, name: str) -> Callable[..., Any]:
        """Return the constructor registered under ``name``.

        Raises
        ------
        RegistryError
            If no constructor is registered.
        """
        try:
            return self._constructors[name]
        except KeyError as e:
            raise RegistryError(
                f"{self._name}: no constructor named {name!r}; "
                f"available: {sorted(self._constructors)}"
            ) from e

    def has(self, name: str) -> bool:
        """Return True iff ``name`` is registered."""
        return name in self._constructors

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Build an instance by calling the registered constructor.

        Equivalent to ``self.get(name)(*args, **kwargs)``.
        """
        return self.get(name)(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._constructors

    def __iter__(self) -> Any:
        return iter(self._constructors)

    def __len__(self) -> int:
        return len(self._constructors)

    def __repr__(self) -> str:
        n = len(self._constructors)
        return f"Registry({self._name!r}, {n} entries)"

    def keys(self) -> list[str]:
        return sorted(self._constructors)

    def values(self) -> list[Callable[..., Any]]:
        return [self._constructors[k] for k in self.keys()]


__all__ = ["Registry", "RegistryError"]
