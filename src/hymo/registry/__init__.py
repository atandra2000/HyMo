"""Public registry: named-constructor pattern for models, optimizers, etc."""

from __future__ import annotations
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

class Registry(dict[str, Callable[..., Any]]):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def register(self, name: str) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            self[name] = fn
            return fn
        return decorator

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self[name](*args, **kwargs)

    def has(self, name: str) -> bool:
        return name in self

MODELS = Registry("model")
OPTIMIZERS = Registry("optimizer")
SCHEDULERS = Registry("scheduler")
TOKENIZERS = Registry("tokenizer")
DATA_SOURCES = Registry("data_source")

__all__ = [
    "MODELS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "TOKENIZERS",
    "DATA_SOURCES",
    "Registry",
]
