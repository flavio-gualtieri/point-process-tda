# src/cloudforger/core/registry.py

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> class lookup shared by every pluggable pipeline component
    (point process, filtration, feature, method). Adding a new
    implementation is: write the class, decorate it with
    ``@some_registry.register("name")`` -- nothing else needs to be touched
    to make it selectable from a RunConfig."""

    def __init__(self, kind: str):
        self._kind = kind
        self._classes: dict[str, type[T]] = {}

    def register(self, *names: str):
        def deco(cls: type[T]) -> type[T]:
            for name in names:
                self._classes[name] = cls
            return cls
        return deco

    def get(self, name: str) -> type[T]:
        try:
            return self._classes[name]
        except KeyError:
            raise ValueError(
                f"Unknown {self._kind} '{name}'. Registered: {self.names()}"
            ) from None

    def build(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._classes)
