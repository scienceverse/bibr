"""Stateless factory registry for layout detectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

Factory = Callable[..., Any]
F = TypeVar("F", bound=Factory)

_BACKENDS: dict[str, Factory] = {}


def register(factory: F) -> F:
    """Register a layout factory under its non-empty class-level ``name``."""
    name = getattr(factory, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("layout backend must define a non-empty name")
    if name in _BACKENDS and _BACKENDS[name] is not factory:
        raise ValueError(f"layout backend {name!r} is already registered")
    _BACKENDS[name] = factory
    return factory


def create(name: str, **kwargs: Any) -> Any:
    """Create a fresh layout detector from a registered factory."""
    try:
        factory = _BACKENDS[name]
    except KeyError as exc:
        known = ", ".join(known_backends())
        raise ValueError(f"unknown layout backend {name!r}; known: {known}") from exc
    return factory(**kwargs)


def known_backends() -> list[str]:
    """Return registered layout backend names in deterministic order."""
    return cast(list[str], sorted(_BACKENDS))
