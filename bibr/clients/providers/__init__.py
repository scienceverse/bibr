"""Registry of LLM provider adapters.

Each provider module imports ``register`` from this package and applies
it as a class decorator. ``get(name)`` returns an instantiated provider
ready for ``build_client()`` / ``call_kwargs(...)``.

Importing this package triggers registration of all bundled providers.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from bibr.clients.providers.base import LlmProvider

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

_PROVIDERS: dict[str, type[LlmProvider]] = {}


def register(cls: type[LlmProvider]) -> type[LlmProvider]:
    """Class decorator that registers a provider under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must declare a class-level `name` attribute")
    if name in _PROVIDERS and _PROVIDERS[name] is not cls:
        raise ValueError(f"LLM provider {name!r} already registered to {_PROVIDERS[name].__name__}")
    _PROVIDERS[name] = cls
    return cls


def get(name: str, settings: GlobalSettings | None = None) -> LlmProvider:
    """Return a fresh provider instance by name."""
    cls = _PROVIDERS.get(name)
    if cls is None:
        known = sorted(_PROVIDERS)
        raise ValueError(f"Unknown LLM provider: {name!r}. Known: {known}")
    if settings is None:
        return cls()

    # Bundled adapters accept instance settings. Keep registered third-party
    # adapters with legacy no-argument constructors working unchanged.
    try:
        parameters = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):
        return cls()
    accepts_settings = any(
        parameter.name == "settings" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    return cls(settings=settings) if accepts_settings else cls()


def known_providers() -> list[str]:
    """Return a sorted list of registered provider names (for diagnostics)."""
    return sorted(_PROVIDERS)


# Eager-import bundled providers so ``@register`` runs at package load. The
# inner imports are deferred until after the registry is defined to avoid
# circular imports.
from bibr.clients.providers import (  # noqa: E402, F401
    anthropic,
    google,
    groq,
    ollama,
    openai,
)

__all__ = ["LlmProvider", "get", "known_providers", "register"]
