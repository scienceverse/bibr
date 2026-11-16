"""The optional extras that carry the cloud provider SDKs, and how to say one is missing.

bibr's core install calls no cloud provider: a fast run, ``--no-llm``, a managed
local model server and any OpenAI-compatible endpoint (through ``instructor``,
whose own dependency is the ``openai`` SDK) need nothing more. The Google and
Anthropic SDKs come with extras:

- ``google``: ``google-genai`` (``LLM_PROVIDER=google``, the provider a run without
  a ``.env`` uses, and ``OCR_BACKEND=gemini``) and ``jsonref``, which Instructor's
  Gemini path imports to map response models to tool schemas;
- ``anthropic``: ``anthropic`` (``LLM_PROVIDER=anthropic``, ``OCR_BACKEND=anthropic``
  and the offline Message Batches client), also reached through the ``batch`` alias.

Every check here is a module lookup: nothing is imported, so the preflights that
use it stay cheap. Messages never say "key", so ``bibr doctor`` tells them apart
from a missing credential.
"""

from __future__ import annotations

import importlib.util
import sys

# extra -> (distribution names, import names), in the order they are reported.
_EXTRA_SDKS: dict[str, tuple[tuple[str, str], ...]] = {
    "google": (("google-genai", "google.genai"), ("jsonref", "jsonref")),
    "anthropic": (("anthropic", "anthropic"),),
}

# LLM_PROVIDER value -> extra.
_LLM_PROVIDER_EXTRAS = {"google": "google", "anthropic": "anthropic"}
# OCR_BACKEND value (the cloud vision backends) -> extra.
_OCR_BACKEND_EXTRAS = {"gemini": "google", "anthropic": "anthropic"}


def module_installed(name: str) -> bool:
    """Whether *name* can be imported, without importing it.

    ``find_spec("google.genai")`` imports the ``google`` namespace and raises
    when that is absent, rather than returning ``None``. A module already in
    ``sys.modules`` counts as installed, and a ``None`` entry there (which makes
    ``import`` fail) as missing.
    """
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_packages(extra: str) -> list[str]:
    """The distributions of *extra* that are not installed (empty when all are)."""
    return [dist for dist, module in _EXTRA_SDKS.get(extra, ()) if not module_installed(module)]


def install_hint(extra: str) -> str:
    """How to install *extra* in each kind of environment bibr is run from."""
    return (
        f'pip install "bibr[{extra}]" (in a uv project: uv add "bibr[{extra}]"; '
        f"in a bibr source checkout: uv sync --extra {extra})"
    )


def _reason(setting: str, extra: str) -> str | None:
    missing = missing_packages(extra)
    if not missing:
        return None
    packages = " and ".join(missing)
    noun = "package" if len(missing) == 1 else "packages"
    return (
        f"{setting} needs the {packages} {noun}, which the {extra!r} extra installs "
        f"and this environment lacks. Install it with: {install_hint(extra)}"
    )


def llm_provider_extra(provider: str | None) -> str | None:
    """The extra ``LLM_PROVIDER=provider`` needs, or ``None`` for a core provider."""
    return _LLM_PROVIDER_EXTRAS.get((provider or "").strip().lower())


def ocr_backend_extra(backend: str | None) -> str | None:
    """The extra ``OCR_BACKEND=backend`` needs, or ``None`` when it needs none."""
    return _OCR_BACKEND_EXTRAS.get((backend or "").strip().lower())


def missing_llm_sdk(provider: str | None) -> str | None:
    """Why ``LLM_PROVIDER=provider`` cannot build a client here, or ``None``."""
    extra = llm_provider_extra(provider)
    if extra is None:
        return None
    return _reason(f"LLM_PROVIDER={(provider or '').strip().lower()}", extra)


def missing_ocr_sdk(backend: str | None) -> str | None:
    """Why the cloud vision backend ``OCR_BACKEND=backend`` cannot start here, or ``None``."""
    extra = ocr_backend_extra(backend)
    if extra is None:
        return None
    return _reason(f"OCR_BACKEND={(backend or '').strip().lower()}", extra)
