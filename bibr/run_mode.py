"""Fast and full run modes: which one a run takes, and what fast switches off.

Fast mode is the normal pipeline with every LLM call skipped (the front-matter
tagger reads the title, authors and abstract); full mode is the pipeline with
the configured LLM. ``PIPELINE_MODE`` (``auto``, ``fast`` or ``full``) and the
``--fast`` / ``--full`` flags choose, and ``auto`` follows what is configured.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, cast

from bibr.config import GlobalSettings, snapshot_settings
from bibr.exceptions import InputValidationError

if TYPE_CHECKING:
    from bibr.pipeline.context import RunMode

logger = logging.getLogger(__name__)

# Reference-parse strategies that call an LLM. A per-run ``--refs`` or
# ``--ref-seg`` that names one is refused by the entry points; only the
# setting is swapped here.
_LLM_PARSE_STRATEGIES = frozenset({"llm", "llm-chunked", "ref-llm"})


def llm_configured(settings: GlobalSettings, backend: str | None = None) -> bool:
    """Whether an LLM is set up to be called, judged from the settings alone.

    A managed local backend always counts. For the cloud backend the provider
    needs what its adapter asks for: an API key (``LLM_API_KEY`` or the
    provider's own), or for ``openai`` a ``LLM_BASE_URL``. Ollama and adapters
    registered by other packages count as configured.

    This does not build a client. The credential check (``preflight_credentials``)
    also raises for a key that would cross the internet in clear text, and
    reading that as "no LLM" would quietly turn a broken configuration into a
    fast run. Here a half-set-up provider is configured, so the run is full and
    the same error stops it, as it does today.

    ``backend`` is the run's ``--llm`` value; ``None`` uses ``LLM_BACKEND``.
    """
    if (backend or settings.llm.backend) != "cloud":
        return True
    llm = settings.llm
    provider = llm.provider.lower()
    if provider == "google":
        return bool(llm.api_key or settings.GOOGLE_API_KEY)
    if provider == "openai":
        return bool(llm.api_key or llm.base_url)
    if provider == "anthropic":
        return bool(llm.api_key or settings.ANTHROPIC_API_KEY)
    if provider == "groq":
        return bool(llm.api_key or settings.GROQ_API_KEY)
    return True


def resolve_mode(
    settings: GlobalSettings, explicit: str | None = None, backend: str | None = None
) -> RunMode:
    """The mode one run takes: ``explicit`` (the flag), then ``PIPELINE_MODE``, then auto.

    Auto is fast when :func:`llm_configured` is false and full when it is true.
    An explicit ``full`` with no LLM is not changed here; the caller's
    credential check reports it and suggests ``--fast``.
    """
    chosen = explicit if explicit is not None else settings.pipeline.mode
    if chosen == "auto":
        return "full" if llm_configured(settings, backend) else "fast"
    if chosen not in ("fast", "full"):
        raise InputValidationError(f"Unknown run mode {chosen!r}. Valid values: auto, fast, full.")
    return cast("RunMode", chosen)


def fast_settings(settings: GlobalSettings) -> GlobalSettings:
    """A snapshot of ``settings`` with every LLM path turned off, for a fast run.

    The front-matter tagger answers alone (``front_tagger_mode="only"``), the
    section and paper classifiers keep their label instead of escalating (the
    section threshold then follows its no-escalation default of 0.5, unless set),
    and implicit-section detection, the reference segmenter and parser fallbacks,
    and the merged core-metadata call are off. A setting-level reference strategy
    that names the LLM becomes the local one. ``settings`` itself is not changed.
    """
    fast = snapshot_settings(settings)
    fast.ml.front_tagger_mode = "only"
    fast.ml.section_classifier_llm_escalation = False
    fast.ml.paper_classifier_llm_escalation = False
    fast.IMPLICIT_SECTION_DETECTION = False
    fast.REF_SEG_LLM_FALLBACK = False
    fast.REF_NER_LLM_FALLBACK = False
    fast.llm.merged_core_metadata = False
    if fast.REF_SEG_STRATEGY == "llm":
        _log_local_strategy("REF_SEG_STRATEGY", "llm", "geom")
        fast.REF_SEG_STRATEGY = "geom"
    if fast.REF_PARSE_STRATEGY in _LLM_PARSE_STRATEGIES:
        _log_local_strategy("REF_PARSE_STRATEGY", fast.REF_PARSE_STRATEGY, "ner")
        fast.REF_PARSE_STRATEGY = "ner"
    return fast


@functools.cache
def _log_local_strategy(name: str, asked: str, used: str) -> None:
    # Once per process and value: the run makes the same swap every time.
    logger.info("Fast mode makes no LLM call, so %s=%s is run as %s", name, asked, used)
