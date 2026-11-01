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
    from collections.abc import Sequence

    from bibr.pipeline.context import RunMode

logger = logging.getLogger(__name__)

#: What a fast run leaves undone, for the one-line summaries (``bibr chew``, ``bibr doctor``,
#: ``--dry-run``). It mirrors :func:`fast_settings`: the local models answer alone.
FAST_SKIPS = "implicit sections and every LLM fallback for metadata, sections and references"

# Reference-parse strategies that call an LLM. A per-run ``--refs`` or
# ``--ref-seg`` that names one is refused by the entry points; only the
# setting is swapped here.
LLM_PARSE_STRATEGIES = frozenset({"llm", "llm-chunked", "ref-llm"})

# The LLM settings whose being set says the user chose an LLM.
_LLM_CHOICES = frozenset({"provider", "model", "base_url"})


def llm_configured(settings: GlobalSettings, backend: str | None = None) -> bool:
    """Whether an LLM is set up to be called, judged from the settings alone.

    A managed local backend always counts. For the cloud backend, so does naming
    a provider, a model or an endpoint (``LLM_PROVIDER``, ``LLM_MODEL``,
    ``LLM_BASE_URL``, from the environment, a ``.env`` file or an assignment):
    whoever did meant to call an LLM, and a missing key or a mistyped key
    variable then stops the run at the credential check, instead of quietly
    turning it into a fast one. Naming none of them leaves the default provider
    (Google), which is configured once its key is set (``LLM_API_KEY`` or
    ``GOOGLE_API_KEY``).

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
    # Not the whole set: ``rate_limit_rpm`` and others are filled in on their own.
    if llm.model_fields_set & _LLM_CHOICES:
        return True
    return bool(llm.api_key or settings.GOOGLE_API_KEY)


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


def resolve_run_mode(
    settings: GlobalSettings,
    *,
    explicit: str | None = None,
    llm_backend: str | None = None,
    llm_flags: Sequence[str] = (),
    no_llm: bool = False,
    fast_label: str = "--fast",
) -> RunMode:
    """The mode a user-facing entry point (``bibr chew``, ``bibr batch``, ``bibr.chew``) takes.

    :func:`resolve_mode` plus the rules about what the user asked for:

    * ``no_llm`` is a structure-only run. It stays ``full``: its LLM stages are
      skipped by ``no_llm`` itself, and it never takes the fast overlay.
    * ``llm_flags`` names the options given that only make sense with an LLM
      (``--llm``, ``--refs llm``, ``--ref-seg llm``). Under auto they are asking
      for an LLM, so the run is ``full`` and the credential check reports a missing
      key. In a fast run they are refused: they would make the call the mode
      promises not to make. ``fast_label`` names how fast was chosen when it was
      explicit.
    """
    if no_llm:
        return "full"
    if explicit == "auto":
        explicit = None
    if explicit is None and llm_flags and settings.pipeline.mode == "auto":
        return "full"
    mode = resolve_mode(settings, explicit, backend=llm_backend)
    if mode == "fast" and llm_flags:
        origin = fast_label if explicit == "fast" else f"PIPELINE_MODE={settings.pipeline.mode}"
        raise ValueError(
            f"{', '.join(llm_flags)} calls an LLM, which fast mode ({origin}) never does. "
            "Run in full mode to use the LLM, or drop the option."
        )
    return mode


def fast_settings(settings: GlobalSettings) -> GlobalSettings:
    """A snapshot of ``settings`` with every LLM path turned off, for a fast run.

    The front-matter tagger answers alone (``front_tagger_mode="only"``), the
    section and paper classifiers keep their label instead of escalating (the
    section threshold then follows its no-escalation default of 0.5, unless set),
    and implicit-section detection, the reference segmenter and parser fallbacks,
    and the merged core-metadata call are off. The local rules that stand in for what
    an LLM would put right (front matter, statistic text, citation linking) are on.
    A setting-level reference strategy that names the LLM becomes the local one.
    ``settings`` itself is not changed.
    """
    fast = snapshot_settings(settings)
    fast.ml.front_tagger_mode = "only"
    fast.ml.section_classifier_llm_escalation = False
    fast.ml.paper_classifier_llm_escalation = False
    fast.IMPLICIT_SECTION_DETECTION = False
    fast.FRONT_TAGGER_RULES = True
    fast.EQUATION_REPAIR_GLITCHES = True
    fast.CITATION_RULE_TIER3 = True
    fast.REF_SEG_LLM_FALLBACK = False
    fast.REF_NER_LLM_FALLBACK = False
    fast.llm.merged_core_metadata = False
    if fast.REF_SEG_STRATEGY == "llm":
        _log_local_strategy("REF_SEG_STRATEGY", "llm", "geom")
        fast.REF_SEG_STRATEGY = "geom"
    if fast.REF_PARSE_STRATEGY in LLM_PARSE_STRATEGIES:
        _log_local_strategy("REF_PARSE_STRATEGY", fast.REF_PARSE_STRATEGY, "ner")
        fast.REF_PARSE_STRATEGY = "ner"
    return fast


@functools.cache
def _log_local_strategy(name: str, asked: str, used: str) -> None:
    # Once per process and value: the run makes the same swap every time.
    logger.info("Fast mode makes no LLM call, so %s=%s is run as %s", name, asked, used)
