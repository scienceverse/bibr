"""Second opinion for the front matter: re-ask a call the registry record disagrees with.

With ``LLM_REVIEW_TRIGGER=registry_disagree`` and a review model
(``LLM_REVIEW_MODEL``), the core extractor compares the model's title and
author list with the paper's own confident registry record, looked up early
under the core call (:mod:`bibr.pipeline.registry_check`). Only a field that
disagrees with the record (or came back empty while the record has it) is
asked again, on the review model, with the same front-matter text and the
same call. The review's answer is used only when it is printed in the front
matter and either agrees with the record or the first answer was not
printed; otherwise the first answer stays. The answer used then runs through
the unchanged grounding chain. The record never supplies a value: it only
decides which of two readings of the page to keep.

Titles are compared loosely for routing: as in
:mod:`bibr.enrich.verify_identity`, and also equal once spaces are dropped,
so a spacing or subtitle difference does not cost a call. On two development
sets (252 papers), a simulation of this routing that took a 27B model's
answer only when it agreed with the record lost no paper: a 4B extraction
model went from 216 to 219 papers with title, authors and the rest right,
and a 2B one from 207 to 216, re-asking one or two calls on 8-12% of the
papers. Multilingual papers are over half of the routed ones, because the
registry often holds a parallel English title, so a review title that is a
translation printed after the first one is refused even when it agrees with
the record.

``extraction.diagnostics.front_matter_review`` counts the outcomes. The
review's token usage is recorded under its own labels (``review_`` prefixed)
and the review model in ``extraction.qualification``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.models import PaperAuthor

logger = logging.getLogger(__name__)

# Label prefix of the review calls' usage rows: the same call on another model.
REVIEW_LABEL_PREFIX = "review_"

TITLE = "title"
AUTHOR = "author"


def review_wanted(settings: GlobalSettings) -> bool:
    """Whether the review is switched on (it runs only with a model configured)."""
    return getattr(getattr(settings, "llm", None), "review_trigger", "off") == "registry_disagree"


def review_configured(settings: GlobalSettings) -> bool:
    """Whether the review is on and has a model to ask."""
    return review_wanted(settings) and bool(settings.llm.review_model)


def review_settings(settings: GlobalSettings) -> GlobalSettings:
    """*settings* with the core LLM switched to the review model (``LLM_REVIEW_*``).

    Without ``LLM_REVIEW_PROVIDER`` the review runs on the core provider, and
    without ``LLM_REVIEW_BASE_URL`` too on its endpoint and key. An explicit
    provider or base URL is another endpoint: ``LLM_API_KEY`` is not sent
    there, and the provider's own key applies unless ``LLM_REVIEW_API_KEY`` is
    set. The structured backend is Instructor and the core model's pins
    (revision, template hash, merged call) are dropped.
    """
    llm = settings.llm
    same_endpoint = llm.review_provider is None and llm.review_base_url is None
    if llm.review_base_url is not None:
        base_url = llm.review_base_url
    else:
        base_url = llm.base_url if llm.review_provider is None else None
    review_llm = llm.model_copy(
        update={
            "provider": llm.review_provider or llm.provider,
            "model": llm.review_model,
            "base_url": base_url,
            "api_key": llm.review_api_key
            if llm.review_api_key is not None
            else (llm.api_key if same_endpoint else None),
            "max_concurrency": llm.review_max_concurrency,
            "timeout_seconds": llm.review_timeout_seconds,
            "structured_backend": "instructor",
            "model_revision": None,
            "jinja_sha256": None,
            "merged_core_metadata": False,
        }
    )
    return settings.model_copy(update={"llm": review_llm})


def build_review_client(settings: GlobalSettings):
    """An ``LLMClient`` for the review model; None when the review is not configured."""
    if not review_configured(settings):
        return None
    from bibr.clients.llm import LLMClient

    return LLMClient(settings=review_settings(settings))


def review_label(label: str) -> str:
    return label if label.startswith(REVIEW_LABEL_PREFIX) else REVIEW_LABEL_PREFIX + label


def pop_review_usage(client, key: str) -> dict[str, Any]:
    """Pop the review client's usage, labels, hashes and trace for *key*, relabelled.

    Returned only when the client tracks usage; the buckets are evicted either
    way so a shared client stays bounded.
    """
    popped = {
        "usage": client.usage_pop_file(key) if hasattr(client, "usage_pop_file") else {},
        "labels": (
            client.usage_labels_pop_file(key) if hasattr(client, "usage_labels_pop_file") else {}
        ),
        "hashes": (
            client.protocol_hashes_pop_file(key)
            if hasattr(client, "protocol_hashes_pop_file")
            else {}
        ),
        "trace": client.traces_pop_file(key) if hasattr(client, "traces_pop_file") else [],
    }
    if not getattr(client, "_track_usage", False):
        return {"usage": {}, "labels": {}, "hashes": {}, "trace": []}
    popped["labels"] = {
        (review_label(label), provider, model): counts
        for (label, provider, model), counts in popped["labels"].items()
    }
    popped["hashes"] = {review_label(label): hashes for label, hashes in popped["hashes"].items()}
    popped["trace"] = [
        {**row, "label": review_label(row["label"])}
        if isinstance(row, dict) and isinstance(row.get("label"), str)
        else row
        for row in popped["trace"]
    ]
    return popped


def merge_counts(into: dict, extra: Mapping) -> None:
    """Add *extra*'s ``{key: {counter: n}}`` buckets to *into*."""
    for key, counts in extra.items():
        bucket = into.setdefault(key, {})
        for name, value in counts.items():
            if isinstance(value, int) and not isinstance(value, bool):
                bucket[name] = bucket.get(name, 0) + value
            else:
                bucket.setdefault(name, value)


@dataclass
class ReviewOutcome:
    """What the review did to the core call's answer."""

    llm_metadata: Any
    # Fields whose review answer was used, and of them the ones chosen because
    # they agree with the record (undone when the final DOI is another one).
    accepted: set[str] = field(default_factory=set)
    record_backed: set[str] = field(default_factory=set)
    receipt: dict | None = None


def _title_key(text: str | None) -> str:
    from bibr.enrich.verify_identity import _title_key as key

    return key(text or "")


def titles_agree_loosely(title: str, record: Mapping[str, Any]) -> bool:
    """*title* agrees with the record's title (alone or with its subtitle),
    also when only spaces differ ("1 H-MRS" and "1H-MRS")."""
    from bibr.enrich.verify_identity import _record_titles, titles_agree

    for record_title in _record_titles(dict(record)):
        if titles_agree(title, record_title):
            return True
        squeezed = _title_key(title).replace(" ", "")
        if squeezed and squeezed == _title_key(record_title).replace(" ", ""):
            return True
    return False


def later_language_version(first: str, other: str, printed_text: str) -> bool:
    """Whether *other* is a language version of *first*, printed after it in
    *printed_text* (the printed-first rule of ``drop_parallel_title``)."""
    from bibr.extract.title_subtitle import _are_language_versions, drop_parallel_title

    if not first.strip() or not _are_language_versions(first, other):
        return False
    kept, _ = drop_parallel_title(f"{first} / {other}", printed_text)
    return kept.strip() != other.strip()


def route(
    record: Mapping[str, Any],
    *,
    title: str,
    authors: Sequence[PaperAuthor],
) -> dict[str, str]:
    """The fields that disagree with *record*, with why: ``{field: reason}``."""
    from bibr.enrich.verify_identity import _record_titles, authors_agree
    from bibr.pipeline.registry_check import author_rows

    routed: dict[str, str] = {}
    if _record_titles(dict(record)):
        if not title.strip():
            routed[TITLE] = "title_missing"
        elif not titles_agree_loosely(title, record):
            routed[TITLE] = "title_disagrees"
    record_authors = [row for row in record.get("author") or [] if isinstance(row, dict)]
    if record_authors:
        rows = author_rows(authors)
        if not rows:
            routed[AUTHOR] = "authors_missing"
        elif not authors_agree(rows, record_authors):
            routed[AUTHOR] = "authors_disagree"
    return routed


async def review_front_matter(
    llm_metadata: Any,
    *,
    record: Mapping[str, Any],
    client: Any,
    settings: GlobalSettings,
    file_hash: str,
    title_text: str,
    authors_text: str,
    failed_fields: Sequence[str],
    convert_authors: Callable[[Any], list[PaperAuthor]],
    title_printed: Callable[[str], bool],
    authors_printed: Callable[[list[PaperAuthor]], bool],
) -> ReviewOutcome:
    """Re-ask the core calls whose answer disagrees with *record*; keep a better answer.

    *convert_authors* turns the model's author rows into ``PaperAuthor``s the
    way the extractor does; *title_printed* and *authors_printed* say whether
    a value is printed in the front matter the call read. A review title that
    is a translation printed after a printed first title is refused, even when
    the record holds it. Never raises for a failed review; the caller's
    cancellation propagates.
    """
    from bibr.enrich.verify_identity import authors_agree
    from bibr.input.consolidate_text import strip_affiliation_markers
    from bibr.pipeline.registry_check import author_rows

    primary_title = strip_affiliation_markers(llm_metadata.title or "").strip()
    primary_authors = convert_authors(llm_metadata.authors)
    routed = route(record, title=primary_title, authors=primary_authors)
    receipt = {
        "model": settings.llm.review_model,
        "status": "configured" if client is not None else "not_configured",
        "routed": len(routed),
        "attempted": 0,
        "accepted": 0,
        "rejected": 0,
        "failed": 0,
        "skipped": 0,
        "reasons": dict(Counter(routed.values())),
    }
    outcome = ReviewOutcome(llm_metadata, receipt=receipt)
    failed = set(failed_fields)
    asked: list[str] = []
    for name in routed:
        call_failed = (name == TITLE and "title" in failed) or (
            name == AUTHOR and "authors" in failed
        )
        if client is None or call_failed:
            receipt["skipped"] += 1
        else:
            asked.append(name)
    if not asked:
        return outcome

    timeout = float(settings.llm.review_timeout_seconds)

    async def ask(name: str):
        if name == TITLE:
            call = client.extract_title_keywords(title_text, file_hash=file_hash)
        else:
            call = client.extract_authors(authors_text, file_hash=file_hash)
        return await asyncio.wait_for(call, timeout=timeout)

    results = await asyncio.gather(*(ask(name) for name in asked), return_exceptions=True)
    record_authors = [row for row in record.get("author") or [] if isinstance(row, dict)]
    update: dict[str, Any] = {}
    for name, result in zip(asked, results, strict=True):
        receipt["attempted"] += 1
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
        if isinstance(result, Exception):
            logger.info("Front-matter review of the %s failed: %s", name, result)
            receipt["failed"] += 1
            continue
        if name == TITLE:
            answer = strip_affiliation_markers(getattr(result, "title", None) or "").strip()
            agrees = titles_agree_loosely(answer, record)
            answer_key, primary_key = _title_key(answer), _title_key(primary_title)
            primary_printed = bool(primary_title) and title_printed(primary_title)
            usable = (
                bool(answer_key)
                and answer_key != primary_key
                # A shorter answer the first one extends drops a printed part.
                and not primary_key.startswith(answer_key + " ")
                and title_printed(answer)
                and (agrees or not primary_printed)
                # The record often holds a later-printed English version.
                and not (
                    primary_printed and later_language_version(primary_title, answer, title_text)
                )
            )
            if usable:
                update["title"] = answer
        else:
            raw = list(getattr(result, "authors", None) or [])
            answer_authors = convert_authors(raw)
            rows = author_rows(answer_authors)
            agrees = bool(rows) and authors_agree(rows, record_authors)
            usable = (
                bool(answer_authors)
                and rows != author_rows(primary_authors)
                and authors_printed(answer_authors)
                and (agrees or not (primary_authors and authors_printed(primary_authors)))
            )
            if usable:
                update["authors"] = raw
        if usable:
            receipt["accepted"] += 1
            outcome.accepted.add(name)
            if agrees:
                outcome.record_backed.add(name)
        else:
            receipt["rejected"] += 1
    if update:
        outcome.llm_metadata = llm_metadata.model_copy(update=update)
    logger.info(
        "Front-matter review: %d routed, %d attempted, %d accepted, %d rejected, %d failed",
        receipt["routed"],
        receipt["attempted"],
        receipt["accepted"],
        receipt["rejected"],
        receipt["failed"],
    )
    return outcome


__all__ = [
    "REVIEW_LABEL_PREFIX",
    "ReviewOutcome",
    "build_review_client",
    "later_language_version",
    "merge_counts",
    "pop_review_usage",
    "review_configured",
    "review_front_matter",
    "review_label",
    "review_settings",
    "review_wanted",
    "route",
    "titles_agree_loosely",
]
