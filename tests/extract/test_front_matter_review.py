"""Second opinion for the front matter (``bibr.extract.front_matter_review``).

With ``LLM_REVIEW_TRIGGER=registry_disagree`` a title or author call whose
answer disagrees with the paper's own registry record is asked again on the
review model. The review's answer is kept only when it is printed and agrees
with the record (or the first answer was not printed); the record never
supplies a value.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from bibr.config import GlobalSettings
from bibr.export.models import FrontMatterReviewExport
from bibr.extract.front_matter_review import (
    build_review_client,
    merge_counts,
    pop_review_usage,
    review_configured,
    review_front_matter,
    review_settings,
    review_wanted,
    route,
    titles_agree_loosely,
)
from bibr.models import PaperAuthor
from bibr.schemas import AuthorLLM, AuthorsLLM, CoreMetadataLLM, TitleKeywordsLLM

TITLE = "Sleep and memory in older adults"
DOI = "10.1234/self.1"
ADA = ("Ada", "Lovelace")
ALAN = ("Alan", "Turing")
RECORD = {
    "service": "crossref",
    "score": 1.0,
    "doi": DOI,
    "title": TITLE,
    "subtitle": None,
    "author": [{"given": g, "family": f} for g, f in (ADA, ALAN)],
}


def _settings(**llm) -> GlobalSettings:
    settings = GlobalSettings()
    settings.llm.review_trigger = "registry_disagree"
    settings.llm.review_model = "review-27b"
    for name, value in llm.items():
        setattr(settings.llm, name, value)
    return settings


def _authors(*names) -> list[AuthorLLM]:
    return [AuthorLLM(given=given, family=family) for given, family in names]


def _convert(rows) -> list[PaperAuthor]:
    return [
        PaperAuthor(author_id=index, given=row.given, family=row.family, affiliation="")
        for index, row in enumerate(rows, start=1)
    ]


def _primary(title: str | None = TITLE, authors=(ADA, ALAN)) -> CoreMetadataLLM:
    return CoreMetadataLLM(title=title, authors=_authors(*authors))


class FakeReview:
    """The review model: canned answers, a failure, or a call that never returns."""

    def __init__(self, *, title=None, authors=(), error=None, hang=False, track=True):
        self.title = title
        self.authors = authors
        self.error = error
        self.hang = hang
        self._track_usage = track
        self.calls: list[tuple[str, str, str]] = []

    async def _answer(self, value):
        if self.hang:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return value

    async def extract_title_keywords(self, text, file_hash="unknown", *, boundary=None):
        self.calls.append(("title", text, file_hash))
        return await self._answer(TitleKeywordsLLM(title=self.title))

    async def extract_authors(self, text, file_hash="unknown", *, json_mode=False, boundary=None):
        self.calls.append(("authors", text, file_hash))
        return await self._answer(AuthorsLLM(authors=_authors(*self.authors)))


async def _review(
    primary,
    client,
    *,
    printed_titles=(TITLE,),
    printed_names=("Lovelace", "Turing"),
    failed=(),
    settings=None,
    record=RECORD,
    title_text="TITLE TEXT",
    **guards,
):
    outcome = await review_front_matter(
        primary,
        record=record,
        client=client,
        settings=settings or _settings(),
        file_hash="h",
        title_text=title_text,
        authors_text="AUTHORS TEXT",
        failed_fields=failed,
        convert_authors=_convert,
        title_printed=lambda title: title in printed_titles,
        authors_printed=lambda authors: all(a.family in printed_names for a in authors),
        **guards,
    )
    # Every receipt reconciles under the export model.
    FrontMatterReviewExport.model_validate(outcome.receipt)
    return outcome


def _counts(receipt) -> tuple[int, ...]:
    return tuple(
        receipt[name]
        for name in ("routed", "attempted", "accepted", "rejected", "failed", "skipped")
    )


# --- settings ------------------------------------------------------------------


def test_review_is_off_by_default():
    settings = GlobalSettings()
    assert settings.llm.review_trigger == "off"
    assert not review_wanted(settings)
    assert build_review_client(settings) is None


def test_review_needs_a_model():
    settings = _settings(review_model=None)
    assert review_wanted(settings)
    assert not review_configured(settings)
    assert build_review_client(settings) is None


def test_review_settings_inherit_the_core_endpoint():
    settings = _settings(provider="openai", base_url="http://core:8000/v1", api_key="core-key")
    review = review_settings(settings).llm
    assert (review.provider, review.model, review.base_url) == (
        "openai",
        "review-27b",
        "http://core:8000/v1",
    )
    assert review.api_key == "core-key"
    assert review.structured_backend == "instructor"
    assert review.merged_core_metadata is False
    assert review.model_revision is None and review.jinja_sha256 is None
    assert review.max_concurrency == settings.llm.review_max_concurrency
    assert review.timeout_seconds == settings.llm.review_timeout_seconds
    # The core settings are untouched.
    assert settings.llm.model != "review-27b"


def test_review_settings_never_send_the_core_key_to_another_endpoint():
    settings = _settings(
        provider="openai",
        base_url="http://core:8000/v1",
        api_key="core-key",
        review_base_url="http://review:8000/v1",
    )
    review = review_settings(settings).llm
    assert review.base_url == "http://review:8000/v1"
    assert review.api_key is None

    settings.llm.review_base_url = None
    settings.llm.review_provider = "google"
    review = review_settings(settings).llm
    assert (review.provider, review.base_url, review.api_key) == ("google", None, None)

    settings.llm.review_api_key = "review-key"
    assert review_settings(settings).llm.api_key == "review-key"


async def test_build_review_client_runs_on_the_review_model():
    client = build_review_client(_settings(provider="openai", base_url="http://core:8000/v1"))
    try:
        assert client._settings.llm.model == "review-27b"
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("name", "a", "b"),
    [
        ("LLM_REVIEW_MODEL", "review-a", "review-b"),
        ("LLM_REVIEW_TRIGGER", "off", "registry_disagree"),
        ("FRONT_MATTER_REGISTRY_TITLE", "false", "true"),
    ],
)
def test_review_and_title_rule_settings_change_the_settings_digest(monkeypatch, name, a, b):
    from bibr.config import compute_behavior_fingerprint

    monkeypatch.setenv(name, a)
    first = compute_behavior_fingerprint(GlobalSettings())
    monkeypatch.setenv(name, b)
    assert compute_behavior_fingerprint(GlobalSettings()) != first


def test_review_api_key_stays_out_of_the_settings_digest(monkeypatch):
    from bibr.config import compute_behavior_fingerprint

    monkeypatch.setenv("LLM_REVIEW_API_KEY", "one")
    first = compute_behavior_fingerprint(GlobalSettings())
    monkeypatch.setenv("LLM_REVIEW_API_KEY", "two")
    assert compute_behavior_fingerprint(GlobalSettings()) == first
    assert "two" not in repr(GlobalSettings().llm)


# --- routing -------------------------------------------------------------------


def test_agreeing_fields_are_not_routed():
    assert route(RECORD, title=TITLE, authors=_convert(_authors(ADA, ALAN))) == {}


def test_spacing_and_subtitle_differences_are_not_routed():
    record = {**RECORD, "title": "Brain 1H-MRS in ageing", "subtitle": "a cohort study"}
    assert titles_agree_loosely("Brain 1 H-MRS in ageing", record)
    assert titles_agree_loosely("Brain 1H-MRS in ageing: A cohort study", record)
    assert not titles_agree_loosely("Sleep and memory", record)


@pytest.mark.parametrize(
    ("title", "authors", "expected"),
    [
        ("", (ADA, ALAN), {"title": "title_missing"}),
        ("Die Wirkung von Schlaf", (ADA, ALAN), {"title": "title_disagrees"}),
        (TITLE, (), {"author": "authors_missing"}),
        (TITLE, (ADA,), {"author": "authors_disagree"}),
    ],
)
def test_disagreeing_fields_are_routed_with_their_reason(title, authors, expected):
    assert route(RECORD, title=title, authors=_convert(_authors(*authors))) == expected


def test_a_record_without_the_field_routes_nothing():
    record = {**RECORD, "title": None, "author": []}
    assert route(record, title="", authors=[]) == {}


# --- the review ----------------------------------------------------------------


async def test_review_agreeing_with_the_record_is_kept():
    client = FakeReview(title=TITLE)
    outcome = await _review(
        _primary("Sleep in older adults"),
        client,
        printed_titles=(TITLE, "Sleep in older adults"),
    )
    # Only the disagreeing call is asked again, with the same text.
    assert client.calls == [("title", "TITLE TEXT", "h")]
    assert outcome.llm_metadata.title == TITLE
    assert outcome.accepted == {"title"} and outcome.record_backed == {"title"}
    assert _counts(outcome.receipt) == (1, 1, 1, 0, 0, 0)
    assert outcome.receipt["reasons"] == {"title_disagrees": 1}
    assert outcome.receipt["model"] == "review-27b"
    assert outcome.receipt["status"] == "configured"


async def test_printed_review_answer_replaces_an_unprinted_first_answer():
    printed = "Schlaf und Gedächtnis im Alter"
    outcome = await _review(
        _primary("Sleep & memory of the old"),
        FakeReview(title=printed),
        printed_titles=(printed,),
        # The rows print no version of it before it.
        later_version=lambda answer, first: False,
    )
    assert outcome.llm_metadata.title == printed
    assert outcome.accepted == {"title"}
    # Chosen because it is printed, not because the record agrees.
    assert outcome.record_backed == set()


async def test_without_rows_a_version_the_text_cannot_place_is_refused():
    """The default guard compares with the first answer in the text the call
    read; a first answer the text does not print cannot show which version is
    printed first, so the review's other-language title is refused."""
    primary = _primary("Sleep & memory of the old")
    outcome = await _review(
        primary,
        FakeReview(title="Schlaf und Gedächtnis im Alter"),
        printed_titles=("Schlaf und Gedächtnis im Alter",),
        title_text="Schlaf und Gedächtnis im Alter\nAda Lovelace, Alan Turing",
    )
    assert outcome.llm_metadata is primary
    assert outcome.receipt["rejected"] == 1


async def test_agreeing_review_over_an_unprinted_first_answer_is_not_record_backed():
    """The record did not decide: the first answer was not printed."""
    outcome = await _review(
        _primary("Sleep in olders adult"), FakeReview(title=TITLE), printed_titles=(TITLE,)
    )
    assert outcome.llm_metadata.title == TITLE
    assert outcome.accepted == {"title"} and outcome.record_backed == set()


async def test_first_answer_counts_as_printed_once_grounded():
    """A first answer that drops a printed character grounds to its printed row,
    so a review that disagrees with the record cannot replace it."""
    printed = "Sleep in older adults"
    primary = _primary("Sleep in older adult")
    outcome = await _review(
        primary,
        FakeReview(title="Memory in older adults"),
        printed_titles=(printed, "Memory in older adults"),
        ground_title=lambda title: printed if title == "Sleep in older adult" else title,
    )
    assert outcome.llm_metadata is primary
    assert outcome.receipt["rejected"] == 1


async def test_review_answer_grounding_to_the_first_one_is_rejected():
    outcome = await _review(
        _primary("Sleep in older adults"),
        FakeReview(title="Sleep in older adult"),
        printed_titles=("Sleep in older adults",),
        ground_title=lambda title: "Sleep in older adults",
    )
    assert outcome.accepted == set() and outcome.receipt["rejected"] == 1


async def test_a_later_version_is_refused_whatever_the_first_answer():
    """The rows say the review title is printed after another version: refused,
    even though the first answer is not printed and the record agrees."""
    primary = _primary("Wirkung des Schlafs bei alteren Erwachsenen")
    outcome = await _review(
        primary,
        FakeReview(title=ENGLISH),
        printed_titles=(ENGLISH,),
        record={**RECORD, "title": ENGLISH},
        later_version=lambda answer, first: answer == ENGLISH,
    )
    assert outcome.llm_metadata is primary
    assert outcome.receipt["rejected"] == 1


async def test_unprinted_review_answer_is_rejected():
    primary = _primary("Sleep in older adults")
    outcome = await _review(
        primary,
        FakeReview(title=TITLE),
        printed_titles=("Sleep in older adults",),
    )
    assert outcome.llm_metadata is primary
    assert outcome.accepted == set()
    assert _counts(outcome.receipt) == (1, 1, 0, 1, 0, 0)


async def test_review_disagreeing_with_the_record_keeps_a_printed_first_answer():
    primary = _primary("Sleep in older adults")
    outcome = await _review(
        primary,
        FakeReview(title="Memory in older adults"),
        printed_titles=("Sleep in older adults", "Memory in older adults"),
    )
    assert outcome.llm_metadata is primary
    assert outcome.receipt["rejected"] == 1


async def test_shortened_review_title_is_rejected():
    full = "Sleep and memory: evidence from a long cohort of older adults in rural areas"
    record = {**RECORD, "title": "Sleep and memory"}
    outcome = await _review(
        _primary(full),
        FakeReview(title="Sleep and memory"),
        printed_titles=(full, "Sleep and memory"),
        record=record,
    )
    assert outcome.llm_metadata.title == full
    assert outcome.receipt["rejected"] == 1


GERMAN = "Die Wirkung von Schlaf auf das Gedächtnis bei älteren Erwachsenen"
ENGLISH = "The effect of sleep on memory in older adults"
BILINGUAL = f"{GERMAN}\n{ENGLISH}\nAda Lovelace, Alan Turing"


async def test_record_backed_translation_printed_second_is_rejected():
    primary = _primary(GERMAN)
    outcome = await _review(
        primary,
        FakeReview(title=ENGLISH),
        printed_titles=(GERMAN, ENGLISH),
        record={**RECORD, "title": ENGLISH},
        title_text=BILINGUAL,
    )
    assert outcome.llm_metadata is primary
    assert outcome.receipt["rejected"] == 1


async def test_record_backed_version_printed_first_replaces_a_later_one():
    outcome = await _review(
        _primary(ENGLISH),
        FakeReview(title=GERMAN),
        printed_titles=(GERMAN, ENGLISH),
        record={**RECORD, "title": GERMAN},
        title_text=BILINGUAL,
    )
    assert outcome.llm_metadata.title == GERMAN
    assert outcome.record_backed == {"title"}


@pytest.mark.parametrize(
    "client",
    [FakeReview(error=RuntimeError("503")), FakeReview(error=TimeoutError())],
    ids=["error", "timeout_error"],
)
async def test_failed_review_keeps_the_first_answer(client):
    primary = _primary("Sleep in older adults")
    outcome = await _review(primary, client, printed_titles=("Sleep in older adults",))
    assert outcome.llm_metadata is primary
    assert _counts(outcome.receipt) == (1, 1, 0, 0, 1, 0)


async def test_review_is_bounded_by_its_timeout():
    settings = _settings()
    settings.llm.__dict__["review_timeout_seconds"] = 0.05  # below the setting's floor
    client = FakeReview(title=TITLE, hang=True)
    outcome = await asyncio.wait_for(
        _review(_primary("Sleep in older adults"), client, settings=settings), timeout=5
    )
    assert outcome.receipt["failed"] == 1
    assert outcome.accepted == set()


async def test_without_a_review_model_routed_calls_are_skipped():
    primary = _primary("Sleep in older adults", authors=(ADA,))
    outcome = await _review(primary, None)
    assert outcome.llm_metadata is primary
    assert outcome.receipt["status"] == "not_configured"
    assert _counts(outcome.receipt) == (2, 0, 0, 0, 0, 2)
    assert outcome.receipt["reasons"] == {"title_disagrees": 1, "authors_disagree": 1}


async def test_a_call_that_failed_the_first_time_is_not_asked_again():
    client = FakeReview(title=TITLE, authors=(ADA, ALAN))
    outcome = await _review(_primary("", authors=(ADA,)), client, failed=("title",))
    assert [call[0] for call in client.calls] == ["authors"]
    assert outcome.receipt["skipped"] == 1
    assert outcome.accepted == {"author"}


async def test_author_review_agreeing_with_the_record_is_kept():
    client = FakeReview(authors=(ADA, ALAN))
    outcome = await _review(_primary(authors=(ADA,)), client)
    assert client.calls == [("authors", "AUTHORS TEXT", "h")]
    assert [a.family for a in outcome.llm_metadata.authors] == ["Lovelace", "Turing"]
    assert outcome.record_backed == {"author"}


async def test_author_review_over_an_unprinted_first_list_is_not_record_backed():
    outcome = await _review(
        _primary(authors=(("Ada", "Lovelance"),)),
        FakeReview(authors=(ADA, ALAN)),
    )
    assert [a.family for a in outcome.llm_metadata.authors] == ["Lovelace", "Turing"]
    assert outcome.accepted == {"author"} and outcome.record_backed == set()


async def test_unprinted_author_review_is_rejected():
    primary = _primary(authors=(ADA,))
    outcome = await _review(primary, FakeReview(authors=(ADA, ALAN)), printed_names=("Lovelace",))
    assert outcome.llm_metadata is primary
    assert outcome.receipt["rejected"] == 1


async def test_cancellation_propagates():
    with pytest.raises(asyncio.CancelledError):
        await _review(_primary("Sleep in older adults"), FakeReview(error=asyncio.CancelledError()))


# --- receipt -------------------------------------------------------------------


def _receipt(**changes) -> dict:
    receipt = {
        "model": "review-27b",
        "status": "configured",
        "routed": 2,
        "attempted": 2,
        "accepted": 1,
        "rejected": 0,
        "failed": 1,
        "skipped": 0,
        "reasons": {"title_disagrees": 1, "authors_missing": 1},
    }
    return {**receipt, **changes}


def test_receipt_validates_when_it_reconciles():
    FrontMatterReviewExport.model_validate(_receipt())


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted": 2},
        {"skipped": 1},
        {"reasons": {"title_disagrees": 1}},
        {"status": "maybe"},
        {"extra": 1},
    ],
    ids=["outcomes", "routed", "reasons", "status", "extra"],
)
def test_receipt_that_does_not_reconcile_is_refused(changes):
    with pytest.raises(ValidationError):
        FrontMatterReviewExport.model_validate(_receipt(**changes))


# --- usage ---------------------------------------------------------------------


class UsageClient:
    """A client's per-file usage buckets, as ``LLMClient`` keeps them."""

    def __init__(self, label: str, model: str, *, track: bool = True):
        self._track_usage = track
        self.usage = {"k": {model: {"attempts": 1, "input_tokens": 10}}}
        self.labels = {"k": {(label, "openai", model): {"attempts": 1, "logical_calls": 1}}}
        self.hashes = {"k": {label: {"prompt_sha256": "a" * 64}}}
        self.trace = {"k": [{"label": label, "model": model}]}

    def usage_pop_file(self, key):
        return self.usage.pop(key, {})

    def usage_labels_pop_file(self, key):
        return self.labels.pop(key, {})

    def protocol_hashes_pop_file(self, key):
        return self.hashes.pop(key, {})

    def traces_pop_file(self, key):
        return self.trace.pop(key, [])

    async def close(self):
        pass


def test_review_usage_is_relabelled():
    client = UsageClient("title_keywords", "review-27b")
    popped = pop_review_usage(client, "k")
    assert popped["labels"] == {
        ("review_title_keywords", "openai", "review-27b"): {"attempts": 1, "logical_calls": 1}
    }
    assert set(popped["hashes"]) == {"review_title_keywords"}
    assert popped["trace"] == [{"label": "review_title_keywords", "model": "review-27b"}]
    assert popped["usage"] == {"review-27b": {"attempts": 1, "input_tokens": 10}}
    assert client.labels == {}


def test_untracked_review_usage_is_evicted_and_not_reported():
    client = UsageClient("authors", "review-27b", track=False)
    assert pop_review_usage(client, "k") == {"usage": {}, "labels": {}, "hashes": {}, "trace": []}
    assert client.usage == {} and client.labels == {} and client.trace == {}


def test_merge_counts_adds_counters():
    into = {"m": {"attempts": 1, "name": "a"}}
    merge_counts(into, {"m": {"attempts": 2, "name": "b"}, "n": {"attempts": 1}})
    assert into == {"m": {"attempts": 3, "name": "a"}, "n": {"attempts": 1}}


async def test_post_parse_records_review_usage_and_model(monkeypatch):
    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.post_parse import post_parse
    from tests.pipeline.test_registry_check import LOOKUP, OWN_DOI, _contents, _found

    core = UsageClient("title_keywords", "core-4b")
    review = UsageClient("title_keywords", "review-27b")
    seen = {}

    async def extract(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        from bibr.clients.llm import _usage_file_hash

        key = _usage_file_hash.get()
        seen["review_client"] = kwargs.get("review_client")
        for client in (core, review):
            for bucket in ("usage", "labels", "hashes", "trace"):
                store = getattr(client, bucket)
                store[key] = store.pop("k")
        return PaperMetadata(doi="", title="A printed title", published="2021")

    settings = _settings()
    settings.llm.model = "core-4b"
    pp = "bibr.pipeline.stages.post_parse"
    monkeypatch.setattr(
        f"{pp}._build_qualification_provenance",
        lambda *_a, usage_by_label, **_k: {"model_id": "core-4b", "labels": usage_by_label},
    )
    with (
        mock.patch(f"{pp}._classify_sections", mock.AsyncMock(return_value=None)),
        mock.patch("bibr.structure.implicit_sections.detect_implicit_sections", mock.AsyncMock()),
        mock.patch(f"{pp}._extract_metadata_and_equations", extract),
        mock.patch(f"{pp}._link_citations", mock.AsyncMock(return_value=None)),
        mock.patch(
            "bibr.extract.research_integrity.extract_structured_integrity", mock.AsyncMock()
        ),
        mock.patch(LOOKUP, _found()),
    ):
        paper = await post_parse(
            contents=_contents(OWN_DOI),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=core,
            settings=settings,
            registry_lookup=True,
            review_client=review,
        )

    assert seen["review_client"] is review
    assert set(paper.llm_usage_labels) == {
        ("title_keywords", "openai", "core-4b"),
        ("review_title_keywords", "openai", "review-27b"),
    }
    assert [row["label"] for row in paper.llm_trace] == ["title_keywords", "review_title_keywords"]
    assert paper.qualification_provenance["review_model_id"] == "review-27b"
    assert set(paper.qualification_provenance["labels"]) == {
        "title_keywords",
        "review_title_keywords",
    }


async def test_post_parse_builds_and_closes_its_own_review_client(monkeypatch):
    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.post_parse import post_parse
    from tests.pipeline.test_registry_check import LOOKUP, OWN_DOI, _contents, _found

    built = UsageClient("authors", "review-27b")
    built.close = mock.AsyncMock()
    seen = {}

    async def extract(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        seen["review_client"] = kwargs.get("review_client")
        return PaperMetadata(doi="", title="A printed title")

    pp = "bibr.pipeline.stages.post_parse"
    monkeypatch.setattr("bibr.extract.front_matter_review.build_review_client", lambda _s: built)
    with (
        mock.patch(f"{pp}._classify_sections", mock.AsyncMock(return_value=None)),
        mock.patch("bibr.structure.implicit_sections.detect_implicit_sections", mock.AsyncMock()),
        mock.patch(f"{pp}._extract_metadata_and_equations", extract),
        mock.patch(f"{pp}._link_citations", mock.AsyncMock(return_value=None)),
        mock.patch(
            "bibr.extract.research_integrity.extract_structured_integrity", mock.AsyncMock()
        ),
        mock.patch(LOOKUP, _found()),
    ):
        await post_parse(
            contents=_contents(OWN_DOI),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=None,
            settings=_settings(),
            registry_lookup=True,
        )
    assert seen["review_client"] is built
    built.close.assert_awaited_once()


async def test_resources_build_and_close_the_review_client(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    client = mock.MagicMock()
    client.close = mock.AsyncMock()
    monkeypatch.setattr("bibr.extract.front_matter_review.build_review_client", lambda _s: client)
    resources = ResourceManager(settings=_settings())
    assert resources.review_llm_client is client
    assert resources.review_llm_client is client  # built once
    await resources.close_llm_client()
    client.close.assert_awaited_once()
    assert resources._review_llm_client is None


def test_resources_have_no_review_client_by_default():
    from bibr.pipeline.resources import ResourceManager

    assert ResourceManager(settings=GlobalSettings()).review_llm_client is None


# --- the core extractor --------------------------------------------------------


class SettledCheck:
    """A finished early lookup that found *record*."""

    def __init__(self, record):
        self._record = record
        self.record_backed: set[str] = set()
        self.review_receipt = None
        self.primary_title = None
        self.settled_with = "unset"

    async def settle(self, published):
        self.settled_with = published

    def record(self):
        return self._record


def _extractor(monkeypatch, *, first_title: str, review: FakeReview, settings=None):
    from bibr.config import snapshot_settings
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from tests.extract.test_core_metadata_front_matter import (
        _candidate,
        _paper_contents,
        _resolution,
    )

    rows = [(1, TITLE, 0), (2, "Ada Lovelace and Alan Turing", 0)]
    title = _candidate("c1", TITLE, roles=frozenset({"title"}), text_ids=(1,))
    byline = _candidate("c2", rows[1][1], roles=frozenset({"byline"}), text_ids=(2,))
    resolution = _resolution(title, byline)
    contents = _paper_contents([*rows, (99, "Body begins", 1)])
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(title=first_title, authors=_authors(ADA, ALAN))
    )
    if settings is None:
        settings = snapshot_settings()
        settings.llm.review_trigger = "registry_disagree"
        settings.llm.review_model = "review-27b"
    settings.llm.per_task_context = True
    settings.llm.merged_core_metadata = False
    check = SettledCheck(RECORD)
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        settings=settings,
        front_matter_resolution=resolution,
        registry_check=check,
        review_client=review,
    )
    monkeypatch.setattr(
        extractor, "_classify_paper", mock.AsyncMock(return_value=("", "", "", None, None))
    )
    return extractor, check, contents


async def test_core_extractor_keeps_an_agreeing_review_title(monkeypatch):
    from bibr.extract.field_decisions import field_decisions_of

    review = FakeReview(title=TITLE)
    # The first answer is a heading of the printed title, not the record's title.
    extractor, check, contents = _extractor(
        monkeypatch, first_title="Sleep and memory", review=review
    )
    metadata = await extractor.extract()

    assert [call[0] for call in review.calls] == ["title"]
    assert metadata.title == TITLE
    proposal = field_decisions_of(metadata).proposal("title")
    assert proposal.source == "llm_review"
    assert check.record_backed == {"title"}
    assert check.primary_title.value == "Sleep and memory"
    assert contents.front_matter_review_receipt == check.review_receipt
    assert check.review_receipt["accepted"] == 1


async def test_core_extractor_without_disagreement_asks_nothing(monkeypatch):
    from bibr.extract.field_decisions import field_decisions_of

    review = FakeReview(title="never asked")
    extractor, check, contents = _extractor(monkeypatch, first_title=TITLE, review=review)
    metadata = await extractor.extract()

    assert review.calls == []
    assert field_decisions_of(metadata).proposal("title").source == "llm"
    assert contents.front_matter_review_receipt["routed"] == 0
    assert check.primary_title is None


async def test_core_extractor_without_the_trigger_writes_no_receipt(monkeypatch):
    from bibr.config import snapshot_settings

    review = FakeReview(title=TITLE)
    extractor, _check, contents = _extractor(
        monkeypatch, first_title="Sleep and memory", review=review, settings=snapshot_settings()
    )
    metadata = await extractor.extract()
    assert review.calls == []
    assert metadata.title == "Sleep and memory"
    assert contents.front_matter_review_receipt is None


# --- the printed-first rule over the selected record's rows ----------------------

GERMAN_MAIN = "Die Wirkung von Schlaf auf das Gedächtnis"
GERMAN_SUB = "eine Studie bei älteren Erwachsenen in der Pflege"
ENGLISH_FULL = "The effect of sleep on memory: a study in older adults in care"


def _title_rows(*texts, roles=frozenset({"title"})):
    from tests.extract.test_core_metadata_front_matter import _candidate, _resolution

    rows = [_candidate(f"c{index}", text, roles=roles) for index, text in enumerate(texts, 1)]
    return _resolution(*rows)


def test_a_title_over_two_rows_is_printed():
    from bibr.extract.title_candidates import grounded_in_selected_title, printed_title_rows

    resolution = _title_rows(GERMAN_MAIN, GERMAN_SUB, ENGLISH_FULL)
    joined = f"{GERMAN_MAIN}: {GERMAN_SUB}"
    assert not grounded_in_selected_title(joined, resolution)
    assert [row.candidate_id for row in printed_title_rows(joined, resolution)] == ["c1", "c2"]
    assert [row.candidate_id for row in printed_title_rows(ENGLISH_FULL, resolution)] == ["c3"]
    assert printed_title_rows("A title the page never prints", resolution) is None


@pytest.mark.parametrize(
    ("texts", "title", "later"),
    [
        # The original over two rows, its translation on a later row.
        ((GERMAN_MAIN, GERMAN_SUB, ENGLISH_FULL), ENGLISH_FULL, True),
        ((GERMAN_MAIN, GERMAN_SUB, ENGLISH_FULL), f"{GERMAN_MAIN}: {GERMAN_SUB}", False),
        ((GERMAN, ENGLISH), ENGLISH, True),
        ((ENGLISH, GERMAN), ENGLISH, False),
        ((ENGLISH, GERMAN), GERMAN, True),
        # Both versions on one row.
        ((f"{GERMAN} / {ENGLISH}",), ENGLISH, True),
        ((f"{GERMAN} / {ENGLISH}",), GERMAN, False),
        # Same language: not a version.
        (("Sleep and memory in shift workers", ENGLISH), ENGLISH, False),
        # Not printed: nothing to say.
        ((GERMAN,), ENGLISH, False),
    ],
)
def test_a_version_printed_after_another_is_found(texts, title, later):
    from bibr.extract.title_candidates import printed_after_language_version

    assert printed_after_language_version(title, _title_rows(*texts)) is later


DOI_RECORD = {**RECORD, "title": ENGLISH}


def _bilingual_extractor(monkeypatch, rows, *, first_title, review_title):
    """The core extractor over a front matter of *rows* (title rows, then a byline)."""
    from bibr.config import snapshot_settings
    from bibr.extract.core_metadata import CoreMetadataExtractor
    from tests.extract.test_core_metadata_front_matter import (
        _candidate,
        _paper_contents,
        _resolution,
    )

    byline = "Ada Lovelace and Alan Turing"
    candidates = [
        _candidate(f"c{index}", text, roles=frozenset({"title"}), text_ids=(index,))
        for index, text in enumerate(rows, 1)
    ]
    candidates.append(
        _candidate(
            f"c{len(rows) + 1}", byline, roles=frozenset({"byline"}), text_ids=(len(rows) + 1,)
        )
    )
    contents = _paper_contents(
        [*((index, text, 0) for index, text in enumerate([*rows, byline], 1)), (99, "Body", 1)]
    )
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(title=first_title, authors=_authors(ADA, ALAN))
    )
    settings = snapshot_settings()
    settings.llm.review_trigger = "registry_disagree"
    settings.llm.review_model = "review-27b"
    settings.llm.per_task_context = True
    settings.llm.merged_core_metadata = False
    check = SettledCheck(DOI_RECORD)
    review = FakeReview(title=review_title)
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm,
        settings=settings,
        front_matter_resolution=_resolution(*candidates),
        registry_check=check,
        review_client=review,
    )
    monkeypatch.setattr(
        extractor, "_classify_paper", mock.AsyncMock(return_value=("", "", "", None, None))
    )
    return extractor, check, review


async def test_review_keeps_a_near_copy_of_the_original_over_the_records_translation(
    monkeypatch,
):
    """The first answer drops a diacritic of the German title printed first; the
    record holds the English title printed second. The review's English answer
    is refused and the grounded German title stays."""
    extractor, check, review = _bilingual_extractor(
        monkeypatch,
        [GERMAN, ENGLISH],
        first_title=GERMAN.replace("älteren", "alteren"),
        review_title=ENGLISH,
    )
    metadata = await extractor.extract()

    assert [call[0] for call in review.calls] == ["title"]
    assert metadata.title == GERMAN
    assert check.review_receipt["rejected"] == 1
    assert check.record_backed == set()


async def test_review_keeps_an_original_printed_over_two_rows(monkeypatch):
    """The German title is printed as a main title and a subtitle row, the
    English one on a later row that the record holds."""
    first = f"{GERMAN_MAIN}: {GERMAN_SUB}"
    extractor, check, _review = _bilingual_extractor(
        monkeypatch,
        [GERMAN_MAIN, GERMAN_SUB, ENGLISH_FULL],
        first_title=first,
        review_title=ENGLISH_FULL,
    )
    check._record = {**DOI_RECORD, "title": ENGLISH_FULL}
    metadata = await extractor.extract()

    assert metadata.title == first
    assert check.review_receipt["rejected"] == 1


async def test_review_takes_the_original_printed_first(monkeypatch):
    """The English title is printed first and the record holds it; the first
    answer took the German translation printed after it."""
    extractor, check, _review = _bilingual_extractor(
        monkeypatch, [ENGLISH, GERMAN], first_title=GERMAN, review_title=ENGLISH
    )
    metadata = await extractor.extract()

    assert metadata.title == ENGLISH
    assert check.review_receipt["accepted"] == 1
    assert check.record_backed == {"title"}
    assert check.primary_title.value == GERMAN


# --- export --------------------------------------------------------------------


def test_title_kept_from_an_agreeing_review_is_reported_corrected():
    from bibr.enrich.verify_identity import verify_identity

    payload = {
        "metadata": {"title": TITLE, "doi": DOI},
        "author": [{"given": "Ada", "family": "Lovelace", "literal": None}],
        "metadata_match": [{**RECORD, "author": RECORD["author"][:1]}],
        "extraction": {
            "fields": {
                "title": {"state": "extracted", "source": "llm_review", "issues": []},
                "author": {"state": "extracted", "source": "llm_review", "issues": []},
            }
        },
    }
    assert verify_identity(payload) == {"title": "corrected", "author": "corrected"}


def test_a_value_whose_record_was_dropped_is_not_reported_corrected():
    """After the final DOI turned out to be another one, the kept review list
    is compared with the final record like any extracted value."""
    from bibr.enrich.verify_identity import verify_identity

    dropped = ["VAL_REGISTRY_DOI_CHANGED"]
    payload = {
        "metadata": {"title": TITLE, "doi": DOI},
        "author": [{"given": "Ada", "family": "Lovelace", "literal": None}],
        "metadata_match": [{**RECORD, "author": RECORD["author"][:1]}],
        "extraction": {
            "fields": {
                "title": {"state": "extracted", "source": "llm", "issues": dropped},
                "author": {"state": "extracted", "source": "llm_review", "issues": dropped},
            }
        },
    }
    assert verify_identity(payload) == {"title": "agree", "author": "agree"}


def test_dropped_record_issue_qualifies_the_title_and_the_author_list():
    from bibr.field_states import FieldScope, build_field_states
    from bibr.validation import IssueSeverity, ValidationIssue

    issue = ValidationIssue("VAL_REGISTRY_DOI_CHANGED", IssueSeverity.WARNING, "changed")
    records = build_field_states(
        present={"title": True, "author": True, "abstract": True},
        sources={"title": "llm", "author": "llm_review", "abstract": "llm"},
        scope=FieldScope(),
        issues=[issue],
        warnings=[],
    )
    assert records["title"].issues == ("VAL_REGISTRY_DOI_CHANGED",)
    assert records["author"].issues == ("VAL_REGISTRY_DOI_CHANGED",)
    assert records["abstract"].issues == ()


def test_receipt_reaches_the_export_diagnostics():
    from bibr.export.json_export import export_paper_to_json
    from bibr.input.file import InputFile
    from bibr.models import PaperMetadata
    from bibr.paper import Paper
    from tests.pipeline.test_registry_check import _contents

    contents = _contents()
    contents.front_matter_review_receipt = _receipt()
    paper = Paper(
        input_file=InputFile(path=Path("a.txt")),
        metadata=PaperMetadata(doi="", title="A printed title"),
        contents=contents,
    )
    data = export_paper_to_json(paper)
    assert data["extraction"]["diagnostics"]["front_matter_review"] == _receipt()
