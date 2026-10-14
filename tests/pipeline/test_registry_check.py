"""The early lookup of the paper's own DOI (``bibr.pipeline.registry_check``).

When a front-matter rule reads the registry record, ``post_parse`` looks up
the DOI the parsed text names while the core metadata call runs. The lookup
is bounded, never fails the paper, is cancelled with it, spares the later
lookup of the same DOI, and is undone when the final DOI is another one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.config import GlobalSettings
from bibr.enrich.references import EnrichmentReport
from bibr.extract.field_decisions import (
    FieldCandidate,
    FieldDecision,
    apply_decision,
    field_decisions_of,
)
from bibr.input.file import InputFile
from bibr.models import BibAuthor, ExternalMatch, MatchSource, PaperMetadata
from bibr.paper import Paper
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.pipeline.enricher import CrossrefEnricher, needs_identity_lookup
from bibr.pipeline.registry_check import (
    RegistryCheck,
    preliminary_doi,
    reconcile_with_final_doi,
    registry_check_wanted,
    start_registry_check,
)
from bibr.pipeline.state import FileState

LOOKUP = "bibr.enrich.references.enrich_paper_identity"
OWN_DOI = "10.1234/self.1"


def _settings(*, title_rule: bool = True, timeout: float = 15.0) -> GlobalSettings:
    settings = GlobalSettings()
    settings.FRONT_MATTER_REGISTRY_TITLE = title_rule
    settings.crossref.request_timeout = timeout
    return settings


def _contents(*dois: str) -> PaperContents:
    sections = [PaperSection(0, "Root", 0, None, CanonicalSection.TITLE)]
    sentences = []
    rows = [("Title", "A printed title"), *(("Article", f"Article DOI: {d}") for d in dois)]
    for index, (header, text) in enumerate(rows, start=1):
        sections.append(PaperSection(index, header, 1, 0, CanonicalSection.TITLE))
        sentences.append(
            PaperSentence(
                text_id=index,
                text=text,
                section_id=index,
                paragraph_id=index,
                page_number=1,
                region_meta={"region_type": "text", "region_page": 1, "region_index": index},
            )
        )
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
    )


def _record(doi: str = OWN_DOI, *, year: int | None = 2021) -> ExternalMatch:
    return ExternalMatch(
        doi=doi,
        score=100.0,
        year=year,
        title="The record title",
        authors=[BibAuthor(given="Ada", family="Lovelace")],
    )


def _found(doi: str = OWN_DOI, *, year: int | None = 2021):
    async def lookup(metadata, **_kwargs):
        metadata.match[MatchSource.CROSSREF] = _record(metadata.doi, year=year)
        return EnrichmentReport(attempted=1, matched=1)

    return AsyncMock(side_effect=lookup)


def _pending_tasks() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not current and not t.done()]


# --- preliminary DOI -----------------------------------------------------------


def test_preliminary_doi_is_the_unambiguous_selection():
    assert preliminary_doi(_contents("https://doi.org/10.1234/SELF.1")) == OWN_DOI


@pytest.mark.parametrize(
    "dois",
    [(), ("https://doi.org/10.1234/self.1", "https://doi.org/10.5555/other.2")],
    ids=["none", "ambiguous"],
)
def test_preliminary_doi_abstains_without_a_single_doi(dois):
    assert preliminary_doi(_contents(*dois)) is None


def test_wanted_only_for_a_rule_that_reads_the_record():
    settings = GlobalSettings()
    assert not registry_check_wanted(settings)
    settings.FRONT_MATTER_REGISTRY_TITLE = True
    assert registry_check_wanted(settings)
    settings = GlobalSettings()
    settings.llm.review_trigger = "registry_disagree"
    assert registry_check_wanted(settings)
    assert not registry_check_wanted(MagicMock())


# --- the lookup ------------------------------------------------------------------


async def test_found_record_is_kept_and_covers_its_doi():
    spy = _found()
    with patch(LOOKUP, spy):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        assert check is not None
        assert await check.settle("2021-03-01") == "found"
    # Idempotent: settling again neither waits nor looks up again.
    assert await check.settle("1990") == "found"
    spy.assert_awaited_once()
    assert check.matches[MatchSource.CROSSREF].doi == OWN_DOI
    assert check.complete and check.covers("10.1234/SELF.1")
    assert not check.covers("10.1234/other")
    row = check.record()
    assert row["service"] == "crossref" and row["score"] == 1.0
    assert row["author"] == [{"given": "Ada", "family": "Lovelace"}]
    assert check.seconds is not None


async def test_no_doi_starts_nothing():
    with patch(LOOKUP, _found()) as spy:
        assert start_registry_check(_contents(), settings=_settings()) is None
    spy.assert_not_awaited()


async def test_timeout_leaves_no_record_and_the_later_lookup_runs():
    async def hang(metadata, **_kwargs):
        await asyncio.sleep(30)

    with patch(LOOKUP, AsyncMock(side_effect=hang)):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings(timeout=0.05))
        assert await check.settle() == "failed"
    assert check.matches == {} and check.record() is None
    assert not check.complete and not check.covers(OWN_DOI)
    assert _pending_tasks() == []


async def test_failed_lookup_is_not_complete():
    async def fail(metadata, **_kwargs):
        return EnrichmentReport(attempted=1, matched=0, failed=1)

    with patch(LOOKUP, AsyncMock(side_effect=fail)):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        assert await check.settle() == "failed"
    assert not check.complete


async def test_raising_lookup_is_a_failure_not_an_error():
    with patch(LOOKUP, AsyncMock(side_effect=RuntimeError("boom"))):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        assert await check.settle() == "failed"


async def test_no_record_is_a_complete_answer():
    async def miss(metadata, **_kwargs):
        return EnrichmentReport(attempted=1, matched=0, failed=0)

    with patch(LOOKUP, AsyncMock(side_effect=miss)):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        assert await check.settle() == "not_found"
    assert check.covers(OWN_DOI) and check.record() is None


async def test_record_of_another_year_is_dropped():
    with patch(LOOKUP, _found(year=2010)):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        assert await check.settle("2021") == "year_mismatch"
    assert check.matches == {} and not check.complete


async def test_cancelled_lookup_settles_as_cancelled():
    async def hang(metadata, **_kwargs):
        await asyncio.sleep(30)

    with patch(LOOKUP, AsyncMock(side_effect=hang)):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        check.cancel()
        assert await check.settle() == "cancelled"
    assert _pending_tasks() == []


async def test_callers_cancellation_propagates_and_cancels_the_lookup():
    async def hang(metadata, **_kwargs):
        await asyncio.sleep(30)

    with patch(LOOKUP, AsyncMock(side_effect=hang)):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        waiter = asyncio.create_task(check.settle())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.wait([check.task])
    assert check.task.cancelled()
    assert _pending_tasks() == []


# --- the later lookup ------------------------------------------------------------


def _paper_with_check(check: RegistryCheck | None, doi: str = OWN_DOI) -> Paper:
    metadata = PaperMetadata(doi=doi, title="A printed title")
    if check is not None:
        metadata.match.update(check.matches)
    paper = Paper(
        input_file=InputFile(path="a.txt"),
        contents=_contents(f"https://doi.org/{doi}"),
        metadata=metadata,
    )
    paper.registry_check = check
    return paper


async def test_covered_paper_is_not_looked_up_again():
    with patch(LOOKUP, _found()):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        await check.settle()
    paper = _paper_with_check(check)
    assert not needs_identity_lookup(paper)

    fs = FileState(path=Path("a.pdf"))
    fs.paper = paper
    with patch(LOOKUP, AsyncMock()) as later:
        await CrossrefEnricher(settings=_settings()).enrich(fs)
    later.assert_not_awaited()
    # The record survives into the enriched export.
    assert paper.metadata.match[MatchSource.CROSSREF].doi == OWN_DOI


async def test_failed_early_lookup_leaves_the_later_lookup_on():
    with patch(LOOKUP, AsyncMock(side_effect=RuntimeError("down"))):
        check = start_registry_check(_contents(OWN_DOI), settings=_settings())
        await check.settle()
    paper = _paper_with_check(check)
    assert needs_identity_lookup(paper)
    fs = FileState(path=Path("a.pdf"))
    fs.paper = paper
    with patch(LOOKUP, _found()) as later:
        await CrossrefEnricher(settings=_settings()).enrich(fs)
    later.assert_awaited_once()


# --- the final DOI -----------------------------------------------------------------


async def _settled_check(doi: str = OWN_DOI) -> RegistryCheck:
    with patch(LOOKUP, _found(doi)):
        check = start_registry_check(_contents(f"https://doi.org/{doi}"), settings=_settings())
        await check.settle()
    return check


async def test_same_final_doi_keeps_the_record():
    from bibr.pipeline.stages.identity import IdentityValidationStage

    check = await _settled_check()
    paper = _paper_with_check(check)
    fs = FileState(path=Path("a.txt"))
    fs.paper = paper
    IdentityValidationStage._validate(fs)

    assert paper.metadata.doi == OWN_DOI
    assert paper.registry_check is check
    assert MatchSource.CROSSREF in paper.metadata.match
    assert not any(i.code == "VAL_REGISTRY_DOI_CHANGED" for i in paper.validation_issues)


async def test_changed_final_doi_drops_the_record_and_its_title():
    from bibr.pipeline.stages.identity import IdentityValidationStage

    check = await _settled_check("10.5555/early.1")
    paper = _paper_with_check(check, doi="")
    # The final selection reads another DOI than the early one.
    paper.contents = _contents(f"https://doi.org/{OWN_DOI}")
    printed = FieldCandidate("title", "llm", "The model title")
    check.title_fallback = FieldDecision("title", printed.value, printed, "extracted")
    check.record_backed = {"title"}
    chosen = FieldCandidate("title", "registry_printed", "The record title")
    apply_decision(paper.metadata, FieldDecision("title", chosen.value, chosen, "registry"))

    fs = FileState(path=Path("a.txt"))
    fs.paper = paper
    IdentityValidationStage._validate(fs)

    assert paper.metadata.doi == OWN_DOI
    assert paper.metadata.match == {}
    assert paper.registry_check is None
    assert paper.metadata.title == "The model title"
    assert field_decisions_of(paper.metadata).get("title").source == "llm"
    (issue,) = [i for i in paper.validation_issues if i.code == "VAL_REGISTRY_DOI_CHANGED"]
    assert issue.evidence_ids == ("doi:10.5555/early.1", f"doi:{OWN_DOI}")
    assert "title was decided again" in issue.message
    assert not issue.blocking
    # Its DOI was never looked up: the later lookup runs.
    assert needs_identity_lookup(paper)


async def test_changed_doi_keeps_a_reviewed_author_list_and_says_so():
    check = await _settled_check("10.5555/early.1")
    paper = _paper_with_check(check)
    check.record_backed = {"author"}
    issue = reconcile_with_final_doi(paper)
    assert issue is not None and "author list chosen with it is kept" in issue.message


# --- post_parse wiring -------------------------------------------------------------


def _post_parse_patches(extract):
    return (
        patch("bibr.pipeline.stages.post_parse._classify_sections", AsyncMock(return_value=None)),
        patch(
            "bibr.structure.implicit_sections.detect_implicit_sections",
            AsyncMock(return_value=None),
        ),
        patch("bibr.pipeline.stages.post_parse._extract_metadata_and_equations", extract),
        patch("bibr.pipeline.stages.post_parse._link_citations", AsyncMock(return_value=None)),
        patch(
            "bibr.extract.research_integrity.extract_structured_integrity",
            AsyncMock(return_value=None),
        ),
    )


async def test_post_parse_looks_the_record_up_under_extraction():
    from bibr.pipeline.stages.post_parse import post_parse

    seen = {}

    async def extract(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        seen["check"] = kwargs.get("registry_check")
        return PaperMetadata(doi="", title="T", published="2021")

    a, b, c, d, e = _post_parse_patches(extract)
    with a, b, c, d, e, patch(LOOKUP, _found()):
        paper = await post_parse(
            contents=_contents(OWN_DOI),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=MagicMock(),
            settings=_settings(),
            registry_lookup=True,
        )
    check = paper.registry_check
    assert isinstance(check, RegistryCheck) and seen["check"] is check
    assert check.status == "found"
    assert paper.metadata.match[MatchSource.CROSSREF].doi == OWN_DOI
    assert _pending_tasks() == []


@pytest.mark.parametrize(
    ("kwargs", "title_rule"),
    [
        ({"registry_lookup": False}, True),
        ({"registry_lookup": True}, False),
        ({"registry_lookup": True, "no_llm": True}, True),
    ],
    ids=["no-lookup-in-run", "no-rule-reads-it", "no-llm"],
)
async def test_post_parse_starts_nothing_when_not_wanted(kwargs, title_rule):
    from bibr.pipeline.stages.post_parse import post_parse

    async def extract(contents, file_hash, no_llm, llm_client, **kw):  # noqa: ARG001
        assert "registry_check" not in kw or kw["registry_check"] is None
        return PaperMetadata(doi="", title="T")

    a, b, c, d, e = _post_parse_patches(extract)
    with a, b, c, d, e, patch(LOOKUP, _found()) as spy:
        paper = await post_parse(
            contents=_contents(OWN_DOI),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=MagicMock(),
            settings=_settings(title_rule=title_rule),
            **kwargs,
        )
    assert paper.registry_check is None
    spy.assert_not_awaited()


async def test_post_parse_failure_cancels_the_lookup():
    from bibr.exceptions import ProcessingError
    from bibr.pipeline.stages.post_parse import post_parse

    started = []

    async def hang(metadata, **_kwargs):
        await asyncio.sleep(30)

    real_start = start_registry_check

    def recording_start(contents, **kwargs):
        check = real_start(contents, **kwargs)
        started.append(check)
        return check

    async def extract(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        await asyncio.sleep(0)
        raise ProcessingError("extraction died", error_code="extraction_failed")

    a, b, c, d, e = _post_parse_patches(extract)
    with (
        a,
        b,
        c,
        d,
        e,
        patch(LOOKUP, AsyncMock(side_effect=hang)),
        patch("bibr.pipeline.registry_check.start_registry_check", recording_start),
    ):
        with pytest.raises(ProcessingError):
            await post_parse(
                contents=_contents(OWN_DOI),
                file_name="x.pdf",
                file_hash="deadbeef",
                llm_client=MagicMock(),
                settings=_settings(),
                registry_lookup=True,
            )
        (check,) = started
        await asyncio.wait([check.task])
    assert check.task.cancelled()
    assert _pending_tasks() == []


async def test_stage_asks_for_the_lookup_only_when_the_run_makes_it():
    from bibr.pipeline.context import PipelineContext, RunConfig
    from bibr.pipeline.progress import NullProgress
    from bibr.pipeline.stages.post_parse import PostParseStage

    seen: list[bool] = []

    async def fake_post_parse(contents, file_name, file_hash, **kwargs):  # noqa: ARG001
        seen.append(kwargs["registry_lookup"])
        return MagicMock(name="paper")

    for config, expected in (
        (RunConfig(), False),
        (RunConfig(crossref_identity=True), True),
        (RunConfig(crossref=True), True),
        (RunConfig(crossref=True, no_llm=True), False),
    ):
        fs = FileState(path=Path("a.pdf"))
        fs.contents = MagicMock(layout_hints=None)
        ctx = PipelineContext(
            file_states=[fs],
            progress=NullProgress(),
            resources=MagicMock(),
            config=config,
            settings=GlobalSettings(),
        )
        with patch("bibr.pipeline.stages.post_parse.post_parse", fake_post_parse):
            await PostParseStage().run(ctx)
        assert seen[-1] is expected, config
