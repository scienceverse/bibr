"""The registry title rule (FRONT_MATTER_REGISTRY_TITLE).

When the extracted title disagrees with the paper's own registry record, the
selected record's printed row that is the record's title stands in. The
record only chooses between printed rows; a later translation, a row the
extracted title extends, or a record title no row prints never wins.
"""

from __future__ import annotations

import pytest

from bibr.enrich.verify_identity import verify_identity

ENGLISH = "The effect of sleep on memory in older adults"
GERMAN = "Die Wirkung von Schlaf auf das Gedächtnis bei älteren Erwachsenen"


def _candidate(candidate_id, text, roles=("title",), *, reading_order=0):
    from bibr.extract.front_matter import FrontMatterCandidate

    return FrontMatterCandidate(
        candidate_id=candidate_id,
        source_kind="paragraph",
        reading_order=reading_order,
        page=1,
        bbox=None,
        region_label="text",
        font_size=None,
        font_bold=None,
        section_id=1,
        text_ids=(),
        paragraph_id=1,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=frozenset(roles),
    )


def _record(title=ENGLISH, subtitle=None, service="crossref"):
    return {
        "service": service,
        "score": 1.0,
        "doi": "10.1234/self",
        "title": title,
        **({"subtitle": subtitle} if subtitle else {}),
    }


def _decide(rows, title, record=None, *, scoped=True, abstained=False):
    from bibr.extract.field_decisions import FieldCandidate, decide_title
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    block = FrontMatterBlock(
        block_id="selected",
        candidate_ids=tuple(row.candidate_id for row in rows),
        title_candidate_ids=tuple(row.candidate_id for row in rows if "title" in row.roles),
    )
    resolution = FrontMatterResolution(
        candidates=tuple(rows),
        blocks=(block,),
        selected_block_id=None if abstained else "selected",
        selection_method="test",
        reason_flags=(),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )
    kwargs = {} if record is None else {"registry": record}
    return decide_title(
        FieldCandidate("title", "llm", title),
        resolution=resolution,
        detected_title="",
        sections=[],
        journal=None,
        publisher=None,
        scoped=scoped,
        abstained=abstained,
        prefer_byline_adjacent=False,
        doc_info=None,
        **kwargs,
    )


def _rejection(decision) -> str:
    (verdict,) = [v for v in decision.considered if v.candidate.source == "registry_printed"]
    assert not verdict.accepted
    return verdict.reason


def test_printed_row_of_the_record_title_replaces_a_disagreeing_title():
    rows = [
        _candidate("running", "Original research", roles=("other",)),
        _candidate("t1", ENGLISH, reading_order=1),
        _candidate("byline", "Ada Lovelace", roles=("byline",), reading_order=2),
    ]
    decision = _decide(rows, "Sleep in older adults: memory effects", _record(ENGLISH.upper()))

    assert decision.value == ENGLISH  # the printed row, not the record's casing
    assert decision.rule == "registry_printed_title"
    assert decision.source == "registry_printed"
    assert decision.selected.evidence_ids == ("t1", "metadata_match:crossref")
    (issue,) = decision.issues
    assert issue.code == "VAL_TITLE_REGISTRY_PRINTED" and not issue.blocking
    replaced = [v for v in decision.considered if v.candidate.source == "llm"]
    assert replaced[0].reason == "replaced by registry_printed"


def test_record_subtitle_must_be_printed_with_the_title():
    rows = [_candidate("t1", "Sleep and memory: A meta-analysis")]
    record = _record("Sleep and memory", subtitle="A meta-analysis")
    decision = _decide(rows, "Memory, sleep and the ageing brain", record)
    assert decision.value == "Sleep and memory: A meta-analysis"

    # Printed without its subtitle, the row is not the record's title.
    rows = [_candidate("t1", "Sleep and memory")]
    decision = _decide(rows, "Memory, sleep and the ageing brain", record)
    assert decision.value == "Memory, sleep and the ageing brain"
    assert _rejection(decision) == "no printed row is the record's title"


def test_record_subtitle_on_the_row_below_is_joined_as_printed():
    rows = [
        _candidate("t1", "Sleep and memory"),
        _candidate("t2", "A meta-analysis", roles=("other",), reading_order=1),
        _candidate("byline", "Ada Lovelace", roles=("byline",), reading_order=2),
    ]
    record = _record("Sleep and Memory", subtitle="a meta-analysis")
    decision = _decide(rows, "Memory, sleep and the ageing brain", record)
    assert decision.value == "Sleep and memory: A meta-analysis"
    assert decision.selected.evidence_ids == ("t1", "t2", "metadata_match:crossref")

    # The row below must print the record's subtitle.
    rows[1] = _candidate("t2", "A systematic review", roles=("other",), reading_order=1)
    decision = _decide(rows, "Memory, sleep and the ageing brain", record)
    assert _rejection(decision) == "no printed row is the record's title"


def test_agreeing_title_is_kept():
    decision = _decide([_candidate("t1", ENGLISH)], ENGLISH, _record())
    assert decision.value == ENGLISH and decision.rule == "extracted"
    assert _rejection(decision) == "agrees with the record"


def test_record_typo_prints_nowhere_and_is_refused():
    rows = [_candidate("t1", ENGLISH)]
    decision = _decide(rows, "Sleep in older adults", _record(ENGLISH.replace("memory", "memroy")))
    assert decision.value == "Sleep in older adults"
    assert _rejection(decision) == "no printed row is the record's title"


def test_title_extending_the_row_is_kept():
    rows = [
        _candidate("t1", "Sleep and memory"),
        _candidate("t2", "A meta-analysis of studies in older adults", reading_order=1),
    ]
    title = "Sleep and memory: A meta-analysis of studies in older adults"
    decision = _decide(rows, title, _record("Sleep and memory"))
    assert decision.value == title
    assert _rejection(decision) == "the title extends the row"


def test_later_parallel_language_title_is_refused():
    """The record holds the English version; the paper prints the German first."""
    rows = [_candidate("de", GERMAN), _candidate("en", ENGLISH, reading_order=1)]
    decision = _decide(rows, GERMAN, _record())
    assert decision.value == GERMAN
    assert _rejection(decision) == "the title is printed before the row"


def test_translation_not_printed_first_is_refused():
    """The extracted German title differs slightly from the printed one, so it is
    not found before the row; the two still read as language versions."""
    rows = [_candidate("de", GERMAN), _candidate("en", ENGLISH, reading_order=1)]
    extracted = GERMAN.replace("älteren ", "")
    decision = _decide(rows, extracted, _record())
    assert decision.value == extracted
    assert _rejection(decision) == "the row is a translation not printed first"


def test_row_printed_after_another_version_is_refused_whatever_was_extracted():
    """The extracted title is neither printed nor a version of the record's; the
    record's English row still follows the German title printed first."""
    rows = [
        _candidate("de", GERMAN),
        _candidate("en", ENGLISH, reading_order=1),
    ]
    decision = _decide(rows, "Sleep, memory and ageing", _record())
    assert decision.value == "Sleep, memory and ageing"
    assert _rejection(decision) == "the row is a translation not printed first"


def test_same_language_title_printed_before_the_row_is_kept():
    """The printed-first rule is not limited to translations: a series line the
    model took, printed above the record's title row, stays."""
    series = "Lecture notes on the cognitive science of everyday reasoning"
    rows = [
        _candidate("s", series),
        _candidate("t", ENGLISH, reading_order=1),
    ]
    decision = _decide(rows, series, _record())
    assert decision.value == series
    assert _rejection(decision) == "the title is printed before the row"


def test_title_printed_before_the_row_outside_title_rows_is_kept():
    rows = [
        _candidate("de", GERMAN, roles=("other",)),
        _candidate("en", ENGLISH, reading_order=1),
    ]
    decision = _decide(rows, GERMAN, _record())
    assert decision.value == GERMAN
    assert _rejection(decision) == "the title is printed before the row"


def test_registry_row_printed_first_wins_over_a_later_translation():
    rows = [_candidate("en", ENGLISH), _candidate("de", GERMAN, roles=("other",), reading_order=1)]
    decision = _decide(rows, GERMAN, _record())
    assert decision.value == ENGLISH


@pytest.mark.parametrize(
    ("kwargs", "record"),
    [
        ({}, None),
        ({"scoped": False}, _record()),
        ({"abstained": True}, _record()),
    ],
    ids=["rule-off", "unscoped", "abstained"],
)
def test_rule_runs_only_when_on_and_scoped(kwargs, record):
    rows = [_candidate("t1", ENGLISH)]
    decision = _decide(rows, "Sleep in older adults", record, **kwargs)
    assert decision.value == "Sleep in older adults"
    assert not any(v.candidate.source == "registry_printed" for v in decision.considered)


def test_no_title_in_hand_leaves_the_null_title_fallbacks_alone():
    decision = _decide([_candidate("t1", ENGLISH)], None, _record())
    assert decision.rule == "selected_record_title"


def test_flags_are_unchanged_by_the_rule():
    decision = _decide([_candidate("t1", ENGLISH)], "Sleep in older adults", _record())
    assert dict(decision.flags) == {
        "scoped": True,
        "abstained": False,
        "prefer_byline_adjacent": False,
    }


# --- the export ---------------------------------------------------------------


def _payload(source: str, title: str = ENGLISH) -> dict:
    return {
        "metadata": {"title": title, "doi": "10.1234/self"},
        "author": [{"given": "Ada", "family": "Lovelace", "literal": None}],
        "metadata_match": [
            {
                "service": "crossref",
                "score": 1.0,
                "doi": "10.1234/self",
                "title": ENGLISH,
                "author": [{"given": "Ada", "family": "Lovelace"}],
            }
        ],
        "extraction": {
            "fields": {
                "title": {"state": "extracted", "source": source, "issues": []},
                "author": {"state": "extracted", "source": "llm", "issues": []},
            }
        },
    }


def test_title_chosen_by_the_rule_is_reported_corrected():
    payload = _payload("registry_printed")
    assert verify_identity(payload) == {"title": "corrected", "author": "agree"}
    assert payload["extraction"]["fields"]["title"]["registry"] == {
        "service": "crossref",
        "outcome": "corrected",
    }


def test_title_extracted_as_is_still_agrees():
    assert verify_identity(_payload("llm"))["title"] == "agree"


def test_corrected_needs_agreement():
    payload = _payload("registry_printed", title="Something else entirely here")
    assert verify_identity(payload)["title"] == "disagree"


def test_corrected_outcome_validates_in_the_export_model():
    from bibr.export.models import FieldRecordExport

    payload = _payload("registry_printed")
    verify_identity(payload)
    FieldRecordExport.model_validate(payload["extraction"]["fields"]["title"], strict=True)


# --- post_parse -----------------------------------------------------------------


def _resolution(rows):
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    block = FrontMatterBlock(
        block_id="selected",
        candidate_ids=tuple(row.candidate_id for row in rows),
        title_candidate_ids=tuple(row.candidate_id for row in rows if "title" in row.roles),
    )
    return FrontMatterResolution(
        candidates=tuple(rows),
        blocks=(block,),
        selected_block_id="selected",
        selection_method="test",
        reason_flags=(),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )


async def _post_parse_with_record(*, rule_on: bool, extract=None):
    from unittest.mock import AsyncMock, MagicMock, patch

    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.post_parse import post_parse
    from tests.pipeline.test_registry_check import LOOKUP, OWN_DOI, _contents, _settings

    resolution = _resolution([_candidate("t1", ENGLISH)])

    def attach(contents, *_args, **_kwargs):
        contents.front_matter_resolution = resolution
        return ()

    async def extracted(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        return PaperMetadata(doi="", title="Sleep in older adults", published="2021")

    extract = extract or extracted

    async def lookup(metadata, **_kwargs):
        from bibr.enrich.references import EnrichmentReport
        from bibr.models import ExternalMatch, MatchSource

        metadata.match[MatchSource.CROSSREF] = ExternalMatch(
            doi=metadata.doi, score=100.0, year=2021, title=ENGLISH
        )
        return EnrichmentReport(attempted=1, matched=1)

    settings = _settings(title_rule=rule_on)
    settings.llm.review_trigger = "registry_disagree"  # the lookup runs either way
    pp = "bibr.pipeline.stages.post_parse"
    with (
        patch(f"{pp}._classify_sections", AsyncMock(return_value=None)),
        patch(f"{pp}._attach_front_matter_resolution", attach),
        patch("bibr.structure.implicit_sections.detect_implicit_sections", AsyncMock()),
        patch(f"{pp}._extract_metadata_and_equations", extract),
        patch(f"{pp}._link_citations", AsyncMock(return_value=None)),
        patch("bibr.extract.research_integrity.extract_structured_integrity", AsyncMock()),
        patch(LOOKUP, AsyncMock(side_effect=lookup)),
    ):
        paper = await post_parse(
            contents=_contents(OWN_DOI),
            file_name="x.pdf",
            file_hash="deadbeef",
            llm_client=MagicMock(),
            settings=settings,
            registry_lookup=True,
        )
    return paper


async def test_post_parse_applies_the_rule_and_keeps_its_fallback():
    paper = await _post_parse_with_record(rule_on=True)
    assert paper.metadata.title == ENGLISH
    assert paper.field_decisions.get("title").rule == "registry_printed_title"
    check = paper.registry_check
    assert check.record_backed == {"title"}
    assert check.title_fallback.value == "Sleep in older adults"
    assert any(i.code == "VAL_TITLE_REGISTRY_PRINTED" for i in paper.validation_issues)

    # A final DOI other than the early one restores the title decided without it.
    from pathlib import Path

    from bibr.pipeline.stages.identity import IdentityValidationStage
    from bibr.pipeline.state import FileState
    from tests.pipeline.test_registry_check import _contents

    paper.contents = _contents("https://doi.org/10.5555/final.9")
    fs = FileState(path=Path("x.txt"))
    fs.paper = paper
    IdentityValidationStage._validate(fs)
    assert paper.metadata.doi == "10.5555/final.9"
    assert paper.metadata.title == "Sleep in older adults"
    assert paper.metadata.match == {}
    assert any(i.code == "VAL_REGISTRY_DOI_CHANGED" for i in paper.validation_issues)


async def test_post_parse_without_the_rule_keeps_the_extracted_title():
    paper = await _post_parse_with_record(rule_on=False)
    assert paper.metadata.title == "Sleep in older adults"
    assert paper.registry_check.record_backed == set()
    # The record is still there for the comparison at export.
    assert paper.metadata.match


async def test_post_parse_keeps_the_first_answer_of_a_record_backed_review():
    """The review's title won over a printed first answer because the record
    agreed; a final DOI other than the early one restores the first answer
    with its own issues, and the review title's issues leave with it."""
    from pathlib import Path

    from bibr.extract.field_decisions import FieldCandidate, field_decisions_of
    from bibr.models import PaperMetadata
    from bibr.pipeline.stages.identity import IdentityValidationStage
    from bibr.pipeline.state import FileState
    from bibr.validation import IssueSeverity, ValidationIssue
    from tests.pipeline.test_registry_check import _contents

    first_issue = ValidationIssue("VAL_TITLE_REGROUNDED", IssueSeverity.WARNING, "first answer")
    review_issue = ValidationIssue("VAL_TITLE_REGROUNDED", IssueSeverity.WARNING, "review")

    async def reviewed(contents, file_hash, no_llm, llm_client, **kwargs):  # noqa: ARG001
        check = kwargs["registry_check"]
        await check.settle("2021")
        # What the core extractor leaves when the review's answer was kept.
        check.record_backed.add("title")
        check.primary_title = FieldCandidate(
            "title", "llm", "Sleep in older adults", issues=(first_issue,)
        )
        metadata = PaperMetadata(doi="", title=ENGLISH, published="2021")
        field_decisions_of(metadata).propose(
            FieldCandidate("title", "llm_review", ENGLISH, issues=(review_issue,))
        )
        return metadata

    paper = await _post_parse_with_record(rule_on=False, extract=reviewed)
    assert paper.metadata.title == ENGLISH
    assert paper.field_decisions.get("title").source == "llm_review"
    assert paper.registry_check.title_fallback.value == "Sleep in older adults"
    assert review_issue in paper.validation_issues

    paper.contents = _contents("https://doi.org/10.5555/final.9")
    fs = FileState(path=Path("x.txt"))
    fs.paper = paper
    IdentityValidationStage._validate(fs)
    assert paper.metadata.title == "Sleep in older adults"
    assert paper.field_decisions.get("title").source == "llm"
    assert review_issue not in paper.validation_issues
    assert first_issue in paper.validation_issues
    assert any(i.code == "VAL_REGISTRY_DOI_CHANGED" for i in paper.validation_issues)
