from unittest.mock import Mock

import pytest

from bibr.clients import ref_llm
from bibr.config import GlobalSettings
from bibr.export.models import ReferenceReviewExport
from bibr.extract.ref_extractor import ReferenceExtractor, _finalize_reference_fields
from bibr.extract.ref_review import coverage, improves, reasons
from bibr.models import PaperReference
from bibr.paper_contents import PaperContents


def reference(title, **kwargs):
    return PaperReference(
        bib_id=1,
        title=title,
        authors=None,
        year=None,
        container=None,
        first_page=None,
        volume=None,
        **kwargs,
    )


def test_plausible_but_incomplete_ner_parse_is_flagged():
    text = "Smith J. 2020. A long source title. Journal of Tests."
    parsed = reference("A long source title")
    flags = reasons(text, parsed, min_coverage=0.8, long_chars=1200)
    assert "missing_printed_year" in flags and "unassigned_words" in flags
    assert coverage(text, parsed) < 0.8


def test_complete_parse_stays_on_fast_path():
    parsed = PaperReference(
        bib_id=1,
        title="A test",
        authors="Smith J",
        year=2020,
        container="Journal",
        first_page=None,
        volume=None,
    )
    assert (
        reasons("Smith J. 2020. A test. Journal.", parsed, min_coverage=0.8, long_chars=1200) == ()
    )


def test_supported_primary_fields_cannot_disappear_during_review():
    current = PaperReference(
        bib_id=1,
        title="Work",
        authors="Smith J",
        year=2020,
        container=None,
        first_page=None,
        volume=None,
    )
    candidate = reference("Work")
    assert not improves(
        "Smith J. 2020. Work.", current, candidate, min_coverage=0.8, long_chars=1200
    )


def test_printed_suffix_restored_only_when_unambiguous():
    assert (
        _finalize_reference_fields({"title": "Work", "year": 2005}, "Smith (2005a). Work.")[
            "year_suffix"
        ]
        == "a"
    )
    assert "year_suffix" not in _finalize_reference_fields(
        {"title": "Work", "year": 2005}, "2005a and 2005b. Work."
    )
    assert "year_suffix" not in _finalize_reference_fields(
        {"title": "Work", "year": 2005}, "2005abc. Work."
    )


async def test_cascade_is_aligned_bounded_and_retains_failed_ner(monkeypatch):
    config = GlobalSettings(
        REF_LLM_BASE_URL="http://reference.test/v1",
        REF_LLM_MODEL="trained-ref",
        REF_NER_LLM_MAX_REVIEWS=2,
    )
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    extractor = ReferenceExtractor(contents, settings=config, llm_client=Mock())
    original = [None, reference("Second"), reference("Third")]
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", Mock(return_value=original))
    calls = []

    async def parse_batch(strings, settings, **kwargs):
        calls.append(strings)
        return [dict.fromkeys(ref_llm.FIELDS) | {"title": "First"}, None]

    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    result = await extractor._parse_references_ner_review_aligned(
        [
            "First",
            "Second work 2020",
            "Third work 2021",
        ]
    )
    assert calls == [["First", "Second work 2020"]]
    assert result[0].title == "First"
    assert result[1] is original[1] and result[2] is original[2]
    receipt = ReferenceReviewExport.model_validate(contents.reference_review_receipt)
    assert (
        receipt.flagged,
        receipt.attempted,
        receipt.accepted,
        receipt.failed,
        receipt.skipped,
    ) == (3, 2, 1, 1, 1)


async def test_weaker_review_is_rejected(monkeypatch):
    config = GlobalSettings(
        REF_LLM_BASE_URL="http://reference.test/v1", REF_LLM_MODEL="trained-ref"
    )
    contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
    extractor = ReferenceExtractor(contents, settings=config, llm_client=Mock())
    original = reference("Supported work")
    monkeypatch.setattr(extractor, "_parse_references_ner_aligned", Mock(return_value=[original]))

    async def parse_batch(strings, settings, **kwargs):
        return [dict.fromkeys(ref_llm.FIELDS) | {"title": "work"}]

    monkeypatch.setattr(ref_llm, "parse_batch", parse_batch)
    result = await extractor._parse_references_ner_review_aligned(
        ["Smith J. 2020. Supported work."]
    )
    assert result == [original]
    assert contents.reference_review_receipt["rejected"] == 1


async def test_unconfigured_and_disabled_review_never_calls_server(monkeypatch):
    for enabled in [True, False]:
        config = GlobalSettings(REF_NER_LLM_FALLBACK=enabled)
        contents = PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={})
        extractor = ReferenceExtractor(contents, settings=config, llm_client=Mock())
        original = reference("Work")
        monkeypatch.setattr(
            extractor, "_parse_references_ner_aligned", Mock(return_value=[original])
        )
        monkeypatch.setattr(
            ref_llm, "parse_batch", Mock(side_effect=AssertionError("unexpected request"))
        )
        assert await extractor._parse_references_ner_review_aligned(["Work 2020"]) == [original]
        if enabled:
            assert contents.reference_review_receipt["status"] == "not_configured"


def test_review_counts_are_verified_by_export_model():
    with pytest.raises(ValueError, match="review attempts"):
        ReferenceReviewExport(
            model="trained",
            status="configured",
            flagged=2,
            attempted=2,
            accepted=1,
            failed=0,
            rejected=0,
            skipped=0,
            reasons={},
        )


ZH = "张三, 李四. 深度学习研究[J]. 计算机学报, 2020, 43(1): 1-10."


def test_cjk_reference_is_routed_even_when_one_span_covers_it():
    lumped = PaperReference(
        bib_id=1,
        title="张三, 李四. 深度学习研究",
        authors=None,
        year=2020,
        container="计算机学报",
        first_page=None,
        volume=None,
    )
    assert coverage(ZH, lumped) == 1.0
    assert reasons(ZH, lumped, min_coverage=0.8, long_chars=1200) == ("cjk_script",)


def test_cjk_review_is_kept_when_no_check_gets_worse():
    current = PaperReference(
        bib_id=1,
        title="张三, 李四. 深度学习研究",
        authors=None,
        year=2020,
        container="计算机学报",
        first_page=None,
        volume=None,
    )
    candidate = PaperReference(
        bib_id=1,
        title="深度学习研究",
        authors="张三, 李四",
        year=2020,
        container="计算机学报",
        first_page=None,
        volume=None,
    )
    assert improves(ZH, current, candidate, min_coverage=0.8, long_chars=1200)
    weaker = PaperReference(
        bib_id=1,
        title="深度学习研究",
        authors="张三, 李四",
        year=2020,
        container="计算机学报 增刊",
        first_page=None,
        volume=None,
    )
    assert not improves(ZH, current, weaker, min_coverage=0.8, long_chars=1200)


def test_equal_latin_review_still_needs_a_gain():
    current = PaperReference(
        bib_id=1,
        title="A test",
        authors="Smith J",
        year=2020,
        container="Journal",
        first_page=None,
        volume=None,
    )
    candidate = current.model_copy(update={"title": "A test", "container": "Journal"})
    assert not improves(
        "Smith J. 2020. A test. Journal.", current, candidate, min_coverage=0.8, long_chars=1200
    )
