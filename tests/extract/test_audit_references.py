"""Reference-extraction audit fixes: the finalize crash and the in-press
detector.

* A superscript or circled digit in a page ("4²", "①") passed the
  ``isdigit()`` guard of the compact page-range expansion and crashed
  ``int()``; with no per-reference guard, that one entry made the paper's
  whole reference list incomplete.
* The in-press detector matched "in press" anywhere: "caveats in press
  releases" and "The Darwin Press" read as in press, which suppressed the
  Vancouver year backfill and the year-from-text repair.
"""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest

from bibr.extract import ref_extractor
from bibr.extract.ref_extractor import (
    ReferenceExtractor,
    _expand_compact_last_page,
    _finalize_reference_fields,
    _is_in_press,
)
from bibr.extract.ref_field_repair import _rule_year_from_text
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import WarningCode
from bibr.schemas import PaperReferenceLLM

SEGMENTS = [
    "Smith, J. (2020). A study. Journal of Things, 55(7), 782-92.",
    "Doe, A. (2021). Another study. Journal of Stuff, 12(3), 782-92.",
]


def _extractor() -> ReferenceExtractor:
    contents = mock.Mock(spec=PaperContents)
    contents.processing_warnings = []
    contents.region_summaries = []
    return ReferenceExtractor(contents, llm_client=mock.Mock())


def _ner_parser(*parsed: dict):
    class _Parser:
        @staticmethod
        def parse_batch(segments):
            return [dict(fields) for fields in parsed]

    return patch.object(ref_extractor, "_get_ner_parser", return_value=_Parser())


def _llm_ref(index: int, title: str) -> PaperReferenceLLM:
    return PaperReferenceLLM(
        index=index,
        title=title,
        authors="Smith, J.",
        year=2020,
        container="Journal of Things",
        volume="55",
        first_page="782",
        last_page="92",
    )


def _finalize_failing_on(title: str):
    """``_finalize_reference_fields`` that raises for the reference titled *title*."""
    real = _finalize_reference_fields

    def finalize(fields, segment):
        if fields.get("title") == title:
            raise ValueError("unforeseen shape")
        return real(fields, segment)

    return patch.object(ref_extractor, "_finalize_reference_fields", side_effect=finalize)


def _finalize_warnings(ext: ReferenceExtractor) -> list[str]:
    return [
        w.message
        for w in ext.contents.processing_warnings
        if w.code == WarningCode.REF_PARSE_FINALIZE_FAILED
    ]


class TestNonAsciiPageDigits:
    @pytest.mark.parametrize(
        ("first_page", "last_page"),
        [("123", "4²"), ("12³", "4"), ("123", "①"), ("1²3", "45")],
    )
    def test_a_digit_int_rejects_is_left_unexpanded(self, first_page, last_page):
        assert _expand_compact_last_page(first_page, last_page) == last_page

    def test_ascii_ranges_still_expand(self):
        assert _expand_compact_last_page("782", "92") == "792"

    def test_finalize_keeps_a_superscript_last_page(self):
        fields = {"title": "A study", "first_page": "123", "last_page": "4²"}
        assert _finalize_reference_fields(fields, None)["last_page"] == "4²"

    def test_the_ner_path_parses_the_reference(self):
        parsed = {"title": "A study", "authors": "Smith, J.", "first_page": "123"}
        with _ner_parser({**parsed, "last_page": "4²"}, {**parsed, "last_page": "45"}):
            refs = _extractor()._parse_references_ner(SEGMENTS)

        assert [(r.first_page, r.last_page) for r in refs] == [("123", "4²"), ("123", "145")]


class TestOneFailingFinalizeKeepsTheOthers:
    """A finalize step that raises on one entry keeps that entry as parsed."""

    def test_ner_path(self):
        ext = _extractor()
        parsed = {"authors": "Smith, J.", "first_page": "782", "last_page": "92"}
        with (
            _ner_parser({**parsed, "title": "Broken"}, {**parsed, "title": "Fine"}),
            _finalize_failing_on("Broken"),
        ):
            refs = ext._parse_references_ner(SEGMENTS)

        assert [(r.title, r.last_page) for r in refs] == [("Broken", "92"), ("Fine", "792")]
        (message,) = _finalize_warnings(ext)
        assert message.startswith("reference 1 kept as parsed: ValueError")

    async def test_llm_batched_path(self, monkeypatch):
        monkeypatch.setattr("bibr.config.Settings.REF_PARSE_BATCH_SIZE", 15)
        ext = _extractor()
        ext.llm_client.extract_references = AsyncMock(
            return_value=[_llm_ref(1, "Broken"), _llm_ref(2, "Fine")]
        )
        with _finalize_failing_on("Broken"):
            refs = await ext._parse_references_llm("\n".join(SEGMENTS), SEGMENTS)

        assert [(r.title, r.last_page) for r in refs] == [("Broken", "92"), ("Fine", "792")]
        assert len(_finalize_warnings(ext)) == 1

    async def test_llm_chunked_path(self):
        ext = _extractor()
        ext.llm_client.extract_references_chunk = AsyncMock(
            return_value=[_llm_ref(1, "Broken"), _llm_ref(2, "Fine")]
        )
        with _finalize_failing_on("Broken"):
            refs = await ext._parse_references_llm_chunked("\n".join(SEGMENTS), SEGMENTS)

        assert [(r.title, r.last_page) for r in refs] == [("Broken", "92"), ("Fine", "792")]
        assert len(_finalize_warnings(ext)) == 1


SUMNER = (
    "Sumner P, Vivian-Griffiths S, Boivin J, Williams A, Bott L, Adams R, et al. "
    "Exaggerations and caveats in press releases and health-related science news. "
    "PLoS One. 2016;11(12):e0168217."
)


class TestInPressNeedsAStatusPosition:
    @pytest.mark.parametrize(
        "text",
        [
            SUMNER,
            "Darwin, C. (1859). On the origin of species. London: The Darwin Press.",
            "Smith, J. 2007. A book. Berlin Press.",
            "Smith, J. (2019). A book. New York: Penguin Press.",
            "Smith, J. (2020). The forthcoming election. Journal of Politics, 3, 1-2.",
            "Smith, J. (2020). Women in press photography. Journal, 2, 3.",
            "Smith J. Advance online learning in schools. J Ed. 2019;3:1-2.",
        ],
    )
    def test_the_words_inside_a_title_or_name_are_not_a_status(self, text):
        assert not _is_in_press(text)

    @pytest.mark.parametrize(
        "text",
        [
            "in press",
            "In  Press",
            "forthcoming",
            "advance online",
            "manuscript submitted",
            "epub ahead",
            "Robertson, C. E., & Van Bavel, J. J. (in press). Inside the funhouse mirror factory.",
            "Smith, J. (in press-a). Title. Journal.",
            "Smith, J. (2021, in press). Title. Journal.",
            "Smith J. A forthcoming study. J Synth Garden Res in press.",
            "Smith J. Title. J Med. In press. doi:10.1000/xyz",
            "Smith J. Title. Proc Natl Acad Sci U S A. In press 2002.",
            "Smith, J. (2020). Title. Journal. Advance online publication. https://doi.org/10.1/x",
            "Smith, J. (2020). Title. Manuscript submitted for publication.",
            "Smith J. Title. Nature. 2020 Jan 5. [Epub ahead of print]",
            "Smith, J. Title. Forthcoming in Journal of Philosophy.",
            "Smith, J. Title. Journal of X, Article in Press.",
        ],
    )
    def test_a_status_of_its_own_is_in_press(self, text):
        assert _is_in_press(text)

    def test_a_title_mentioning_press_releases_keeps_its_vancouver_year(self):
        fields = {"title": "Exaggerations", "year": None, "volume": None, "first_page": None}
        assert _finalize_reference_fields(fields, SUMNER)["year"] == 2016

    def test_a_publisher_named_press_keeps_its_year_from_the_text(self):
        fields = {"authors": "Smith, J.", "title": "A book"}
        assert _rule_year_from_text(fields, "Smith, J. A book. Berlin: Berlin Press, 2007.")
        assert fields["year"] == 2007

    def test_the_ner_path_does_not_flag_it(self):
        with _ner_parser({"title": "Exaggerations and caveats", "authors": "Sumner P"}):
            (ref,) = _extractor()._parse_references_ner([SUMNER])

        assert ref.is_in_press is False
        assert ref.year == 2016
