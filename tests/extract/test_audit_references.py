"""Reference-extraction audit fixes: the finalize crash, the in-press detector
and the splitters that rescanned the whole text.

* A superscript or circled digit in a page ("4²", "①") passed the
  ``isdigit()`` guard of the compact page-range expansion and crashed
  ``int()``; with no per-reference guard, that one entry made the paper's
  whole reference list incomplete.
* The in-press detector matched "in press" anywhere: "caveats in press
  releases" and "The Darwin Press" read as in press, which suppressed the
  Vancouver year backfill and the year-from-text repair.
* The merged-reference splitter re-tokenised the whole prefix for every date
  (and searched it for an in-title citation), and the layout-line aligner
  searched the whole text once per line, so both were quadratic.
"""

from __future__ import annotations

import time
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest

from bibr.extract import anchor_snap, ref_extractor
from bibr.extract.anchor_snap import align_line_starts
from bibr.extract.merge_split import find_interior_onsets, split_merged_refs
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
            "Smith, J. (2020). In press and online. Journal, 2, 3.",
            "Smith, J. (2020). Freedom: in press a free voice. Journal, 2, 3.",
            "Smith, J. (2020). In press A study of things. Journal, 2, 3.",
            "Smith, J. (2016) in press releases. Journal, 2, 3.",
            "Smith, J. (2020). Elections 2020 in press coverage. J, 2, 3.",
            "Smith, J. (2020). A book. Berlin: Berlin Press. doi:10.1000/xyz",
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
            # PDF text keeps the typographic hyphen or dash, or a space.
            "Smith, J. (in press‐a). Title. Journal.",
            "Smith, J. (in press‑b). Title. Journal.",
            "Smith, J. (in press–a). Title. Journal.",
            "Smith, J. (In Press B). Title. Journal.",
            "Smith, J. (2021, in press). Title. Journal.",
            "Smith, J. (2021 in press). Title. Journal.",
            "Smith, J. (2021 forthcoming). Title. Journal.",
            "Smith J. Title. J Med in press. doi:10.1000/xyz",
            "Smith J. Title. J Med. 2021 Epub ahead of print. doi:10.1000/xyz",
            "Smith, J. (2021). Title. Journal. Advance online publication 12 March 2021.",
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

    def test_the_ner_path_flags_a_dash_suffixed_status(self):
        segment = "Robertson, C. E. (in press‐a). Inside the funhouse mirror factory. Journal."
        with _ner_parser({"title": "Inside the funhouse mirror factory", "authors": "Robertson"}):
            (ref,) = _extractor()._parse_references_ner([segment])

        assert ref.is_in_press is True
        assert ref.year is None


def _author_date_entry(i: int) -> str:
    return (
        f"Author{i}, A. B., Other{i}, C. ({1950 + i % 70}). A study of item {i} in the "
        f"memory literature. Journal of Studies, {i % 90 + 1}(2), {i}-{i + 9}."
    )


def _numbered_entry(i: int) -> str:
    return (
        f"{i}. Author{i} A, Other{i} B. A study of item {i} in the memory literature"
        + " with a title that goes on" * 12
        + f". J Stud. 2001;{i % 90 + 1}:{i}-{i + 9}."
    )


class TestMergedSplitIsLinear:
    """The old splitter took ~40 s on the author-date string and ~30 s on the
    numbered one; both now take a small fraction of a second."""

    @pytest.mark.parametrize(
        ("entry", "count"), [(_author_date_entry, 2000), (_numbered_entry, 999)]
    )
    def test_a_long_merged_string_splits_quickly(self, entry, count):
        entries = [entry(i) for i in range(1, count + 1)]
        start = time.perf_counter()
        split, added = split_merged_refs([" ".join(entries)])
        elapsed = time.perf_counter() - start

        assert split == entries
        assert added == count - 1
        assert elapsed < 2.0

    def test_a_lead_after_a_long_capitalized_url_still_opens_a_reference(self):
        """The bounded walk reads whole tokens: cutting the text a fixed width
        before the date would start inside the URL on "Abcdef…", read that
        fragment as a capitalized name and lose the onset."""
        url = "https://example.org/" + "Abcdefghij" * 15 + "xx"
        text = (
            f"Smith, J. (2001). A long report on things. {url} "
            "World Health Organization. (2005). Global report on falls."
        )
        assert find_interior_onsets(text) == [text.index("World")]

    @pytest.mark.parametrize(
        "text",
        [
            "Smith, J. (2001). A correction to Cousineau (2005). Journal, 1, 2.",
            "Brown, T. (2018). Beyond Kahneman and Tversky (1979): A reply. Journal, 2, 3.",
        ],
    )
    def test_an_in_title_citation_is_still_not_an_onset(self, text):
        assert find_interior_onsets(text) == []


class _CountingText(str):
    """Reference text that counts the characters ``find`` scans and slicing
    reads, whichever way the aligner walks it."""

    scanned: int

    def __new__(cls, value: str) -> _CountingText:
        text = super().__new__(cls, value)
        text.scanned = 0
        return text

    def find(self, sub, start=0, end=None):  # type: ignore[override]
        stop = len(self) if end is None else end
        pos = super().find(sub, start, stop)
        self.scanned += (stop if pos == -1 else pos + len(sub)) - start
        return pos

    def __getitem__(self, key):  # type: ignore[override]
        part = super().__getitem__(key)
        if isinstance(key, slice):
            self.scanned += len(part)
        return part


def _layout(count: int) -> tuple[str, list[str], list[bool], list[int]]:
    entries = [_author_date_entry(i) for i in range(1, count + 1)]
    text = "\n".join(entries)
    lines: list[str] = []
    bounds: list[bool] = []
    for entry in entries:
        for k in range(0, len(entry), 45):
            lines.append(entry[k : k + 45])
            bounds.append(k == 0)
    starts = [text.index(entry) for entry in entries]
    return text, lines, bounds, starts


class TestLineAlignmentIsLinear:
    def test_a_long_list_is_not_searched_once_per_line(self):
        raw, lines, bounds, expected = _layout(1000)
        text = _CountingText(raw)

        assert align_line_starts(text, lines, bounds) == expected
        # One pass reads a probe-sized window at each offset (30x the text);
        # the old walk searched the whole text for every one of the 3000 lines
        # (3000x).
        assert text.scanned <= 2 * anchor_snap.ANCHOR_LEN * len(text)

    @pytest.mark.parametrize(
        ("text", "lines", "bounds"),
        [
            (
                "Gamma, D., Delta, E. F., & Zeta, I. (2001). One thing. Journal A, 1, 1-9.\n"
                "Gamma, D., Delta, E. F., & Zeta, I. (2002). Other thing. Journal B, 2, 2-8.\n"
                "Alpha, B., Gamma, D., Delta, E. F., & Zeta, I. (2003). Third. J C, 3, 3-7.",
                [
                    "Gamma, D., Delta, E. F., & Zeta, I. (2002).",
                    "Other thing. Journal B, 2, 2-8.",
                    "Gamma, D., Delta, E. F., & Zeta, I. (2001).",
                    "One thing. Journal A, 1, 1-9.",
                    "Alpha, B., Gamma, D., Delta, E. F., & Zeta, I.",
                    "(2003). Third. J C, 3, 3-7.",
                    "12",
                    "Gamma, D., Delta, E. F., & Zeta, I. (2004).",
                ],
                [True, False, True, False, True, False, False, True],
            ),
            _layout(30)[:3],
        ],
        ids=["shuffled-repeats", "generated"],
    )
    def test_the_index_finds_what_the_per_line_search_found(self, text, lines, bounds, monkeypatch):
        searched = align_line_starts(text, lines, bounds)
        monkeypatch.setattr(anchor_snap, "_INDEX_MIN_PROBES", 0)
        assert align_line_starts(text, lines, bounds) == searched
