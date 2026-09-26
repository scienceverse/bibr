"""Tests for MetadataExtractor helper methods.

Tests pure logic methods (DOI regex, cutoff index, row collection)
without needing LLM models.
"""

import asyncio
import logging
import math
import threading
from unittest import mock

import pandas as pd
import pytest

from bibr import __version__ as bibr_version
from bibr.config import GlobalSettings
from bibr.exceptions import UpstreamServiceError
from bibr.extract.author_email_harvester import AuthorEmailHarvester
from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.extract.extractor import (
    MetadataExtractor,
    _await_core_and_reference_tasks,
    _build_ref_text,
    _infer_bibtype,
    _parse_refs_via_ner,
    _strip_enum_markers,
)
from bibr.extract.training_capture import save_ref_training_data, save_seg_training_data
from bibr.paper import PaperMetadata
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.utils.text import normalize_doi
from bibr.validation import ValidationIssue


def _make_extractor(
    sections, texts, detected_headers=None, detected_footers=None, paper_sections=None
):
    """Create a MetadataExtractor with a mock PaperContents."""
    df = pd.DataFrame({"section_name": sections, "text": texts})
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = detected_headers or []
    contents.detected_footers = detected_footers or []
    contents.layout_hints = []
    contents.sections = paper_sections or []
    contents.sentences = []
    return MetadataExtractor(contents)


def _make_extractor_pages(rows):
    """Extractor whose sentences_df carries section_name/text/page_number.

    rows: list of (section_name, text, page_number).
    """
    df = pd.DataFrame(rows, columns=["section_name", "text", "page_number"])
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    return MetadataExtractor(contents)


def _make_reclaim_extractor(rows):
    """Extractor whose sentences_df carries text_id/page_number (reclaim needs both).

    rows: list of (text_id, page_number, section_name, text).
    Returns (extractor, references-only DataFrame).
    """
    df = pd.DataFrame(rows, columns=["text_id", "page_number", "section_name", "text"])
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    ext = MetadataExtractor(contents)
    return ext, df[df["section_name"] == "References"]


def test_metadata_extractor_threads_front_matter_resolution_to_core():
    from bibr.extract.front_matter import FrontMatterResolution

    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {"text_id": [], "section_name": [], "text": [], "page_number": []}
    )
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    resolution = FrontMatterResolution(
        candidates=(),
        blocks=(),
        selected_block_id=None,
        selection_method="abstained",
        reason_flags=("multiple_plausible_blocks",),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )

    extractor = MetadataExtractor(contents, front_matter_resolution=resolution)

    assert extractor.core._front_matter_resolution is resolution


def test_metadata_extractor_compatibility_path_has_no_front_matter_resolution():
    extractor = _make_extractor(["Title"], ["A paper"])

    assert extractor.core._front_matter_resolution is None


async def test_metadata_extractor_scopes_email_harvester_to_selected_text_ids(monkeypatch):
    from bibr.extract.front_matter import (
        FrontMatterBlock,
        FrontMatterCandidate,
        FrontMatterResolution,
    )
    from bibr.paper_contents import PaperSentence
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    sentences = [
        PaperSentence(1, "Selected title", 0, 1, page_number=1),
        PaperSentence(2, "Alice Example", 0, 2, page_number=1),
        PaperSentence(
            3,
            "Corresponding author: Alice Example alice@outside.test",
            0,
            3,
            page_number=1,
        ),
        PaperSentence(4, "Body", 1, 4, page_number=1),
    ]
    sections = [
        PaperSection(0, "Title", 0, None, CanonicalSection.TITLE, 1.0),
        PaperSection(1, "Introduction", 1, None, CanonicalSection.INTRODUCTION, 1.0),
    ]
    contents = PaperContents(sentences, sections, [], [], {0: "", 1: ""})
    contents.detected_headers = ["Adjacent record DOI: 10.9999/outside.record"]
    contents.detected_footers = ["Wrong footer DOI: 10.9999/outside.footer"]

    def candidate(candidate_id, raw_text, roles, text_ids):
        return FrontMatterCandidate(
            candidate_id=candidate_id,
            source_kind="paragraph",
            reading_order=int(candidate_id[-1]),
            page=1,
            bbox=None,
            region_label="text",
            font_size=None,
            font_bold=None,
            section_id=0,
            text_ids=text_ids,
            paragraph_id=None,
            raw_text=raw_text,
            normalized_text=raw_text.casefold(),
            roles=frozenset(roles),
        )

    selected = (
        candidate("c1", "Selected title", {"title"}, (1,)),
        candidate("c2", "Alice Example", {"byline"}, (2,)),
    )
    outside = candidate(
        "c3",
        "Corresponding author: Alice Example alice@outside.test",
        set(),
        (3,),
    )
    resolution = FrontMatterResolution(
        candidates=(*selected, outside),
        blocks=(FrontMatterBlock("b1", ("c1", "c2"), ("c1",)),),
        selected_block_id="b1",
        selection_method="unique_block",
        reason_flags=(),
        allowed_text_ids=frozenset({1, 2}),
        allowed_section_ids=frozenset({0}),
    )
    llm = mock.MagicMock()
    llm.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="Selected title",
            authors=[AuthorLLM(given="Alice", family="Example")],
        )
    )
    extractor = MetadataExtractor(
        contents,
        llm_client=llm,
        front_matter_resolution=resolution,
    )
    monkeypatch.setattr(
        extractor.core,
        "_classify_paper",
        mock.AsyncMock(return_value=("", "", "", None, None)),
    )

    await extractor.extract_core_metadata()

    assert extractor.metadata.authors[0].email is None
    assert extractor.metadata.authors[0].corresponding is False
    assert extractor.metadata.doi == ""
    (llm_context,), _ = llm.extract_core_metadata.await_args
    assert "outside.record" not in llm_context
    assert "outside.footer" not in llm_context


class TestBuildRefText:
    """``ref_text`` must collapse OCR wide-letter-spacing before segmentation.

    Some preprints (eyecolor.pdf) are typeset with wide letter spacing; OCR
    renders the reference list with tabs, carriage returns, and non-breaking
    spaces between every token (``"1.\\t\\r \\xa0Zietsch,\\t\\r \\xa0B."``). The
    late whitespace cleanup (``finalize_text``) runs only AFTER extraction, so
    the segmenter would otherwise receive the raw spacing — anchor snapping then
    fails the exact match, the fuzzy fallback mis-locates the boundary, and the
    leading reference is dropped or mangled (ref #1 ``Zietsch`` → ``etsch``).
    Collapsing each row to single spaces, then joining with newlines, preserves
    the one-reference-per-line structure anchor snapping relies on.
    """

    def test_collapses_pathological_inter_token_whitespace(self):
        rows = [
            "1.\t\r \xa0Zietsch,\t\r \xa0B.\t\r \xa0P.\t\r \xa0Variation\t\r \xa0\r\nin mate (2011).",
            "2.\t\r \xa0Nojo,\t\r \xa0S. Human homogamy (2012).",
        ]
        assert _build_ref_text(rows) == (
            "1. Zietsch, B. P. Variation in mate (2011).\n2. Nojo, S. Human homogamy (2012)."
        )

    def test_preserves_one_reference_per_line(self):
        rows = ["Smith, J. (2020). A.", "Jones, A. (2019). B."]
        assert _build_ref_text(rows) == "Smith, J. (2020). A.\nJones, A. (2019). B."

    def test_drops_blank_and_whitespace_only_rows(self):
        rows = ["Smith, J. (2020).", "   \t \xa0 ", "Jones, A. (2019)."]
        assert _build_ref_text(rows) == "Smith, J. (2020).\nJones, A. (2019)."

    def test_leaves_already_clean_rows_untouched(self):
        rows = ["Baldauf, D., & Desimone, R. (2014). Neural mechanisms. Science, 344, 424-427."]
        assert _build_ref_text(rows) == rows[0]


class TestStripEnumMarkers:
    """Printed list-numbering must not reach the NER parser.

    The v4 GIANT checkpoint never learned bare dot-markers as O: it absorbs
    them into the adjacent field span (``1.`` → authors, ``42.`` → title,
    kelders-et-al-2024.pdf). Stripping is gated at the bibliography level —
    a numbered style numbers every entry, so a lone number-leading segment
    in an unnumbered bibliography is left alone.
    """

    def test_strips_dot_markers_from_numbered_bibliography(self):
        refs = [
            "1. Wongvibulsin S, Habeos EE, et al. Digital health. J Med Internet Res 2021.",
            "2. Wang Y, Min J, et al. Mobile health. JMIR Mhealth Uhealth 2020.",
            "42. Moshe I, Terhorst Y, et al. Digital interventions. Psychol Bull 2021.",
        ]
        assert _strip_enum_markers(refs) == [
            "Wongvibulsin S, Habeos EE, et al. Digital health. J Med Internet Res 2021.",
            "Wang Y, Min J, et al. Mobile health. JMIR Mhealth Uhealth 2020.",
            "Moshe I, Terhorst Y, et al. Digital interventions. Psychol Bull 2021.",
        ]

    def test_strips_bracketed_markers(self):
        refs = ["[1] Smith J. A paper. 2020.", "[2] Doe A. Another. 2019."]
        assert _strip_enum_markers(refs) == [
            "Smith J. A paper. 2020.",
            "Doe A. Another. 2019.",
        ]

    def test_unnumbered_bibliography_unchanged(self):
        refs = [
            "Smith, J. (2020). A paper. Journal, 5(3), 1-10.",
            "Doe, A. (2019). Another paper. Journal, 4(1), 11-20.",
        ]
        assert _strip_enum_markers(refs) == refs

    def test_minority_marked_segments_not_stripped(self):
        # One number-leading segment among unnumbered refs (e.g. a stray
        # numbered footnote that survived segment filtering) must stay verbatim.
        refs = [
            "1. Smith, J. (2020). A paper. Journal, 5(3), 1-10.",
            "Doe, A. (2019). Another paper. Journal, 4(1), 11-20.",
            "Roe, R. (2018). A third paper. Journal, 3(2), 21-30.",
        ]
        assert _strip_enum_markers(refs) == refs

    def test_year_leading_entry_untouched_even_in_numbered_bibliography(self):
        # 4-digit years don't match the 1-3 digit marker pattern; a year-leading
        # corporate entry inside a numbered bibliography must survive intact.
        refs = [
            "1. Smith J. A paper. J Test 2020.",
            "2. Doe A. Another. J Test 2019.",
            "1990. Annual report of the agency. Government Press.",
        ]
        assert _strip_enum_markers(refs)[2] == refs[2]

    def test_empty_list_passthrough(self):
        assert _strip_enum_markers([]) == []


class _FakeRefParser:
    """Captures parse_batch input; returns a minimal valid field dict per ref."""

    def __init__(self):
        self.seen: list[str] | None = None

    def parse_batch(self, ref_texts):
        self.seen = list(ref_texts)
        return [{"title": f"T{i}", "authors": f"A{i}"} for i in range(len(ref_texts))]


class TestParseReferencesNerStripsMarkers:
    def test_parser_receives_stripped_strings(self, monkeypatch):
        fake = _FakeRefParser()
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_parser", lambda *_a: fake)
        ext = _make_extractor([], [])
        refs = [
            "1. Wongvibulsin S, et al. Digital health. J Med Internet Res 2021; 23: e18773.",
            "2. Wang Y, et al. Mobile health. JMIR Mhealth Uhealth 2020; 8: e15400.",
        ]

        parsed = ext.refs._parse_references_ner(refs)

        assert fake.seen == [
            "Wongvibulsin S, et al. Digital health. J Med Internet Res 2021; 23: e18773.",
            "Wang Y, et al. Mobile health. JMIR Mhealth Uhealth 2020; 8: e15400.",
        ]
        assert [r.bib_id for r in parsed] == [1, 2]

    def test_aligned_parser_keeps_empty_prediction_slot(self, monkeypatch):
        class _Parser:
            def parse_batch(self, ref_texts):
                return [
                    {"title": "First", "authors": "Alpha A"},
                    {},
                    {"title": "Third", "authors": "Gamma G"},
                ]

        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_parser", lambda *_a: _Parser())
        ext = _make_extractor([], [])

        aligned = ext.refs._parse_references_ner_aligned(["first", "empty", "third"])

        assert aligned[1] is None
        assert [ref.bib_id for ref in aligned if ref is not None] == [1, 2]
        assert [
            ref.title for ref in ext.refs._parse_references_ner(["first", "empty", "third"])
        ] == [
            "First",
            "Third",
        ]


class TestParseReferencesNerDoiRescue:
    """The NER path must apply the same segment-scoped DOI rescue as the LLM
    path: a printed doi:/doi.org token in the ref's own segment fills a DOI the
    parser under-emitted, but never overrides one the parser did emit."""

    @staticmethod
    def _parser_returning(fields: dict):
        class _Parser:
            def parse_batch(self, ref_texts):
                return [dict(fields) for _ in ref_texts]

        return _Parser()

    def test_printed_doi_rescued_when_ner_misses_it(self, monkeypatch):
        parser = self._parser_returning({"title": "Digital health", "authors": "Wongvibulsin S"})
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_parser", lambda *_a: parser)
        ext = _make_extractor([], [])
        refs = [
            "Wongvibulsin S, et al. Digital health. J Med Internet Res 2021; "
            "23: e18773. doi:10.2196/18773"
        ]

        parsed = ext.refs._parse_references_ner(refs)

        assert parsed[0].doi == "10.2196/18773"

    def test_ner_emitted_doi_wins_over_segment_token(self, monkeypatch):
        parser = self._parser_returning(
            {"title": "Digital health", "authors": "Wongvibulsin S", "doi": "10.1000/real.1"}
        )
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_parser", lambda *_a: parser)
        ext = _make_extractor([], [])
        refs = ["Wongvibulsin S. Digital health. 2021. doi:10.9999/other.2"]

        parsed = ext.refs._parse_references_ner(refs)

        assert parsed[0].doi == "10.1000/real.1"


class TestReclaimBoundaryOrphans:
    """The reclaim must rescue genuine head fragments, not preceding front-matter.

    False-orphan strings are verbatim from the exp #2 / B″ re-gate judged
    failures: an acknowledgment sentence, a numbered footnote, and the paper's
    own byline were each reclaimed into ref_text and emitted as phantom bib
    stubs. In all three papers the first reference row was a COMPLETE entry —
    nothing was orphaned, so nothing should have been reclaimed.
    """

    ACK_PROSE = (
        "A reviewer pointed out that statistical support for the direct contrast of theta "
        "and alpha effects in the original study is weak and so should be interpreted with "
        "caution (Gelman & Stern, 2006; Nieuwenhuis et al., 2011)."
    )
    FOOTNOTE = (
        "1. Following Wuensch's (2009) recommendations, we report 90% CIs for squared "
        "effect-size estimates."
    )
    BYLINE = "Delaram Farzanfar, Dirk B. Walther"
    COMPLETE_FIRST_REF = (
        "Baldauf, D., & Desimone, R. (2014). Neural mechanisms of object-based "
        "attention. Science, 344(6182), 424-427."
    )

    def _reclaimed(self, orphan_text, first_ref_text):
        ext, ref_df = _make_reclaim_extractor(
            [
                (132, 17, "Discussion", "Some unrelated discussion sentence."),
                (133, 17, "Acknowledgments", orphan_text),
                (134, 17, "References", first_ref_text),
                (
                    135,
                    17,
                    "References",
                    "Capilla, A., & Gross, J. (2011). Steady-state "
                    "visual evoked potentials. NeuroImage, 54(2), 836-851.",
                ),
            ]
        )
        out = ext.locator._reclaim_boundary_orphans(ref_df, "References")
        return list(out["text_id"])

    # --- false orphans: first reference row is complete → reclaim nothing ---

    def test_prose_sentence_with_cites_not_reclaimed(self):
        assert self._reclaimed(self.ACK_PROSE, self.COMPLETE_FIRST_REF) == [134, 135]

    def test_numbered_footnote_not_reclaimed(self):
        assert self._reclaimed(self.FOOTNOTE, self.COMPLETE_FIRST_REF) == [134, 135]

    def test_byline_not_reclaimed(self):
        assert self._reclaimed(self.BYLINE, self.COMPLETE_FIRST_REF) == [134, 135]

    # --- genuine head fragments: first reference row is a tail → reclaim ---

    def test_lowercase_continuation_tail_reclaims_head(self):
        head = (
            "Oostenveld, R., Fries, P., Maris, E., & Schoffelen, J.-M. (2011). "
            "FieldTrip: Open source software for advanced analysis"
        )
        tail = (
            "of MEG, EEG, and invasive electrophysiological data. Computational "
            "Intelligence and Neuroscience, 2011, Article 156869."
        )
        assert self._reclaimed(head, tail) == [133, 134, 135]

    def test_yearless_container_tail_reclaims_head(self):
        head = (
            "Hixon, J. G., & Swann, W. B. (1993). When does introspection bear fruit? "
            "Self-reflection, self-insight, and interpersonal choices."
        )
        tail = "Journal of Personality and Social Psychology, 64(1), 35-43."
        assert self._reclaimed(head, tail) == [133, 134, 135]

    # --- numbered reference styles still count as complete entry starts ---

    def test_numbered_first_ref_is_complete_no_reclaim(self):
        first = "[3] Wuensch, K. L. (2009). Standardized effect sizes. Journal, 14, 1-9."
        assert self._reclaimed(self.ACK_PROSE, first) == [134, 135]


class TestFindDoi:
    """Tests for _find_doi regex extraction."""

    def test_standard_doi(self):
        ext = _make_extractor([], [])
        assert ext.core._find_doi("DOI: 10.1038/nature12373") == "10.1038/nature12373"

    def test_doi_in_url(self):
        ext = _make_extractor([], [])
        result = ext.core._find_doi("https://doi.org/10.1016/j.cell.2020.01.001")
        assert result == "10.1016/j.cell.2020.01.001"

    def test_doi_with_special_chars(self):
        ext = _make_extractor([], [])
        assert ext.core._find_doi("10.1000/xyz123-abc") == "10.1000/xyz123-abc"

    def test_no_doi(self):
        ext = _make_extractor([], [])
        assert ext.core._find_doi("No DOI here") is None

    def test_empty_string(self):
        ext = _make_extractor([], [])
        assert ext.core._find_doi("") is None

    def test_doi_with_parentheses(self):
        ext = _make_extractor([], [])
        result = ext.core._find_doi("10.1002/(SICI)1097-0258")
        assert result == "10.1002/(SICI)1097-0258"

    def test_hyphen_space_split_doi_is_bridged(self):
        # exp #2 (ja.2018-26): EOL-hyphenated DOI joined as "hyphen space" —
        # the regex must bridge the gap instead of truncating at the space.
        ext = _make_extractor([], [])
        text = (
            "Economics E-Journal, 12 (2018-26): 1-9. "
            "http://dx.doi.org/10.5018/economics- ejournal.ja.2018-26"
        )
        assert ext.core._find_doi(text) == "10.5018/economics-ejournal.ja.2018-26"

    def test_trailing_hyphen_not_bridged_into_prose(self):
        # The guard is that the hyphen-space bridge above must not swallow the
        # following prose. It previously fell back to the stub "10.1234/abc-";
        # normalize_doi now rejects a hyphen-terminated suffix, so the decline is
        # a clean abstention instead of a DOI that resolves to nothing.
        ext = _make_extractor([], [])
        assert ext.core._find_doi("DOI: 10.1234/abc- The Journal of Things") is None

    def test_truncated_prefix_loses_to_full_doi(self):
        # wrap-truncated copy earlier in the text must not shadow the full DOI.
        # Resolved by priority ranking (marker > doi.org/ URL), not by
        # prefix-dropping — the latter was removed because it also discarded
        # correct DOIs that other candidates merely extend (e.g. figure DOIs).
        ext = _make_extractor([], [])
        text = (
            "Citation: https://doi.org/10.1371/journal.\n"
            "some prose here\n"
            "DOI: 10.1371/journal.pone.0279511\n"
        )
        assert ext.core._find_doi(text) == "10.1371/journal.pone.0279511"

    def test_article_doi_beats_journal_doi(self):
        ext = _make_extractor([], [])
        text = (
            "Journal DOI: https://doi.org/10.46654/RJMP\n"
            "Article DOI: https://doi.org/10.46654/RJMP.14033\n"
        )
        assert ext.core._find_doi(text) == "10.46654/RJMP.14033"

    def test_journal_doi_alone_is_still_found(self):
        ext = _make_extractor([], [])
        text = "Journal DOI: https://doi.org/10.46654/RJMP\n"
        assert ext.core._find_doi(text) == "10.46654/RJMP"

    def test_unrelated_longer_doi_does_not_shadow(self):
        ext = _make_extractor([], [])
        text = "DOI: 10.1234/abc\nsee also 10.1234/abcdef.99\n"
        # 10.1234/abc is a DOI: marker match (priority 3), 10.1234/abcdef.99 is bare (priority 1)
        assert ext.core._find_doi(text) == "10.1234/abc"

    def test_figure_doi_does_not_shadow_paper_doi(self):
        # PLOS prints figure DOIs in body captions, inside the pre-references
        # window. They extend the paper's own DOI, so a prefix-dropping rule
        # would discard the correct DOI in favour of a figure's.
        ext = _make_extractor([], [])
        text = (
            "PLoS ONE 10(6): e0130688. doi:10.1371/journal.pone.0130688\n"
            "Fig 1. Screening strategy.\ndoi:10.1371/journal.pone.0130688.g001\n"
            "Fig 5. Median fin fold.\ndoi:10.1371/journal.pone.0130688.g005\n"
        )
        assert ext.core._find_doi(text) == "10.1371/journal.pone.0130688"


class TestGetCutoffIndex:
    """Tests for _get_cutoff_index, which finds where metadata ends."""

    def test_cutoff_at_introduction(self):
        """Cutoff should be at the first Introduction row."""
        sections = ["Abstract"] * 5 + ["1. Introduction"] * 10 + ["2. Methods"] * 10
        texts = [f"Sentence {i}" for i in range(25)]

        paper_sections = [
            PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
            PaperSection(1, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(2, "2. Methods", 2, None, CanonicalSection.METHODS, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        cutoff = ext.locator.get_cutoff_index()
        assert cutoff == 5  # Introduction starts at index 5

    def test_cutoff_at_methods_when_no_introduction(self):
        """If no Introduction found, fall back to Methods."""
        sections = ["Abstract"] * 5 + ["Methods"] * 10
        texts = [f"Sentence {i}" for i in range(15)]

        paper_sections = [
            PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
            PaperSection(1, "Methods", 2, None, CanonicalSection.METHODS, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        cutoff = ext.locator.get_cutoff_index()
        assert cutoff == 5

    def test_fallback_250(self):
        """When no standard sections found, returns 250."""
        sections = ["Unknown"] * 300
        texts = [f"Sentence {i}" for i in range(300)]

        paper_sections = [
            PaperSection(0, "Unknown", 2, None, CanonicalSection.UNKNOWN, 0.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        cutoff = ext.locator.get_cutoff_index()
        assert cutoff == 250

    def test_early_cutoff_below_cap_unchanged(self):
        """A found cutoff below the cap is returned untouched."""
        sections = ["Abstract"] * 5 + ["1. Introduction"] * 10
        texts = [f"Sentence {i}" for i in range(15)]

        paper_sections = [
            PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
            PaperSection(1, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        assert ext.locator.get_cutoff_index() == 5  # 5 < default cap 250

    def test_late_cutoff_capped_and_warns(self, monkeypatch, caplog):
        """A found cutoff above the cap returns the cap and logs a warning."""
        from bibr.config import Settings

        monkeypatch.setattr(Settings.llm, "core_cutoff_max_sentences", 4)
        # Intro misclassified (UNKNOWN); only Discussion matches, at index 10.
        sections = ["Body"] * 10 + ["Discussion"] * 5
        texts = [f"Sentence {i}" for i in range(15)]
        paper_sections = [
            PaperSection(0, "Body", 2, None, CanonicalSection.UNKNOWN, 0.0),
            PaperSection(1, "Discussion", 2, None, CanonicalSection.DISCUSSION, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        with caplog.at_level(logging.WARNING):
            cutoff = ext.locator.get_cutoff_index()
        assert cutoff == 4
        assert any(
            "DISCUSSION" in r.message and "10" in r.message and "4" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_no_match_uses_setting(self, monkeypatch):
        """When no IMRaD header matches, the cap setting is used as the fallback."""
        from bibr.config import Settings

        monkeypatch.setattr(Settings.llm, "core_cutoff_max_sentences", 42)
        sections = ["Unknown"] * 100
        texts = [f"Sentence {i}" for i in range(100)]
        paper_sections = [
            PaperSection(0, "Unknown", 2, None, CanonicalSection.UNKNOWN, 0.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        assert ext.locator.get_cutoff_index() == 42

    def test_setting_zero_disables_cap(self, monkeypatch):
        """Setting 0 preserves legacy behavior: uncapped cutoff, 250 fallback."""
        from bibr.config import Settings

        monkeypatch.setattr(Settings.llm, "core_cutoff_max_sentences", 0)
        # Late cutoff is NOT capped.
        sections = ["Body"] * 10 + ["Discussion"] * 5
        texts = [f"Sentence {i}" for i in range(15)]
        paper_sections = [
            PaperSection(0, "Body", 2, None, CanonicalSection.UNKNOWN, 0.0),
            PaperSection(1, "Discussion", 2, None, CanonicalSection.DISCUSSION, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        assert ext.locator.get_cutoff_index() == 10

        # Fallback stays 250 when no IMRaD header matches.
        sections2 = ["Unknown"] * 300
        texts2 = [f"Sentence {i}" for i in range(300)]
        paper_sections2 = [
            PaperSection(0, "Unknown", 2, None, CanonicalSection.UNKNOWN, 0.0),
        ]
        ext2 = _make_extractor(sections2, texts2, paper_sections=paper_sections2)
        assert ext2.locator.get_cutoff_index() == 250


class TestCollectCoreMetadataRows:
    """Tests for _collect_core_metadata_rows."""

    def test_includes_rows_before_cutoff(self):
        sections = ["Title"] * 5 + ["Introduction"] * 10
        texts = [f"Sentence {i}" for i in range(15)]

        ext = _make_extractor(sections, texts)
        result = ext.locator.collect_core_metadata_rows(cutoff_iloc=5)
        assert len(result) == 5

    def test_includes_orcid_rows(self):
        """ORCID rows from anywhere in the document should be included."""
        sections = ["Title"] * 3 + ["Methods"] * 5 + ["Author Info"] * 2
        texts = [
            "Title line 1",
            "Title line 2",
            "Title line 3",
            "Method 1",
            "Method 2",
            "Method 3",
            "Method 4",
            "Method 5",
            "John Doe https://orcid.org/0000-0001-2345-6789",
            "Jane Doe https://orcid.org/0000-0002-3456-7890",
        ]

        ext = _make_extractor(sections, texts)
        result = ext.locator.collect_core_metadata_rows(cutoff_iloc=3)
        # 3 title rows + 2 ORCID rows (in "Author Info" section)
        assert len(result) >= 3
        # The ORCID rows should be included
        orcid_texts = result["text"].str.contains("orcid.org", na=False)
        assert orcid_texts.sum() == 2

    def test_no_duplicates(self):
        """Rows that are both before cutoff and contain ORCID shouldn't be duplicated."""
        sections = ["Title"] * 3
        texts = [
            "Title with https://orcid.org/0000-0001-2345-6789",
            "Other title text",
            "More title text",
        ]

        ext = _make_extractor(sections, texts)
        result = ext.locator.collect_core_metadata_rows(cutoff_iloc=3)
        assert len(result) == 3  # No duplicates

    def test_pulls_page1_corresponding_footnote_below_cutoff(self):
        """De Gruyter affiliation/ORCID footnotes land below the cutoff in
        stream order but on page 1; they must be pulled into the metadata
        context so the LLM can extract affiliation + ORCID (qual-grade30 M6)."""
        rows = [
            ("Title", "Macroprudential Policy and the Financial Cycle", 1),
            ("Title", "Martina Basarac Sertić, Valentina Vučković, Ana Andabaka", 1),
            ("Introduction", "This paper studies macroprudential policy.", 1),
            ("Body", "Body text continues on page 2.", 2),
            (
                "Footnote",
                "* Corresponding author: Martina Basarac Sertić, Economic Research "
                "Division, Croatian Academy of Sciences and Arts, Zagreb, Croatia",
                1,
            ),
            (
                "Footnote",
                "Valentina Vučković, Ana Andabaka: Faculty of Economics and Business, "
                "University of Zagreb 0000-0002-5438-0665",
                1,
            ),
        ]
        ext = _make_extractor_pages(rows)
        result = ext.locator.collect_core_metadata_rows(cutoff_iloc=2)
        joined = " || ".join(result["text"])
        assert "Corresponding author: Martina" in joined  # corresp footnote pulled
        assert "0000-0002-5438-0665" in joined  # bare-ORCID footnote pulled
        assert "Body text continues on page 2" not in joined  # page-2 body excluded
        assert "This paper studies" not in joined  # post-cutoff page-1 body excluded

    def test_does_not_pull_page2_correspondence(self):
        """The footnote rescue is scoped to page 1; a later-page correspondence
        line is not pulled (avoids dragging body/reference text into metadata)."""
        rows = [
            ("Title", "A Title", 1),
            ("Introduction", "Intro text.", 1),
            ("Acknowledgements", "Correspondence concerning this article, page 2.", 2),
        ]
        ext = _make_extractor_pages(rows)
        result = ext.locator.collect_core_metadata_rows(cutoff_iloc=1)
        joined = " || ".join(result["text"])
        assert "Correspondence concerning this article" not in joined

    def test_orcid_org_url_still_pulled_with_page_column(self):
        """The existing orcid.org rescue keeps working when page_number exists,
        regardless of page (a bare-ORCID gate must not replace it)."""
        rows = [
            ("Title", "A Title", 1),
            ("Introduction", "Intro text.", 1),
            ("Footnote", "Jane Doe https://orcid.org/0000-0002-3456-7890", 2),
        ]
        ext = _make_extractor_pages(rows)
        result = ext.locator.collect_core_metadata_rows(cutoff_iloc=1)
        joined = " || ".join(result["text"])
        assert "orcid.org/0000-0002-3456-7890" in joined


class TestCollectReferenceRows:
    """Tests for _collect_reference_rows."""

    def test_collects_reference_section(self):
        sections = ["Introduction"] * 5 + ["References"] * 3
        texts = [f"Sentence {i}" for i in range(8)]

        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        result = ext._collect_reference_rows()
        assert len(result) == 3

    def test_raises_when_no_references(self):
        sections = ["Introduction"] * 5
        texts = [f"Sentence {i}" for i in range(5)]

        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)

        import pytest

        with pytest.raises(ValueError, match="No reference section"):
            ext._collect_reference_rows()


class TestHeaderAliasPrecedence:
    """A printed references heading must win over a misclassified canonical map.

    exp #2 (09567976241258149): the classifier typed a body section
    ("Flipped preferences") as REFERENCES while the literal "References"
    header was typed UNKNOWN — the canonical-map path then short-circuited
    and extraction collapsed to in-text citation stubs in both arms.
    """

    def test_header_alias_overrides_misclassified_canonical_map(self):
        sections = ["Introduction"] * 3 + ["Flipped preferences"] * 2 + ["References"] * 4
        texts = (
            [f"Intro {i}" for i in range(3)] + ["Body A", "Body B"] + [f"Ref {i}" for i in range(4)]
        )
        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            # misclassified body section
            PaperSection(1, "Flipped preferences", 2, None, CanonicalSection.REFERENCES, 0.6),
            # the real references list, typed UNKNOWN by the classifier
            PaperSection(2, "References", 2, None, CanonicalSection.UNKNOWN, 0.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        result = ext._collect_reference_rows()
        assert len(result) == 4
        assert all(t.startswith("Ref ") for t in result["text"])

    def test_agreeing_header_and_map_unchanged(self):
        sections = ["Introduction"] * 5 + ["References"] * 3
        texts = [f"Sentence {i}" for i in range(8)]
        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        result = ext._collect_reference_rows()
        assert len(result) == 3

    def test_no_alias_header_falls_back_to_canonical_map(self):
        """Without a literal references heading, the canonical map is trusted as before."""
        sections = ["Introduction"] * 2 + ["Literatur"] * 3
        texts = [f"Sentence {i}" for i in range(5)]
        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            # non-English heading classified as REFERENCES — no alias match
            PaperSection(1, "Literatur", 2, None, CanonicalSection.REFERENCES, 0.9),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        result = ext._collect_reference_rows()
        assert len(result) == 3

    def test_substring_header_does_not_trigger_override(self):
        """Headers merely containing 'references' mid-string must not pre-empt the map."""
        sections = ["Comparing references across cultures"] * 2 + ["Literatur"] * 3
        texts = ["Body A", "Body B"] + [f"Ref {i}" for i in range(3)]
        paper_sections = [
            PaperSection(
                0, "Comparing references across cultures", 2, None, CanonicalSection.UNKNOWN, 0.0
            ),
            PaperSection(1, "Literatur", 2, None, CanonicalSection.REFERENCES, 0.9),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        result = ext._collect_reference_rows()
        assert len(result) == 3
        assert all(t.startswith("Ref ") for t in result["text"])


class TestFindDoiHeadersFooters:
    """Tests for DOI extraction from headers and footers."""

    def test_doi_found_in_footer(self):
        ext = _make_extractor(
            [],
            [],
            detected_footers=["Journal of Testing | https://doi.org/10.1234/test.2024"],
        )
        assert ext.core._find_doi("No DOI in body") == "10.1234/test.2024"

    def test_doi_found_in_header(self):
        ext = _make_extractor(
            [],
            [],
            detected_headers=["DOI: 10.5678/header-doi.001"],
        )
        assert ext.core._find_doi("No DOI in body") == "10.5678/header-doi.001"

    def test_body_doi_takes_precedence(self):
        """DOI in the main text should be preferred over header/footer."""
        ext = _make_extractor(
            [],
            [],
            detected_footers=["https://doi.org/10.9999/footer-doi"],
        )
        assert ext.core._find_doi("Paper DOI: 10.1111/body-doi") == "10.1111/body-doi"

    def test_no_doi_anywhere(self):
        ext = _make_extractor(
            [],
            [],
            detected_headers=["Volume 42, Issue 3"],
            detected_footers=["Page 1 of 10"],
        )
        assert ext.core._find_doi("No DOI here either") is None

    def test_footer_checked_before_header(self):
        """Footers are more common for DOIs, so they are checked first."""
        ext = _make_extractor(
            [],
            [],
            detected_footers=["10.1000/footer-first"],
            detected_headers=["10.2000/header-second"],
        )
        assert ext.core._find_doi("No body DOI") == "10.1000/footer-first"


class TestFindDoiWithFallback:
    """Tests for _find_doi_with_fallback's early-page (pages 1-2) rescue."""

    def test_page2_doi_found_via_early_page_fallback(self):
        rows = [
            ("Title", "A Study of Reproductive Health Outcomes", 1),
            ("Introduction", "This paper examines maternal outcomes.", 1),
            (
                "Body",
                "Full text available at http://dx.doi.org/10.5935/1981-2965.20170016",
                2,
            ),
        ]
        ext = _make_extractor_pages(rows)
        front_text = (
            "A Study of Reproductive Health Outcomes This paper examines maternal outcomes."
        )
        assert ext.core._find_doi_with_fallback(front_text) == "10.5935/1981-2965.20170016"

    def test_page1_doi_still_found_via_early_page_fallback(self):
        """Widening to pages 1-2 must not regress the original page-1 case."""
        rows = [
            ("Title", "A Title With No DOI In The Front Matter", 1),
            ("Footer", "DOI: 10.1234/page1-doi", 1),
        ]
        ext = _make_extractor_pages(rows)
        front_text = "A Title With No DOI In The Front Matter"
        assert ext.core._find_doi_with_fallback(front_text) == "10.1234/page1-doi"

    def test_no_doi_anywhere_returns_none(self):
        rows = [
            ("Title", "A Title With No DOI Anywhere", 1),
            ("Body", "More text, still no DOI.", 2),
        ]
        ext = _make_extractor_pages(rows)
        front_text = "A Title With No DOI Anywhere"
        assert ext.core._find_doi_with_fallback(front_text) is None

    def test_page2_reference_doi_is_not_taken_as_paper_doi(self):
        """A short paper with no front-matter DOI whose reference list begins on
        page 2 must not surface a reference DOI as its own — the fallback drops
        rows printed under a references heading."""
        rows = [
            ("Title", "A Brief Report With No Front-Matter DOI", 1),
            ("Introduction", "This short report prints no DOI on page 1.", 1),
            (
                "References",
                "Smith J. Prior work. J Things 2020;1:2. https://doi.org/10.9999/reference.only",
                2,
            ),
        ]
        ext = _make_extractor_pages(rows)
        front_text = (
            "A Brief Report With No Front-Matter DOI This short report prints no DOI on page 1."
        )
        assert ext.core._find_doi_with_fallback(front_text) is None

    def test_page2_footnote_doi_survives_reference_exclusion(self):
        """An early-page footnote DOI remains eligible when reference rows are excluded."""
        rows = [
            ("Title", "A Study of Reproductive Health Outcomes", 1),
            ("Introduction", "This paper examines maternal outcomes.", 1),
            ("References", "Smith J. Prior work. https://doi.org/10.9999/reference.only", 2),
            # Footnote appended at document end, page 2, no references heading.
            (None, "http://dx.doi.org/10.5935/1981-2965.20170016", 2),
        ]
        ext = _make_extractor_pages(rows)
        front_text = (
            "A Study of Reproductive Health Outcomes This paper examines maternal outcomes."
        )
        assert ext.core._find_doi_with_fallback(front_text) == "10.5935/1981-2965.20170016"


class TestDoiPoisoning:
    """M2: ORCID rows pulled from the reference list must not poison DOI extraction.

    `_collect_core_metadata_rows` includes every row containing "orcid.org"
    from the whole document. A reference-list row carrying both an ORCID URL
    and a doi.org URL would inject a priority-2 DOI candidate that outranks
    the paper's own DOI when the real DOI prints only bare (priority 1).
    """

    @staticmethod
    def _poisoned_extractor():
        sections = ["Title"] * 2 + ["1. Introduction"] * 2 + ["References"] * 2
        texts = [
            "A Study of Things",
            # Paper's own DOI prints bare (priority 1) in the banner line.
            "Journal of Stuff 12, 34-56 (2024) 10.1234/real.paper",
            "Intro sentence one.",
            "Intro sentence two.",
            # Reference row with ORCID + doi.org URL (priority 2).
            "Smith J. https://orcid.org/0000-0001-2345-6789. "
            "Dataset. https://doi.org/10.9999/poison.ref",
            "Another reference.",
        ]
        paper_sections = [
            PaperSection(0, "Title", 2, None, CanonicalSection.UNKNOWN, 0.0),
            PaperSection(1, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(2, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
        ]
        return _make_extractor(sections, texts, paper_sections=paper_sections)

    async def test_ref_row_doi_does_not_outrank_paper_doi(self):
        ext = self._poisoned_extractor()
        with mock.patch.object(ext.core, "_call_core_llm", return_value=None):
            await ext.extract_core_metadata()
        assert ext.metadata.doi == "10.1234/real.paper"

    async def test_orcid_rows_still_reach_llm_text(self):
        """The ORCID rows must stay in the LLM input (author<->ORCID mapping)."""
        ext = self._poisoned_extractor()
        with mock.patch.object(ext.core, "_call_core_llm", return_value=None) as llm:
            await ext.extract_core_metadata()
        (full_text,), _ = llm.call_args
        assert "orcid.org/0000-0001-2345-6789" in full_text


class TestCoreMetadataFailurePolicy:
    async def test_upstream_service_failure_propagates(self):
        ext = _make_extractor(["Title"], ["Test paper"])
        ext.core.llm_client = mock.MagicMock()
        ext.core.llm_client.extract_core_metadata = mock.AsyncMock(
            side_effect=UpstreamServiceError("LLM", "service down")
        )

        with pytest.raises(UpstreamServiceError):
            await ext.core._call_core_llm("Test paper")

    async def test_invalid_output_processing_failure_preserves_identity(self):
        from bibr.exceptions import ProcessingError

        error = ProcessingError(
            "LLM returned invalid structured output",
            error_code="llm_invalid_output",
        )
        ext = _make_extractor(["Title"], ["Test paper"])
        ext.core.llm_client = mock.MagicMock()
        ext.core.llm_client.extract_core_metadata = mock.AsyncMock(side_effect=error)

        with pytest.raises(ProcessingError) as raised:
            await ext.core._call_core_llm("Test paper")

        assert raised.value is error


async def test_completed_typed_reference_failure_wins_over_core_failure():
    from bibr.exceptions import ProcessingError

    error = ProcessingError(
        "LLM returned invalid structured output",
        error_code="llm_invalid_output",
    )
    reference_finished = asyncio.Event()

    async def fail_references():
        reference_finished.set()
        raise error

    async def fail_core_later():
        await reference_finished.wait()
        await asyncio.sleep(0.01)
        raise RuntimeError("ordinary core failure must not mask typed references")

    core_task = asyncio.create_task(fail_core_later())
    ref_task = asyncio.create_task(fail_references())

    with pytest.raises(ProcessingError) as raised:
        await _await_core_and_reference_tasks(core_task, ref_task)

    assert raised.value is error
    assert core_task.done()


class TestCollectReferenceRowsLayoutFallback:
    """Tests for layout-hint fallback in _collect_reference_rows."""

    def test_fallback_to_last_unknown_section(self):
        """When no REFERENCES section is found but layout hints indicate references,
        the last UNKNOWN section should be used as fallback."""
        sections = ["Introduction"] * 5 + ["Literaturverzeichnis"] * 3
        texts = [f"Sentence {i}" for i in range(8)]

        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "Literaturverzeichnis", 2, None, CanonicalSection.UNKNOWN, 0.3),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        ext.contents.layout_hints = [("reference", 5), ("reference_content", 6)]

        result = ext._collect_reference_rows()
        assert len(result) == 3

    def test_no_fallback_without_layout_hints(self):
        """Without layout hints, missing REFERENCES section raises ValueError."""
        # A heading that is not a references heading in any language: the
        # header-text fallback would take "Literaturverzeichnis".
        sections = ["Introduction"] * 5 + ["Anhang"] * 3
        texts = [f"Sentence {i}" for i in range(8)]

        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "Anhang", 2, None, CanonicalSection.UNKNOWN, 0.3),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        ext.contents.layout_hints = []

        import pytest

        with pytest.raises(ValueError, match="No reference section"):
            ext._collect_reference_rows()

    def test_no_fallback_when_all_sections_classified(self):
        """If all sections are classified (no UNKNOWN), layout fallback raises."""
        sections = ["Introduction"] * 5 + ["Discussion"] * 3
        texts = [f"Sentence {i}" for i in range(8)]

        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "Discussion", 2, None, CanonicalSection.DISCUSSION, 0.9),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        ext.contents.layout_hints = [("reference", 3)]

        import pytest

        with pytest.raises(ValueError, match="layout hints indicate"):
            ext._collect_reference_rows()

    def test_normal_references_section_preferred(self):
        """When REFERENCES is properly classified, layout fallback is not used."""
        sections = ["Introduction"] * 3 + ["References"] * 4 + ["Unknown End"] * 2
        texts = [f"Sentence {i}" for i in range(9)]

        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
            PaperSection(2, "Unknown End", 2, None, CanonicalSection.UNKNOWN, 0.2),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        ext.contents.layout_hints = [("reference", 3)]

        result = ext._collect_reference_rows()
        # Should get the References section, not the Unknown End
        assert len(result) == 4


class TestInferBibtype:
    """Tests for _infer_bibtype — returns v6.0 BibType strings."""

    def test_journal_article(self):
        assert _infer_bibtype("Nature", None, "") == "journal_article"

    def test_book_by_isbn(self):
        assert _infer_bibtype(None, "978-0-123", "") == "book"

    def test_conference_paper(self):
        assert _infer_bibtype(None, None, "In Proceedings of ACM") == "conference_paper"

    def test_conference(self):
        assert _infer_bibtype(None, None, "Proc. of a conference") == "conference_paper"

    def test_thesis(self):
        assert _infer_bibtype(None, None, "PhD Thesis, MIT") == "thesis"

    def test_dissertation(self):
        assert _infer_bibtype(None, None, "Doctoral dissertation, Stanford") == "thesis"

    def test_masters_thesis(self):
        assert _infer_bibtype(None, None, "Master's thesis, ETH Zurich") == "thesis"

    def test_preprint_arxiv(self):
        assert _infer_bibtype(None, None, "arXiv:2301.12345") == "preprint"

    def test_preprint_biorxiv(self):
        assert _infer_bibtype(None, None, "bioRxiv preprint") == "preprint"

    def test_preprint_ssrn(self):
        assert _infer_bibtype(None, None, "Available at SSRN") == "preprint"

    def test_report(self):
        assert _infer_bibtype(None, None, "Technical Report TR-2020-01") == "report"

    def test_working_paper(self):
        assert _infer_bibtype(None, None, "Working paper series") == "report"

    def test_book_chapter(self):
        assert _infer_bibtype(None, None, "In: Smith (Ed.) Handbook, Chapter 3") == "book_chapter"

    def test_chapter_keyword(self):
        assert _infer_bibtype(None, None, "Book chapter on topic X") == "book_chapter"

    def test_crossref_type_journal_article(self):
        assert _infer_bibtype(None, None, "", crossref_type="journal-article") == "journal_article"

    def test_crossref_type_dissertation(self):
        assert _infer_bibtype(None, None, "", crossref_type="dissertation") == "thesis"

    def test_crossref_type_posted_content(self):
        assert _infer_bibtype(None, None, "", crossref_type="posted-content") == "preprint"

    def test_crossref_type_overrides_heuristic(self):
        # Even if container is set, crossref_type takes priority
        assert _infer_bibtype("Nature", None, "", crossref_type="book") == "book"

    def test_crossref_type_proceedings(self):
        assert _infer_bibtype(None, None, "", crossref_type="proceedings") == "conference_paper"

    def test_crossref_type_component_dataset(self):
        assert _infer_bibtype(None, None, "", crossref_type="component") == "other"
        assert _infer_bibtype(None, None, "", crossref_type="dataset") == "dataset"

    def test_unknown(self):
        assert _infer_bibtype(None, None, "Some random text") is None

    def test_crossref_type_unknown_falls_through(self):
        # Unknown crossref type maps to "other" via migrate_bib_type
        assert _infer_bibtype("Nature", None, "", crossref_type="unknown-type") == "other"


class TestNormalizeDoi:
    """Tests for DOI normalization — covers the exact errors from CrossRef logs."""

    def test_bare_doi_unchanged(self):
        assert (
            normalize_doi("10.1016/j.compppsych.2017.04.004") == "10.1016/j.compppsych.2017.04.004"
        )

    def test_strips_https_doi_org(self):
        assert (
            normalize_doi("https://doi.org/10.1016/j.compppsych.2017.04.004")
            == "10.1016/j.compppsych.2017.04.004"
        )

    def test_strips_partial_url_prefix(self):
        # Exact error from logs: NER subword artifact strips "https" leaving "://"
        assert (
            normalize_doi("://doi.org/10.1016/j.compppsych.2017.04.004")
            == "10.1016/j.compppsych.2017.04.004"
        )

    def test_strips_http_doi_org(self):
        assert (
            normalize_doi("http://doi.org/10.1186/s12916-015-0325-4") == "10.1186/s12916-015-0325-4"
        )

    def test_strips_dx_doi_org(self):
        assert normalize_doi("https://dx.doi.org/10.1002/cpp.1929") == "10.1002/cpp.1929"

    def test_strips_doi_colon_prefix(self):
        assert normalize_doi("doi:10.1080/14737175.2017.1307737") == "10.1080/14737175.2017.1307737"

    def test_strips_bare_doi_org(self):
        assert normalize_doi("doi.org/10.1002/cpp.1929") == "10.1002/cpp.1929"

    def test_rejects_parenthesis(self):
        # Exact error from logs: NER tagged "(" as DOI
        assert normalize_doi("(") is None

    def test_rejects_empty(self):
        assert normalize_doi("") is None
        assert normalize_doi(None) is None

    def test_rejects_garbage(self):
        assert normalize_doi("not-a-doi") is None
        assert normalize_doi("https://example.com") is None

    def test_strips_trailing_punctuation(self):
        assert normalize_doi("10.1016/j.foo.2024.") == "10.1016/j.foo.2024"
        assert normalize_doi("10.1016/j.foo.2024,") == "10.1016/j.foo.2024"

    def test_strips_embedded_url_www(self):
        assert (
            normalize_doi("10.1177/0956797618796480www.psychologicalscience.org/PS")
            == "10.1177/0956797618796480"
        )

    def test_strips_embedded_url_https(self):
        assert (
            normalize_doi("10.1177/0956797618796480https://journals.sagepub.com")
            == "10.1177/0956797618796480"
        )

    def test_strips_embedded_url_http(self):
        assert (
            normalize_doi("10.1038/s41586-024-07386-0http://www.nature.com/reprints")
            == "10.1038/s41586-024-07386-0"
        )

    def test_rejects_truncation_at_a_trailing_hyphen(self):
        # A printed DOI line-wrapped after one of its own hyphens ("10.1037/0033-"
        # / "2909.115.1.102") leaves a stub that satisfies "10.NNNN/<non-space>".
        # Accepting it hands back a DOI that resolves to nothing AND starves the
        # wrap rescue, which gates on this returning None. No DOI ends on a hyphen:
        # 0 of the 4787 distinct DOIs printed across the gold corpus do.
        assert normalize_doi("10.1037/0033-") is None
        assert normalize_doi("https://doi.org/10.1016/S0140-") is None
        # Stripping instead of rejecting would be worse: "10.1037/0033" is a
        # syntactically valid DOI that is not the one on the page.
        assert normalize_doi("10.1037/0033-.") is None

    def test_preserves_legitimate_doi_with_letters(self):
        assert normalize_doi("10.1016/j.foo.2024.01.001") == "10.1016/j.foo.2024.01.001"
        assert normalize_doi("10.1038/s41586-024-07386-0") == "10.1038/s41586-024-07386-0"
        assert normalize_doi("10.1093/brain/110.3.747") == "10.1093/brain/110.3.747"

    def test_collapses_repeated_slashes(self):
        assert normalize_doi("10.1037//0096-1523.2.4.567") == "10.1037/0096-1523.2.4.567"
        assert normalize_doi("10.1093///brain/110.3.747") == "10.1093/brain/110.3.747"


class TestRefExtractionStrategy:
    """Tests for LLM reference extraction in extract_all_metadata."""

    def _make_extractor_with_refs(self):
        """Create extractor with a mock reference section."""
        sections = ["Introduction"] * 3 + ["References"] * 5
        texts = [f"Intro {i}" for i in range(3)] + [
            "Smith J. (2020). Paper A. Nature, 10, 1-5.",
            "Jones A. (2019). Paper B. Science, 20, 10-15.",
            "Brown B. (2021). Paper C. Cell, 30, 100-110.",
            "Davis D. (2018). Paper D. PNAS, 40, 200-210.",
            "Wilson W. (2017). Paper E. BMJ, 50, 300-310.",
        ]
        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
        ]
        return _make_extractor(sections, texts, paper_sections=paper_sections)

    async def test_llm_extracts_references(self):
        """LLM extraction should produce references."""
        pytest.importorskip("torch")
        ext = self._make_extractor_with_refs()

        # Mock core metadata extraction
        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])

        # Mock LLM client
        from bibr.schemas import PaperReferenceLLM

        mock_ref = PaperReferenceLLM(
            index=1,
            title="Paper A",
            first_page="1",
            volume="10",
            authors="Smith, J.",
            year="2020",
            container="Nature",
        )
        ext.llm_client = mock.MagicMock()
        ext.llm_client.segment_references = mock.AsyncMock(
            return_value=[
                "Smith J. (2020).",
                "Jones A. (2019).",
                "Brown B. (2021).",
                "Davis D. (2018).",
                "Wilson W. (2017).",
            ]
        )
        ext.llm_client.extract_references = mock.AsyncMock(return_value=[mock_ref])

        await ext.extract_all_metadata()

        # The LLM returned one ref for five segmented entries. The other four
        # are no longer silently dropped — they are recovered from their own
        # segments via the NER parser and recorded as a parse-fallback warning.
        titles = [r.title for r in ext.metadata.references]
        assert len(ext.metadata.references) == 5
        assert titles[0] == "Paper A"

    async def test_llm_failure_logged_not_fatal(self):
        """When LLM reference extraction fails, core metadata should still succeed."""
        ext = self._make_extractor_with_refs()

        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(side_effect=Exception("LLM unavailable"))

        # Total ref failure (LLM down AND NER fallback down) so the ref task raises;
        # core metadata must still survive (logged-not-fatal in extract_all_metadata).
        with mock.patch.object(
            ext.refs, "_parse_references_ner_aligned", side_effect=Exception("NER down")
        ):
            await ext.extract_all_metadata()

        # Core metadata should still be set, references empty
        assert ext.metadata.title == "Test"
        assert len(ext.metadata.references) == 0

    async def test_swallowed_ref_failure_surfaces_warning(self):
        """A swallowed (non-BibrError) reference-extraction failure — the raw
        CUDA OOM / model-load exception the local NER parser can raise — must be
        recorded on processing_warnings, not merely logged. Without this, a total
        reference wipeout is invisible in the export apart from the downstream
        VAL_REF_COUNT_MISMATCH, which flags the symptom but not the cause."""
        from bibr.processing_warnings import WarningCode

        ext = self._make_extractor_with_refs()
        ext.contents.processing_warnings = []
        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])

        # Raw non-BibrError mirroring a torch CUDA OOM from the NER ref parser.
        ext._extract_references = mock.AsyncMock(
            side_effect=RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        )

        await ext.extract_all_metadata()

        # Preserved behaviour: non-fatal, core metadata survives, refs empty.
        assert ext.metadata.title == "Test"
        assert len(ext.metadata.references) == 0
        # New behaviour: the failure is surfaced on processing_warnings.
        hits = [
            w
            for w in ext.contents.processing_warnings
            if w.code == WarningCode.REF_EXTRACTION_ERROR
        ]
        assert hits, "expected a REF_EXTRACTION_ERROR warning"
        assert hits[0].message.startswith("RuntimeError: CUDA out of memory")

    async def test_systemic_reference_failure_preserves_core_and_marks_incomplete(self):
        ext = self._make_extractor_with_refs()
        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(
            doi="10.1234/core",
            title="Durable Core",
            keywords=["survives"],
            authors=[],
        )
        ext._extract_references = mock.AsyncMock(
            side_effect=UpstreamServiceError("LLM", "service down")
        )

        result = await ext.extract_all_metadata()

        assert result.title == "Durable Core"
        assert result.doi == "10.1234/core"
        assert result.keywords == ["survives"]
        assert result.references == []
        assert result.references_incomplete is True
        assert "UpstreamServiceError" in result._references_incomplete_diagnostic
        assert "service down" in result._references_incomplete_diagnostic
        assert len(result._references_incomplete_diagnostic) <= 256

    async def test_invalid_output_reference_failure_is_not_marked_incomplete(self):
        from bibr.exceptions import ProcessingError

        error = ProcessingError(
            "LLM returned invalid structured output",
            error_code="llm_invalid_output",
        )
        ext = self._make_extractor_with_refs()
        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(
            doi="10.1234/core",
            title="Durable Core",
            keywords=[],
            authors=[],
        )
        ext._extract_references = mock.AsyncMock(side_effect=error)

        with pytest.raises(ProcessingError) as raised:
            await ext.extract_all_metadata()

        assert raised.value is error
        assert ext.metadata.references_incomplete is not True

    async def test_reference_row_collection_failure_runs_core_and_marks_incomplete(self):
        ext = self._make_extractor_with_refs()

        async def extract_core():
            ext.metadata = PaperMetadata(doi="10.1234/core", title="Durable Core")

        ext.extract_core_metadata = mock.AsyncMock(side_effect=extract_core)
        ext._collect_reference_rows = mock.MagicMock(
            side_effect=RuntimeError("reference locator crashed")
        )

        result = await ext.extract_all_metadata()

        ext.extract_core_metadata.assert_awaited_once()
        assert result.title == "Durable Core"
        assert result.doi == "10.1234/core"
        assert result.references == []
        assert result.references_incomplete is True
        assert "reference locator crashed" in result._references_incomplete_diagnostic

    async def test_reference_low_yield_issue_is_preserved_alongside_core_issues(self):
        extractor = _make_extractor(["References"], ["1. Smith J. Example. 2020."])
        extractor.core.extract = mock.AsyncMock(
            return_value=PaperMetadata(doi="", title="Example", keywords=[], authors=[])
        )
        extractor.core.validation_issues = []
        extractor.locator.collect_reference_rows = mock.Mock(
            return_value=pd.DataFrame({"text": ["1. Smith J. Example. 2020."]})
        )
        low_yield = ValidationIssue(
            "VAL_REF_LOW_YIELD",
            "warning",
            "credible reference starts sharply exceed valid parsed output",
            origin_stage="extract",
        )
        extractor.refs.validation_issues = []

        async def extract_refs(_ref_df):
            extractor.refs.validation_issues.append(low_yield)
            return []

        extractor.refs.extract = mock.AsyncMock(side_effect=extract_refs)

        await extractor.extract_all_metadata()

        assert extractor.validation_issues == [low_yield]

    async def test_reference_issue_merge_accepts_collaborator_double_without_issue_sink(self):
        extractor = _make_extractor(["Introduction"], ["Body text"])
        extractor.core.extract = mock.AsyncMock(
            return_value=PaperMetadata(doi="", title="Example", keywords=[], authors=[])
        )
        extractor.core.validation_issues = []
        extractor.locator.collect_reference_rows = mock.Mock(
            side_effect=ValueError("No reference section found")
        )
        extractor.refs = mock.Mock(spec=[])

        result = await extractor.extract_all_metadata()

        assert result.title == "Example"
        assert extractor.validation_issues == []

    async def test_reused_metadata_extractor_does_not_merge_stale_ref_issue(self):
        extractor = _make_extractor(["Introduction"], ["Body text"])
        stale = ValidationIssue("VAL_REF_LOW_YIELD", "warning", "stale", origin_stage="extract")
        extractor.refs.validation_issues = [stale]
        extractor.core.extract = mock.AsyncMock(
            return_value=PaperMetadata(doi="", title="Second call", keywords=[], authors=[])
        )
        extractor.core.validation_issues = []
        extractor.locator.collect_reference_rows = mock.Mock(
            side_effect=ValueError("No reference section found")
        )

        result = await extractor.extract_all_metadata()

        assert result.title == "Second call"
        assert extractor.refs.validation_issues == []
        assert extractor.validation_issues == []

    async def test_reused_empty_input_clears_facade_and_reference_issue_state(self):
        extractor = _make_extractor([], [])
        stale = ValidationIssue("VAL_REF_LOW_YIELD", "warning", "stale", origin_stage="extract")
        extractor.validation_issues = [stale]
        extractor.refs.validation_issues = [stale]
        extractor.metadata = PaperMetadata(doi="10.1234/old", title="Old metadata")

        result = await extractor.extract_all_metadata()

        assert result == PaperMetadata(doi="", title="", keywords=[], authors=[])
        assert extractor.validation_issues == []
        assert extractor.refs.validation_issues == []

    @pytest.mark.parametrize(
        "control_error",
        [asyncio.CancelledError(), SystemExit("stop"), KeyboardInterrupt()],
    )
    async def test_reference_row_collection_control_errors_propagate(self, control_error):
        ext = self._make_extractor_with_refs()
        ext.extract_core_metadata = mock.AsyncMock()
        ext._collect_reference_rows = mock.MagicMock(side_effect=control_error)

        with pytest.raises(type(control_error)):
            await ext.extract_all_metadata()

        ext.extract_core_metadata.assert_not_awaited()

    async def test_reference_cancellation_cancels_hung_core_promptly(self):
        ext = self._make_extractor_with_refs()
        ext.metadata = PaperMetadata(doi="10.1234/core", title="Durable Core")
        core_started = asyncio.Event()
        core_cancelled = asyncio.Event()
        never_finishes = asyncio.Event()

        async def hung_core():
            core_started.set()
            try:
                await never_finishes.wait()
            except asyncio.CancelledError:
                core_cancelled.set()
                raise

        async def cancelled_references(_ref_df):
            await core_started.wait()
            raise asyncio.CancelledError

        ext.extract_core_metadata = mock.AsyncMock(side_effect=hung_core)
        ext._extract_references = mock.AsyncMock(side_effect=cancelled_references)

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(ext.extract_all_metadata(), timeout=0.25)

        assert core_cancelled.is_set()

    async def test_references_from_gather_result_land_on_metadata(self):
        """References transfer via the gather return value, not a side channel."""
        ext = self._make_extractor_with_refs()

        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])

        from bibr.paper import PaperReference

        refs = [
            PaperReference(
                bib_id=1,
                title="Paper A",
                first_page=None,
                volume=None,
                authors="Smith, J.",
                year=2020,
                container=None,
            )
        ]
        ext._extract_references = mock.AsyncMock(return_value=refs)

        await ext.extract_all_metadata()

        assert ext.metadata.references == refs
        assert not hasattr(ext, "_extracted_references")

    async def test_metadata_none_at_assignment_logs_error(self, caplog):
        """References with no metadata to attach to must log an error, not vanish."""
        ext = self._make_extractor_with_refs()

        ext.extract_core_metadata = mock.AsyncMock()  # never sets ext.metadata

        from bibr.paper import PaperReference

        refs = [
            PaperReference(
                bib_id=1,
                title="Paper A",
                first_page=None,
                volume=None,
                authors="Smith, J.",
                year=2020,
                container=None,
            )
        ]
        ext._extract_references = mock.AsyncMock(return_value=refs)

        with caplog.at_level(logging.ERROR), pytest.raises(AssertionError):
            await ext.extract_all_metadata()

        assert any("metadata" in r.message.lower() for r in caplog.records)

    async def test_completeness_filter(self):
        """References missing both title and author should be filtered out."""
        pytest.importorskip("torch")
        ext = self._make_extractor_with_refs()

        ext.extract_core_metadata = mock.AsyncMock()
        ext.metadata = PaperMetadata(doi="", title="Test", keywords=[], authors=[])

        from bibr.schemas import PaperReferenceLLM

        refs = [
            PaperReferenceLLM(
                index=1,
                title="Good Paper",
                first_page=None,
                volume=None,
                authors="Smith",
                year="2020",
                container=None,
            ),
            PaperReferenceLLM(
                index=2,
                title="",
                first_page=None,
                volume=None,
                authors=None,
                year=None,
                container=None,
            ),
            PaperReferenceLLM(
                index=3,
                title="",
                first_page=None,
                volume=None,
                authors="Jones",
                year="2019",
                container=None,
            ),
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.segment_references = mock.AsyncMock(
            return_value=[
                "Smith J. (2020).",
                "Jones A. (2019).",
                "Brown B. (2021).",
                "Davis D. (2018).",
                "Wilson W. (2017).",
            ]
        )
        ext.llm_client.extract_references = mock.AsyncMock(return_value=refs)

        with mock.patch("bibr.config.Settings.REF_PARSE_STRATEGY", "llm"):
            await ext.extract_all_metadata()

        # No reference survives with neither a title nor authors — that is what
        # the completeness filter is for.
        # (Ref 3 has an empty title but real authors, so it legitimately stays.)
        titles = {r.title for r in ext.metadata.references}
        assert "Good Paper" in titles
        assert not any(not r.title and not r.authors for r in ext.metadata.references)
        # All five printed entries are represented. Ref 2 came back from the
        # LLM with no title and no authors; the filter below drops such a row,
        # so its slot counts as *missing* and goes to NER like segments 4 and
        # 5, which the LLM never returned at all. Counting it as covered used
        # to delete the entry outright and shift every later bib_id.
        assert len(ext.metadata.references) == 5
        assert [r.bib_id for r in ext.metadata.references] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Chunk reference text
# ---------------------------------------------------------------------------


class TestNerLoaderSplit:
    """Segmenter and parser load independently (LLM-parse path must not load
    the parser)."""

    def test_get_ner_segmenter_does_not_load_parser(self, monkeypatch):
        pytest.importorskip("torch")
        import bibr.extract.ref_extractor as ex

        monkeypatch.setattr(ex, "_NER_SEGMENTER", None, raising=False)
        monkeypatch.setattr(ex, "_NER_PARSER", None, raising=False)

        seg_sentinel = object()
        with (
            mock.patch("bibr.ner.segmenter.RefSegmenter", return_value=seg_sentinel) as seg_cls,
            mock.patch("bibr.ner.parser.RefParser") as parser_cls,
        ):
            seg = ex._get_ner_segmenter()
        assert seg is seg_sentinel
        seg_cls.assert_called_once()
        parser_cls.assert_not_called()


class TestNerGetterConcurrency:
    """Concurrent calls to the lazy NER getters initialize the model once."""

    def _race(self, getter, n_threads=8):
        import threading

        barrier = threading.Barrier(n_threads)
        results = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            obj = getter()
            with lock:
                results.append(obj)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def _counting_cls(self):
        import time as time_mod

        calls = []

        class CountingModel:
            def __init__(self, *args, **kwargs):
                calls.append(1)
                time_mod.sleep(0.05)

        return CountingModel, calls

    def test_segmenter_initialized_once(self, monkeypatch):
        pytest.importorskip("torch")
        import bibr.extract.ref_extractor as ex

        monkeypatch.setattr(ex, "_NER_SEGMENTER", None, raising=False)
        counting_cls, calls = self._counting_cls()
        with mock.patch("bibr.ner.segmenter.RefSegmenter", counting_cls):
            results = self._race(ex._get_ner_segmenter)
        assert len(calls) == 1
        assert all(r is results[0] for r in results)

    def test_parser_initialized_once(self, monkeypatch):
        pytest.importorskip("torch")
        import bibr.extract.ref_extractor as ex

        monkeypatch.setattr(ex, "_NER_PARSER", None, raising=False)
        counting_cls, calls = self._counting_cls()
        with mock.patch("bibr.ner.parser.RefParser", counting_cls):
            results = self._race(ex._get_ner_parser)
        assert len(calls) == 1
        assert all(r is results[0] for r in results)


class TestChunk:
    def test_chunks_of_five(self):
        from bibr.extract.extractor import _chunk

        assert list(_chunk(list(range(12)), 5)) == [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11],
        ]

    def test_empty(self):
        from bibr.extract.extractor import _chunk

        assert list(_chunk([], 5)) == []


# ---------------------------------------------------------------------------
# LLM reference extraction (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------


class TestLLMReferenceExtraction:
    """Tests for _parse_references_llm — batched LLM parse path."""

    def _make_extractor_with_ref_df(self, ref_texts):
        """Create extractor and a ref_df from a list of text strings."""
        sections = ["Introduction"] * 3 + ["References"] * len(ref_texts)
        texts = [f"Intro {i}" for i in range(3)] + ref_texts
        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)
        ref_df = pd.DataFrame({"text": ref_texts})
        return ext, ref_df

    def _mock_ref(self, index, title, author, year):
        from bibr.schemas import PaperReferenceLLM

        return PaperReferenceLLM(
            index=index,
            title=title,
            authors=author,
            year=year,
            first_page=None,
            volume=None,
            container=None,
        )

    async def test_small_list(self):
        """Short reference list processed and yields correct bib_ids."""
        ref_texts = [
            "Smith, J. (2020). Paper A. Nature, 10, 1-5.",
            "Jones, A. (2019). Paper B. Science, 20, 10-15.",
        ]
        ext, ref_df = self._make_extractor_with_ref_df(ref_texts)

        mock_refs = [
            self._mock_ref(1, "Paper A", "Smith, J.", "2020"),
            self._mock_ref(2, "Paper B", "Jones, A.", "2019"),
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(return_value=mock_refs)

        ref_text = "\n".join(ref_texts)
        result = await ext.refs._parse_references_llm(ref_text, ref_texts)

        assert len(result) == 2
        assert result[0].bib_id == 1
        assert result[1].bib_id == 2

    async def test_large_list_batched(self):
        """Large reference list is batched; all refs extracted and reindexed."""
        ref_texts = [
            f"[{i + 1}] Smith, J. ({2000 + i % 25}). Title {i} about an important topic. "
            f"Journal of Testing {i}, {i}, 100-{100 + i}."
            for i in range(80)
        ]
        ext, ref_df = self._make_extractor_with_ref_df(ref_texts)

        async def mock_extract(text, file_hash="", start_index=1, expected_count=None):
            n = len([line for line in text.split("\n") if line.strip()])
            return [
                self._mock_ref(
                    start_index + i, f"Title {start_index + i - 1}", f"Author {i}", f"200{i % 10}"
                )
                for i in range(n)
            ]

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(side_effect=mock_extract)

        ref_text = "\n".join(ref_texts)
        result = await ext.refs._parse_references_llm(ref_text, ref_texts)

        # All refs extracted
        assert len(result) == 80
        # bib_ids re-indexed sequentially
        ids = [r.bib_id for r in result]
        assert ids == list(range(1, 81))

        # Verify batching actually happened
        from bibr.config import Settings

        n_refs = 80
        expected_calls = math.ceil(n_refs / Settings.REF_PARSE_BATCH_SIZE)
        assert ext.llm_client.extract_references.call_count == expected_calls, (
            f"Expected {expected_calls} batched calls, got "
            f"{ext.llm_client.extract_references.call_count}"
        )
        # Every call must have received at most REF_PARSE_BATCH_SIZE lines
        for call in ext.llm_client.extract_references.call_args_list:
            numbered_arg = call.args[0]
            lines_in_batch = len([ln for ln in numbered_arg.split("\n") if ln.strip()])
            assert lines_in_batch <= Settings.REF_PARSE_BATCH_SIZE, (
                f"Batch had {lines_in_batch} refs, exceeding limit {Settings.REF_PARSE_BATCH_SIZE}"
            )

    async def test_uneven_final_batch_start_index_continuity(self):
        """n not divisible by batch size: short tail batch keeps start_index contiguous."""
        from bibr.config import Settings

        batch_size = Settings.REF_PARSE_BATCH_SIZE
        n_refs = batch_size + 2  # one full batch + a 2-ref tail
        ref_texts = [
            f"[{i + 1}] Smith, J. (2020). Title {i} about a topic. Journal {i}, {i}, 1-10."
            for i in range(n_refs)
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)

        seen_start_indices = []

        async def mock_extract(text, file_hash="", start_index=1, expected_count=None):
            seen_start_indices.append(start_index)
            n = len([line for line in text.split("\n") if line.strip()])
            return [
                self._mock_ref(start_index + i, f"Title {start_index + i - 1}", f"A {i}", "2020")
                for i in range(n)
            ]

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(side_effect=mock_extract)

        result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert seen_start_indices == [1, batch_size + 1]
        assert len(result) == n_refs
        assert [r.bib_id for r in result] == list(range(1, n_refs + 1))

    async def test_all_batches_failed_raises(self):
        """Total failure (every batch fails the LLM *and* its NER fallback) raises
        instead of silently returning []."""
        ref_texts = [
            "Smith, J. (2020). Paper A. Nature, 10, 1-5.",
            "Jones, A. (2019). Paper B. Science, 20, 10-15.",
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(
            side_effect=UpstreamServiceError("LLM", "service down")
        )

        # NER fallback also fails → nothing recovered → fail loud.
        with mock.patch.object(
            ext.refs, "_parse_references_ner_aligned", side_effect=RuntimeError("NER down")
        ):
            with pytest.raises(UpstreamServiceError):
                await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

    async def test_all_llm_batches_fail_ner_recovers_no_raise(self):
        """When every LLM batch fails but NER recovers them, return the NER refs —
        do NOT raise (the result is partial-but-valid, not empty)."""
        ref_texts = [
            "Smith, J. (2020). Paper A. Nature, 10, 1-5.",
            "Jones, A. (2019). Paper B. Science, 20, 10-15.",
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(
            side_effect=UpstreamServiceError("LLM", "service down")
        )

        from bibr.paper import PaperReference

        def fake_ner(segments):
            return [
                PaperReference(
                    bib_id=i + 1,
                    title=f"NER {i}",
                    first_page=None,
                    volume=None,
                    authors=f"A {i}",
                    year=None,
                    container=None,
                )
                for i in range(len(segments))
            ]

        with mock.patch.object(ext.refs, "_parse_references_ner_aligned", side_effect=fake_ner):
            result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert len(result) == 2
        assert [r.bib_id for r in result] == [1, 2]
        assert [r.title for r in result] == ["NER 0", "NER 1"]

    async def test_partial_batch_failure_recovers_via_ner(self):
        """A hard-failed batch is retried once, then recovered via the NER parser —
        never silently dropped. The successful batch is preserved and every ref
        survives, re-indexed in document order."""
        ref_texts = [
            f"[{i + 1}] Smith, J. (200{i}). Title {i} about a topic. Journal {i}, {i}, 1-10."
            for i in range(10)
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)

        calls = {"n": 0}

        async def mock_extract(text, file_hash="", start_index=1, expected_count=None):
            # The first batch fails on every attempt (initial + retry); the second
            # batch always succeeds.
            calls["n"] += 1
            if start_index == 1:
                raise UpstreamServiceError("LLM", "transient failure")
            n = len([line for line in text.split("\n") if line.strip()])
            return [
                self._mock_ref(start_index + i, f"Title {start_index + i - 1}", f"A {i}", "2020")
                for i in range(n)
            ]

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(side_effect=mock_extract)

        # Deterministic NER fallback: recover one ref per failed-batch segment so the
        # test asserts the merge wiring, not the real NER model.
        from bibr.paper import PaperReference

        def fake_ner(segments):
            return [
                PaperReference(
                    bib_id=i + 1,
                    title=f"NER Title {i}",
                    first_page=None,
                    volume=None,
                    authors=f"N {i}",
                    year=None,
                    container=None,
                )
                for i in range(len(segments))
            ]

        # Pin batch size so the 10 refs span exactly two batches regardless
        # of the REF_PARSE_BATCH_SIZE default.
        ext.refs._settings.REF_PARSE_BATCH_SIZE = 5
        with mock.patch.object(ext.refs, "_parse_references_ner_aligned", side_effect=fake_ner):
            result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        # 3 LLM calls: 2 initial batches + 1 retry of the failed first batch.
        assert calls["n"] == 3
        # No refs lost: 5 NER-recovered (failed batch) + 5 from the surviving batch.
        assert len(result) == 10
        assert [r.bib_id for r in result] == list(range(1, 11))
        # Failed batch (indices 1-5) was recovered from NER and kept first in order.
        assert [r.title for r in result[:5]] == [f"NER Title {i}" for i in range(5)]

    async def _assert_degenerate_skips_retry(self, failure_exc):
        """Shared body: a batch failing with *failure_exc* (a degenerate
        timeout/output-cap error, as wrapped by extract_references) is NOT
        retried — it falls straight to NER — while the other batch is kept."""
        ref_texts = [
            f"[{i + 1}] Smith, J. (200{i}). Title {i} about a topic. Journal {i}, {i}, 1-10."
            for i in range(10)
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)

        calls = {"n": 0}

        async def mock_extract(text, file_hash="", start_index=1, expected_count=None):
            calls["n"] += 1
            if start_index == 1:
                raise failure_exc
            n = len([line for line in text.split("\n") if line.strip()])
            return [
                self._mock_ref(start_index + i, f"Title {start_index + i - 1}", f"A {i}", "2020")
                for i in range(n)
            ]

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(side_effect=mock_extract)

        from bibr.paper import PaperReference

        def fake_ner(segments):
            return [
                PaperReference(
                    bib_id=i + 1,
                    title=f"NER Title {i}",
                    first_page=None,
                    volume=None,
                    authors=f"N {i}",
                    year=None,
                    container=None,
                )
                for i in range(len(segments))
            ]

        ext.refs._settings.REF_PARSE_BATCH_SIZE = 5
        with mock.patch.object(ext.refs, "_parse_references_ner_aligned", side_effect=fake_ner):
            result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        # The degenerate first batch is never retried at the SAME size (the
        # identical prompt would just reproduce the loop) — it is split and
        # re-tried smaller. The mock fails every start_index==1 call, so:
        # 2 initial batches + halves [1-3] (fails) / [4-5] (ok) + quarters
        # [1-2] (fails) / [3] (ok) = 6 calls, and only refs 1-2 fall to NER.
        assert calls["n"] == 6
        # No refs lost: 2 NER-recovered + 8 LLM-parsed, in document order.
        assert len(result) == 10
        assert [r.bib_id for r in result] == list(range(1, 11))
        assert [r.title for r in result[:2]] == [f"NER Title {i}" for i in range(2)]
        assert [r.title for r in result[2:]] == [f"Title {i}" for i in range(2, 10)]

    async def test_timeout_batch_skips_retry_falls_back_to_ner(self):
        """A timed-out batch (greedy decode looped past the per-request budget)
        is never retried at the same size — re-asking the identical prompt just
        reproduces the loop. It is split and re-tried smaller; only the span
        that keeps failing is recovered via NER."""
        await self._assert_degenerate_skips_retry(
            UpstreamServiceError(
                "LLM",
                "Failed to extract references",
                TimeoutError("LLM call timed out after 240s (per-request timeout: 120s)"),
            )
        )

    async def test_incomplete_output_batch_skips_retry_falls_back_to_ner(self):
        """A batch that ran to the output-token cap (IncompleteOutputException —
        on small-context servers usually a size-induced overflow) likewise skips
        the same-size retry and goes through the split ladder before NER."""
        from instructor.core.exceptions import IncompleteOutputException

        await self._assert_degenerate_skips_retry(
            UpstreamServiceError(
                "LLM",
                "Failed to extract references",
                IncompleteOutputException("output incomplete due to max_tokens limit"),
            )
        )

    async def test_empty_ref_strings_returns_empty(self):
        """No segments → no LLM calls → empty result, no raise."""
        ext, _ = self._make_extractor_with_ref_df(["placeholder"])
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock()

        result = await ext.refs._parse_references_llm("", [])

        assert result == []
        ext.llm_client.extract_references.assert_not_called()

    async def test_reindex_sequential(self):
        """bib_ids are re-indexed 1..N sequentially."""
        ref_texts = [
            f"[{i + 1}] Smith, J. ({2000 + i % 25}). Title {i} about a topic. "
            f"Journal {i}, {i}, 100-{100 + i}."
            for i in range(60)
        ]
        ext, ref_df = self._make_extractor_with_ref_df(ref_texts)

        async def mock_extract(text, file_hash="", start_index=1, expected_count=None):
            n = len([line for line in text.split("\n") if line.strip()])
            return [
                self._mock_ref(
                    start_index + i, f"Title {start_index + i - 1}", f"Author {i}", "2020"
                )
                for i in range(n)
            ]

        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(side_effect=mock_extract)

        ref_text = "\n".join(ref_texts)
        result = await ext.refs._parse_references_llm(ref_text, ref_texts)

        ids = [r.bib_id for r in result]
        assert ids == list(range(1, len(result) + 1))

    async def test_whole_batch_null_authors_rescued_from_segment(self, caplog):
        """A batch the LLM returns with EVERY author null gets authors rescued
        from each ref's own author-date segment (qual-grade30 M2)."""
        ref_texts = [
            "Drehmann, M., and Tsatsaronis, K. (2014). The credit-to-GDP Gap. "
            "BIS Quarterly Review, 2014, 55-73.",
            "ECB (2010). Survey on Access to Finance. European Central Bank.",
            "Fagiolo, G., and Roventini, A. (2012). Scientific Status. Journal, 1, 1-10.",
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)
        # LLM parsed titles/years but dropped EVERY author in the batch.
        mock_refs = [
            self._mock_ref(1, "The credit-to-GDP Gap", None, "2014"),
            self._mock_ref(2, "Survey on Access to Finance", None, "2010"),
            self._mock_ref(3, "Scientific Status", None, "2012"),
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(return_value=mock_refs)

        with caplog.at_level(logging.WARNING, logger="bibr.extract.ref_extractor"):
            result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert [r.authors for r in result] == [
            "Drehmann, M., and Tsatsaronis, K",
            "ECB",
            "Fagiolo, G., and Roventini, A",
        ]
        # A machine-greppable warning is emitted for the all-null batch.
        assert any("null author" in r.message.lower() for r in caplog.records)

    async def test_lone_null_author_not_rescued(self):
        """A single null-author ref in an otherwise-populated batch is left as-is
        (whole-batch-null gating — smallest blast radius)."""
        ref_texts = [
            "Drehmann, M., and Tsatsaronis, K. (2014). The credit-to-GDP Gap. BIS, 2014, 55-73.",
            "ECB (2010). Survey on Access to Finance. ECB.",
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)
        mock_refs = [
            self._mock_ref(1, "The credit-to-GDP Gap", "Drehmann, M., and Tsatsaronis, K", "2014"),
            self._mock_ref(2, "Survey on Access to Finance", None, "2010"),  # lone null
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(return_value=mock_refs)

        result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert result[0].authors == "Drehmann, M., and Tsatsaronis, K"
        assert result[1].authors is None

    async def test_whole_batch_null_rescue_declines_vancouver(self):
        """Even in an all-null batch, Vancouver refs (no leading (YEAR)) are not
        rescued — cutting at the first paren would over-capture title/container."""
        ref_texts = [
            "Smith J, Jones K. A randomized trial of the thing. Lancet. 2020;15(3):55-60.",
            "Doe A, Roe B. Another clinical study of interest. BMJ. 2019;12(2):11-20.",
        ]
        ext, _ = self._make_extractor_with_ref_df(ref_texts)
        mock_refs = [
            self._mock_ref(1, "A randomized trial of the thing", None, "2020"),
            self._mock_ref(2, "Another clinical study of interest", None, "2019"),
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(return_value=mock_refs)

        result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert [r.authors for r in result] == [None, None]


# ---------------------------------------------------------------------------
# Header-text fallback in _collect_reference_rows
# ---------------------------------------------------------------------------


class TestCollectReferenceRowsHeaderFallback:
    """Tests for Fallback 2: header-text pattern matching in _collect_reference_rows."""

    def test_bibliography_header_detected(self):
        """Section named 'Bibliography' is found by header-text fallback."""
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Introduction",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
            PaperSection(
                section_id=2,
                header="Bibliography",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
        ]
        ext = _make_extractor(
            sections=["Introduction", "Introduction", "Bibliography", "Bibliography"],
            texts=["Intro sentence 1.", "Intro sentence 2.", "Ref 1.", "Ref 2."],
            paper_sections=sections,
        )
        ref_df = ext._collect_reference_rows()
        assert len(ref_df) == 2
        assert list(ref_df["text"]) == ["Ref 1.", "Ref 2."]

    def test_works_cited_header_detected(self):
        """Section named 'Works Cited' is found by header-text fallback."""
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Body",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=2,
                header="Works Cited",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
        ]
        ext = _make_extractor(
            sections=["Body", "Works Cited", "Works Cited"],
            texts=["Body text.", "Cited 1.", "Cited 2."],
            paper_sections=sections,
        )
        ref_df = ext._collect_reference_rows()
        assert len(ref_df) == 2

    def test_literature_cited_header_detected(self):
        """Section named 'Literature Cited' is found by header-text fallback."""
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Body",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
            PaperSection(
                section_id=2,
                header="Literature Cited",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
        ]
        ext = _make_extractor(
            sections=["Body", "Literature Cited"],
            texts=["Body text.", "Ref 1."],
            paper_sections=sections,
        )
        ref_df = ext._collect_reference_rows()
        assert len(ref_df) == 1

    def test_case_insensitive_header_match(self):
        """Header matching should be case-insensitive."""
        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="BIBLIOGRAPHY",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
        ]
        ext = _make_extractor(
            sections=["BIBLIOGRAPHY", "BIBLIOGRAPHY"],
            texts=["Ref 1.", "Ref 2."],
            paper_sections=sections,
        )
        ref_df = ext._collect_reference_rows()
        assert len(ref_df) == 2


# ---------------------------------------------------------------------------
# _save_ref_training_data filesystem writes
# ---------------------------------------------------------------------------


class TestSaveRefTrainingData:
    """Tests for training_capture.save_ref_training_data filesystem writes."""

    def test_skips_when_dir_not_set(self, monkeypatch):
        """No-op when REF_TRAINING_DATA_DIR is None."""
        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", None)
        # Should not raise
        save_ref_training_data("some text", [])

    def test_writes_json_file(self, tmp_path, monkeypatch):
        """Writes a JSON file with input/output/n_references keys."""
        import json

        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", str(tmp_path))

        # _save_ref_training_data expects objects with model_dump(); use a mock
        ref_mock = mock.MagicMock()
        ref_mock.model_dump.return_value = {"title": "Test", "authors": "Smith", "year": 2020}

        save_ref_training_data("raw bib text", [ref_mock])

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["input"] == "raw bib text"
        assert data["n_references"] == 1
        assert len(data["output"]) == 1

    def test_records_llm_provenance(self, tmp_path):
        """The record names the provider, model, and prompt that produced it."""
        import json

        settings = GlobalSettings()
        settings.REF_TRAINING_DATA_DIR = str(tmp_path)
        settings.llm.provider = "anthropic"
        settings.llm.model = "claude-test"

        save_ref_training_data("raw bib text", [], settings=settings)

        data = json.loads(next(tmp_path.glob("*.json")).read_text())
        provenance = data["provenance"]
        prompt_sha256 = provenance.pop("prompt_sha256")
        assert len(prompt_sha256) == 64 and set(prompt_sha256) <= set("0123456789abcdef")
        assert provenance == {
            "source": "llm",
            "provider": "anthropic",
            "model": "claude-test",
            "prompt": "references_parse",
            "bibr_version": bibr_version,
        }

    def test_skips_duplicate(self, tmp_path, monkeypatch):
        """Doesn't overwrite if file already exists (content-hash dedup)."""
        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", str(tmp_path))

        save_ref_training_data("same text", [])
        files_after_first = list(tmp_path.glob("*.json"))
        assert len(files_after_first) == 1
        mtime = files_after_first[0].stat().st_mtime

        save_ref_training_data("same text", [])
        files_after_second = list(tmp_path.glob("*.json"))
        assert len(files_after_second) == 1
        assert files_after_second[0].stat().st_mtime == mtime  # not rewritten

    def test_logs_warning_on_failure(self, tmp_path, monkeypatch, caplog):
        """Logs a warning (no exception) when write fails."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("occupied", encoding="utf-8")
        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", str(blocked / "child"))

        with caplog.at_level(logging.WARNING):
            save_ref_training_data("text", [])
        assert any("Failed to save ref training data" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _save_seg_training_data filesystem writes (CRF segmenter training data)
# ---------------------------------------------------------------------------


class TestSaveSegTrainingData:
    """Tests for training_capture.save_seg_training_data filesystem writes."""

    def test_skips_when_dir_not_set(self, monkeypatch):
        """No-op when REF_TRAINING_DATA_DIR is None."""
        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", None)
        # Should not raise
        save_seg_training_data("some text", ["a"])

    def test_writes_json_file_to_segmentation_subdir(self, tmp_path, monkeypatch):
        """Writes {input, segments, n_segments} under a segmentation/ subdir."""
        import json

        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", str(tmp_path))

        save_seg_training_data(
            "Smith, J. (2020). A.\nDoe, A. (2019). B.",
            ["Smith, J. (2020). A.", "Doe, A. (2019). B."],
        )

        # Parser training data and segmentation training data stay separated.
        assert list(tmp_path.glob("*.json")) == []
        files = list((tmp_path / "segmentation").glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["input"] == "Smith, J. (2020). A.\nDoe, A. (2019). B."
        assert data["segments"] == ["Smith, J. (2020). A.", "Doe, A. (2019). B."]
        assert data["n_segments"] == 2

    def test_records_llm_provenance(self, tmp_path):
        """The record names the provider, model, and prompt that produced it."""
        import json

        settings = GlobalSettings()
        settings.REF_TRAINING_DATA_DIR = str(tmp_path)
        settings.llm.provider = "anthropic"
        settings.llm.model = "claude-test"

        save_seg_training_data("A.\nB.", ["A.", "B."], settings=settings)

        data = json.loads(next((tmp_path / "segmentation").glob("*.json")).read_text())
        provenance = data["provenance"]
        prompt_sha256 = provenance.pop("prompt_sha256")
        assert len(prompt_sha256) == 64 and set(prompt_sha256) <= set("0123456789abcdef")
        assert provenance == {
            "source": "llm_anchor",
            "provider": "anthropic",
            "model": "claude-test",
            "prompt": "references_segment",
            "bibr_version": bibr_version,
        }

    def test_prompt_hash_tracks_the_prompt_not_the_input(self, tmp_path, monkeypatch):
        """Same prompt → same hash across blocks; an edited prompt → a new hash."""
        import json
        from dataclasses import replace

        from bibr.clients.prompts import PROMPTS

        settings = GlobalSettings()
        settings.REF_TRAINING_DATA_DIR = str(tmp_path)
        save_seg_training_data("first block", ["first block"], settings=settings)
        save_seg_training_data("second block", ["second block"], settings=settings)
        spec = PROMPTS["references_segment"]
        monkeypatch.setitem(
            PROMPTS, "references_segment", replace(spec, system=spec.system + " Edited.")
        )
        save_seg_training_data("third block", ["third block"], settings=settings)

        records = [json.loads(p.read_text()) for p in (tmp_path / "segmentation").glob("*.json")]
        hashes = {r["input"]: r["provenance"]["prompt_sha256"] for r in records}
        assert hashes["first block"] == hashes["second block"]
        assert hashes["third block"] != hashes["first block"]

    def test_skips_duplicate(self, tmp_path, monkeypatch):
        """Doesn't overwrite if file already exists (content-hash dedup)."""
        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", str(tmp_path))
        seg_dir = tmp_path / "segmentation"

        save_seg_training_data("same text", ["same text"])
        files_after_first = list(seg_dir.glob("*.json"))
        assert len(files_after_first) == 1
        mtime = files_after_first[0].stat().st_mtime

        save_seg_training_data("same text", ["same text"])
        files_after_second = list(seg_dir.glob("*.json"))
        assert len(files_after_second) == 1
        assert files_after_second[0].stat().st_mtime == mtime  # not rewritten

    def test_logs_warning_on_failure(self, tmp_path, monkeypatch, caplog):
        """Logs a warning (no exception) when write fails."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("occupied", encoding="utf-8")
        monkeypatch.setattr("bibr.config.Settings.REF_TRAINING_DATA_DIR", str(blocked / "child"))

        with caplog.at_level(logging.WARNING):
            save_seg_training_data("text", ["text"])
        assert any("Failed to save seg training data" in r.message for r in caplog.records)


class TestConvertLlmAuthorsSuffixTrim:
    """_convert_llm_authors — trim a trailing family (including a generational
    suffix) the LLM echoed into the given-name field."""

    @staticmethod
    def _convert(given, family):
        from bibr.schemas import AuthorLLM

        return CoreMetadataExtractor._convert_llm_authors([AuthorLLM(given=given, family=family)])

    def test_suffix_duplicated_into_given_trimmed(self):
        out = self._convert("John J. Sollers III", "Sollers III")
        assert out[0].given == "John J."
        assert out[0].family == "Sollers III"

    def test_family_duplicated_into_given_trimmed(self):
        out = self._convert("Lisa van de Ven", "van de Ven")
        assert out[0].given == "Lisa"

    def test_substring_not_at_token_boundary_kept(self):
        # 'son' is a tail substring of 'Jonson' but not a whole trailing token
        out = self._convert("Jonson", "son")
        assert out[0].given == "Jonson"

    def test_exact_consortium_echo_still_blanked(self):
        out = self._convert("WHO Collaborators", "WHO Collaborators")
        assert out[0].given == ""

    def test_clean_author_unchanged(self):
        out = self._convert("Saskia M.", "Kelders")
        assert out[0].given == "Saskia M."
        assert out[0].family == "Kelders"


class TestHarvestCorrespondingAuthorEmails:
    """Tests for _harvest_corresponding_author_emails — backfills author emails
    from body text when the LLM didn't surface them."""

    def _build_extractor(self, sentences, sections, authors):
        """Build a real extractor wired up to look at synthetic sentences."""
        from bibr.models import PaperAuthor, PaperMetadata
        from bibr.paper_contents import PaperContents, PaperSection, PaperSentence

        sentence_objs = [
            PaperSentence(text_id=i, text=text, section_id=sec_id, paragraph_id=1)
            for i, (text, sec_id) in enumerate(sentences, start=1)
        ]
        section_objs = [
            PaperSection(
                section_id=sid, header=hdr, level=1, parent_section_id=0, section_type=stype
            )
            for sid, hdr, stype in sections
        ]
        contents = PaperContents(
            sentences=sentence_objs,
            sections=section_objs,
            tables=[],
            links=[],
            sections_text={},
        )
        author_objs = [
            PaperAuthor(
                author_id=i,
                given=given,
                family=family,
                affiliation="",
                email=None,
                corresponding=corresponding,
                orcid=None,
            )
            for i, (given, family, corresponding) in enumerate(authors, start=1)
        ]
        ext = MetadataExtractor(contents)
        ext.metadata = PaperMetadata(doi="", title="", keywords=[], authors=author_objs)
        return ext

    def test_email_assigned_when_family_in_window(self):
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                ("Corresponding author: Jane Smith.", 1),
                ("Email: jane.smith@uni.edu", 1),
            ],
            sections=[(1, "Authors", CanonicalSection.UNKNOWN)],
            authors=[("Jane", "Smith", False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].email == "jane.smith@uni.edu"
        assert ext.metadata.authors[0].corresponding is True

    def test_existing_email_not_overwritten(self):
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("Smith email: new@x.com", 1)],
            sections=[(1, "Authors", CanonicalSection.UNKNOWN)],
            authors=[("Jane", "Smith", False)],
        )
        ext.metadata.authors[0].email = "original@x.com"
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].email == "original@x.com"

    def test_skips_when_no_authors(self):
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("Email: x@y.com", 1)],
            sections=[(1, "Authors", CanonicalSection.UNKNOWN)],
            authors=[],
        )
        ext._email_harvester.harvest(ext.metadata.authors)  # no-op, no crash

    def test_references_section_skipped(self):
        """Emails inside the references section must not bleed into authors."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                ("Smith published this work.", 2),  # references
                ("Smith correspondence: ref-paper@x.com", 2),  # references
            ],
            sections=[(2, "References", CanonicalSection.REFERENCES)],
            authors=[("Jane", "Smith", False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].email is None

    def test_single_corresponding_author_gets_email_by_elimination(self):
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("Contact: solo@uni.edu", 1)],
            sections=[(1, "Authors", CanonicalSection.UNKNOWN)],
            authors=[("Solo", "McAuthor", True)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].email == "solo@uni.edu"

    def test_multiple_corresponding_unknown_email_unassigned(self):
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("Contact: solo@uni.edu", 1)],
            sections=[(1, "Authors", CanonicalSection.UNKNOWN)],
            authors=[("Alice", "First", True), ("Bob", "Second", True)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        # Neither author matches the family name in the sentence and there are
        # multiple corresponding authors → no by-elimination assignment.
        assert all(a.email is None for a in ext.metadata.authors)

    def test_pg_prefix_stripped_before_match(self):
        """GLM-OCR emits 'pg ' prefixes for ✉ glyphs — those must be stripped."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("pg jane@uni.edu", 1), ("Jane Smith, corresponding.", 1)],
            sections=[(1, "Authors", CanonicalSection.UNKNOWN)],
            authors=[("Jane", "Smith", False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].email == "jane@uni.edu"

    def test_byline_emails_do_not_promote_corresponding(self):
        """Many CS / arXiv papers list every author's email in the byline.

        The harvester must backfill the emails but must NOT promote every
        bylined author to ``corresponding=True``. That promotion requires an
        explicit anchor like "Corresponding author" / "✉" / "Address
        correspondence" in the surrounding window.
        """
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                (
                    "Ashish Vaswani avaswani@google.com Noam Shazeer noam@google.com "
                    "Niki Parmar nikip@google.com Jakob Uszkoreit usz@google.com",
                    1,
                ),
                ("Google Brain", 1),
                ("Provided proper attribution is provided ...", 1),
            ],
            sections=[(1, "Title", CanonicalSection.TITLE)],
            authors=[
                ("Ashish", "Vaswani", False),
                ("Noam", "Shazeer", False),
                ("Niki", "Parmar", False),
                ("Jakob", "Uszkoreit", False),
            ],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        emails = [a.email for a in ext.metadata.authors]
        corrs = [a.corresponding for a in ext.metadata.authors]
        # Emails were harvested
        assert "avaswani@google.com" in emails
        assert "noam@google.com" in emails
        # But NONE were promoted to corresponding=True (no anchor in window)
        assert all(c is False for c in corrs), corrs

    def test_corresponding_marker_promotes_corresponding(self):
        """When the window contains a 'corresponding' / 'correspondence' anchor
        (or the ✉ glyph / GLM-OCR's 'pg ' misread of it), the matched author
        gets ``corresponding=True`` along with the email."""
        from bibr.paper_contents import CanonicalSection

        # Explicit "corresponding author" anchor
        ext = self._build_extractor(
            sentences=[("Address correspondence to Jane Smith jane@uni.edu", 1)],
            sections=[(1, "Footer", CanonicalSection.UNKNOWN)],
            authors=[("Jane", "Smith", False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].email == "jane@uni.edu"
        assert ext.metadata.authors[0].corresponding is True

        # ✉ glyph anchor
        ext2 = self._build_extractor(
            sentences=[("✉ Jane Smith jane@uni.edu", 1)],
            sections=[(1, "Footer", CanonicalSection.UNKNOWN)],
            authors=[("Jane", "Smith", False)],
        )
        ext2._email_harvester.harvest(ext2.metadata.authors)
        assert ext2.metadata.authors[0].email == "jane@uni.edu"
        assert ext2.metadata.authors[0].corresponding is True

    def test_email_already_claimed_by_other_author_not_duplicated(self):
        """A nearby citation must not steal an email already claimed by another author."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                ("pg Sophie von Stumm, Department of Psychology, Goldsmiths.", 1),
                ("E-mail: s.vonstumm@gold.ac.uk", 1),
                ("partly due to genetic influence (Krapohl & Plomin, 2016).", 1),
            ],
            sections=[(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT)],
            authors=[
                ("Ziada", "Ayorech", False),
                ("Eva", "Krapohl", False),
                ("Robert", "Plomin", False),
                ("Sophie", "von Stumm", True),
            ],
        )
        # Mimic the LLM having already assigned the email correctly to Sophie.
        ext.metadata.authors[3].email = "s.vonstumm@gold.ac.uk"

        ext._email_harvester.harvest(ext.metadata.authors)

        emails = [a.email for a in ext.metadata.authors]
        # Sophie keeps her email; nobody else gets it
        assert emails == [None, None, None, "s.vonstumm@gold.ac.uk"], emails
        # Only Sophie remains corresponding
        flags = [a.corresponding for a in ext.metadata.authors]
        assert flags == [False, False, False, True], flags

    def test_email_local_part_affinity_over_distant_surname(self):
        """When two authors' surnames sit in the harvester window, prefer the
        one whose surname matches the email's local-part — beats pure
        proximity ordering and beats author-list iteration order.
        """
        from bibr.paper_contents import CanonicalSection

        # Both surnames appear in adjacent sentences. Without affinity, the
        # author iterated first (Krapohl) wins. With affinity, von Stumm wins.
        ext = self._build_extractor(
            sentences=[
                ("(Krapohl & Plomin, 2016).", 1),
                ("Sophie von Stumm — correspondence: s.vonstumm@gold.ac.uk", 1),
            ],
            sections=[(1, "Footer", CanonicalSection.UNKNOWN)],
            authors=[
                ("Eva", "Krapohl", False),
                ("Sophie", "von Stumm", False),
            ],
        )
        ext._email_harvester.harvest(ext.metadata.authors)

        assert ext.metadata.authors[0].email is None
        assert ext.metadata.authors[1].email == "s.vonstumm@gold.ac.uk"
        assert ext.metadata.authors[1].corresponding is True

    def test_proximity_beats_iteration_order(self):
        """When no surname affinity helps, prefer the surname appearing in the
        same/adjacent sentence as the email over one that's 2+ sentences away.
        """
        from bibr.paper_contents import CanonicalSection

        # Email sits right next to Smith. Doe's name is mentioned 2 sentences
        # earlier in an unrelated context. Smith should win even though Doe
        # is iterated first.
        ext = self._build_extractor(
            sentences=[
                ("Earlier work by Doe established the baseline.", 1),
                ("Methods were preregistered.", 1),
                ("Corresponding author: Jane Smith, jane@example.com", 1),
            ],
            sections=[(1, "Footer", CanonicalSection.UNKNOWN)],
            authors=[
                ("John", "Doe", False),
                ("Jane", "Smith", False),
            ],
        )
        ext._email_harvester.harvest(ext.metadata.authors)

        assert ext.metadata.authors[0].email is None
        assert ext.metadata.authors[1].email == "jane@example.com"


class TestPromoteCorrespondingFromAnchors:
    """Tests for the post-pass that promotes corresponding=True for authors
    whose LLM-extracted email sits next to a "Corresponding Author(s):" anchor.
    """

    def _build_extractor(self, sentences, sections, authors):
        from bibr.models import PaperAuthor, PaperMetadata
        from bibr.paper_contents import PaperContents, PaperSection, PaperSentence

        sentence_objs = [
            PaperSentence(text_id=i, text=t, section_id=sid, paragraph_id=1)
            for i, (t, sid) in enumerate(sentences, start=1)
        ]
        section_objs = [
            PaperSection(section_id=sid, header=h, level=1, parent_section_id=0, section_type=st)
            for sid, h, st in sections
        ]
        contents = PaperContents(
            sentences=sentence_objs,
            sections=section_objs,
            tables=[],
            links=[],
            sections_text={},
        )
        ext = MetadataExtractor(contents)
        author_objs = [
            PaperAuthor(
                author_id=i,
                given=g,
                family=f,
                affiliation="",
                email=e,
                corresponding=c,
                orcid=None,
            )
            for i, (g, f, e, c) in enumerate(authors, start=1)
        ]
        ext.metadata = PaperMetadata(doi="", title="", keywords=[], authors=author_objs)
        return ext

    def test_promotes_both_authors_under_plural_anchor(self):
        """Multi-corresponding block: LLM gives emails but flags both False."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                ("Corresponding Authors:", 1),
                ("Caroline Charpentier, caroline.charpentier@ucl.ac.uk", 1),
                ("Tali Sharot, t.sharot@ucl.ac.uk", 1),
            ],
            sections=[(1, "Footer", CanonicalSection.UNKNOWN)],
            authors=[
                ("Caroline", "Charpentier", "caroline.charpentier@ucl.ac.uk", False),
                ("Other", "Coauthor", None, False),
                ("Tali", "Sharot", "t.sharot@ucl.ac.uk", False),
            ],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].corresponding is True
        assert ext.metadata.authors[2].corresponding is True
        assert ext.metadata.authors[1].corresponding is False

    def test_no_promotion_without_anchor(self):
        """An email mentioned without a corresponding-author anchor must NOT
        promote — otherwise byline-only emails would silently become
        corresponding=True. (Multi-author paper: a sole author with an email
        is de-facto corresponding and IS promoted.)"""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                ("Author byline: Jane Smith, jane@uni.edu", 1),
                ("Methods were preregistered.", 1),
            ],
            sections=[(1, "Body", CanonicalSection.UNKNOWN)],
            authors=[("Jane", "Smith", "jane@uni.edu", False), ("John", "Doe", None, False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].corresponding is False

    def test_promotes_via_section_header_anchor(self):
        """PSS layout: the corresponding-author callout becomes the section
        heading itself (e.g. 'Corresponding Author:'); the body sentences then
        carry just 'E-mail: x@y' with no lexical anchor."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("E-mail: emily.holmes@mrc-cbu.cam.ac.uk", 1)],
            sections=[(1, "Corresponding Author:", CanonicalSection.UNKNOWN)],
            authors=[("Emily A.", "Holmes", "emily.holmes@mrc-cbu.cam.ac.uk", False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].corresponding is True

    def test_promotes_via_footnote_family_in_text(self):
        """Sage strips the 'Corresponding Author:' header and renames the
        section to 'Footnote N'; the family + email co-occurrence is what
        survives."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[
                ("Tali Sharot, Affective Brain Lab, UCL E-mail: t.sharot@ucl.ac.uk", 1),
            ],
            sections=[(1, "Footnote 2", CanonicalSection.FOOTNOTE)],
            authors=[("Tali", "Sharot", "t.sharot@ucl.ac.uk", False)],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].corresponding is True

    def test_promotes_via_footnote_family_in_email_localpart(self):
        """Bare 'E-mail: x@y' footnote with no surrounding name: the email
        local-part itself contains the author's surname (e.g.
        caroline.charpentier.11@ucl.ac.uk for Charpentier)."""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("E-mail: caroline.charpentier.11@ucl.ac.uk", 1)],
            sections=[(1, "Footnote 1", CanonicalSection.FOOTNOTE)],
            authors=[
                ("Caroline J.", "Charpentier", "caroline.charpentier.11@ucl.ac.uk", False),
            ],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].corresponding is True

    def test_no_footnote_promotion_when_family_unmatched(self):
        """Footnote affinity must NOT fire when neither the section text nor
        the email local-part references the author's family — otherwise a
        random footnote-housed email would falsely promote any email-bearing
        author by section-type alone. (Multi-author paper: the sole-author
        rule would legitimately promote a single author with an email.)"""
        from bibr.paper_contents import CanonicalSection

        ext = self._build_extractor(
            sentences=[("Some unrelated note: contact@example.com", 1)],
            sections=[(1, "Footnote 1", CanonicalSection.FOOTNOTE)],
            authors=[
                ("Jane", "Smith", "contact@example.com", False),
                ("John", "Doe", None, False),
            ],
        )
        ext._email_harvester.harvest(ext.metadata.authors)
        assert ext.metadata.authors[0].corresponding is False


class TestDemoteImplausibleCorrespondingFlags:
    """Tests for the post-LLM backstop that catches "every author marked
    corresponding" responses (a common failure mode on papers where every
    author lists their email in the byline)."""

    def _authors(self, n_total: int, n_corresp: int):
        from bibr.models import PaperAuthor

        return [
            PaperAuthor(
                author_id=i + 1,
                given=f"G{i}",
                family=f"F{i}",
                affiliation="",
                email=None,
                corresponding=(i < n_corresp),
                orcid=None,
            )
            for i in range(n_total)
        ]

    def test_all_authors_marked_corresponding_three_or_more_demoted(self):
        authors = self._authors(8, 8)
        AuthorEmailHarvester.demote_implausible_flags(authors)
        assert all(a.corresponding is False for a in authors)

    def test_subset_marked_corresponding_unchanged(self):
        # 4 authors, 2 corresponding — plausible
        authors = self._authors(4, 2)
        AuthorEmailHarvester.demote_implausible_flags(authors)
        flags = [a.corresponding for a in authors]
        assert flags == [True, True, False, False]

    def test_two_authors_both_corresponding_unchanged(self):
        authors = self._authors(2, 2)
        AuthorEmailHarvester.demote_implausible_flags(authors)
        assert all(a.corresponding for a in authors)

    def test_single_author_corresponding_unchanged(self):
        authors = self._authors(1, 1)
        AuthorEmailHarvester.demote_implausible_flags(authors)
        assert authors[0].corresponding is True

    def test_empty_authors_no_op(self):
        AuthorEmailHarvester.demote_implausible_flags([])  # no crash


class TestExtractReferencesOrchestration:
    """segment → CRF-fallback → batch → parse → assemble."""

    def _llm_ref(self, idx, title, authors="Author, A."):
        from bibr.schemas import PaperReferenceLLM

        return PaperReferenceLLM(
            index=idx,
            title=title,
            authors=authors,
            first_page=None,
            last_page=None,
            volume=None,
            issue=None,
            year=2020,
            container=None,
            doi=None,
        )

    def _extractor_with_refs(self, llm_client):
        ref_texts = [
            "Smith, J. (2020). A. Journal, 1, 1-10.",
            "Doe, A. (2019). B. Journal, 2, 11-20.",
        ]
        df = pd.DataFrame({"section_name": ["References"] * 2, "text": ref_texts})
        contents = mock.Mock(spec=PaperContents)
        contents.sentences_df = df
        contents.detected_headers = []
        contents.detected_footers = []
        contents.layout_hints = []
        contents.sections = []
        contents.sentences = []
        return MetadataExtractor(contents, llm_client=llm_client), df

    async def test_llm_segments_then_batch_parses(self, monkeypatch):
        monkeypatch.setattr(
            "bibr.extract.ref_extractor._resolve_ref_strategies",
            lambda *a, **kw: ("llm", "llm"),
        )
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(
            return_value=["Smith, J. (2020).", "Doe, A. (2019)."]
        )
        llm.extract_references = mock.AsyncMock(
            return_value=[self._llm_ref(1, "A"), self._llm_ref(2, "B")]
        )
        ext, df = self._extractor_with_refs(llm)
        refs = await ext._extract_references(df)

        llm.segment_references.assert_awaited_once()
        assert len(refs) == 2
        assert [r.bib_id for r in refs] == [1, 2]

    async def test_crf_fallback_on_segmentation_exception(self, monkeypatch):
        monkeypatch.setattr(
            "bibr.extract.ref_extractor._resolve_ref_strategies",
            lambda *a, **kw: ("llm", "llm"),
        )
        fake_seg = mock.Mock()
        fake_seg.segment = mock.Mock(return_value=["Smith, J. (2020). A.", "Doe, A. (2019). B."])
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake_seg)
        llm = mock.Mock()
        llm.segment_references = mock.AsyncMock(side_effect=RuntimeError("seg down"))
        llm.extract_references = mock.AsyncMock(
            return_value=[self._llm_ref(1, "A"), self._llm_ref(2, "B")]
        )
        ext, df = self._extractor_with_refs(llm)
        refs = await ext._extract_references(df)

        fake_seg.segment.assert_called_once()
        assert len(refs) == 2

    async def test_crf_fallback_on_zero_snapped_spans(self, monkeypatch):
        monkeypatch.setattr(
            "bibr.extract.ref_extractor._resolve_ref_strategies",
            lambda *a, **kw: ("llm", "llm"),
        )
        fake_seg = mock.Mock()
        fake_seg.segment = mock.Mock(return_value=["Smith, J. (2020). A."])
        monkeypatch.setattr("bibr.extract.ref_extractor._get_ner_segmenter", lambda *_a: fake_seg)
        llm = mock.Mock()
        # Anchors that do not occur in the ref text → 0 snapped spans.
        llm.segment_references = mock.AsyncMock(return_value=["NONEXISTENT ANCHOR"])
        llm.extract_references = mock.AsyncMock(return_value=[self._llm_ref(1, "A")])
        ext, df = self._extractor_with_refs(llm)
        await ext._extract_references(df)

        fake_seg.segment.assert_called_once()


class TestResolveRefStrategies:
    """The decoupled (seg, parse) resolution table."""

    def _resolve(self, monkeypatch, *, seg=None, parse=None):
        from bibr.config import Settings
        from bibr.extract.extractor import _resolve_ref_strategies

        monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", seg, raising=False)
        monkeypatch.setattr(Settings, "REF_PARSE_STRATEGY", parse, raising=False)
        return _resolve_ref_strategies()

    def test_default_is_geom_ner(self, monkeypatch):
        # The local geometry GBM is the default segmenter (LLM cascade); the
        # local ModernBERT-CRF parser is the default parser. Explicit None
        # (knobs unset) resolves to the same defaults as the config fields.
        assert self._resolve(monkeypatch) == ("geom", "ner")

    def test_per_run_overrides_win_over_settings(self, monkeypatch):
        # RunConfig-carried strategies (chew(refs=...), CLI --refs) beat every
        # Settings knob without mutating the global.
        from bibr.config import Settings
        from bibr.extract.extractor import _resolve_ref_strategies

        monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "geom", raising=False)
        monkeypatch.setattr(Settings, "REF_PARSE_STRATEGY", "ner", raising=False)
        assert _resolve_ref_strategies(seg_override="llm", parse_override="llm") == (
            "llm",
            "llm",
        )

    def test_partial_override_keeps_settings_for_the_rest(self, monkeypatch):
        from bibr.config import Settings
        from bibr.extract.extractor import _resolve_ref_strategies

        monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", None, raising=False)
        monkeypatch.setattr(Settings, "REF_PARSE_STRATEGY", None, raising=False)
        assert _resolve_ref_strategies(parse_override="LLM") == ("geom", "llm")

    def test_explicit_crf_seg_stays_reachable(self, monkeypatch):
        # CRF segmentation remains opt-in via the explicit knob (with NER parse).
        assert self._resolve(monkeypatch, seg="crf", parse="ner") == ("crf", "ner")

    def test_case_insensitive(self, monkeypatch):
        assert self._resolve(monkeypatch, seg="CRF", parse="LLM") == ("crf", "llm")


class TestExpandCompactLastPage:
    """_expand_compact_last_page — bibliographic compact ranges ("782-92" → 792)."""

    @pytest.mark.parametrize(
        ("first_page", "last_page", "expected"),
        [
            # Judged MDPI cases (207607601421051739369458)
            ("782", "92", "792"),
            ("956", "65", "965"),
            ("205", "10", "210"),
            ("356", "60", "360"),
            # Single-digit tail: "98-9" → 98-99
            ("98", "9", "99"),
            ("782", "9", "789"),
            # Two digits dropped: "105-13" → 105-113
            ("105", "13", "113"),
        ],
    )
    def test_expands_compact_ranges(self, first_page, last_page, expected):
        from bibr.extract.extractor import _expand_compact_last_page

        assert _expand_compact_last_page(first_page, last_page) == expected

    @pytest.mark.parametrize(
        ("first_page", "last_page"),
        [
            ("105", "113"),  # normal full range
            ("9", "12"),  # normal range, lp longer
            ("100", "100"),  # single page printed as range
            (None, "92"),  # missing first page
            ("782", None),  # missing last page
            ("e13067", "92"),  # article-number first page
            ("782", "e92"),  # non-numeric last page
            ("218", "05"),  # prefix-fill would go backwards (205 < 218)
            ("381", "12 "),  # equal-ish garbage stays put after strip → expands? no: "12" valid
        ],
    )
    def test_leaves_non_compact_untouched(self, first_page, last_page):
        from bibr.extract.extractor import _expand_compact_last_page

        assert _expand_compact_last_page(first_page, last_page) == last_page


class TestBackfillIssue:
    """_backfill_issue — recover issue the LLM dropped from the printed "vol(issue)"."""

    @pytest.mark.parametrize(
        ("volume", "segment", "expected"),
        [
            # Judged cases (0956797621995197, 09567976241258149)
            ("55", "communicative situations. Psychophysiology, 55(7), Article e13067.", "7"),
            ("121", "A meta-analysis. Psychological Bulletin, 121(3), 371-394.", "3"),
            ("16", "Cognitive, Affective, & Behavioral Neuroscience, 16(5), 836-847.", "5"),
            ("8", "Science Advances, 8(39), Article eabn9418.", "39"),
            ("2", "Affective Science, 2(4), 379-390.", "4"),
            # Issue ranges keep the dash
            ("12", "Journal of Things, 12(3-4), 100-115.", "3-4"),
            ("12", "Journal of Things, 12(3–4), 100-115.", "3–4"),
        ],
    )
    def test_backfills_from_segment(self, volume, segment, expected):
        from bibr.extract.extractor import _backfill_issue

        assert _backfill_issue(None, volume, segment) == expected

    def test_existing_issue_never_overwritten(self):
        from bibr.extract.extractor import _backfill_issue

        seg = "Psychophysiology, 55(7), Article e13067."
        assert _backfill_issue("3", "55", seg) == "3"

    @pytest.mark.parametrize(
        ("volume", "segment"),
        [
            # No parenthesized issue printed
            ("41", "An ERP analysis. Psychophysiology, 41, 441-449."),
            # Volume missing → nothing to anchor on
            (None, "Psychophysiology, 55(7), Article e13067."),
            # Year in parens must not be mistaken for an issue of volume "20"
            ("20", "Smith, J. (2020). A title. Journal, 20, 1-10."),
            # Non-numeric content in parens (supplements) stays conservative
            ("55", "Psychophysiology, 55(Suppl 1), 1-10."),
            # Volume printed but the only "NN(" belongs to a different number
            ("55", "Psychological Bulletin, 121(3), 371-394."),
        ],
    )
    def test_no_backfill_without_clean_match(self, volume, segment):
        from bibr.extract.extractor import _backfill_issue

        assert _backfill_issue(None, volume, segment) is None


class TestSplitVolIssue:
    """_split_vol_issue — strip bib punctuation noise and split a combined
    'N(M)' (printed in either the volume or issue span) into volume + issue."""

    @pytest.mark.parametrize(
        ("volume", "issue", "expected"),
        [
            # trailing punctuation stripped
            ("25,", None, ("25", None)),
            ("25", "", ("25", None)),
            # combined "N(M)" dumped into the volume span → split
            ("21(4),", None, ("21", "4")),
            ("6(1)", None, ("6", "1")),
            ("55(7):", None, ("55", "7")),  # Vancouver "55(7):782-92"
            # combined form landed in the issue span instead
            (None, "6(1)", ("6", "1")),
            # issue range keeps the dash
            ("21(4-5)", None, ("21", "4-5")),
            # already-clean pair is untouched
            ("12", "3", ("12", "3")),
            # parens-only issue cleaned to bare number
            ("12", "(4)", ("12", "4")),
            # nothing to do
            (None, None, (None, None)),
        ],
    )
    def test_split(self, volume, issue, expected):
        from bibr.extract.extractor import _split_vol_issue

        assert _split_vol_issue(volume, issue) == expected


class TestNerParseNormalization:
    """vol/issue normalization wired into the NER parse path."""

    def test_ner_parse_splits_combined_vol_issue(self):
        ext = _make_extractor(["References"], ["ref"])
        parser = mock.MagicMock()
        parser.parse_batch.return_value = [
            {
                "title": "A title",
                "authors": "Smith, J.",
                "year": 2020,
                "container": "Journal",
                "volume": "21(4),",  # combined form dumped into the volume span
                "issue": None,
                "first_page": "100",
                "last_page": "110",
            }
        ]
        with mock.patch("bibr.extract.ref_extractor._get_ner_parser", return_value=parser):
            refs = ext.refs._parse_references_ner(
                ["Smith, J. (2020). A title. Journal, 21(4), 100-110."]
            )
        assert refs[0].volume == "21"
        assert refs[0].issue == "4"


class TestParsedRefNormalization:
    """Compact-page + issue normalization wired into the LLM parse path."""

    async def test_llm_parse_normalizes_pages_and_issue(self):
        from bibr.schemas import PaperReferenceLLM

        ref_texts = [
            "Alves, S. (2016). Views of detained women. Social Sciences, 5(4), 782-92.",
            "Baess, P. (2015). My partner. Brain Research, 233(1), 105-113.",
        ]
        sections = ["Introduction"] * 3 + ["References"] * len(ref_texts)
        texts = [f"Intro {i}" for i in range(3)] + ref_texts
        paper_sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
            PaperSection(1, "References", 2, None, CanonicalSection.REFERENCES, 1.0),
        ]
        ext = _make_extractor(sections, texts, paper_sections=paper_sections)

        mock_refs = [
            PaperReferenceLLM(
                index=1,
                title="Views of detained women",
                authors="Alves, S.",
                year=2016,
                container="Social Sciences",
                volume="5",
                issue=None,  # LLM dropped the printed (4)
                first_page="782",
                last_page="92",  # LLM parsed the compact range verbatim
            ),
            PaperReferenceLLM(
                index=2,
                title="My partner",
                authors="Baess, P.",
                year=2015,
                container="Brain Research",
                volume="233",
                issue="1",
                first_page="105",
                last_page="113",
            ),
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(return_value=mock_refs)

        result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert result[0].last_page == "792"
        assert result[0].issue == "4"
        # Clean entry untouched
        assert result[1].last_page == "113"
        assert result[1].issue == "1"


# ---------------------------------------------------------------------------
# Shared reference finalization (strategy-independent)
# ---------------------------------------------------------------------------


class TestFinalizeReferenceFields:
    """_finalize_reference_fields — the strategy-independent finalize shared by
    every parse path (LLM, NER, future NuExtract): DOI rescue, compact
    last-page expansion, bib_type migration/inference."""

    def _finalize(self, fields, segment):
        from bibr.extract.extractor import _finalize_reference_fields

        return _finalize_reference_fields(fields, segment)

    def test_doi_rescued_from_segment_when_parser_missed_it(self):
        out = self._finalize(
            {"title": "Digital health", "authors": "Wongvibulsin S"},
            "Wongvibulsin S. Digital health. 2021. doi:10.2196/18773",
        )
        assert out["doi"] == "10.2196/18773"

    def test_parser_emitted_doi_truncated_at_a_wrap_falls_through_to_rescue(self):
        # The parser copies what it sees, so a DOI wrapped after a hyphen comes
        # back truncated rather than absent. The rescue is reached through
        # ``normalize_doi(...) or ...``, so a stub that validates suppresses it —
        # the reference then exports a DOI that resolves to nothing.
        out = self._finalize(
            {"title": "Some thoughts", "authors": "Miller N E", "doi": "10.1037/0033-"},
            "Miller, N. E. (1994). Some thoughts. Psychological Bulletin, "
            "115(1), 102-115. doi:10.1037/0033- 2909.115.1.102",
        )
        assert out["doi"] == "10.1037/0033-2909.115.1.102"

    def test_parser_emitted_doi_wins_over_segment_token(self):
        out = self._finalize(
            {"title": "T", "authors": "A", "doi": "10.1000/real.1"},
            "T. A. 2021. doi:10.9999/other.2",
        )
        assert out["doi"] == "10.1000/real.1"

    def test_compact_last_page_expanded(self):
        out = self._finalize(
            {"title": "T", "authors": "A", "first_page": "346", "last_page": "9"},
            None,
        )
        assert out["last_page"] == "349"

    def test_explicit_bib_type_kept_and_migrated(self):
        out = self._finalize({"title": "T", "bib_type": "journal-article"}, None)
        assert out["bib_type"] == "journal_article"

    def test_bib_type_inferred_from_container_when_missing(self):
        out = self._finalize({"title": "T", "container": "Nature"}, None)
        assert out["bib_type"] == "journal_article"

    @pytest.mark.parametrize(
        ("tail", "expected"),
        [
            ("PhD thesis, Example University, 2020.", "thesis"),
            ("arXiv:2301.12345, 2023.", "preprint"),
            ("In Proceedings of ACM, 2020.", "conference_paper"),
            ("Technical report TR-2020-01.", "report"),
        ],
    )
    def test_bib_type_uses_the_printed_segment(self, tail, expected):
        out = self._finalize({"title": "A useful method"}, f"Smith A. A useful method. {tail}")
        assert out["bib_type"] == expected

    def test_segment_does_not_override_explicit_type(self):
        out = self._finalize(
            {"title": "A useful method", "bib_type": "journal_article"},
            "Smith A. A useful method. Earlier version: arXiv:2301.12345.",
        )
        assert out["bib_type"] == "journal_article"


class TestSequenceReferences:
    """_sequence_references — drop stub refs (no title AND no authors), then
    assign contiguous bib_ids. Single filter/sequence point for all paths."""

    @staticmethod
    def _ref(bib_id, title, authors):
        from bibr.paper import PaperReference

        return PaperReference.model_validate(
            {
                "bib_id": bib_id,
                # PaperReference.title is a required str; parse paths coerce
                # a missing title to "" (falsy for the stub filter).
                "title": title or "",
                "authors": authors,
                "year": None,
                "container": None,
                "volume": None,
                "first_page": None,
            }
        )

    def test_stub_filtered_and_ids_contiguous(self):
        from bibr.extract.extractor import _sequence_references

        refs = [self._ref(1, "A", "X"), self._ref(2, None, None), self._ref(3, "C", "Z")]
        out = _sequence_references(refs)
        assert [(r.bib_id, r.title) for r in out] == [(1, "A"), (2, "C")]

    def test_all_valid_kept_in_order(self):
        from bibr.extract.extractor import _sequence_references

        refs = [self._ref(9, "A", None), self._ref(7, None, "X")]
        out = _sequence_references(refs)
        assert [r.bib_id for r in out] == [1, 2]


class TestLLMPathContiguousBibIds:
    """A junk ref the LLM emits (no title, no authors) must not leave a gap in
    bib_ids: filter first, then sequence — same contract as the NER path."""

    async def test_junk_ref_filtered_without_bib_id_gap(self):
        from bibr.schemas import PaperReferenceLLM

        ref_texts = [
            "Smith, J. (2020). Paper A. Nature, 10, 1-5.",
            "— garbled fragment —",
            "Jones, A. (2019). Paper B. Science, 20, 10-15.",
        ]
        sections = ["References"] * len(ref_texts)
        ext = _make_extractor(sections, ref_texts)

        def _llm_ref(index, title, authors, year):
            return PaperReferenceLLM(
                index=index,
                title=title,
                authors=authors,
                year=year,
                first_page=None,
                volume=None,
                container=None,
            )

        mock_refs = [
            _llm_ref(1, "Paper A", "Smith, J.", "2020"),
            _llm_ref(2, None, None, None),
            _llm_ref(3, "Paper B", "Jones, A.", "2019"),
        ]
        ext.llm_client = mock.MagicMock()
        ext.llm_client.extract_references = mock.AsyncMock(return_value=mock_refs)

        result = await ext.refs._parse_references_llm("\n".join(ref_texts), ref_texts)

        assert [r.title for r in result] == ["Paper A", "Paper B"]
        assert [r.bib_id for r in result] == [1, 2]


class TestRefParseStrategyRegistry:
    """Parse-strategy dispatch is a registry, not an if/elif — adding a
    strategy (e.g. NuExtract) is one entry, and unknown names fall back to
    the LLM parser."""

    async def test_ner_entry_routes_to_ner_parser(self, monkeypatch):
        from bibr.extract.extractor import REF_PARSE_STRATEGIES

        ext = _make_extractor([], [])
        seen = {}
        monkeypatch.setattr(
            ext.refs,
            "_parse_references_ner",
            lambda ref_strings: seen.setdefault("ner", ref_strings),
        )
        out = await REF_PARSE_STRATEGIES["ner"](ext.refs, "full text", ["r1", "r2"])
        assert seen["ner"] == ["r1", "r2"]
        assert out == ["r1", "r2"]

    async def test_ner_parse_runs_off_event_loop(self):
        """The synchronous ModernBERT-CRF parse must not block the event loop:
        the NER adapter offloads ``_parse_references_ner`` via
        ``asyncio.to_thread``, so it runs on a worker thread (a different ident
        than the loop thread) and its result flows through unchanged."""
        loop_ident = threading.get_ident()
        recorded: dict[str, object] = {}
        sentinel = ["parsed-ref"]

        def fake_parse(ref_strings):
            recorded["ident"] = threading.get_ident()
            recorded["arg"] = ref_strings
            return sentinel

        ext = mock.Mock()
        ext._parse_references_ner.side_effect = fake_parse
        out = await _parse_refs_via_ner(ext, "full text", ["r1", "r2"])
        assert out is sentinel
        assert recorded["arg"] == ["r1", "r2"]
        assert recorded["ident"] != loop_ident

    async def test_unknown_strategy_falls_back_to_llm(self, monkeypatch):
        from bibr.extract import extractor as extractor_mod

        ext = _make_extractor([], [])

        async def fake_llm(ref_text, ref_strings):
            return ["llm-called"]

        def fail_ner(ref_strings):
            raise AssertionError("NER must not run for unknown strategies")

        monkeypatch.setattr(ext.refs, "_parse_references_llm", fake_llm)
        monkeypatch.setattr(ext.refs, "_parse_references_ner", fail_ner)
        parser = extractor_mod.REF_PARSE_STRATEGIES.get(
            "no-such-strategy", extractor_mod.REF_PARSE_STRATEGIES["llm"]
        )
        assert await parser(ext.refs, "text", ["r1"]) == ["llm-called"]
