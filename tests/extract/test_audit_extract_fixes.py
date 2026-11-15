"""Regressions: typeset-hyphen DOIs, display-math runs, NaN gate scores, unit-vs-marker emails."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_doi_identity_pdf_evidence import _contents, _evidence, _line, _select  # noqa: E402

from bibr.extract.doi_identity import collect_doi_candidates, select_doi_candidates  # noqa: E402
from bibr.extract.equation_extractor import EquationExtractor  # noqa: E402
from bibr.extract.front_matter import (  # noqa: E402
    FrontMatterBlock,
    FrontMatterCandidate,
    _matching_doi_blocks,
)
from bibr.extract.geom_tagger import expected_entry_precision  # noqa: E402
from bibr.extract.pdf_doi_evidence import MetadataDoi  # noqa: E402
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence  # noqa: E402


def _extract(text: str):
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(section_id=1, header="Results", level=1, parent_section_id=0),
    ]
    sentence = PaperSentence(text_id=1, text=text, section_id=1, paragraph_id=1)
    return EquationExtractor().extract_from_sentences([sentence], sections)


_TYPESET_HYPHENS = ["‐", "‑", "–", "−"]
_DOI = "10.1234/s1-2020-01838-x"


@pytest.mark.parametrize("hyphen", _TYPESET_HYPHENS)
def test_a_text_layer_doi_with_typeset_hyphens_is_read_whole(hyphen):
    contents = _contents([(CanonicalSection.TITLE, "A study of examples", 1)])
    line = _line(1, "Published: 01.09.2024 DOI: " + _DOI.replace("-", hyphen))

    _candidates, selection = _select(contents, _evidence([line]))

    assert selection.selected is not None
    assert selection.selected.normalized == _DOI


@pytest.mark.parametrize("hyphen", _TYPESET_HYPHENS)
def test_a_parsed_doi_with_typeset_hyphens_is_read_whole(hyphen):
    text = "Published 2024. DOI: " + _DOI.replace("-", hyphen)
    contents = _contents([(CanonicalSection.TITLE, text, 1)])

    selection = select_doi_candidates(collect_doi_candidates(contents))

    assert selection.selected is not None
    assert selection.selected.normalized == _DOI
    assert selection.selected.raw.endswith(_DOI.replace("-", hyphen))


@pytest.mark.parametrize("hyphen", _TYPESET_HYPHENS)
def test_a_text_layer_reading_with_typeset_hyphens_is_the_parsed_doi(hyphen):
    contents = _contents([(CanonicalSection.TITLE, "Published 2024. DOI: " + _DOI, 1)])
    line = _line(1, "Published: 2024 DOI: " + _DOI.replace("-", hyphen))

    candidates, selection = _select(contents, _evidence([line]))

    assert selection.selected.normalized == _DOI
    assert [c.source_kind for c in candidates] == ["sentence"]


@pytest.mark.parametrize("hyphen", _TYPESET_HYPHENS)
def test_a_metadata_doi_with_typeset_hyphens_agrees_whole(hyphen):
    contents = _contents([(CanonicalSection.TITLE, "Published 2024. DOI: " + _DOI, 1)])
    evidence = _evidence(metadata=[MetadataDoi("pdf_info", "doi", _DOI.replace("-", hyphen))])

    candidates = collect_doi_candidates(contents, evidence)

    assert {c.normalized for c in candidates} == {_DOI}


@pytest.mark.parametrize("hyphen", _TYPESET_HYPHENS)
def test_an_expected_doi_is_found_in_a_block_printing_typeset_hyphens(hyphen):
    candidate = FrontMatterCandidate(
        candidate_id="c1",
        source_kind="sentence",
        reading_order=0,
        page=1,
        bbox=None,
        region_label=None,
        font_size=None,
        font_bold=None,
        section_id=None,
        text_ids=(1,),
        paragraph_id=None,
        raw_text="DOI: " + _DOI.replace("-", hyphen),
        normalized_text="doi",
        roles=frozenset({"doi"}),
    )
    block = FrontMatterBlock(block_id="b1", candidate_ids=("c1",), title_candidate_ids=())

    found = _matching_doi_blocks((block,), {"c1": candidate}, _DOI, None)

    assert [b.block_id for b in found] == ["b1"]


def test_a_nan_gate_score_stays_nan():
    assert np.isnan(expected_entry_precision(np.array([0.9, np.nan, 0.1])))
    assert expected_entry_precision(np.array([0.1, 0.9, 0.2])) == pytest.approx(0.7)


@pytest.mark.parametrize(
    "text",
    ["$$t = 2.1, p = .03$$", "We found $$t = 2.1, p = .03$$ here."],
)
def test_a_display_formula_holding_a_taken_stat_is_read_in_runs(text):
    got = [(e.lhs, e.comp, e.rhs) for e in _extract(text)]

    assert got == [("t", "=", "2.1"), ("p", "=", ".03")]


def test_a_display_formula_with_a_taken_stat_leaves_no_fragment():
    got = [(e.lhs, e.comp, e.rhs) for e in _extract("We found $$p < 0.001^{***}$$ here.")]

    assert got == [("p", "<", "0.001")]


def test_a_pg_unit_is_not_a_corresponding_marker():
    from test_author_email_harvester import _make_contents_with_text

    from bibr.extract.author_email_harvester import AuthorEmailHarvester
    from bibr.models import PaperAuthor

    def harvest(middle: str):
        contents = _make_contents_with_text(
            ["Title", "Jane Doe, John Smith", middle, "Jane Doe: jane.doe@university.edu"]
        )
        authors = [
            PaperAuthor(author_id=1, given="Jane", family="Doe", affiliation="X"),
            PaperAuthor(author_id=2, given="John", family="Smith", affiliation="X"),
        ]
        AuthorEmailHarvester(contents).harvest(authors)
        return authors[0]

    assert harvest("Serum levels were 10 pg ml in controls.").corresponding is False
    marked = harvest("pg Jane Doe")
    assert marked.corresponding is True


def test_an_unbroken_cjk_row_is_scanned_for_addresses_in_linear_time():
    import time

    from bibr.extract.front_matter import _ADDRESS_TOKEN_RE

    row = "中文摘要内容" * 1000
    started = time.perf_counter()
    assert _ADDRESS_TOKEN_RE.sub(" ", row) == row
    assert time.perf_counter() - started < 0.5
    old = re.compile(r"\S*(?:@|\[at\]|\(at\)|://|www\.|[^\W\d_]\.[^\W\d_])\S*", re.IGNORECASE)
    for text in (
        "mail jan.peeters@example.org now",
        "see https://x.org/a, www.b.org",
        "x[at]y(at)z a.b 1.2 ab",
        "plain words only",
    ):
        assert _ADDRESS_TOKEN_RE.sub(" ", text) == old.sub(" ", text)
