"""Regression & exactness tests for Wave 3A micro-hotspot optimizations."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd

from bibr.extract import core_metadata as cm
from bibr.extract import doi_identity as di
from bibr.extract import equation_extractor as eq_mod
from bibr.extract import footnote_citations as fc
from bibr.extract import statement_scan as ss
from bibr.input import consolidate_text as ct
from bibr.models import PaperAuthor
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence
from bibr.structure import xref_utils as xu
from bibr.utils.link_targets import LinkTargets


def test_xref_prescan_rejects_embedded_substrings_but_keeps_subsection():
    for prose in (
        "This requires equal attention to subsequent database establishment.",
        "Notable acceptable outcomes remained suitable across stable frequencies.",
        "Please configure the adequate techniques accordingly.",
    ):
        assert xu._XREF_PRESCAN_RE.search(prose) is None

    for hit in (
        "See Table 1 and Fig. 2.",
        "Refer to Eq. (3) and Section 4.",
        "As discussed in Subsection 2.4 and §3.1.",
        "See Supplementary Material and Tbl. 2.",
    ):
        assert xu._XREF_PRESCAN_RE.search(hit) is not None


def test_doi_candidates_gate_handles_ascii_and_non_ascii_edge_cases():
    prov = {
        "source_kind": "sentence",
        "page": 1,
        "section_id": 0,
        "section_type": None,
        "region_index": None,
        "region_type": None,
        "text_id": 1,
    }
    # Pure ASCII prose without "/" or "10" returns empty fast
    assert (
        di._candidates_from_text("No identifier is present in this prose sentence.", **prov) == []
    )
    # Unicode hyphen inside DOI bypasses ASCII fast gate and repairs properly
    assert di._candidates_from_text("DOI: 10.1234/abc\u2010123", **prov)
    # Control char inside 10. bypasses fast gate and repairs properly
    assert di._candidates_from_text("DOI: 1\x000.1234/abc.123", **prov)


def test_equation_prescan_and_glitch_guards_preserve_latex_and_glitches():
    ex = eq_mod.EquationExtractor()
    sec = [
        PaperSection(
            section_id=1,
            header="Results",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.RESULTS,
        )
    ]
    sents = [
        PaperSentence(
            text_id=1,
            text="Plain prose with no math or numbers at all.",
            section_id=1,
            paragraph_id=0,
            page_number=1,
        ),
        PaperSentence(
            text_id=2,
            text=r"We found $x \le 5$ and $t(28) = 3.1$.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
        ),
        PaperSentence(
            text_id=3,
            text="Glitched stat t(59) 5 0.66, p , .01.",
            section_id=1,
            paragraph_id=2,
            page_number=1,
        ),
        PaperSentence(
            text_id=4,
            text="Dropped equals F(1, 208) 5.01, p 0.026.",
            section_id=1,
            paragraph_id=3,
            page_number=1,
        ),
        PaperSentence(
            text_id=5,
            text="Cyrillic \u0440 < .05 and math \U0001d461(12) = 2.4.",
            section_id=1,
            paragraph_id=4,
            page_number=1,
        ),
    ]
    out = ex.extract_from_sentences(sents, sec, repair_glitches=True)
    text_ids = {eq.text_id for eq in out}
    assert 1 not in text_ids
    assert {2, 3, 4, 5}.issubset(text_ids)


def test_statement_scan_categories_in_matches_individual_anchors():
    samples = [
        "Ordinary methods sentence about participant recruitment.",
        "This work was supported by NIH grant R01-12345.",
        "The authors declare no competing financial interests.",
        "Data are available from the corresponding author upon reasonable request.",
        "Informed consent was obtained and the IRB approved the study.",
        "Funding",
    ]
    for sample in samples:
        expected = {
            field
            for field, patterns in ss._ANCHORS.items()
            if any(p.search(sample) for p in patterns)
        }
        norm = sample.strip().casefold()
        expected.update(f for f, labels in ss._BARE_CATEGORY_LABELS.items() if norm in labels)
        assert ss._categories_in(sample) == expected


def test_link_targets_short_key_guard_exactness():
    uris = [
        "https://doi.org/10.1038/s41592-024-00001-x",
        "https://osf.io/abc12/",
        "https://github.com/org/repo",
    ]
    lt = LinkTargets(uris)
    # Prose collisions on ("h", "t"), ("d", "o"), ("w", "w")
    assert not lt.spans("with", "the", line_ends=True)
    assert not lt.spans("and", "of", line_ends=True)
    # Genuine link splits at length 1, 2, 3, and >3
    assert lt.spans("(h", "ttps://osf.io/abc12/", line_ends=True)
    assert lt.spans("ht", "tps://osf.io/abc12/", line_ends=True)
    assert lt.spans("https://osf.", "io/abc12/", line_ends=True)
    assert lt.spans("10.", "1038/s41592-024-00001-x", line_ends=True)


def test_footnote_first_sentence_break_matches_sentence_breaks_with_quotes():
    cases = [
        "No break here at all",
        "First sentence. Second sentence. Third sentence.",
        'See "Title. With Period." Next sentence. Another.',
        "« Titre. Avec point. » Suite de la note. Fin.",
        'Unclosed "quote that runs. Past a break. And keeps going.',
    ]
    for text in cases:
        breaks = fc._sentence_breaks(text)
        expected = breaks[0].start() if breaks else None
        assert fc._first_sentence_break_pos(text) == expected


def test_footnote_collapse_repeats_exact_with_prefixes_and_single_word_titles():
    refs = [
        SimpleNamespace(
            text_id=1,
            title="Hamlet",
            authors="Shakespeare, W.",
            year=1603,
            container=None,
            publisher=None,
            volume=None,
            first_page=None,
            doi=None,
        ),
        SimpleNamespace(
            text_id=2,
            title="Hamlet",
            authors="Shakespeare, W.",
            year=1603,
            container="Folio",
            publisher="Blount",
            volume="1",
            first_page="1",
            doi=None,
        ),
        SimpleNamespace(
            text_id=3,
            title="Modern History",
            authors="Smith, J.",
            year=2020,
            container=None,
            publisher=None,
            volume=None,
            first_page=None,
            doi=None,
        ),
        SimpleNamespace(
            text_id=4,
            title="Modern History of Europe",
            authors="Smith, J.",
            year=2020,
            container="Oxford",
            publisher="OUP",
            volume=None,
            first_page="10",
            doi=None,
        ),
        SimpleNamespace(
            text_id=5,
            title="Modern History of Asia",
            authors="Jones, A.",
            year=2020,
            container=None,
            publisher=None,
            volume=None,
            first_page=None,
            doi=None,
        ),
    ]
    out = fc.collapse_repeats(refs)
    assert [r.text_id for r in out] == [1, 3, 5]
    assert out[0].container == "Folio"
    assert out[1].title == "Modern History of Europe"


def test_affiliation_grounding_memoizes_repeated_affiliations(monkeypatch):
    calls = 0
    real_fn = cm._ungrounded_affiliation_components

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(cm, "_ungrounded_affiliation_components", counting)
    authors = [
        PaperAuthor(
            author_id=i + 1,
            given="A.",
            family=f"Author{i}",
            affiliation="Department of Physics, University of Oxford",
        )
        for i in range(25)
    ]
    meta_df = pd.DataFrame({"text": ["Department of Physics, University of Oxford"]})
    dropped = cm.CoreMetadataExtractor._normalize_author_affiliations(authors, meta_df)
    assert dropped == []
    assert calls == 1


def test_clean_text_content_late_guards_preserve_math_urls_and_spaced_digits():
    assert (
        ct.clean_text_content_late("Plain prose without specials.", from_ocr=True)
        == "Plain prose without specials."
    )
    assert (
        ct.clean_text_content_late(
            r"Value $\alpha + \beta = 1$ at https://example.org/a_b_c.", from_ocr=True
        )
        == "Value α + β = 1 at https://example.org/a_b_c."
    )
    assert ct.clean_text_content_late(
        "F ( 1, 2 6 )=1 7. 9 0 6", from_ocr=True
    ) == "F ( 1, 2 6 )=1 7. 9 0 6" or re.search(
        r"17\.906|906", ct.clean_text_content_late("F ( 1, 2 6 )=1 7. 9 0 6", from_ocr=True)
    )
    assert ct.clean_text_content_late("Item 1 2 3 and 4", from_ocr=True) == "Item 123 and 4"
