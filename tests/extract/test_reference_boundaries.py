"""Conservative terminal trimming for reference-section spill."""

from unittest import mock

import pandas as pd
import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection


def _locator(texts: list[str]) -> RefLocator:
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = pd.DataFrame(
        {
            "text_id": range(89, 89 + len(texts)),
            "section_id": [8] * len(texts),
            "section_name": ["References"] * len(texts),
            "page_number": [10] * len(texts),
            "text": texts,
        }
    )
    contents.sections = [PaperSection(8, "References", 1, None, CanonicalSection.REFERENCES, 1.0)]
    contents.layout_hints = []
    return RefLocator(contents)


def test_ord324_publisher_note_starts_terminal_non_reference_record():
    refs = [
        "9. Locatelli F, Dallapiccola B et al. Le priorità Del piano nazionale Della genomica. 2021.",
        "10. Stark Z, Scott RH. Genomic newborn screening for rare diseases. 2023.",
    ]
    spill = [
        "Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "Francesco Andrea Causio1 · Sara Farina1 · Alessandra Maio1",
        "1 Section of Hygiene, Department of Life Sciences & Public Health, Rome, Italy",
    ]

    located = _locator([*refs, *spill]).collect_reference_rows()

    assert located["text"].tolist() == refs


def test_explicit_author_information_heading_trims_only_the_tail():
    refs = [
        "1. Smith J. Author biography methods in historical research. Journal of Tests. 2020.",
        "2. Doe A. Institutional affiliations and citation behavior. Scientometrics. 2021.",
    ]
    located = _locator([*refs, "Author information", "Alice Smith is professor of sociology."])

    assert located.collect_reference_rows()["text"].tolist() == refs


def test_ord324_authors_and_affiliations_heading_trims_tail():
    refs = [
        "9. Locatelli F, Dallapiccola B et al. Le priorità Del piano nazionale Della genomica. 2021.",
        "10. Stark Z, Scott RH. Genomic newborn screening for rare diseases. 2023.",
    ]
    spill = [
        "Authors and Affiliations",
        "Francesco Andrea Causio1 · Sara Farina1 · Alessandra Maio1",
        "1 Section of Hygiene, Department of Life Sciences & Public Health, Rome, Italy",
    ]

    located = _locator([*refs, *spill]).collect_reference_rows()

    assert located["text"].tolist() == refs


def test_keyword_occurrences_inside_reference_titles_do_not_cut():
    refs = [
        "1. Smith J. Author biography methods in historical research. Journal of Tests. 2020.",
        "2. Doe A. Institutional affiliations and citation behavior. Scientometrics. 2021.",
        "3. Roe B. New record linkage methods for cohort studies. Epidemiology. 2022.",
    ]

    located = _locator(refs).collect_reference_rows()

    assert located["text"].tolist() == refs


def test_unnumbered_publisher_note_title_is_not_a_terminal_heading():
    refs = [
        "Smith J. (2020). Institutional change. Journal of Economics 1:1-5.",
        "Doe A. (2021). Regulation and growth. Economic Review 2:6-9.",
        "Publisher's note on economic institutions in transition economies. (2022). Policy 3:10-15.",
    ]

    assert _locator(refs).collect_reference_rows()["text"].tolist() == refs


def test_one_reference_plus_continuation_is_insufficient_boundary_evidence():
    rows = [
        "Smith J. (2020). A long reference title.",
        "Journal of Tests 1:1-5.",
        "Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "Authors and Affiliations",
    ]

    assert _locator(rows).collect_reference_rows()["text"].tolist() == rows


def test_year_bearing_journal_continuation_is_not_a_second_reference_start():
    rows = [
        "Smith J. (2020). A long reference title.",
        "Journal of Tests. Online publication 2021;12:1-5.",
        "Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "Authors and Affiliations",
    ]

    assert _locator(rows).collect_reference_rows()["text"].tolist() == rows


def test_strong_unnumbered_author_bylines_support_terminal_boundary():
    refs = [
        "Locatelli F, Dallapiccola B et al. (2022). Le priorità del piano nazionale della genomica.",
        "Stark Z, Scott RH. (2023). Genomic newborn screening for rare diseases.",
    ]
    tail = [
        "Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "Authors and Affiliations",
    ]

    assert _locator([*refs, *tail]).collect_reference_rows()["text"].tolist() == refs


def test_paper190_full_given_names_and_accents_support_terminal_boundary():
    refs = [
        "Bühler, Charlotte (ed.) 1922. Quellen und Studien zur Jugendkunde. Jena: Fischer.",
        "Bühring, Gerald 2007. Charlotte Bühler oder Der Lebenslauf. Frankfurt: Lang.",
    ]
    tail = [
        "Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "Chair of Romance Cultural Studies Saarland University 66123 Saarbrücken Germany.",
    ]

    assert _locator([*refs, *tail]).collect_reference_rows()["text"].tolist() == refs


@pytest.mark.parametrize(
    "boilerplate",
    [
        "Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "PUBLISHER’S NOTE SPRINGER NATURE REMAINS NEUTRAL WITH REGARD TO JURISDICTIONAL CLAIMS IN PUBLISHED MAPS AND INSTITUTIONAL AFFILIATIONS.",
        "Publisher's note:",
    ],
)
def test_two_genuine_starts_allow_unmistakable_publisher_boundary(boilerplate):
    refs = [
        "9. Locatelli F, Dallapiccola B et al. Le priorità Del piano nazionale Della genomica. 2021.",
        "10. Stark Z, Scott RH. Genomic newborn screening for rare diseases. 2023.",
    ]
    tail = [boilerplate, "Authors and Affiliations", "1 Section of Hygiene, Rome, Italy"]

    assert _locator([*refs, *tail]).collect_reference_rows()["text"].tolist() == refs


def test_reused_locator_clears_stale_terminal_boundary_flags():
    locator = _locator(
        [
            "1. Smith J. First source. 2020. Journal 1:1-5.",
            "2. Doe A. Second source. 2021. Journal 2:6-9.",
            "Authors and Affiliations",
        ]
    )
    locator.collect_reference_rows()
    assert locator.contents.reference_boundary_reason_flags == ["terminal_boundary_trimmed"]

    clean = _locator(
        [
            "1. Smith J. First source. 2020. Journal 1:1-5.",
            "2. Doe A. Second source. 2021. Journal 2:6-9.",
        ]
    ).sentences_df
    locator.sentences_df = clean
    locator.contents.sentences_df = clean

    locator.collect_reference_rows()

    assert locator.contents.reference_boundary_reason_flags == []


@pytest.mark.parametrize(
    "numbering",
    [
        "(1) ",
        "1- ",
        "1 – ",
        "1 ",
        "1.",
    ],
)
def test_non_bracket_entry_numbering_is_recognised(numbering):
    from bibr.extract.ref_locator import _ENTRY_NUMBERING_RE

    assert _ENTRY_NUMBERING_RE.match(f"{numbering}European Commission. A strategy. 2020.")


@pytest.mark.parametrize(
    "text",
    [
        "1.5 million people were affected by the intervention nationwide.",
        "12 patients were enrolled in the trial and followed for two years.",
        "2020 Census results for the metropolitan area were published late.",
        "1234. Something far past a plausible entry number.",
        "(2019) Smith and colleagues reported a null effect.",
    ],
)
def test_entry_numbering_refuses_measurements_and_years(text):
    from bibr.extract.ref_locator import _ENTRY_NUMBERING_RE

    assert not _ENTRY_NUMBERING_RE.match(text)


def test_space_numbered_vancouver_refs_support_terminal_boundary():
    refs = [
        "1 European Commission. A pharmaceutical strategy for Europe. Brussels; 2020.",
        "2 Lindsay S, Cagliostro E. A systematic review. Disabil Rehabil. 2018;40:1.",
    ]
    tail = [
        "Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.",
        "Authors and Affiliations",
    ]

    assert _locator([*refs, *tail]).collect_reference_rows()["text"].tolist() == refs
