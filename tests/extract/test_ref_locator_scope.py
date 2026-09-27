"""RefLocator keeps the paper's own reference list when the page layout hides or doubles it.

Each case is a dev-set scan or native PDF whose list the locator took from the
wrong place or missed (texts shortened from the real rows):

* W2122660901 / W2095946572: the scanned page opens with the end of
  the previous article, whose list sits above this paper's title under the
  same "References" heading.
* W2037590930: "Bibliography.—1." is printed run-in inside the
  closing section, so no section is headed by it.
* W4312442100: the layout model read the hanging-indent list as
  two tables, so the "References" heading heads no rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    PaperTable,
)


def _contents(sections, rows, *, tables=None, layout_hints=None) -> PaperContents:
    """PaperContents from ``(header, type)`` sections and ``(section_id, page, text)`` rows."""
    paper_sections = [PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN)]
    for section_id, (header, section_type) in enumerate(sections, start=1):
        paper_sections.append(PaperSection(section_id, header, 1, 0, section_type))
    sentences = [
        PaperSentence(
            text_id=text_id,
            text=text,
            section_id=section_id,
            paragraph_id=text_id,
            page_number=page,
        )
        for text_id, (section_id, page, text) in enumerate(rows, start=1)
    ]
    return PaperContents(
        sentences=sentences,
        sections=paper_sections,
        tables=tables or [],
        links=[],
        sections_text={s.section_id: "" for s in paper_sections},
        layout_hints=layout_hints or [],
    )


def _texts(ref_df: pd.DataFrame) -> list[str]:
    return ref_df["text"].tolist()


# --- the previous article's list above the title ---------------------------


def test_previous_articles_list_above_the_title_is_dropped():
    """W2122660901: the page-1 rows under "References" belong to the article before."""
    contents = _contents(
        [
            ("References", CanonicalSection.REFERENCES),
            ("Community mental health services in Malaysia", CanonicalSection.TITLE),
            ("Comment", CanonicalSection.UNKNOWN),
        ],
        [
            (0, 1, "The model evolving in Oman may represent an alternative."),
            (1, 1, "WHO/UNICEF (1978) Alma-Ata Declaration 1978. Geneva: WHO."),
            (1, 1, "WIG, N. N. et al (1980) Community reactions to mental disorders."),
            (1, 1, "Psychiatric Bulletin (1992), 16, 648 650"),
            (2, 1, "MOHD. RAZALI SALLEH, Lecturer in Psychiatry"),
            (2, 1, "The need to confine and restrain psychotic patients."),
            (3, 3, "The current ratio is one of the lowest (Tan & Lipton, 1988)."),
            (
                1,
                3,
                "ANDREWS, G. (1991) Psychiatry in Australia. Psychiatric Bulletin, 15, 446-449.",
            ),
            (1, 3, "TAN, E. S. & LIPTON, G. (1988) Mental Health Services. Manila: WHO."),
        ],
    )

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == [
        "ANDREWS, G. (1991) Psychiatry in Australia. Psychiatric Bulletin, 15, 446-449.",
        "TAN, E. S. & LIPTON, G. (1988) Mental Health Services. Manila: WHO.",
    ]
    assert contents.reference_boundary_reason_flags == ["preceding_article_rows_dropped"]


def test_list_only_above_the_title_gives_way_to_the_papers_own_list():
    """W2095946572: the paper's single entry sits under a singular "Reference" heading."""
    contents = _contents(
        [
            ("References", CanonicalSection.REFERENCES),
            ("Remark on strongly additive set functions", CanonicalSection.TITLE),
            ("Reference", CanonicalSection.UNKNOWN),
        ],
        [
            (1, 1, "[1] K. Fan, Fixed-point and minimax theorems, Proc. Nat. Acad. 38 (1952)."),
            (1, 1, "[2] J. L. Kelley, General Topology, Princeton 1955."),
            (2, 1, "by J. Kisyński (Warszawa)"),
            (2, 1, "A set function with values in an abelian group is called additive if"),
            (2, 4, "Thus the proof of our theorem is completed."),
            (3, 4, "[1] P. R. Halmos, Measure Theory, New York 1950."),
        ],
        layout_hints=[("reference_content", 1)],
    )

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == ["[1] P. R. Halmos, Measure Theory, New York 1950."]


def test_rows_above_a_title_on_a_later_page_are_kept():
    """Only the first page can open with another article: a later title changes nothing."""
    contents = _contents(
        [
            ("References", CanonicalSection.REFERENCES),
            ("A Title Printed on Page Two", CanonicalSection.TITLE),
        ],
        [
            (1, 1, "Smith, J. (2019). Choice. Journal of Econ, 1, 1-2."),
            (2, 2, "Body text of the paper."),
        ],
    )

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == ["Smith, J. (2019). Choice. Journal of Econ, 1, 1-2."]
    assert contents.reference_boundary_reason_flags == []


# --- a run-in heading -------------------------------------------------------

_RUN_IN_ROWS = [
    (1, 5, "8. The results obtained will also possess a greater validity."),
    (1, 5, "Bibliography.&mdash;1."),
    (1, 5, "Dreyer, Georges, and Walker, E. W. Ainley : Proc. Roy. Soc., 1914, B., p. 319."),
    (1, 5, "2."),
    (1, 5, "Ostwald, Wo., and Dernoscheck, A. : Z. Chem. Ind. Kolloide, 1910, S. 297."),
    (1, 5, "3."),
    (1, 5, "Arrhenius, Svante:\r\nAnwendung der physikalischen Chemie, Berlin, 1904."),
    (1, 5, "4."),
    (1, 5, "Fraser, J. R., and Elliott. R. H. : Ibid., p. 249."),
    (1, 5, "THE special annual service for members of the\r\nUniversity will be held."),
    (1, 5, "The possibility of cheapening radium is foreshadowed by a scheme."),
]


def test_run_in_bibliography_heading_opens_the_list():
    """W2037590930: the list starts after the heading and ends at the next news item."""
    contents = _contents([("Conclusions.", CanonicalSection.DISCUSSION)], _RUN_IN_ROWS)

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == ["1."] + [text for _, _, text in _RUN_IN_ROWS[2:9]]
    assert contents.reference_boundary_reason_flags == ["run_in_reference_heading"]
    # The section the heading sits in stays a body section.
    assert contents.sections[1].section_type == CanonicalSection.DISCUSSION


@pytest.mark.parametrize(
    "row",
    [
        "References: see the supplementary material.",
        "references. 1. Smith, J. (2019). Choice.",
        "References to earlier work, 1990, 1995 and 2001, are given below.",
    ],
)
def test_prose_about_references_is_not_a_run_in_heading(row):
    rows = [
        (1, 1, row),
        (1, 1, "Smith, J. 1990. A title. Journal 1, 1-2."),
        (1, 1, "Jones, K. 1995. A title. Journal 2, 3-4."),
        (1, 1, "Brown, L. 2001. A title. Journal 3, 5-6."),
    ]
    contents = _contents([("Discussion", CanonicalSection.DISCUSSION)], rows)

    with pytest.raises(ValueError, match="No reference section found"):
        RefLocator(contents).collect_reference_rows()


# --- a list read as tables ---------------------------------------------------


def _table(table_id: int, rows: list[list[str]], *, body_section_id: int, page: int) -> PaperTable:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return PaperTable(
        table_id=table_id,
        df=df,
        tbl_html="<table/>",
        section_id=10 + table_id,
        page_number=page,
        _body_section_id=body_section_id,
    )


def test_reference_list_read_as_tables_becomes_rows():
    """W4312442100: the "References" heading heads no rows; its tables hold the entries."""
    tables = [
        _table(
            4,
            [
                ["Abramson, A. S. 1962:", "The Vowels and Tones of Standard Thai, Bloomington"],
                ["Basbøll, H. 1968:", '"The phoneme system of Advanced Standard Copenhagen"'],
                ["Brink, L. and J. Lund 1974:", "Udtaleforskelle i Danmark, Copenhagen"],
            ],
            body_section_id=2,
            page=20,
        ),
        _table(
            5,
            [
                ["Maack, A. 1949:", '"Die spezifische Lautdauer deutscher Sonanten"'],
                ["Reinholt Petersen, N.", '1974: "The influence of tongue height"'],
            ],
            body_section_id=2,
            page=21,
        ),
    ]
    contents = _contents(
        [
            ("5.3 Short and long vowels", CanonicalSection.RESULTS),
            ("References", CanonicalSection.UNKNOWN),
        ],
        [(1, 19, "Long vowels are longer than short vowels at every height.")],
        tables=tables,
    )

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == [
        "Abramson, A. S. 1962: The Vowels and Tones of Standard Thai, Bloomington",
        'Basbøll, H. 1968: "The phoneme system of Advanced Standard Copenhagen"',
        "Brink, L. and J. Lund 1974: Udtaleforskelle i Danmark, Copenhagen",
        'Maack, A. 1949: "Die spezifische Lautdauer deutscher Sonanten"',
        'Reinholt Petersen, N. 1974: "The influence of tongue height"',
    ]
    assert ref_df["page_number"].tolist() == [20, 20, 20, 21, 21]
    assert not set(ref_df["text_id"]) & {s.text_id for s in contents.sentences}
    assert contents.sections[2].section_type == CanonicalSection.REFERENCES
    assert contents.reference_boundary_reason_flags == ["reference_table_rows"]


def test_data_table_under_an_empty_references_heading_is_not_a_list():
    tables = [
        _table(
            1,
            [
                ["", "short", "long"],
                ["i", "9.5", "14.4"],
                ["e", "10.4", "14.7"],
                ["a", "13.2", "16.9"],
            ],
            body_section_id=2,
            page=20,
        )
    ]
    contents = _contents(
        [("Results", CanonicalSection.RESULTS), ("References", CanonicalSection.REFERENCES)],
        [(1, 19, "Durations are given in table 1.")],
        tables=tables,
    )

    with pytest.raises(ValueError, match="No reference section found"):
        RefLocator(contents).collect_reference_rows()
