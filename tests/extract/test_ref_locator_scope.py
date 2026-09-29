"""RefLocator keeps the paper's own reference list when the page layout hides or doubles it.

Each case is a dev-set scan or native PDF whose list the locator took from the
wrong place or missed (texts shortened from the real rows):

* Two scanned papers: the first page opens with the end of
  the previous article, whose list sits above this paper's title under the
  same "References" heading.
* A scanned journal note: "Bibliography.—1." is printed run-in inside the
  closing section, so no section is headed by it.
* A native PDF: the layout model read the hanging-indent list as
  two tables, so the "References" heading heads no rows.
* A scanned French report: "BIBLIOGRAPHIE." is printed on a line of its own
  inside the closing section, and the OCR misread it.
* A scanned newsletter: the next item is printed right below a numbered list,
  under the same "References" heading.
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
    """The first scan: the page-1 rows under "References" belong to the article before."""
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
    """The second scan: the paper's single entry sits under a singular "Reference" heading."""
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
    """The journal note: the list starts after the heading and ends at the next news item."""
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
    """The native PDF: the "References" heading heads no rows; its tables hold the entries."""
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


# --- a heading row the OCR misread --------------------------------------------

_MISREAD_ROWS = [
    (1, 15, "La P.S. est hostile à la nationalisation du secteur, dans sa sphère."),
    (1, 15, "BIBLIOGRfU'HIE."),
    (1, 15, "Rapport sur les Assurances, présenté au gouvernement, Bruxelles 1940."),
    (1, 15, "L'Assurance en Belgique, quelques problèmes actuels, Jacques Basyn, 1947."),
    (1, 15, "Rapport de la Commission des Assurances Privées, Bruxelles 1958."),
    (1, 15, "Le Recueil Financier, 65ème année, tome II, Edition E. Bruylant, Bruxelles 1958."),
    (1, 16, "Les bénéfices des compagnies d'assurance sur la Vie en Belgique."),
    (1, 16, "D'intéressantes considérations sur les bénéfices ont paru, en mars 1955."),
]


def _misread_contents(heading: str) -> PaperContents:
    rows = [_MISREAD_ROWS[0], (1, 15, heading), *_MISREAD_ROWS[2:]]
    return _contents([("En résumé", CanonicalSection.UNKNOWN)], rows)


@pytest.mark.parametrize("heading", ["BIBLIOGRfU'HIE.", "BIBLIOGRAPHI€.", "BIBLIOGRAPHIE."])
def test_misread_bibliography_heading_row_opens_the_list(heading):
    """The French report: the list starts after the heading and ends at the next annex title."""
    contents = _misread_contents(heading)

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == [text for _, _, text in _MISREAD_ROWS[2:6]]
    assert contents.reference_boundary_reason_flags == ["misread_reference_heading"]
    assert contents.sections[1].section_type == CanonicalSection.UNKNOWN


@pytest.mark.parametrize(
    "heading",
    [
        "BIOGRAPHIE.",  # three edits from BIBLIOGRAPHIE
        "PREFERENCES",  # one edit from REFERENCES, but another first letter
        "BIBLIOGRAPHIQUE.",  # two edits from BIBLIOGRAPHIE, but two letters longer
        "Bibliogrfu'hie.",  # not set in capitals
        "BIBLIOGRfU'HIE ET SOURCES.",  # not a word on its own
    ],
)
def test_other_heading_rows_do_not_open_a_list(heading):
    contents = _misread_contents(heading)

    with pytest.raises(ValueError, match="No reference section found"):
        RefLocator(contents).collect_reference_rows()


def test_misread_heading_needs_three_dated_rows_after_it():
    rows = _MISREAD_ROWS[:4] + _MISREAD_ROWS[6:]
    contents = _contents([("En résumé", CanonicalSection.UNKNOWN)], rows)

    with pytest.raises(ValueError, match="No reference section found"):
        RefLocator(contents).collect_reference_rows()


def test_last_misread_heading_with_a_list_opens_it():
    """An earlier annex under a misread heading of its own: the list closest to the end is taken."""
    annex = [
        (1, 12, "BIBLIOGRAPHI€."),
        (1, 12, "Statistiques des assurances, Bruxelles 1950."),
        (1, 12, "Annuaire des assurances, Bruxelles 1951."),
        (1, 12, "Rapport annuel de l'Office de Contrôle, Bruxelles 1952."),
        (1, 13, "La seconde partie traite des compagnies étrangères."),
    ]
    contents = _contents([("En résumé", CanonicalSection.UNKNOWN)], annex + _MISREAD_ROWS)

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == [text for _, _, text in _MISREAD_ROWS[2:6]]


# --- the next item below a numbered list ---------------------------------------

_NEWSLETTER_LIST = [
    "1. Baars AJ et al (1992) Lead intoxication in cattle: a case report.",
    "2. Lund LJ and Brown JRH (1989) Lead Poisoning from contaminated feed Vet Rec 125;536.",
    "3. Report of the Chief Veterinary Officer Annual Report 1989 HMSO.",
    "5. Sharma RP, Street JC, Shupe JL and Bourcier DR (1982) J Dairy Sci 65;972.",
    "4. MAFF News Releases November 1989 - February 1990 MAFF London.",
    "6. In Mineral Tolerance of Domestic Animals 1990 Lead Nat Acad of Sciences.",
    "7. Hathaway SC (1993) Risk Assessment procedures used by the Codex. Food Control 4;189-201.",
]
_NEWS_ITEM = [
    "November 13, 1998 Consent Decree Entered in Animal Drug GMP Case",
    "On October 20, 1998, the U.S. District Court incorporated into an order a Consent Decree.",
    "Under the Consent Decree, the firm and its president are permanently restrained.",
]


def _regioned_contents(rows: list[tuple]) -> PaperContents:
    """A "References" section of ``(text, region label, region index[, page])`` rows.

    A row is on page 4 unless it names its page; a row labelled None has no layout region.
    """
    pages = [row[3] if len(row) > 3 else 4 for row in rows]
    contents = _contents(
        [("References", CanonicalSection.REFERENCES)],
        [(1, page, row[0]) for row, page in zip(rows, pages, strict=True)],
    )
    for sentence, row, page in zip(contents.sentences, rows, pages, strict=True):
        if row[1] is not None:
            sentence.region_meta = {
                "region_type": row[1],
                "region_page": page,
                "region_index": row[2],
            }
    return contents


def _newsletter_rows(
    news_item: list[str],
    news_label: str | None = "text",
    *,
    news_index: int = 10,
    list_label: str | None = "reference_content",
    last_label: str | None = None,
) -> list[tuple]:
    """The numbered list (entry 7 in region 7) and the news item below it (in *news_index*)."""
    rows: list[tuple] = [
        (text, list_label, index) for index, text in enumerate(_NEWSLETTER_LIST, 1)
    ]
    if last_label is not None:
        rows[-1] = (rows[-1][0], last_label, rows[-1][2])
    return rows + [(text, news_label, news_index) for text in news_item]


def test_next_item_below_a_complete_numbered_list_is_cut():
    """The newsletter: the list prints 1 to 7 (4 and 5 swapped) and ends where the news item starts."""
    contents = _regioned_contents(_newsletter_rows(_NEWS_ITEM))

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == _NEWSLETTER_LIST
    assert contents.reference_boundary_reason_flags == ["numbered_list_end_trimmed"]


@pytest.mark.parametrize(
    "rows",
    [
        # The last entry's tail after a column break starts in lower case.
        _newsletter_rows(["and advisory bodies. Food Control 4;189-201.", "Received 2 May 1998."]),
        # One row after the list: it may still be the last entry's tail.
        _newsletter_rows(_NEWS_ITEM[:1]),
        # The layout read the rows below as part of the list.
        _newsletter_rows(_NEWS_ITEM, news_label="reference_content"),
        # The printed numbers skip one: the list may go on.
        [row for row in _newsletter_rows(_NEWS_ITEM) if not row[0].startswith("6.")],
        # Two numbered entries are too few to call the list complete.
        [row for row in _newsletter_rows(_NEWS_ITEM) if row[0][:2] in {"1.", "2."} or row[2] == 10],
        # The layout did not read the last entry as part of the list.
        _newsletter_rows(_NEWS_ITEM, last_label="text"),
        # The rows below sit in the last entry's own region.
        _newsletter_rows(_NEWS_ITEM, news_index=7),
        # No layout region is known for the rows below.
        _newsletter_rows(_NEWS_ITEM, news_label=None),
        # No layout region is known for the list.
        _newsletter_rows(_NEWS_ITEM, list_label=None),
    ],
)
def test_rows_below_a_numbered_list_are_kept_without_a_clear_end(rows):
    contents = _regioned_contents(rows)

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == [row[0] for row in rows]
    assert "numbered_list_end_trimmed" not in contents.reference_boundary_reason_flags


def _broken_last_entry_rows(head: str, tail: str, *, tail_page: int) -> list[tuple]:
    """Entries 1 to 6, then entry 7 whose tail and a copyright line sit in regions read as text."""
    rows: list[tuple] = [
        (text, "reference_content", index) for index, text in enumerate(_NEWSLETTER_LIST[:6], 1)
    ]
    return rows + [
        (head, "reference_content", 7),
        (tail, "text", 10, tail_page),
        ("Copyright 1998 by the Association. All rights reserved.", "text", 11, tail_page),
    ]


@pytest.mark.parametrize(
    "rows",
    [
        # The last entry breaks off mid-title and its tail goes on over the page in a capital.
        _broken_last_entry_rows(
            "7. Hathaway SC (1993) Risk Assessment procedures used by the Codex",
            "Alimentarius Commission and its advisory bodies. Food Control 4;189-201.",
            tail_page=5,
        ),
        # The same break at a column on the same page.
        _broken_last_entry_rows(
            "7. Hathaway SC (1993) Risk Assessment procedures used by the Codex",
            "Alimentarius Commission and its advisory bodies. Food Control 4;189-201.",
            tail_page=4,
        ),
        # The title ends on a full stop and the journal is printed over the page.
        _broken_last_entry_rows(
            "7. Hathaway SC (1993) Risk Assessment procedures used by the Codex Commission.",
            "Food Control 4;189-201.",
            tail_page=5,
        ),
    ],
)
def test_last_entrys_tail_opening_in_a_capital_is_kept(rows):
    """A Title Case title or a journal name after the break is the entry's own tail, not what follows."""
    contents = _regioned_contents(rows)

    ref_df = RefLocator(contents).collect_reference_rows()

    assert _texts(ref_df) == [row[0] for row in rows]
    assert "numbered_list_end_trimmed" not in contents.reference_boundary_reason_flags
