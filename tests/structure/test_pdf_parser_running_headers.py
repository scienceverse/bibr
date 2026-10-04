"""Tests for running-header detection in ``PDFParser._mark_running_headers``.

The layout model occasionally tags running headers (an author line at the top
of every page, a banner, a journal name) as ``doc_title`` or
``paragraph_title``. Without de-duplication these become section breaks and
swallow the body — see eyecolor.pdf, where a page-2 author-line ``doc_title``
swallowed the entire Introduction.
"""

import pytest

from bibr.structure.pdf_parser import PDFParser


def _heading(label: str, content: str, y: int = 100) -> dict:
    return {
        "label": label,
        "content": content,
        "bbox_2d": [50, y, 500, y + 30],
    }


def _text(content: str, y: int = 200) -> dict:
    return {
        "label": "text",
        "content": content,
        "bbox_2d": [50, y, 500, y + 50],
    }


def test_subsequent_doc_title_demoted_to_running_header():
    """First doc_title on page 1 is the title; later doc_titles are running headers."""
    pages = [
        [_heading("doc_title", "The Real Paper Title"), _text("Body of page 1.")],
        [_heading("doc_title", "Author A, Author B, Author C"), _text("Page 2 body.")],
        [_heading("doc_title", "Author A, Author B, Author C"), _text("Page 3 body.")],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()
    # Page-1 first doc_title (the real title) stays.
    assert (0, 0) not in parser._running_header_regions
    # Subsequent doc_titles are demoted.
    assert (1, 0) in parser._running_header_regions
    assert (2, 0) in parser._running_header_regions


def test_repeating_paragraph_title_demoted_via_multipage_heuristic():
    """Same heading text on multiple pages = running header."""
    pages = [
        [_heading("paragraph_title", "Journal of Examples")],
        [_heading("paragraph_title", "Journal of Examples")],
        [_heading("paragraph_title", "Methods")],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()
    # "Journal of Examples" appears on 2 pages → demoted on both.
    assert (0, 0) in parser._running_header_regions
    assert (1, 0) in parser._running_header_regions
    # "Methods" appears once → kept.
    assert (2, 0) not in parser._running_header_regions


def test_title_repeated_as_a_running_head_survives_on_page_one():
    """The printed title is not a running header just because it repeats."""
    title = "Serum Electrolyte Levels and Body Mass Index in Adults"
    pages = [
        [_heading("doc_title", title), _text("Body of page 1.")],
        [_heading("paragraph_title", title), _text("Page 2 body.")],
        [_heading("paragraph_title", title), _text("Page 3 body.")],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (0, 0) not in parser._running_header_regions
    assert (1, 0) in parser._running_header_regions
    assert (2, 0) in parser._running_header_regions


def test_repeated_page_one_furniture_stays_demoted():
    """The page-1 reprieve does not extend to banners, mastheads or copyright.

    These repeat on page 1 as well and are never the article title, so keeping
    the first occurrence would only hand front matter a spurious seed.
    """
    for banner in (
        "OPEN ACCESS",
        "Journal of Examples",
        "Copyright 2024 The Authors. All rights reserved.",
    ):
        pages = [
            [_heading("paragraph_title", banner), _text("Body of page 1.")],
            [_heading("paragraph_title", banner), _text("Page 2 body.")],
        ]
        parser = PDFParser(json_result=pages)
        parser._mark_running_headers()

        assert (0, 0) in parser._running_header_regions, banner
        assert (1, 0) in parser._running_header_regions, banner


def test_unique_paragraph_titles_kept():
    """Paragraph titles that appear once each are real section headings."""
    pages = [
        [_heading("doc_title", "The Title")],
        [
            _heading("paragraph_title", "Methods"),
            _heading("paragraph_title", "Results"),
        ],
        [_heading("paragraph_title", "Discussion")],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()
    # No subsequent doc_title and no repeats → nothing demoted.
    assert parser._running_header_regions == set()


def test_repeating_body_text_demoted_as_running_header():
    """A body-text region whose content repeats across pages is a running header.

    Regression: eyecolor.pdf's wide-letter-spaced running header
    ("Sexual imprinting & eye color DeBruine et al. preprint v.2") is tagged
    ``text`` by GLM-OCR, not a heading, so it bypassed the heading-only
    detection and leaked into the References block, corrupting segmentation.
    The banner sits in the top margin band, like real page furniture.
    """
    header = "Sexual imprinting & eye color DeBruine et al. preprint v.2"
    pages = [
        [
            _heading("paragraph_title", "References"),
            _text("Smith, J. (2020). A real reference. Journal of Examples, 1, 1-9."),
        ],
        [
            _text(header, y=40),
            _text("Jones, A. (2019). Another reference here. Journal of Things, 2, 10-20."),
        ],
        [
            _text(header, y=40),
            _text("Brown, B. (2018). A third reference here. Journal of Stuff, 3, 21-30."),
        ],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()
    # The repeated body-text header is demoted on every page it appears.
    assert (1, 0) in parser._running_header_regions
    assert (2, 0) in parser._running_header_regions
    # Unique reference body text is never demoted.
    assert (0, 1) not in parser._running_header_regions
    assert (1, 1) not in parser._running_header_regions
    assert (2, 1) not in parser._running_header_regions


def test_repeating_body_text_with_whitespace_noise_still_demoted():
    """Reflowed running headers (word-per-line / odd spacing) still de-dupe.

    The same header is OCR'd with different internal whitespace on different
    pages; whitespace-normalized comparison must still recognise the repeat.
    The banner sits in the top margin band, like real page furniture.
    """
    pages = [
        [_text("Running  Header\nWith Spacing", y=40)],
        [_text("Running Header With  Spacing", y=40)],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()
    assert (0, 0) in parser._running_header_regions
    assert (1, 0) in parser._running_header_regions


def test_long_repeated_body_block_not_demoted():
    """A long repeated body region is left alone — only short furniture is demoted.

    Guards against demoting a genuine repeated paragraph; furniture lines are
    short, real paragraphs are not.
    """
    long_para = (
        "This is a long body paragraph that, by some quirk of the source "
        "document, happens to appear verbatim on two different pages. It is "
        "far longer than any running header or page footer would ever be, so "
        "the running-header detector must not mistake it for page furniture."
    )
    pages = [[_text(long_para)], [_text(long_para)]]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()
    assert parser._running_header_regions == set()


def test_repeated_body_header_diverted_from_reference_content():
    """End-to-end: the repeated body-text header must not reach section content.

    The banner sits in the top margin band, like real page furniture.
    """
    header = "Sexual imprinting & eye color DeBruine et al. preprint v.2"
    pages = [
        [
            _heading("paragraph_title", "References"),
            _text("Smith, J. (2020). A real reference. Journal of Examples, 1, 1-9."),
        ],
        [
            _text(header, y=40),
            _text("Jones, A. (2019). Another reference here. Journal of Things, 2, 10-20."),
        ],
        [
            _text(header, y=40),
            _text("Brown, B. (2018). A third reference here. Journal of Stuff, 3, 21-30."),
        ],
    ]
    parser = PDFParser(json_result=pages)
    parser.parse()
    # Diverted to structural metadata, not body content.
    assert any("Sexual imprinting" in h for h in parser.detected_headers)
    # Absent from the deferred body text that becomes reference sentences.
    deferred = " ".join(t[0] for t in parser._deferred_texts)
    assert "Sexual imprinting" not in deferred


def test_full_parse_eyecolor_style_running_header_does_not_split_body():
    """End-to-end: a page-2 author-line doc_title should not become a section."""
    pages = [
        [
            _heading("doc_title", "Positive sexual imprinting for human eye color"),
            _text("DeBruine et al. preprint v.2"),
        ],
        [
            _heading("doc_title", "Lisa M. DeBruine, Benedict C. Jones, Anthony C. Little"),
            _text("Human romantic partners tend to have similar physical traits."),
            _text("This is the introduction body."),
        ],
        [_heading("paragraph_title", "METHODS"), _text("Method body sentence.")],
    ]
    parser = PDFParser(json_result=pages)
    contents = parser.parse()
    headers = [s.header for s in contents.sections]
    # The real title and METHODS should still appear.
    assert "Positive sexual imprinting for human eye color" in headers
    assert "METHODS" in headers
    # The author-line running header must NOT have become a section.
    assert "Lisa M. DeBruine, Benedict C. Jones, Anthony C. Little" not in headers


def test_repeated_mid_page_section_headings_are_not_running_headers():
    """Multi-study papers legitimately repeat Method/Results per study.

    Repetition alone used to demote them, deleting the headings and folding
    their body text into the preceding Study section — so the export had no
    METHODS and no RESULTS at all.
    """
    pages = [
        [_heading("paragraph_title", "Study 1", y=380), _text("Study 1 intro.", y=430)],
        [_heading("paragraph_title", "Method", y=300), _text("Study 1 method.", y=350)],
        [_heading("paragraph_title", "Results", y=300), _text("Study 1 results.", y=350)],
        [_heading("paragraph_title", "Study 2", y=380), _text("Study 2 intro.", y=430)],
        [_heading("paragraph_title", "Method", y=300), _text("Study 2 method.", y=350)],
        [_heading("paragraph_title", "Results", y=300), _text("Study 2 results.", y=350)],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert not parser._running_header_regions


def test_repeated_heading_in_the_margin_band_is_still_demoted():
    """The geometry gate must not stop demoting genuine page furniture."""
    pages = [
        [_heading("paragraph_title", "Journal of Examples", y=20), _text("Page 1 body.")],
        [_heading("paragraph_title", "Journal of Examples", y=20), _text("Page 2 body.")],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (0, 0) in parser._running_header_regions
    assert (1, 0) in parser._running_header_regions


def test_footer_band_repeats_are_demoted():
    pages = [
        [_text("Page 1 body.", y=300), _heading("paragraph_title", "Preprint 2026", y=940)],
        [_text("Page 2 body.", y=300), _heading("paragraph_title", "Preprint 2026", y=940)],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    # The page-1 occurrence keeps its existing title reprieve; the repeat in
    # the footer band is demoted.
    assert (1, 1) in parser._running_header_regions


def test_repeated_mid_column_body_sentence_is_not_a_running_header():
    """A short body sentence repeated mid-column on two pages stays in the body.

    Two-study papers legitimately repeat boilerplate (analysis notes, table
    notes) mid-column; repetition alone must not divert it to headers.
    """
    repeated = "All analyses were conducted in R (R Core Team, 2021)."
    pages = [
        [
            _heading("paragraph_title", "Method", y=300),
            _text(repeated, y=500),
            _text("Participants were 120 students.", y=560),
        ],
        [
            _heading("paragraph_title", "Method", y=300),
            _text(repeated, y=500),
            _text("Participants were 200 adults.", y=560),
        ],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (0, 1) not in parser._running_header_regions
    assert (1, 1) not in parser._running_header_regions


def test_repeated_mid_column_body_sentence_survives_full_parse():
    """End-to-end: the repeated mid-column sentence reaches the body text."""
    repeated = "All analyses were conducted in R (R Core Team, 2021)."
    pages = [
        [
            _heading("paragraph_title", "Method", y=300),
            _text(repeated, y=500),
            _text("Participants were 120 students.", y=560),
        ],
        [
            _heading("paragraph_title", "Method", y=300),
            _text(repeated, y=500),
            _text("Participants were 200 adults.", y=560),
        ],
    ]
    parser = PDFParser(json_result=pages)
    parser.parse()
    deferred = " ".join(t[0] for t in parser._deferred_texts)

    assert deferred.count(repeated) == 2
    assert repeated not in parser.detected_headers


def test_repeated_legend_rows_printed_together_stay_demoted():
    """A block of repeated rows (a chart legend reprinted with each float) is
    float furniture, not body text, wherever it sits on the page."""
    pages = [
        [
            _text("Quadro 4 lists the green-area laws.", y=300),
            _text("com interface", y=541),
            _text("□ sem interface", y=562),
            _text("Na composição das áreas verdes, os impactos são frequentes.", y=600),
        ],
        [
            _text("Quadro 5 lists the morphology laws.", y=300),
            _text("com interface", y=428),
            _text("□ sem interface", y=449),
            _text("Aos aspectos morfológicos, Lamas (2014) associa a forma.", y=500),
        ],
    ]
    parser = PDFParser(json_result=pages)
    parser.parse()

    assert {(0, 1), (0, 2), (1, 1), (1, 2)} <= parser._running_header_regions
    assert not any("com interface" in t[0] for t in parser._deferred_texts)


def _sidebar_pages(sidebar_heading: str) -> list[list[dict]]:
    """A magazine article: a boxed sidebar headed by a mid-page ``doc_title`` on page 3."""
    return [
        [
            _heading("doc_title", "Nudge Your Customers Toward Better Choices", y=80),
            _text("Defaults are the options a customer gets without acting.", y=200),
        ],
        [
            _heading("paragraph_title", "Mass Defaults", y=300),
            _text("Mass defaults apply to every customer alike.", y=350),
        ],
        [
            _text("Most firms set them once and never revisit them.", y=300),
            _heading("doc_title", sidebar_heading, y=538),
            _text("Sometimes the best default is no default at all.", y=600),
        ],
    ]


def test_mid_page_doc_title_on_a_later_page_is_a_sidebar_heading():
    """Every later-page doc_title used to be demoted, which dropped the sidebar
    heading and merged the sidebar into the section around it."""
    sidebar = "When No Default Is Your Best Option"
    parser = PDFParser(json_result=_sidebar_pages(sidebar))
    parser._mark_running_headers()

    assert (2, 1) not in parser._running_header_regions

    full = PDFParser(json_result=_sidebar_pages(sidebar))
    contents = full.parse()
    sidebar_section = next(s for s in contents.sections if s.header == sidebar)
    by_section = {text: section_id for text, _page, section_id, *_ in full._deferred_texts}
    assert by_section["Sometimes the best default is no default at all."] == (
        sidebar_section.section_id
    )
    assert by_section["Most firms set them once and never revisit them."] != (
        sidebar_section.section_id
    )


def test_later_page_doc_title_in_the_margin_band_stays_demoted():
    pages = _sidebar_pages("Harvard Business Review")
    pages[2][1] = _heading("doc_title", "Harvard Business Review", y=20)
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (2, 1) in parser._running_header_regions


def test_title_behind_a_cover_sheet_stays_demoted_mid_page():
    """A submission cover sheet prints the title; the article page repeats it mid-page."""
    pages = [
        [
            _heading("doc_title", "Supplier Opportunism in Buyer-Supplier NPD", y=150),
            _text("Manuscript ID DS-2026-0001. Manuscript type: Original Article.", y=300),
        ],
        [
            _heading("doc_title", "SUPPLIER OPPORTUNISM IN BUYER-SUPPLIER NPD:", y=420),
            _text("Collaborating with a supplier exposes the buyer to opportunism.", y=520),
        ],
    ]
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (1, 0) in parser._running_header_regions


def test_mid_page_doc_title_repeated_on_later_pages_stays_demoted():
    pages = _sidebar_pages("Author A, Author B, Author C")
    pages.append(
        [
            _heading("doc_title", "Author A, Author B, Author C", y=450),
            _text("Page 4 body.", y=520),
        ]
    )
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert {(2, 1), (3, 0)} <= parser._running_header_regions


def test_mid_page_copyright_doc_title_on_a_later_page_stays_demoted():
    pages = _sidebar_pages(
        "Copyright 2026 Harvard Business School Publishing. All rights reserved."
    )
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (2, 1) in parser._running_header_regions


_TRANSLATED_TITLE = "Mémoire de l’humidité du sol dans les pâturages d’altitude"


def _two_language_pages(record_rows: list[dict]) -> list[list[dict]]:
    """An article printed with its title and front matter again in a second language.

    Page 2 ends the body, then a mid-page ``doc_title`` heads *record_rows*.
    """
    return [
        [
            _heading("doc_title", "Soil Moisture Memory in Upland Pastures", y=150),
            _text("Ada Field and Ben Moor", y=200),
            {
                "label": "abstract",
                "content": "Pastures keep the moisture of a wet spring well into summer.",
                "bbox_2d": [50, 260, 500, 360],
            },
            _text("Upland pastures dry out late in the season.", y=420),
        ],
        [
            _text("We thank the farmers who let us sample their fields.", y=150),
            _heading("doc_title", _TRANSLATED_TITLE, y=420),
            *record_rows,
        ],
    ]


@pytest.mark.parametrize(
    "record_rows",
    [
        [
            _heading("paragraph_title", "Résumé", y=470),
            _text("Les pâturages gardent l’humidité d’un printemps humide.", y=500),
        ],
        [
            _text("Les pâturages gardent l’humidité d’un printemps humide.", y=470),
            _text("Mots-clés : sol · pâturage · été", y=560),
        ],
        [
            {
                "label": "abstract",
                "content": "Les pâturages gardent l’humidité d’un printemps humide.",
                "bbox_2d": [50, 470, 500, 560],
            }
        ],
        [
            _heading("paragraph_title", "A. Field", y=470),
            _text("Upland Soil Institute, Northtown. E-mail: a.field@example.org", y=500),
        ],
    ],
    ids=["abstract-heading", "keywords-lead-in", "abstract-region", "byline-e-mail"],
)
def test_later_page_title_heading_its_own_record_stays_demoted(record_rows):
    """Kept as a sidebar heading, the translated title opened a second title
    section and front matter could not choose between the two records."""
    parser = PDFParser(json_result=_two_language_pages(record_rows))
    parser._mark_running_headers()

    assert (1, 1) in parser._running_header_regions

    contents = PDFParser(json_result=_two_language_pages(record_rows)).parse()
    assert not any(s.header == _TRANSLATED_TITLE for s in contents.sections)


def test_sidebar_prose_that_mentions_an_abstract_keeps_its_heading():
    pages = _sidebar_pages("When No Default Is Your Best Option")
    pages[2][2] = _text("Abstract defaults rarely help a customer decide.", y=600)
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert (2, 1) not in parser._running_header_regions


# A preprint server banner: the rights line plus the licence and DOI lines, one
# region of about 340 characters at the top of every page.
_PREPRINT_BANNER = (
    "Example preprint doi: https://doi.org/10.0000/2026.01.01.000001; this version posted "
    "January 1, 2026. The copyright holder for this preprint (which was not certified by "
    "peer review) is the author/funder, who has granted the server a license to display "
    "the preprint in perpetuity. It is made available under a CC-BY 4.0 International license."
)


def _banner(y1: int = 0, y2: int = 36) -> dict:
    return {"label": "text", "content": _PREPRINT_BANNER, "bbox_2d": [50, y1, 950, y2]}


def _banner_pages(banner_pages: set[int], *, y1: int = 0, y2: int = 36) -> list[list[dict]]:
    pages = [
        [_heading("doc_title", "A Cohort Study of Venous Disease", y=120)],
        [_heading("paragraph_title", "Data availability", y=300)],
        [_heading("paragraph_title", "References", y=300)],
    ]
    bodies = [
        "Participants were recruited from three hospitals.",
        "Data are available from the authors on request.",
        "Smith, J. (2020). A real reference. Journal of Examples, 1, 1-9.",
    ]
    for page_idx, page in enumerate(pages):
        if page_idx in banner_pages:
            page.insert(0, _banner(y1, y2))
        page.append(_text(bodies[page_idx], y=400))
    return pages


def test_long_preprint_banner_in_the_margin_band_is_demoted():
    """The banner is longer than the 200-character furniture cap, so it was
    never counted and its ``text``-labelled copies landed in the body."""
    assert len(_PREPRINT_BANNER) > 300
    parser = PDFParser(json_result=_banner_pages({1, 2}))
    parser._mark_running_headers()

    assert {(1, 0), (2, 0)} <= parser._running_header_regions

    full = PDFParser(json_result=_banner_pages({1, 2}))
    full.parse()
    deferred = " ".join(t[0] for t in full._deferred_texts)
    assert "copyright holder" not in deferred
    assert "Data are available from the authors on request." in deferred


def test_long_repeated_region_in_mid_page_is_kept():
    parser = PDFParser(json_result=_banner_pages({1, 2}, y1=400, y2=436))
    parser._mark_running_headers()

    assert not {(1, 0), (2, 0)} & parser._running_header_regions


def test_long_repeated_paragraph_reaching_into_the_band_is_kept():
    """A manuscript that prints its body twice: the same paragraph starts at the
    top of two pages, inside the band, and runs down the page."""
    parser = PDFParser(json_result=_banner_pages({1, 2}, y1=91, y2=437))
    parser._mark_running_headers()

    assert not {(1, 0), (2, 0)} & parser._running_header_regions


def test_long_band_region_on_one_page_only_is_kept():
    parser = PDFParser(json_result=_banner_pages({1}))
    parser._mark_running_headers()

    assert (1, 0) not in parser._running_header_regions


def test_long_repeated_region_without_geometry_is_kept():
    """A missing bbox falls back to demotion only for short rows."""
    pages = _banner_pages({1, 2})
    for page in (pages[1], pages[2]):
        page[0] = {"label": "text", "content": _PREPRINT_BANNER}
    parser = PDFParser(json_result=pages)
    parser._mark_running_headers()

    assert not {(1, 0), (2, 0)} & parser._running_header_regions
