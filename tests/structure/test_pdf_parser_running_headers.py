"""Tests for running-header detection in ``PDFParser._mark_running_headers``.

The layout model occasionally tags running headers (an author line at the top
of every page, a banner, a journal name) as ``doc_title`` or
``paragraph_title``. Without de-duplication these become section breaks and
swallow the body — see eyecolor.pdf, where a page-2 author-line ``doc_title``
swallowed the entire Introduction.
"""

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
    """
    header = "Sexual imprinting & eye color DeBruine et al. preprint v.2"
    pages = [
        [
            _heading("paragraph_title", "References"),
            _text("Smith, J. (2020). A real reference. Journal of Examples, 1, 1-9."),
        ],
        [
            _text(header),
            _text("Jones, A. (2019). Another reference here. Journal of Things, 2, 10-20."),
        ],
        [
            _text(header),
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
    """
    pages = [
        [_text("Running  Header\nWith Spacing")],
        [_text("Running Header With  Spacing")],
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
    """End-to-end: the repeated body-text header must not reach section content."""
    header = "Sexual imprinting & eye color DeBruine et al. preprint v.2"
    pages = [
        [
            _heading("paragraph_title", "References"),
            _text("Smith, J. (2020). A real reference. Journal of Examples, 1, 1-9."),
        ],
        [
            _text(header),
            _text("Jones, A. (2019). Another reference here. Journal of Things, 2, 10-20."),
        ],
        [
            _text(header),
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
