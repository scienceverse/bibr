"""The PDF's own DOI evidence: front-page text layer, links, Info and XMP."""

from __future__ import annotations

import pytest

from bibr.extract.pdf_doi_evidence import _to_layout_point, is_pdf, read_pdf_doi_evidence
from tests.extract.pdf_builder import Page, TextRun, build_pdf, xmp_packet


def _paper_pdf(**kwargs) -> bytes:
    return build_pdf(
        [
            Page(
                runs=[
                    TextRun(72, 700, "A Study of Examples"),
                    TextRun(300, 740, "Example Journal 11 (2026) 1-12"),
                    TextRun(72, 60, "https://doi.org/10.1234/abc.5"),
                    TextRun(72, 48, "1234-5678/© 2026 The Authors."),
                    # A repository banner printed up the right margin.
                    TextRun(
                        580,
                        200,
                        "Example J: first published as 10.1234/banner.7 on 1 May 1999.",
                        size=7,
                        angle=90,
                    ),
                ],
                links=[
                    ((70, 57, 230, 70), "https://doi.org/10.1234/abc.5"),
                    ((298, 737, 460, 752), "https://doi.org/10.1234/abc.5"),
                    ((72, 690, 200, 710), "https://example.org/about"),
                ],
            ),
            Page(runs=[TextRun(72, 700, "Body text on page two.")]),
            Page(runs=[TextRun(72, 700, "doi: 10.9999/page.three")]),
        ],
        **kwargs,
    )


def test_reads_the_front_pages_doi_lines_with_the_line_after_each():
    evidence = read_pdf_doi_evidence(_paper_pdf(), (1, 2))

    assert evidence.pages == (1, 2)
    by_text = {line.text: line for line in evidence.lines}
    footer = by_text["https://doi.org/10.1234/abc.5"]
    assert footer.page == 1
    assert footer.next_text == "1234-5678/© 2026 The Authors."
    # Near the page foot: y grows downwards in the layout frame.
    assert all(center[1] > 900 for center in footer.centers if center is not None)
    # Page 3 is not a front page.
    assert not any("page.three" in line.text for line in evidence.lines)


def test_reads_a_rotated_margin_banner_as_one_line():
    evidence = read_pdf_doi_evidence(_paper_pdf(), (1,))

    [banner] = [line for line in evidence.lines if "banner" in line.text]
    assert banner.text == "Example J: first published as 10.1234/banner.7 on 1 May 1999."
    centers = [center for center in banner.centers if center is not None]
    assert centers
    # In the right margin, running up the page.
    assert min(center[0] for center in centers) > 900
    assert centers[0][1] > centers[-1][1]


def test_reads_doi_links_with_the_text_printed_around_them():
    evidence = read_pdf_doi_evidence(_paper_pdf(), (1,))

    assert [(link.page, link.doi) for link in evidence.links] == [
        (1, "10.1234/abc.5"),
        (1, "10.1234/abc.5"),
    ]
    printed, citation_line = evidence.links
    assert "https://doi.org/10.1234/abc.5" in printed.printed_text
    assert citation_line.printed_text == "Example Journal 11 (2026) 1-12"
    x1, y1, x2, y2 = printed.rect
    assert x1 < x2 and y1 < y2 and y2 > 900


def test_reads_the_document_information_dois_and_no_xmp():
    # An XMP packet anywhere in the file may belong to an embedded object that
    # names another article; only the document-information dictionary is read.
    pdf = _paper_pdf(
        info={
            "Subject": "Example Journal 11 (2026) 1-12. doi:10.1234/abc.5",
            "doi": "10.1234/abc.5",
            "Title": "A Study of Examples",
        },
        xmp=xmp_packet(prism_doi="10.5555/embedded.9"),
    )

    evidence = read_pdf_doi_evidence(pdf, (1,))

    assert [(item.source, item.key, item.value) for item in evidence.metadata] == [
        ("pdf_info", "Subject", "Example Journal 11 (2026) 1-12. doi:10.1234/abc.5"),
        ("pdf_info", "doi", "10.1234/abc.5"),
    ]


@pytest.mark.parametrize("render_mode", [3, 7])
def test_text_that_paints_nothing_is_not_read(render_mode):
    # A scan's hidden OCR layer (3) and clip-only text (7) are not printed.
    pdf = build_pdf(
        [
            Page(
                runs=[
                    TextRun(72, 700, "A Study of Examples"),
                    TextRun(72, 60, "https://doi.org/10.1234/shown.5"),
                    TextRun(
                        580, 400, "doi:10.1234/hidden.7", size=6, angle=90, render_mode=render_mode
                    ),
                ],
                links=[((70, 57, 230, 70), "https://doi.org/10.1234/shown.5")],
            )
        ]
    )

    evidence = read_pdf_doi_evidence(pdf, (1,))

    assert [line.text for line in evidence.lines] == ["https://doi.org/10.1234/shown.5"]
    assert "hidden" not in evidence.links[0].printed_text


def test_doi_links_per_page_are_capped():
    links = [
        ((10, 10 + i % 40, 60, 12 + i % 40), f"https://doi.org/10.1234/ref.{i}") for i in range(230)
    ]
    pdf = build_pdf([Page(runs=[TextRun(72, 700, "A Study of Examples")], links=links)])

    evidence = read_pdf_doi_evidence(pdf, (1,))

    assert len(evidence.links) == 200


def test_pages_the_pdf_lacks_are_skipped():
    evidence = read_pdf_doi_evidence(_paper_pdf(), (2, 9))

    assert evidence.pages == (2,)
    assert evidence.lines == () and evidence.links == ()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_text_layer_points_map_back_onto_the_layout_boxes(rotation):
    from bibr.ocr.native_text import _normalized_bbox_to_pdf_points

    crop_box = (10.0, 50.0, 610.0, 850.0)
    box = [120.0, 300.0, 380.0, 340.0]
    left, bottom, right, top = _normalized_bbox_to_pdf_points(box, crop_box, rotation)

    corners = {
        tuple(round(v, 6) for v in _to_layout_point(x, y, crop_box, rotation))
        for x in (left, right)
        for y in (bottom, top)
    }

    assert corners == {(120.0, 300.0), (120.0, 340.0), (380.0, 300.0), (380.0, 340.0)}


def test_is_pdf_checks_the_header():
    assert is_pdf(b"%PDF-1.7\n...")
    assert not is_pdf(b"PK\x03\x04 docx bytes")
    assert not is_pdf(None)
