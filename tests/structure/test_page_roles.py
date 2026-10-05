"""The page-role writer (``bibr.structure.page_roles``) on synthetic pages.

Its evidence, strongest first: the layer's artifacts and the furniture
strip's objects, the layout labels, recurrence across pages, the page labels,
and runs of page numbers. Without the layer the writer decides from the
regions alone; a role is never decided from a margin band alone.
"""

from __future__ import annotations

from bibr.document import ids, views
from bibr.document.harvest import build_document_layer
from bibr.document.rebuild import attach_blocks
from bibr.ocr.types import OcrRegionResult
from bibr.structure.page_roles import (
    LINE_NUMBER,
    PAGE_NUMBER,
    PAGE_ROLES_RULE,
    RUNNING_FOOTER,
    RUNNING_HEADER,
    WATERMARK,
    write_page_roles,
)
from bibr.structure.pdf_parser import PDFParser
from tests.document._pdfs import PageSpec, build_pdf, marked, text
from tests.ocr.test_line_number_column import _reference_page
from tests.ocr.test_watermark_text import _REVIEW_WATERMARK, _body

HEADER = "Journal of Examples 12 (2024) 1-9"
BODY = "The body of the page reads on below its running head."


def _region(content: str, bbox_2d, label: str = "text") -> dict:
    return {"label": label, "native_label": label, "content": content, "bbox_2d": list(bbox_2d)}


def _layer(pdf_bytes: bytes, pages: list[list[dict]]):
    """The layer of *pdf_bytes* with *pages* attached as its blocks."""
    layer = build_document_layer(pdf_bytes, range(len(pages)), budget=None)
    attach_blocks(
        layer,
        [
            [
                OcrRegionResult.from_dict({**region, "index": index})
                for index, region in enumerate(page)
            ]
            for page in pages
        ],
    )
    return layer


def _layout_box(layer, page: int, pdf_box) -> list[float]:
    box = views.to_layout_bbox(layer.page(page), pdf_box)
    assert box is not None
    return list(box)


def _line_position(page_lines, wanted: str) -> int:
    return next(i for i, line in enumerate(page_lines) if line["text"] == wanted)


# --- Recurrence, from the regions alone ---------------------------------------------


def test_a_band_head_printed_on_every_page_is_a_running_header():
    pages = [
        [_region(HEADER, (60, 20, 600, 40)), _region(BODY, (60, 200, 940, 800))],
        [_region(HEADER, (60, 22, 600, 42)), _region(BODY, (60, 200, 940, 800))],
        [_region(HEADER, (60, 21, 600, 41)), _region(BODY, (60, 200, 940, 800))],
    ]

    roles = write_page_roles(pages)

    tags = [roles.block(page, 0) for page in range(3)]
    assert all(tag is not None and tag.role == RUNNING_HEADER for tag in tags)
    first = tags[0]
    assert first.target == "p0.r0"
    assert first.decided.version == PAGE_ROLES_RULE
    assert first.decided.calibrated is False
    # The evidence is the head's other occurrences.
    assert set(first.decided.evidence) == {"p1.r0", "p2.r0"}
    assert all(roles.block(page, 1) is None for page in range(3))


def test_the_title_stays_the_title_where_later_pages_repeat_it():
    title = "Serum Electrolytes and Body Mass in Adults"
    pages = [
        [_region(title, (60, 60, 900, 95), "doc_title"), _region(BODY, (60, 200, 940, 800))],
        [_region(title, (60, 20, 900, 40), "paragraph_title"), _region(BODY, (60, 200, 940, 800))],
        [_region(title, (60, 20, 900, 40), "paragraph_title"), _region(BODY, (60, 200, 940, 800))],
    ]

    roles = write_page_roles(pages)

    assert roles.block(0, 0) is None
    assert roles.block(1, 0) is not None
    assert roles.block(2, 0) is not None


def test_lone_numbers_that_run_with_the_page_are_page_numbers():
    pages = [
        [_region(BODY, (60, 200, 940, 800)), _region(str(page + 11), (480, 950, 520, 970))]
        for page in range(3)
    ]
    # A year alone in the band of one page runs with nothing.
    pages[1].append(_region("2020", (60, 950, 120, 970)))

    roles = write_page_roles(pages)

    for page in range(3):
        tag = roles.block(page, 1)
        assert tag is not None and tag.role == PAGE_NUMBER
        assert tag.decided.component == "page_roles.page_number_run"
    assert roles.block(1, 2) is None


def test_a_numbered_heading_opening_two_pages_is_not_furniture():
    pages = [
        [
            _region("Appendix 1", (60, 30, 400, 60), "paragraph_title"),
            _region(BODY, (60, 200, 940, 800)),
        ],
        [
            _region("Appendix 2", (60, 30, 400, 60), "paragraph_title"),
            _region(BODY, (60, 200, 940, 800)),
        ],
    ]

    roles = write_page_roles(pages)

    assert roles.block(0, 0) is None
    assert roles.block(1, 0) is None


# --- Artifacts -------------------------------------------------------------------


def _artifact_page() -> bytes:
    content = marked("Artifact", text(HEADER, 72.0, 760.0, size=8.0))
    content += text(BODY, 72.0, 600.0)
    content += text("A second line of the body follows.", 72.0, 585.0)
    return build_pdf([PageSpec(content)])


def test_artifact_text_in_a_band_is_furniture_on_a_page_of_its_own():
    pdf_bytes = _artifact_page()
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    pages = [
        [
            _region(HEADER, _layout_box(probe, 0, (60.0, 750.0, 420.0, 772.0))),
            _region(BODY, _layout_box(probe, 0, (60.0, 570.0, 560.0, 615.0))),
        ]
    ]
    layer = _layer(pdf_bytes, pages)
    page_lines = views.page_lines(layer.page(0))

    roles = write_page_roles(pages, page_lines=page_lines, layer=layer)

    head = roles.block(0, 0)
    assert head is not None and head.role == RUNNING_HEADER
    assert head.decided.component == "page_roles.artifact"
    header_line = layer.page(0).blocks[0].lines
    assert head.decided.evidence == tuple(ids.line(0, line) for line in header_line)
    assert roles.block(0, 1) is None
    # The text-layer line the reference stream reads gets the same role.
    line_tag = roles.tag(ids.page_line(0, _line_position(page_lines, HEADER)))
    assert line_tag is not None and line_tag.role == RUNNING_HEADER
    assert line_tag.decided.component == "page_roles.artifact"
    assert line_tag.decided.evidence == head.decided.evidence


def test_without_the_layer_one_page_has_no_evidence_for_its_head():
    pdf_bytes = _artifact_page()
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    pages = [
        [
            _region(HEADER, _layout_box(probe, 0, (60.0, 750.0, 420.0, 772.0))),
            _region(BODY, _layout_box(probe, 0, (60.0, 570.0, 560.0, 615.0))),
        ]
    ]
    page_lines = views.page_lines(probe.page(0))

    roles = write_page_roles(pages, page_lines=page_lines)

    # The head is in the band, but a band is position, not evidence.
    assert roles.tags == ()


def test_artifact_text_outside_the_bands_keeps_its_place():
    content = text(BODY, 72.0, 600.0)
    content += marked("Artifact", text("Downloaded from an archive", 72.0, 400.0))
    pdf_bytes = build_pdf([PageSpec(content)])
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    pages = [
        [
            _region(BODY, _layout_box(probe, 0, (60.0, 590.0, 560.0, 615.0))),
            _region(
                "Downloaded from an archive", _layout_box(probe, 0, (60.0, 390.0, 400.0, 415.0))
            ),
        ]
    ]
    layer = _layer(pdf_bytes, pages)

    roles = write_page_roles(pages, layer=layer)

    assert roles.tags == ()


# --- Page labels -----------------------------------------------------------------


def _numbered_page() -> bytes:
    return build_pdf([PageSpec(text(BODY, 72.0, 600.0) + text("7", 300.0, 30.0))])


def _numbered_regions(probe, number: str) -> list[list[dict]]:
    return [
        [
            _region(BODY, _layout_box(probe, 0, (60.0, 590.0, 560.0, 615.0))),
            _region(number, _layout_box(probe, 0, (290.0, 24.0, 320.0, 44.0))),
        ]
    ]


def test_the_page_label_names_the_page_number():
    pdf_bytes = _numbered_page()
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    pages = _numbered_regions(probe, "7")
    layer = _layer(pdf_bytes, pages)
    layer.page(0).label = "7"
    page_lines = views.page_lines(layer.page(0))

    roles = write_page_roles(pages, page_lines=page_lines, layer=layer)

    number = roles.block(0, 1)
    assert number is not None and number.role == PAGE_NUMBER
    assert number.decided.component == "page_roles.page_label"
    line_tag = roles.tag(ids.page_line(0, _line_position(page_lines, "7")))
    assert line_tag is not None and line_tag.role == PAGE_NUMBER
    assert roles.block(0, 0) is None


def test_a_number_the_label_does_not_give_the_page_is_left_alone():
    pdf_bytes = _numbered_page()
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    pages = _numbered_regions(probe, "7")
    layer = _layer(pdf_bytes, pages)
    layer.page(0).label = "8"

    assert write_page_roles(pages, layer=layer).block(0, 1) is None
    # Without the layer there is no label to match, and one page runs with nothing.
    assert write_page_roles(pages).block(0, 1) is None


# --- The furniture strip -----------------------------------------------------------


def _line_number_column(label: str = "text"):
    """A line-numbered page: its PDF, the strip's numbers, and regions for the column and the body.

    The column's OCR text is the numbers the strip removed from the text layer.
    """
    pdf_bytes = _reference_page()
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    numbers = probe.page(0).furniture
    assert len(numbers) >= 8 and {item.kind for item in numbers} == {"line_number"}
    left = min(item.bbox_pdf[0] for item in numbers) - 1.0
    bottom = min(item.bbox_pdf[1] for item in numbers) - 1.0
    right = max(item.bbox_pdf[2] for item in numbers) + 1.0
    top = max(item.bbox_pdf[3] for item in numbers) + 1.0
    column = _layout_box(probe, 0, (left, bottom, right, top))
    body = _layout_box(probe, 0, (right + 4.0, bottom, 560.0, top))
    printed = " ".join(item.text for item in numbers)
    pages = [[_region(printed, column, label), _region("References", body)]]
    return pdf_bytes, numbers, pages


def test_the_strip_furniture_and_a_line_number_column_read_by_ocr_are_tagged():
    pdf_bytes, numbers, pages = _line_number_column()
    printed = pages[0][0]["content"]
    column = pages[0][0]["bbox_2d"]
    layer = _layer(pdf_bytes, pages)

    roles = write_page_roles(pages, layer=layer)

    for item in layer.page(0).furniture:
        tag = roles.tag(item.furniture_id)
        assert tag is not None and tag.role == LINE_NUMBER
        assert tag.decided.component == "page_roles.strip"
    column_tag = roles.block(0, 0)
    assert column_tag is not None and column_tag.role == LINE_NUMBER
    assert column_tag.decided.component == "page_roles.strip"
    assert set(column_tag.decided.evidence) == {item.furniture_id for item in numbers}
    assert roles.block(0, 1) is None

    # A column the OCR read with words in it is not the numbers alone.
    pages[0][0] = _region(printed + " References", column)
    assert write_page_roles(pages, layer=_layer(pdf_bytes, pages)).block(0, 0) is None


def test_a_watermark_read_by_ocr_is_tagged_with_the_strip_object():
    pdf_bytes = build_pdf([PageSpec(_REVIEW_WATERMARK + _body())])
    probe = build_document_layer(pdf_bytes, range(1), budget=None)
    (mark,) = probe.page(0).furniture
    assert mark.kind == "watermark"
    whole = _layout_box(probe, 0, (36.0, 36.0, 576.0, 756.0))
    stamp = _layout_box(probe, 0, mark.bbox_pdf)
    # The body block comes first, so the lines the stamp's box crosses stay its own.
    pages = [[_region("The body text.", whole), _region("For Review Only", stamp)]]
    layer = _layer(pdf_bytes, pages)
    assert not layer.page(0).blocks[1].lines

    roles = write_page_roles(pages, layer=layer)

    stamp_tag = roles.block(0, 1)
    assert stamp_tag is not None and stamp_tag.role == WATERMARK
    assert stamp_tag.decided.evidence == (mark.furniture_id,)
    assert roles.tag(mark.furniture_id).role == WATERMARK
    assert roles.block(0, 0) is None


def test_a_footer_role_follows_the_position_of_the_block():
    footer = "Preprint posted to an archive under a licence"
    pages = [
        [_region(BODY, (60, 200, 940, 800)), _region(footer, (60, 950, 900, 975))],
        [_region(BODY, (60, 200, 940, 800)), _region(footer, (60, 950, 900, 975))],
    ]

    roles = write_page_roles(pages)

    assert roles.block(0, 1).role == RUNNING_FOOTER


# --- The parser reads the roles ------------------------------------------------------


def test_the_parser_drops_a_page_number_without_filing_it_as_a_running_head():
    pages = [
        [
            _region(HEADER, (60, 20, 600, 40)),
            _region("The first page opens the study.", (60, 200, 940, 800)),
            _region("11", (480, 950, 520, 970)),
        ],
        [
            _region(HEADER, (60, 20, 600, 40)),
            _region("The second page reports its results.", (60, 200, 940, 800)),
            _region("12", (480, 950, 520, 970)),
        ],
    ]
    roles = write_page_roles(pages)
    assert roles.block(0, 2).role == PAGE_NUMBER

    contents = PDFParser(json_result=pages, page_roles=roles).parse()

    assert contents.detected_headers == [HEADER, HEADER]
    assert not any(sentence.text.strip() in {"11", "12"} for sentence in contents.sentences)


def test_the_parser_drops_a_line_number_column_the_layout_calls_a_caption():
    pdf_bytes, numbers, pages = _line_number_column("figure_title")
    roles = write_page_roles(pages, layer=_layer(pdf_bytes, pages))
    assert roles.block(0, 0).role == LINE_NUMBER

    contents = PDFParser(json_result=pages, page_roles=roles).parse()

    # Not a caption candidate, which an unassigned caption would leave in the body.
    receipt = contents.caption_assignment_receipt
    candidates = receipt.candidates if receipt is not None else ()
    assert not any(numbers[0].text in candidate.text for candidate in candidates)
    assert not any(numbers[0].text in sentence.text for sentence in contents.sentences)
    assert contents.detected_headers == []
