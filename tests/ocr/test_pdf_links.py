"""URI link annotations: the reader and the DOI a link targets."""

from __future__ import annotations

import pytest

from bibr.ocr.pdf_links import PdfUriLink, doi_from_uri, read_uri_links


def _pdf_with_links(
    links: list[tuple[tuple[int, int, int, int], str]], *, blank_pages_before: int = 0
) -> bytes:
    """A 200 x 200 pt PDF whose last page carries the given URI link annotations.

    ``blank_pages_before`` blank pages come first. Written by hand, with a
    correct cross-reference table, so the test needs no PDF writer.
    """
    # Objects: 1 catalog, 2 pages, 3 font, 4.. blank pages, then the linked
    # page, then its annotations.
    page_numbers = list(range(4, 4 + blank_pages_before + 1))
    first_annot = page_numbers[-1] + 1
    annots = " ".join(f"{first_annot + i} 0 R" for i in range(len(links)))
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for _ in range(blank_pages_before):
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>")
    objects.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Annots [{annots}] >>"
        ).encode()
    )
    for (x0, y0, x1, y1), uri in links:
        objects.append(
            (
                f"<< /Type /Annot /Subtype /Link /Rect [{x0} {y0} {x1} {y1}] /Border [0 0 0] "
                f"/A << /S /URI /URI ({uri}) >> >>"
            ).encode()
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def test_reader_returns_each_uri_link_with_its_page_and_rectangle():
    pdf = _pdf_with_links(
        [
            ((10, 20, 110, 40), "https://doi.org/10.1234/abc.5"),
            ((10, 60, 90, 70), "https://example.org/page"),
        ]
    )
    links = read_uri_links(pdf)

    assert [link.uri for link in links] == [
        "https://doi.org/10.1234/abc.5",
        "https://example.org/page",
    ]
    assert links[0] == PdfUriLink(
        page_index=0, rect=pytest.approx((10.0, 20.0, 110.0, 40.0)), uri=links[0].uri
    )
    assert read_uri_links(pdf, page_indices=[3]) == []


@pytest.mark.parametrize(
    ("uri", "doi"),
    [
        ("https://doi.org/10.1016/j.ymgme.2017.11.005", "10.1016/j.ymgme.2017.11.005"),
        ("http://dx.doi.org/10.1037/0022-3514.59.5.899", "10.1037/0022-3514.59.5.899"),
        ("doi:10.1002/acp.1722", "10.1002/acp.1722"),
        ("https://doi.org/10.1002%2F%28SICI%291097", "10.1002/(SICI)1097"),
        # Entities escaped twice, as one publisher writes them.
        (
            "https://doi.org/10.1002/(SICI)1097:1&amp;lt;55&amp;gt;3.0",
            "10.1002/(SICI)1097:1<55>3.0",
        ),
        ("https://doi.org/10.1234/ab%00cd", None),
        ("https://doi.org/10.1234/ab\ufffdcd", None),
        ("https://doi.org/not-a-doi", None),
        ("https://www.mdpi.com/2409-515X/7/1/17/s1", None),
        ("https://www.tandfonline.com/doi/full/10.1080/1", None),
        ("", None),
    ],
)
def test_doi_from_uri_reads_only_doi_resolver_links(uri, doi):
    assert doi_from_uri(uri) == doi
