"""A tiny PDF writer for DOI-evidence tests: text runs, links, Info and XMP.

Written by hand, with a correct cross-reference table, so the tests need no
PDF library. Text is set in the standard Helvetica font; a run may be rotated
(a repository banner printed up the page margin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TextRun:
    x: float
    y: float
    text: str
    size: float = 10.0
    angle: float = 0.0  # degrees counter-clockwise


@dataclass
class Page:
    runs: list[TextRun] = field(default_factory=list)
    # (x0, y0, x1, y1) in points and the link's URI.
    links: list[tuple[tuple[float, float, float, float], str]] = field(default_factory=list)
    size: tuple[float, float] = (600.0, 800.0)


def _literal(text: str) -> bytes:
    raw = text.encode("cp1252")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def build_pdf(
    pages: list[Page], *, info: dict[str, str] | None = None, xmp: str | None = None
) -> bytes:
    """Serialize *pages* with optional document-information entries and XMP packet."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled in once the page tree and metadata exist
    pages_ref = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    page_refs = []
    for page in pages:
        content = bytearray()
        for run in page.runs:
            cos, sin = math.cos(math.radians(run.angle)), math.sin(math.radians(run.angle))
            content += (
                f"BT /F1 {run.size:g} Tf {cos:.4f} {sin:.4f} {-sin:.4f} {cos:.4f} "
                f"{run.x:g} {run.y:g} Tm "
            ).encode()
            content += _literal(run.text) + b" Tj ET\n"
        stream = add(
            f"<< /Length {len(content)} >>\nstream\n".encode() + bytes(content) + b"\nendstream"
        )
        annots = []
        for (x0, y0, x1, y1), uri in page.links:
            annots.append(
                add(
                    f"<< /Type /Annot /Subtype /Link /Rect [{x0:g} {y0:g} {x1:g} {y1:g}] "
                    f"/Border [0 0 0] /A << /S /URI /URI ".encode()
                    + _literal(uri)
                    + b" >> >>"
                )
            )
        width, height = page.size
        annot_refs = " ".join(f"{ref} 0 R" for ref in annots)
        page_refs.append(
            add(
                (
                    f"<< /Type /Page /Parent {pages_ref} 0 R /MediaBox [0 0 {width:g} {height:g}] "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R "
                    f"/Annots [{annot_refs}] >>"
                ).encode()
            )
        )
    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects[pages_ref - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode()
    metadata_entry = b""
    if xmp is not None:
        packet = xmp.encode("utf-8")
        metadata = add(
            f"<< /Type /Metadata /Subtype /XML /Length {len(packet)} >>\nstream\n".encode()
            + packet
            + b"\nendstream"
        )
        metadata_entry = f" /Metadata {metadata} 0 R".encode()
    objects[catalog - 1] = (
        f"<< /Type /Catalog /Pages {pages_ref} 0 R".encode() + metadata_entry + b" >>"
    )
    info_ref = None
    if info:
        entries = b" ".join(f"/{key} ".encode() + _literal(value) for key, value in info.items())
        info_ref = add(b"<< " + entries + b" >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"/Size {len(objects) + 1} /Root {catalog} 0 R"
    if info_ref is not None:
        trailer += f" /Info {info_ref} 0 R"
    out += f"trailer\n<< {trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def xmp_packet(**properties: str) -> str:
    """An XMP packet whose rdf:Description carries *properties* (``prism_doi`` → prism:doi)."""
    body = "".join(
        f"<{name.replace('_', ':', 1)}>{value}</{name.replace('_', ':', 1)}>"
        for name, value in properties.items()
    )
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f"<rdf:Description>{body}</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
