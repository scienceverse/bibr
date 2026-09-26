"""Native ePub parser.

ePub is treated as a packaging layer around XHTML/HTML spine documents. The
actual article parsing delegates to :mod:`bibr.input.html_native`.
"""

from __future__ import annotations

import html
import io
import logging
import posixpath
import zipfile
from dataclasses import dataclass, field
from urllib.parse import unquote

from bibr.input.html_native import HtmlParser
from bibr.input.xml_entities import parse_xml
from bibr.input.zip_limits import ZipExpansionLimitError, read_zip_member_capped
from bibr.paper_contents import PaperContents
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.utils.text import normalize_doi

logger = logging.getLogger(__name__)

_EPUB_TOTAL_UNCOMPRESSED_MAX_BYTES = 256 * 1024 * 1024
_EPUB_MAX_ENTRIES = 20_000
_EPUB_MAX_COMPRESSION_RATIO = 100
# Per-member real-byte cap. CPython's ZipExtFile already enforces the declared
# file_size when a member is read — truncating the output and raising
# BadZipFile on the CRC mismatch a lying header causes — so this cap is
# defense in depth: it keeps the rejection at validation, however the archive
# is read downstream (audit L10/M7).
_EPUB_MAX_MEMBER_BYTES = 64 * 1024 * 1024
# Every limit above is per member, so a spine that names one member N times
# multiplies all of them. Bound the spine itself, and the bytes it actually
# accumulates, independently of how the archive is packed.
_EPUB_MAX_SPINE_DOCUMENTS = 2_000
_EPUB_MAX_SPINE_BYTES = 64 * 1024 * 1024


@dataclass
class EpubDocument:
    html_bytes: bytes
    metadata: dict
    # Spine members named by the manifest but absent from the zip. Only
    # missing members are skipped; anything else (an over-cap member, a CRC
    # failure) rejects the book. Callers surface these so the loss is visible.
    skipped_spine_members: list[str] = field(default_factory=list)


class EpubSpineMemberMissingError(ValueError):
    """A spine member named by the manifest is absent from the zip archive."""


def _ln(el) -> str:
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _attr(el, name: str) -> str | None:
    for key, value in el.attrib.items():
        if key.rsplit("}", 1)[-1] == name:
            return value
    return None


def _text(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split()).strip()


def _read_zip_member(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        zf.getinfo(name)
    except KeyError:
        raise EpubSpineMemberMissingError(f"ePub member missing: {name}") from None
    try:
        return read_zip_member_capped(zf, name, max_bytes=_EPUB_MAX_MEMBER_BYTES)
    except ZipExpansionLimitError as exc:
        raise ValueError(str(exc)) from exc


def _check_zip_limits(zf: zipfile.ZipFile) -> bool:
    entries = zf.infolist()
    if len(entries) > _EPUB_MAX_ENTRIES:
        return False
    total_size = sum(info.file_size for info in entries)
    total_compressed = sum(info.compress_size for info in entries)
    if total_size > _EPUB_TOTAL_UNCOMPRESSED_MAX_BYTES:
        return False
    return total_size / max(total_compressed, 1) <= _EPUB_MAX_COMPRESSION_RATIO


def read_epub_document(epub_bytes: bytes) -> EpubDocument:
    """Read package metadata and spine XHTML into one synthetic HTML document."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(epub_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid ePub zip archive") from exc

    with zf:
        if not _check_zip_limits(zf):
            raise ValueError("ePub archive exceeds expansion limits")
        mimetype = _read_zip_member(zf, "mimetype").decode("utf-8", errors="ignore").strip()
        if mimetype != "application/epub+zip":
            raise ValueError("ePub mimetype file is missing or invalid")

        container = parse_xml(_read_zip_member(zf, "META-INF/container.xml"))
        rootfile_path = None
        for el in container.iter():
            if _ln(el) == "rootfile":
                # OCF/OPF hrefs are URL references: a name with a space is
                # percent-encoded, but zip member names are not.
                full_path = _attr(el, "full-path")
                rootfile_path = unquote(full_path) if full_path else None
                break
        if not rootfile_path:
            raise ValueError("ePub container has no rootfile")

        opf = parse_xml(_read_zip_member(zf, rootfile_path))
        base = posixpath.dirname(rootfile_path)

        manifest: dict[str, str] = {}
        spine_ids: list[str] = []
        metadata: dict = {"authors": [], "keywords": []}

        for el in opf.iter():
            ln = _ln(el)
            if ln == "item":
                item_id = _attr(el, "id")
                href = _attr(el, "href")
                if item_id and href:
                    manifest[item_id] = posixpath.normpath(posixpath.join(base, unquote(href)))
            elif ln == "itemref":
                idref = _attr(el, "idref")
                if idref:
                    spine_ids.append(idref)
            elif ln == "title" and not metadata.get("title"):
                metadata["title"] = _text(el)
            elif ln == "creator":
                creator = _text(el)
                if creator:
                    metadata.setdefault("authors", []).append(creator)
            elif ln == "identifier":
                value = _text(el)
                # ePub packagers spell the DOI as a bare '10.…', 'doi:10.…',
                # 'https://doi.org/10.…', or 'urn:doi:10.…' — normalize every
                # form to the bare DOI instead of only the first.
                bare = value
                if bare.lower().startswith("urn:doi:"):
                    bare = bare[len("urn:doi:") :]
                doi = normalize_doi(bare)
                if doi:
                    metadata["doi"] = doi
            elif ln == "publisher" and not metadata.get("publisher"):
                metadata["publisher"] = _text(el)
            elif ln == "date" and not metadata.get("published"):
                metadata["published"] = _text(el)
            elif ln == "subject":
                subject = _text(el)
                if subject:
                    metadata.setdefault("keywords", []).append(subject)
            elif ln == "rights" and not metadata.get("license"):
                metadata["license"] = _text(el)

        if len(spine_ids) > _EPUB_MAX_SPINE_DOCUMENTS:
            raise ValueError("ePub archive exceeds expansion limits")
        # De-duplicate: a repeated idref contributes nothing but a second copy
        # of the same text, and is the cheapest way to multiply the per-member
        # caps above.
        seen_paths: set[str] = set()
        spine_paths: list[str] = []
        for idref in spine_ids:
            path = manifest.get(idref)
            if path is None or path in seen_paths:
                continue
            seen_paths.add(path)
            spine_paths.append(path)
        if not spine_paths:
            raise ValueError("ePub package has no readable spine documents")

        body_parts: list[str] = []
        spine_bytes = 0
        skipped: list[str] = []
        for path in spine_paths:
            try:
                data = _read_zip_member(zf, path)
            except EpubSpineMemberMissingError as exc:
                # Only a missing member is skipped: the expansion-limit
                # ValueError and BadZipFile/CRC failures propagate and reject
                # the book, as before.
                logger.warning("Skipping missing ePub spine member %r: %s", path, exc)
                skipped.append(path)
                continue
            spine_bytes += len(data)
            if spine_bytes > _EPUB_MAX_SPINE_BYTES:
                raise ValueError("ePub archive exceeds expansion limits")
            body_parts.append(data.decode("utf-8", errors="replace"))
        if not body_parts:
            raise ValueError(
                "ePub package has no readable spine documents"
                + (f" (skipped: {', '.join(skipped)})" if skipped else "")
            )

    head_parts = []
    if metadata.get("title"):
        head_parts.append(
            f'<meta name="citation_title" content="{html.escape(metadata["title"])}">'
        )
    if metadata.get("doi"):
        head_parts.append(f'<meta name="citation_doi" content="{html.escape(metadata["doi"])}">')
    for author in metadata.get("authors") or []:
        head_parts.append(f'<meta name="citation_author" content="{html.escape(author)}">')
    for keyword in metadata.get("keywords") or []:
        head_parts.append(f'<meta name="dc.subject" content="{html.escape(keyword)}">')
    for name, meta_name in (
        ("publisher", "dc.publisher"),
        ("published", "dc.date"),
        ("license", "dc.rights"),
    ):
        if metadata.get(name):
            head_parts.append(
                f'<meta name="{meta_name}" content="{html.escape(str(metadata[name]))}">'
            )

    # html5lib sniffs the encoding and falls back to windows-1252 without a
    # declaration, which mojibakes every non-ASCII character in the metadata
    # and body below. Declare it first: the scan only reads the opening bytes.
    combined = (
        '<!doctype html><html><head><meta charset="utf-8">'
        + "".join(head_parts)
        + "</head><body><article>"
        + "\n".join(body_parts)
        + "</article></body></html>"
    )
    return EpubDocument(
        html_bytes=combined.encode("utf-8"), metadata=metadata, skipped_spine_members=skipped
    )


class EpubParser:
    """Parse ePub bytes by delegating spine XHTML to :class:`HtmlParser`."""

    def __init__(self, epub_bytes: bytes, *, document: EpubDocument | None = None) -> None:
        self.epub_bytes = epub_bytes
        self._document = document
        self._html_parser: HtmlParser | None = None

    @property
    def assembler(self):
        if self._html_parser is None:
            raise RuntimeError("parse() must be called before accessing assembler")
        return self._html_parser.assembler

    @property
    def _deferred_texts(self) -> list[tuple[str, int | None, int, bool, bool]]:
        if self._html_parser is None:
            return []
        return self._html_parser._deferred_texts

    def parse(self) -> PaperContents:
        from bibr.exceptions import ProcessingError

        try:
            document = self._document or read_epub_document(self.epub_bytes)
        except Exception as exc:
            raise ProcessingError(f"Failed to parse ePub: {exc}") from exc

        self._html_parser = HtmlParser(document.html_bytes)
        contents = self._html_parser.parse()
        # A skipped chapter leaves no text in the payload, so record each loss
        # as a coded warning. PaperContents.processing_warnings reaches
        # Paper.processing_warnings in post_parse and from there
        # extraction.warnings, so the incomplete book is visible in the export.
        for path in document.skipped_spine_members:
            contents.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.EPUB_SPINE_MEMBER_SKIPPED,
                    f"Skipped missing ePub spine member {path!r}; its text is absent",
                )
            )
        return contents

    def apply_segmentation(self, contents: PaperContents, all_segments: list[list[str]]) -> None:
        if self._html_parser is None:
            raise RuntimeError("parse() must be called before apply_segmentation()")
        self._html_parser.apply_segmentation(contents, all_segments)

    def create_content_sections(self, contents: PaperContents) -> None:
        if self._html_parser is None:
            raise RuntimeError("parse() must be called before create_content_sections()")
        self._html_parser.create_content_sections(contents)
