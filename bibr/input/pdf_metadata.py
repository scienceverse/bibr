"""Native PDF doc-info metadata harvest.

Born-digital PDFs often carry publisher-set document-info metadata (Title,
a DOI in Subject/Keywords, a keyword list). Reading it is free — but
doc-info is also routinely junk: word processors stamp the source filename
as Title ("Microsoft Word - draft_v3.docx") and the submitting user as
Author. Every harvested field therefore passes a guard, and the result is
only ever offered to the fill-empty merge (``_merge_ocr_metadata`` via
``post_parse``'s ``ocr_metadata`` parameter) — LLM/layout extraction always
wins when it produced anything.

Title guard: accepted only when it is verifiably PRINTED on the first page
(token containment against the native first-page text, hyphenation- and
line-wrap-tolerant). Without usable verification text (scanned PDF) the
title is rejected outright — the junk rate of unverifiable doc-info titles
is too high.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bibr.utils.text import normalize_doi

logger = logging.getLogger(__name__)

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>()\[\]{}]+")
_JUNK_TITLE_RE = re.compile(
    r"(?i)(microsoft word|powerpoint|libreoffice|untitled"
    r"|\.docx?\b|\.tex\b|\.pdf\b|\.dvi\b|\.qxd\b|\.indd\b|\.fm\b)"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Below this much first-page text there is nothing meaningful to verify a
# title against (blank/scanned page), so title harvesting is disabled.
_MIN_VERIFICATION_CHARS = 200

# Fraction of the title's tokens that must appear on the first page.
_TITLE_TOKEN_MATCH_RATIO = 0.8


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _title_printed_on_page(title: str, page_text: str) -> bool:
    # Undo hyphenated line wraps ("Atten-\ntion" → "Attention") before
    # tokenizing so a wrapped printed title still matches.
    dehyphenated = re.sub(r"-\s*\n\s*", "", page_text)
    if len(dehyphenated) < _MIN_VERIFICATION_CHARS:
        return False
    page_tokens = set(_tokens(dehyphenated))
    title_tokens = _tokens(title)
    if not title_tokens:
        return False
    matched = sum(1 for t in title_tokens if t in page_tokens)
    return matched / len(title_tokens) >= _TITLE_TOKEN_MATCH_RATIO


def _harvest_title(info: dict[str, Any], first_page_text: str) -> str | None:
    title = str(info.get("Title") or "").strip()
    if len(title) < 8:
        return None
    if _JUNK_TITLE_RE.search(title):
        return None
    if not any(c.isalpha() for c in title):
        return None
    if not _title_printed_on_page(title, first_page_text):
        return None
    return title


def _harvest_doi(info: dict[str, Any]) -> str | None:
    for field in ("Subject", "Keywords"):
        m = _DOI_RE.search(str(info.get(field) or ""))
        if m:
            doi = normalize_doi(m.group(0))
            if doi:
                return doi
    return None


def _harvest_keywords(info: dict[str, Any]) -> list[str] | None:
    raw = str(info.get("Keywords") or "").strip().rstrip(" .;")
    if not raw:
        return None
    sep = ";" if ";" in raw else ","
    candidates = [k.strip() for k in raw.split(sep) if k.strip()]
    # Same shape heuristic as the section-scrape recovery in post_parse:
    # a plausible keyword list is short entries without sentence periods
    # (this also rejects URLs and DOIs masquerading as keywords).
    if not (1 <= len(candidates) <= 15):
        return None
    if not all(len(k) <= 80 and "." not in k and any(c.isalpha() for c in k) for k in candidates):
        return None
    return candidates


def harvest_docinfo(info: dict[str, Any], first_page_text: str) -> dict[str, Any]:
    """Harvest guarded metadata from a PDF document-info dict.

    Returns a dict with any of ``title``/``doi``/``keywords`` that passed
    their guards (possibly empty). Shaped for ``_merge_ocr_metadata``.
    """
    out: dict[str, Any] = {}
    title = _harvest_title(info, first_page_text)
    if title:
        out["title"] = title
    doi = _harvest_doi(info)
    if doi:
        out["doi"] = doi
    keywords = _harvest_keywords(info)
    if keywords:
        out["keywords"] = keywords
    return out


def harvest_pdf_metadata(pdf_bytes: bytes, first_page_text: str) -> dict[str, Any]:
    """Read the doc-info dictionary from *pdf_bytes* and harvest it.

    Best-effort: any pdfium failure returns ``{}``. Callers run this off the
    event loop (it takes the shared pdfium lock).
    """
    from bibr.ocr.utils import pdfium_lock

    try:
        import pypdfium2 as pdfium

        with pdfium_lock:
            doc = pdfium.PdfDocument(pdf_bytes)
            try:
                info = {k: doc.get_metadata_value(k) for k in ("Title", "Subject", "Keywords")}
            finally:
                doc.close()
    except Exception:
        logger.debug("PDF doc-info read failed", exc_info=True)
        return {}
    return harvest_docinfo(info, first_page_text)
