"""The PDF outline (bookmarks), as the PDF declares it.

An entry keeps its title as written, blank or not, its depth and parent, and
where it points: the 0-based page, the position on it in PDF points and the
name of the destination when the entry goes by one. Nothing is filtered here:
:mod:`bibr.document.outline_guard` decides whether the outline is usable.

The walk is bibr's own (first child, next sibling). ``bibr.input.pdf_outline``
reads the same bookmarks for the heading matcher through pypdfium2's
``get_toc``, which stops at depth 15 and has the matcher's own filters; this one
keeps every depth, and a bookmark chain that loops back ends where the loop
is found.

Everything here calls pdfium and needs the caller's ``pdfium_lock``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bibr.document import destinations
from bibr.document.model import OutlineEntry

if TYPE_CHECKING:
    from bibr.document.harvest import _Api

APIS = (
    "FPDFBookmark_GetFirstChild",
    "FPDFBookmark_GetNextSibling",
    "FPDFBookmark_GetTitle",
    "FPDFBookmark_GetDest",
    "FPDFBookmark_GetAction",
    "FPDFAction_GetType",
    "FPDF_GetMetaText",
)

# An outline longer than this is read to this many entries (books have a few
# thousand), so a hostile one cannot hold the lock for long.
MAX_ENTRIES = 20_000


def meta_title(api: _Api, doc) -> str | None:
    """The document information dictionary's /Title (None when it has none)."""
    return destinations.utf16_text(api.FPDF_GetMetaText, doc.raw, b"Title")


def read_outline(
    api: _Api, doc, names: destinations.NamedDests | None, n_pages: int
) -> tuple[list[OutlineEntry], str | None]:
    """The document's bookmarks in document order, and a note when the walk was cut short.

    A failure partway keeps the entries read so far and says so in the note.
    """
    entries: list[OutlineEntry] = []
    seen: set[int] = set()
    note: str | None = None
    # Bookmarks still to visit, as (bookmark, depth, parent entry); the next
    # sibling goes below its subtree, so a subtree is read before it.
    pending = [(api.FPDFBookmark_GetFirstChild(doc.raw, None), 0, None)]
    try:
        while pending:
            bookmark, level, parent = pending.pop()
            if not bookmark:
                continue
            address = destinations.address(bookmark)
            if address in seen:
                note = "circular bookmark reference"
                continue
            if len(entries) >= MAX_ENTRIES:
                note = f"more than {MAX_ENTRIES} bookmarks, the rest unread"
                break
            seen.add(address)
            idx = len(entries)
            entries.append(_entry(api, doc, bookmark, names, n_pages, idx, level, parent))
            pending.append((api.FPDFBookmark_GetNextSibling(doc.raw, bookmark), level, parent))
            pending.append((api.FPDFBookmark_GetFirstChild(doc.raw, bookmark), level + 1, idx))
    except Exception as exc:  # noqa: BLE001 - a layer component never fails the paper
        note = f"{type(exc).__name__}: {exc}"[:500]
    return entries, note


def _entry(
    api: _Api,
    doc,
    bookmark,
    names: destinations.NamedDests | None,
    n_pages: int,
    idx: int,
    level: int,
    parent: int | None,
) -> OutlineEntry:
    title = destinations.utf16_text(api.FPDFBookmark_GetTitle, bookmark) or ""
    page = x = y = name = None
    # An action that is not a jump inside this document (a remote jump, a
    # URI) names no page of it: pdfium would read its destination all the same.
    action = api.FPDFBookmark_GetAction(bookmark)
    internal = not action or api.FPDFAction_GetType(action) == api.c.PDFACTION_GOTO
    dest = api.FPDFBookmark_GetDest(doc.raw, bookmark) if internal else None
    if dest:
        page = destinations.dest_page(api, doc, dest, n_pages)
        if page is not None:
            x, y = destinations.dest_position(api, dest)
        if names is not None:
            name = names.name_of(dest)
    return OutlineEntry(
        idx=idx, parent=parent, level=level, title=title, page=page, x=x, y=y, dest_name=name
    )
