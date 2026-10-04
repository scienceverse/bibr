"""The structure tree of a tagged PDF, as pdfium reads it page by page.

``FPDF_StructTree_GetForPage`` builds the tree of one page: the elements that
hold marked content on that page and each of their ancestors, with the
marked content of that page only. An element that holds content on several
pages is therefore read once per page. Each copy keeps the element's place in
the whole tree, as the index of the element among its parent's kids from the
root down (:attr:`StructElem.path`), because pdfium leaves a slot for every
kid and fills those that belong to the page. On gate192 it is the same on
every copy and tells the elements apart (no two elements with one path had a
different type or alternate text).

What is read of an element is what pdfium offers: its type with /RoleMap
applied for one step (a type mapped through a chain stops at the middle
name), its alternate and actual text, its language and the ids of the marked
content it holds (``(page, mcid)``). pdfium offers nothing for the other kinds
of kid: an object reference (a link annotation or a form widget the element
wraps) and a kid on another page both read as "not an element and not marked
content here". The links an element wraps are found from the text under them
(``bibr.document.views.StructIndex.link_element``).

pdfium gives a page a tree only when the catalog's /MarkInfo says the PDF is
tagged. A PDF with a structure tree root and no /MarkInfo (4 of the 68 papers
of gate192 whose root the census found) reads as untagged, without elements.

Everything here calls pdfium and needs the caller's ``pdfium_lock``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bibr.document import destinations
from bibr.document._ids import page_id
from bibr.document.model import StructElem

if TYPE_CHECKING:
    from bibr.document.harvest import _Api

APIS = (
    "FPDF_StructTree_GetForPage",
    "FPDF_StructTree_Close",
    "FPDF_StructTree_CountChildren",
    "FPDF_StructTree_GetChildAtIndex",
    "FPDF_StructElement_GetType",
    "FPDF_StructElement_CountChildren",
    "FPDF_StructElement_GetChildAtIndex",
    "FPDF_StructElement_GetChildMarkedContentID",
    "FPDF_StructElement_GetAltText",
    "FPDF_StructElement_GetActualText",
    "FPDF_StructElement_GetLang",
)
# Whether the catalog's /MarkInfo says the PDF is tagged.
CATALOG_APIS = ("FPDFCatalog_IsTagged",)

# A page's tree with more elements than this is read to this many (the busiest
# page of gate192 has a few hundred), so a hostile tree cannot hold the lock
# for long.
MAX_ELEMENTS = 50_000


def read_page_tree(api: _Api, page, page_index: int) -> tuple[list[StructElem], bool, str | None]:
    """The page's structure elements in document order, whether it has a tree, and a note.

    A page has a tree when the PDF has a structure tree root that reaches pages
    (a page with no tagged content has a tree and no elements). The note says
    why the walk ended early; the elements read before then are kept.
    """
    tree = api.FPDF_StructTree_GetForPage(page.raw)
    if not tree:
        return [], False, None
    elements: list[StructElem] = []
    note: str | None = None
    try:
        # Elements still to visit, as (element, parent id, path). An element's
        # kids go on the stack in reverse, so they come off in order, each with
        # its subtree before the next one.
        pending: list[tuple[Any, str | None, tuple[int, ...]]] = [
            (api.FPDF_StructTree_GetChildAtIndex(tree, index), None, (index,))
            for index in reversed(range(api.FPDF_StructTree_CountChildren(tree)))
        ]
        seen: set[int] = set()
        while pending:
            handle, parent, path = pending.pop()
            # A slot of the tree whose element has no content on this page is empty.
            if not handle:
                continue
            address = destinations.address(handle)
            if address in seen:
                note = "circular structure reference"
                continue
            if len(elements) >= MAX_ELEMENTS:
                note = f"more than {MAX_ELEMENTS} structure elements, the rest unread"
                break
            seen.add(address)
            elements.append(_element(api, handle, page_index, len(elements), parent, path, pending))
    except Exception as exc:  # noqa: BLE001 - a layer component never fails the paper
        note = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        api.FPDF_StructTree_Close(tree)
    return elements, True, note


def _element(
    api: _Api,
    handle,
    page_index: int,
    number: int,
    parent: str | None,
    path: tuple[int, ...],
    pending: list[tuple[Any, str | None, tuple[int, ...]]],
) -> StructElem:
    """The element at *handle*; its element kids are pushed onto *pending*."""
    elem_id = page_id("st", page_index, number)
    mcrs: list[tuple[int, int]] = []
    kids = []
    for index in range(api.FPDF_StructElement_CountChildren(handle)):
        kid = api.FPDF_StructElement_GetChildAtIndex(handle, index)
        if kid:
            kids.append((kid, elem_id, (*path, index)))
            continue
        # Not an element: marked content of this page, or a kid pdfium does not offer.
        mcid = api.FPDF_StructElement_GetChildMarkedContentID(handle, index)
        if mcid >= 0:
            mcrs.append((page_index, int(mcid)))
    pending.extend(reversed(kids))
    return StructElem(
        elem_id=elem_id,
        parent=parent,
        role=destinations.utf16_text(api.FPDF_StructElement_GetType, handle) or "",
        mcrs=tuple(mcrs),
        page=page_index,
        path=path,
        alt=destinations.utf16_text(api.FPDF_StructElement_GetAltText, handle),
        actual=destinations.utf16_text(api.FPDF_StructElement_GetActualText, handle),
        lang=destinations.utf16_text(api.FPDF_StructElement_GetLang, handle),
    )
