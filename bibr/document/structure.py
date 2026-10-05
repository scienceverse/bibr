"""The structure tree of a tagged PDF, as pdfium reads it page by page.

``FPDF_StructTree_GetForPage`` builds the tree of one page: the elements that
hold marked content on that page and each of their ancestors, with the
marked content of that page only. An element that holds content on several
pages is therefore read once per page, and :func:`merge` joins the copies into
one element. Each copy keeps the element's place in the whole tree, as the
index of the element among its parent's kids from the root down
(:attr:`StructElem.path`), because pdfium leaves a slot for every kid and
fills those that belong to the page. On gate192 and the manuscripts it is the
same on every copy and tells the elements apart (no two elements with one path
had a different type, alternate text, actual text, language or number of kids).
The path is the element's id (``st0.3.2``), so the id depends on the PDF alone.

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

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from bibr.document import destinations, ids
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

# A document with more structure elements than this is read to this many (each page's
# copy of an element counts: the busiest gate192 paper has 3,560, its busiest page 803),
# and an element with more kids than MAX_KIDS is read to that many (the busiest holds 411
# marked-content references), so a hostile tree cannot hold the lock for long: 20,000
# elements take 0.2 s to read. The caps bound what is read, not pdfium, which builds a
# page's whole tree in one call that cannot be cut short: a page with 40,000 kids under
# one parent takes it 0.7 s.
MAX_ELEMENTS = 20_000
MAX_KIDS = 2_000


class Copy(NamedTuple):
    """The page's copy of a structure element, and the number of kids pdfium counts in it.

    The kid count is read for :func:`merge` to compare and is not kept in the layer: pdfium
    gives every copy of an element a slot for each kid of its /K, so it is the same on
    every page.
    """

    elem: StructElem
    kids: int


@dataclass(slots=True)
class PageTree:
    """What was read of the structure tree of one page."""

    # The page's copies of the elements in document order.
    copies: list[Copy] = field(default_factory=list)
    # Whether pdfium gave the page a tree. It does when the PDF has a structure tree
    # root that reaches pages; a page with no tagged content has a tree and no elements.
    has_tree: bool = False
    # What cut the read short, each once: a tree or an element with more kids than
    # MAX_KIDS, a circular reference. The reader of the document says each once for all its
    # pages, not once for each page.
    cuts: list[str] = field(default_factory=list)
    # Whether the allowance of elements ran out on this page: another element was there.
    # The caller says so, with this page, and gives pdfium no later one.
    stopped: bool = False
    # Why the walk failed, and the elements read before then are kept.
    failure: str | None = None

    def cut(self, text: str) -> None:
        if text not in self.cuts:
            self.cuts.append(text)


def read_page_tree(api: _Api, page, page_index: int, *, limit: int = MAX_ELEMENTS) -> PageTree:
    """The page's copies of the structure elements in document order, and what the read found.

    At most *limit* elements are read: what the document's allowance of them has
    left. When the allowance is spent and another element is there, the result
    says so (:attr:`PageTree.stopped`) and the caller gives pdfium no later page:
    pdfium builds the whole tree of a page in one call that cannot be cut short,
    so a document over its allowance would pay for every page for nothing. A
    document whose elements are exactly the allowance is read whole and says
    nothing.
    """
    tree = api.FPDF_StructTree_GetForPage(page.raw)
    if not tree:
        return PageTree()
    read = PageTree(has_tree=True)
    try:
        tops = api.FPDF_StructTree_CountChildren(tree)
        if tops > MAX_KIDS:
            read.cut(f"a tree with more than {MAX_KIDS} top-level elements, the rest unread")
        # Elements still to visit, as (element, path). An element's kids go on
        # the stack in reverse, so they come off in order, each with its subtree
        # before the next one.
        pending: list[tuple[Any, tuple[int, ...]]] = [
            (api.FPDF_StructTree_GetChildAtIndex(tree, index), (index,))
            for index in reversed(range(min(tops, MAX_KIDS)))
        ]
        seen: set[int] = set()
        while pending:
            handle, path = pending.pop()
            # A slot of the tree whose element has no content on this page is empty.
            if not handle:
                continue
            address = destinations.address(handle)
            if address in seen:
                read.cut("circular structure reference")
                continue
            if len(read.copies) >= limit:
                read.stopped = True
                break
            seen.add(address)
            element, wide = _element(api, handle, page_index, path, pending)
            read.copies.append(element)
            if wide:
                read.cut(f"an element with more than {MAX_KIDS} kids, the rest unread")
    except Exception as exc:  # noqa: BLE001 - a layer component never fails the paper
        read.failure = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        api.FPDF_StructTree_Close(tree)
    return read


def _element(
    api: _Api,
    handle,
    page_index: int,
    path: tuple[int, ...],
    pending: list[tuple[Any, tuple[int, ...]]],
) -> tuple[Copy, bool]:
    """The page's copy of the element at *handle*, and whether it has more kids than were read.

    Its element kids are pushed onto *pending*.
    """
    mcrs: list[tuple[int, int]] = []
    kids = []
    count = api.FPDF_StructElement_CountChildren(handle)
    for index in range(min(count, MAX_KIDS)):
        kid = api.FPDF_StructElement_GetChildAtIndex(handle, index)
        if kid:
            kids.append((kid, (*path, index)))
            continue
        # Not an element: marked content of this page, or a kid pdfium does not offer.
        mcid = api.FPDF_StructElement_GetChildMarkedContentID(handle, index)
        if mcid >= 0:
            mcrs.append((page_index, int(mcid)))
    pending.extend(reversed(kids))
    element = StructElem(
        elem_id=ids.struct_element(path),
        parent=ids.struct_element(path[:-1]) if len(path) > 1 else None,
        role=destinations.utf16_text(api.FPDF_StructElement_GetType, handle) or "",
        mcrs=tuple(mcrs),
        path=path,
        alt=destinations.utf16_text(api.FPDF_StructElement_GetAltText, handle),
        actual=destinations.utf16_text(api.FPDF_StructElement_GetActualText, handle),
        lang=destinations.utf16_text(api.FPDF_StructElement_GetLang, handle),
    )
    return Copy(element, int(count)), count > MAX_KIDS


def merge(copies: list[Copy]) -> tuple[list[StructElem], int]:
    """The elements the page-by-page *copies* are of, one each, and how many copies differ from the first of theirs.

    The copies of an element agree in everything but the marked content, which
    is each page's own; the element holds all of it, in the order the pages
    were read. An element comes in the place of its first copy, so it follows
    the element above it. The first copy gives the type, the text and the
    language; a copy that differs from it in those or in its number of kids
    (none on gate192 or the manuscripts) is counted, and its marked content is
    added all the same.
    """
    first: dict[str, Copy] = {}
    held: dict[str, list[tuple[int, int]]] = {}
    differing = 0
    for copy in copies:
        known = first.setdefault(copy.elem.elem_id, copy)
        held.setdefault(copy.elem.elem_id, []).extend(copy.elem.mcrs)
        if _signature(copy) != _signature(known):
            differing += 1
    return [
        replace(copy.elem, mcrs=tuple(held[elem_id])) for elem_id, copy in first.items()
    ], differing


def _signature(copy: Copy) -> tuple[str, str | None, str | None, str | None, int]:
    """What the copies of one element must agree on."""
    elem = copy.elem
    return (elem.role, elem.alt, elem.actual, elem.lang, copy.kids)
