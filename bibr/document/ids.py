"""Ids of layer objects: ``p{page}.{kind}{n}``, and ``{kind}{n}`` for those of the whole PDF.

An id names one object on one page: the absolute 0-based page index, a
lowercase kind prefix, and the object's index among that page's objects of
the kind. D1's kinds are spans ``sp`` (rows of ``PageColumns.span_rec``),
lines ``l`` (``PageColumns.line_span``), blocks ``r`` (the page's post-OCR
regions, in order) and furniture ``f`` (``Page.furniture``). :func:`make`
and :func:`parse` take any lowercase prefix that is not a document kind, so
later kinds need no new parser.

Span, line and furniture ids come from the PDF alone: the same PDF read
under the same :data:`~bibr.document.model.INDEX_FRAME` and pdfium gives the
same ids. Block ids do not: a block's index is its position in the post-OCR
region list, which depends on the layout model and on OCR merges, and D3
moves block ids to the layout slot. Resolve blocks through
:mod:`bibr.document.views`, and do not build block ids from region indexes
or persist them before then.

An object that belongs to the whole PDF, not to a page, has no page in its
id: ``{kind}{n}``, with a ``.{n}`` more for each level of its place in a
tree. D2's document kind is outline entries ``ol`` (``ol5``, the entry's
position in the outline). Such an id depends on the PDF alone: not on the page
range a layer is built for, nor on a page that failed. A kind is a page kind
or a document kind, never both (:data:`DOCUMENT_KINDS`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPAN = "sp"
LINE = "l"
BLOCK = "r"
FURNITURE = "f"
OUTLINE = "ol"

# The kinds of the ids that name no page, so that "r5" or "p3" is no id.
DOCUMENT_KINDS = frozenset({OUTLINE})

_KIND = re.compile(r"[a-z]+")
_ID = re.compile(r"p(0|[1-9][0-9]*)\.([a-z]+)(0|[1-9][0-9]*)")
_DOCUMENT_ID = re.compile(r"([a-z]+)((?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*)")


@dataclass(frozen=True, slots=True)
class LayerId:
    # None for a document id.
    page: int | None
    kind: str
    n: int
    # The numbers before ``n`` in a document id: the indexes of the levels above it.
    above: tuple[int, ...] = ()

    @property
    def path(self) -> tuple[int, ...]:
        """The numbers of a document id, first level first (the index alone for a page id)."""
        return (*self.above, self.n)

    def __str__(self) -> str:
        if self.page is None:
            return make_document(self.kind, *self.path)
        return make(self.page, self.kind, self.n)


def make(page: int, kind: str, n: int) -> str:
    if page < 0 or n < 0 or not _KIND.fullmatch(kind) or kind in DOCUMENT_KINDS:
        raise ValueError(f"no layer id for page {page!r}, kind {kind!r}, index {n!r}")
    return f"p{page}.{kind}{n}"


def make_document(kind: str, *path: int) -> str:
    """The id of the document object of *kind* at *path*, one index for each level of its tree."""
    if kind not in DOCUMENT_KINDS or not path or min(path) < 0:
        raise ValueError(f"no document id for kind {kind!r}, path {path!r}")
    return kind + ".".join(map(str, path))


def parse(layer_id: str) -> LayerId:
    """The page (None for a document id), kind and index *layer_id* names; ValueError when it is not an id."""
    match = _ID.fullmatch(layer_id)
    if match is not None and match.group(2) not in DOCUMENT_KINDS:
        page, kind, n = match.groups()
        return LayerId(int(page), kind, int(n))
    match = _DOCUMENT_ID.fullmatch(layer_id)
    if match is not None and match.group(1) in DOCUMENT_KINDS:
        *above, n = (int(part) for part in match.group(2).split("."))
        return LayerId(None, match.group(1), n, tuple(above))
    raise ValueError(f"not a layer id: {layer_id!r}")


def span(page: int, n: int) -> str:
    return make(page, SPAN, n)


def line(page: int, n: int) -> str:
    return make(page, LINE, n)


def block(page: int, n: int) -> str:
    return make(page, BLOCK, n)


def furniture(page: int, n: int) -> str:
    return make(page, FURNITURE, n)


def outline_entry(n: int) -> str:
    return make_document(OUTLINE, n)
