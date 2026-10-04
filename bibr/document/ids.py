"""Ids of layer objects: ``p{page}.{kind}{n}``.

An id names one object on one page: the absolute 0-based page index, a
lowercase kind prefix, and the object's index among that page's objects of
the kind. The same PDF and page range always give the same ids. D1's kinds
are spans ``sp`` (rows of ``PageColumns.span_rec``), lines ``l``
(``PageColumns.line_span``), blocks ``r`` (the page's post-OCR regions, in
order) and furniture ``f`` (``Page.furniture``). :func:`make` and
:func:`parse` take any lowercase prefix, so later kinds need no new parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPAN = "sp"
LINE = "l"
BLOCK = "r"
FURNITURE = "f"

_KIND = re.compile(r"[a-z]+")
_ID = re.compile(r"p(0|[1-9][0-9]*)\.([a-z]+)(0|[1-9][0-9]*)")


@dataclass(frozen=True, slots=True)
class LayerId:
    page: int
    kind: str
    n: int

    def __str__(self) -> str:
        return make(self.page, self.kind, self.n)


def make(page: int, kind: str, n: int) -> str:
    if page < 0 or n < 0 or not _KIND.fullmatch(kind):
        raise ValueError(f"no layer id for page {page!r}, kind {kind!r}, index {n!r}")
    return f"p{page}.{kind}{n}"


def parse(layer_id: str) -> LayerId:
    """The page, kind and index *layer_id* names; ValueError when it is not an id."""
    match = _ID.fullmatch(layer_id)
    if match is None:
        raise ValueError(f"not a layer id: {layer_id!r}")
    page, kind, n = match.groups()
    return LayerId(int(page), kind, int(n))


def span(page: int, n: int) -> str:
    return make(page, SPAN, n)


def line(page: int, n: int) -> str:
    return make(page, LINE, n)


def block(page: int, n: int) -> str:
    return make(page, BLOCK, n)


def furniture(page: int, n: int) -> str:
    return make(page, FURNITURE, n)
