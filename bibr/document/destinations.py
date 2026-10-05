"""Destinations: the document's named destinations and where a destination points.

The outline and the link reader share these. pdfium resolves a name to its
destination array before a bookmark or link hands it out, so the name is lost
with it. A destination reached by name is the very array the name table holds,
so the array's address finds the name again, exactly: on gate192 that is true
for every one of the 6,887 links whose annotation names its destination.

Everything here calls pdfium and needs the caller's ``pdfium_lock``.
"""

from __future__ import annotations

import ctypes
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.document.harvest import _Api

APIS = (
    "FPDF_CountNamedDests",
    "FPDF_GetNamedDest",
    "FPDFDest_GetDestPageIndex",
    "FPDFDest_GetLocationInPage",
    "FPDFDest_GetView",
)

# A document with more named destinations than this keeps its names unread. pdfium walks
# the name tree from its root for every name it is asked for, so reading the table costs
# the square of the count, and the count is the only bound on the time: 2,000 names take
# 0.04 s in a tree shaped like pdfTeX's (six names to a leaf, six leaves to a node) and
# 0.24 s in the worst shape (one name to a leaf under one root). The most on gate192 is
# 607 names, and its p90 is 199.
MAX_NAMED_DESTS = 2_000
# A string over this many bytes reads as absent, so a hostile one is never copied whole
# (the longest URI or alt text on gate192 and the manuscripts is 872 characters).
MAX_TEXT = 1 << 16

Position = tuple[float | None, float | None]

# What a destination points at is read in a band of the target page: from this many
# points above the destination's y to this many below it, by the top edge of what lies
# there (y grows upwards). Tools put a destination a little above its target.
BAND_ABOVE = 6.0
BAND_BELOW = 40.0


def address(pointer) -> int:
    return ctypes.addressof(pointer.contents)


def utf16_text(function, *args) -> str | None:
    """The string a pdfium getter of ``(*args, buffer, size)`` returns, or None if it is absent.

    The getters return UTF-16LE with a terminating NUL, and their size
    includes it, so an absent string is 0 bytes and an empty one is 2. A string
    over :data:`MAX_TEXT` bytes is absent too.
    """
    size = function(*args, None, 0)
    if size <= 0 or size > MAX_TEXT:
        return None
    # An array of 16-bit units suits the getters that type their buffer as void*
    # and those that type it as ushort*.
    buffer = (ctypes.c_ushort * ((size + 1) // 2))()
    function(*args, buffer, size)
    return bytes(buffer)[: max(size - 2, 0)].decode("utf-16-le", "replace")


class NamedDests:
    """The document's named destinations, found again by the destination they hold.

    The names are read on first use: a document that no link or bookmark
    reaches by name never pays for them.
    """

    def __init__(self, api: _Api, doc) -> None:
        self._api = api
        self._doc = doc
        self.count = int(api.FPDF_CountNamedDests(doc.raw))
        self.error: str | None = None
        self._by_address: dict[int, str] | None = None
        self._by_name: dict[str, Any] = {}

    def name_of(self, dest) -> str | None:
        """The name whose destination is *dest* (the first, if several share it)."""
        if not dest:
            return None
        if self._by_address is None:
            self._by_address = self._read()
        return self._by_address.get(address(dest))

    def dest_of(self, name: str):
        """The destination the table gives *name*, or None.

        pdfium's own lookup by name searches the name tree and misses names a tree
        not sorted as it expects holds (a non-ASCII name among them), while the
        table lists all of them.
        """
        if self._by_address is None:
            self._by_address = self._read()
        return self._by_name.get(name)

    def _read(self) -> dict[int, str]:
        found: dict[int, str] = {}
        if self.count > MAX_NAMED_DESTS:
            self.error = f"{self.count} named destinations, over the limit of {MAX_NAMED_DESTS}"
            return found
        api = self._api
        # One call for each name, into a buffer that holds the longest name read: asking
        # for the size first would walk the name tree twice.
        buffer = ctypes.create_string_buffer(MAX_TEXT)
        for index in range(self.count):
            size = ctypes.c_long(MAX_TEXT)
            dest = api.FPDF_GetNamedDest(self._doc.raw, index, buffer, ctypes.byref(size))
            # No destination at the index: nothing. A name too long for the buffer: pdfium
            # still returns the destination, with a size of -1. The size includes the NUL.
            if not dest or not 2 < size.value <= MAX_TEXT:
                continue
            name = buffer[: size.value - 2].decode("utf-16-le", "replace")
            found.setdefault(address(dest), name)
            self._by_name.setdefault(name, dest)
        return found


def dest_page(api: _Api, doc, dest, n_pages: int) -> int | None:
    """The 0-based page *dest* points at, or None when it names no page of the document."""
    index = api.FPDFDest_GetDestPageIndex(doc.raw, dest)
    return int(index) if 0 <= index < n_pages else None


def finite(value: float) -> float | None:
    """*value*, or None when it is not finite: pdfium reads a number too large for a float as infinity."""
    return value if math.isfinite(value) else None


def dest_position(api: _Api, dest) -> Position:
    """The ``(x, y)`` a destination gives on its page, in PDF points; an open coordinate is None.

    ``[page /XYZ x y zoom]`` says which of x and y it leaves open. The Fit
    variants fix one coordinate (FitH and FitBH the top, FitV and FitBV the
    left, FitR the left and top of its rectangle); Fit and FitB fix none. A
    coordinate that is not finite is open too.
    """
    pdfium_c = api.c
    has_x, has_y, has_zoom = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    x, y, zoom = ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0)
    if api.FPDFDest_GetLocationInPage(dest, has_x, has_y, has_zoom, x, y, zoom):
        return (finite(x.value) if has_x.value else None, finite(y.value) if has_y.value else None)
    count = ctypes.c_ulong(0)
    params = (ctypes.c_float * 4)()
    mode = api.FPDFDest_GetView(dest, count, params)
    n = count.value
    if mode in (pdfium_c.PDFDEST_VIEW_FITH, pdfium_c.PDFDEST_VIEW_FITBH) and n >= 1:
        return (None, finite(params[0]))
    if mode in (pdfium_c.PDFDEST_VIEW_FITV, pdfium_c.PDFDEST_VIEW_FITBV) and n >= 1:
        return (finite(params[0]), None)
    if mode == pdfium_c.PDFDEST_VIEW_FITR and n >= 4:
        return (finite(params[0]), finite(params[3]))
    return (None, None)
