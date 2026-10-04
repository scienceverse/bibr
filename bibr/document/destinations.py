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

# A document with more named destinations than this keeps its names unread, so
# a hostile name tree cannot hold the lock for long.
MAX_NAMED_DESTS = 100_000

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
    includes it, so an absent string is 0 bytes and an empty one is 2.
    """
    size = function(*args, None, 0)
    if size <= 0:
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
        for index in range(self.count):
            # pdfium leaves the size alone when the index has no destination.
            size = ctypes.c_long(0)
            if not api.FPDF_GetNamedDest(self._doc.raw, index, None, ctypes.byref(size)):
                continue
            if size.value <= 2:
                continue
            buffer = ctypes.create_string_buffer(size.value)
            dest = api.FPDF_GetNamedDest(self._doc.raw, index, buffer, ctypes.byref(size))
            if dest:
                name = buffer.raw[: size.value - 2].decode("utf-16-le", "replace")
                found.setdefault(address(dest), name)
                self._by_name.setdefault(name, dest)
        return found


def dest_page(api: _Api, doc, dest, n_pages: int) -> int | None:
    """The 0-based page *dest* points at, or None when it names no page of the document."""
    index = api.FPDFDest_GetDestPageIndex(doc.raw, dest)
    return int(index) if 0 <= index < n_pages else None


def dest_position(api: _Api, dest) -> Position:
    """The ``(x, y)`` a destination gives on its page, in PDF points; an open coordinate is None.

    ``[page /XYZ x y zoom]`` says which of x and y it leaves open. The Fit
    variants fix one coordinate (FitH and FitBH the top, FitV and FitBV the
    left, FitR the left and top of its rectangle); Fit and FitB fix none.
    """
    pdfium_c = api.c
    has_x, has_y, has_zoom = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    x, y, zoom = ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0)
    if api.FPDFDest_GetLocationInPage(dest, has_x, has_y, has_zoom, x, y, zoom):
        return (x.value if has_x.value else None, y.value if has_y.value else None)
    count = ctypes.c_ulong(0)
    params = (ctypes.c_float * 4)()
    mode = api.FPDFDest_GetView(dest, count, params)
    n = count.value
    if mode in (pdfium_c.PDFDEST_VIEW_FITH, pdfium_c.PDFDEST_VIEW_FITBH) and n >= 1:
        return (None, params[0])
    if mode in (pdfium_c.PDFDEST_VIEW_FITV, pdfium_c.PDFDEST_VIEW_FITBV) and n >= 1:
        return (params[0], None)
    if mode == pdfium_c.PDFDEST_VIEW_FITR and n >= 4:
        return (params[0], params[3])
    return (None, None)
