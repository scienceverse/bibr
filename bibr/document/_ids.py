"""The ids D2's records carry, built in one place.

Every id is a pure function of where the thing is in the PDF, so it does not
move when the layer is built for another page range. D1's ``ids`` module takes
over once it lands; these two functions are what it replaces.
"""

from __future__ import annotations


def page_id(kind: str, page: int, number: int) -> str:
    """``p3.lk4``: the *number*-th thing of *kind* on 0-based *page*."""
    return f"p{page}.{kind}{number}"


def document_id(kind: str, number: int) -> str:
    """``ol5``: the *number*-th thing of *kind* in the document."""
    return f"{kind}{number}"
