"""Reading a printed "<first>-<last>" locator as a page range or an article number."""

import re
from typing import NamedTuple

_PREFIXED_FIRST = re.compile(r"([A-Za-z]+)(\d+)")
_BARE_NUMBER = re.compile(r"\d+")


class PagePair(NamedTuple):
    """The first and last page of a locator; ``last`` is None for an article number."""

    first: str
    last: str | None


def read_prefixed_locator(first: str, last: str) -> PagePair | None:
    """Read a letter-prefixed first part followed by a shorter bare number.

    Journals abbreviate the end of a range ("S163-9" for S163-S169, "e120-5"
    for e120-e125) and also print article numbers followed by a year suffix
    ("e00376-18": article e00376, 2018). Both have a letter-prefixed first
    part and a shorter bare second part, so the second part is expanded with
    the leading digits of the first number. When the expanded number is larger
    it is a range and the last page keeps the prefix ("S169"); otherwise the
    whole locator is one article number and stays the first page.

    None when the pair is neither: the first part has no letter prefix, the
    second part is not a bare number, or it has as many digits as the first
    part's number (both parts then read as a plain range, as before).
    """
    opening = _PREFIXED_FIRST.fullmatch(first)
    if opening is None or _BARE_NUMBER.fullmatch(last) is None:
        return None
    prefix, digits = opening.groups()
    if len(last) >= len(digits):
        return None
    expanded = digits[: len(digits) - len(last)] + last
    if int(expanded) > int(digits):
        return PagePair(first, prefix + expanded)
    return PagePair(f"{first}-{last}", None)
