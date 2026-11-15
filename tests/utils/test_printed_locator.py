import pytest

from bibr.utils.printed_locator import PagePair, read_prefixed_locator


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        ("S163", "9", PagePair("S163", "S169")),
        ("e120", "5", PagePair("e120", "e125")),
        ("e00120", "5", PagePair("e00120", "e00125")),
        ("e00376", "18", PagePair("e00376-18", None)),
        ("e120", "20", PagePair("e120-20", None)),
        ("e120", "19", PagePair("e120-19", None)),
        ("1262", "1275", None),
        ("1262", "75", None),
        ("60", "72", None),
        ("e1", "e12", None),
        ("e12", "34", None),
        ("S10", "S12", None),
        ("e100", "e5", None),
    ],
)
def test_prefixed_locator(first, last, expected):
    assert read_prefixed_locator(first, last) == expected
