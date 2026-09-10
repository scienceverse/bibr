"""The in-text citation counter must see accented surnames (Müller, Lévy)."""

from bibr.pipeline.stages.post_parse import _count_distinct_intext_citations


def test_counts_accented_surnames():
    text = "As Müller (2020) showed, and Lévy (2019) confirmed, the effect holds."
    # distinct (surname, year): (müller, 2020), (lévy, 2019)
    assert _count_distinct_intext_citations(text) == 2


def test_ascii_surnames_unchanged():
    text = "Smith (2020) and Jones (2019) both reported it; Smith (2020) again."
    assert _count_distinct_intext_citations(text) == 2


def test_curly_apostrophe_surname_counted():
    # O'Brien with a real right-curly apostrophe (U+2019) — a regression guard:
    # this was supported before the accent fix and must stay supported.
    text = "O" + chr(0x2019) + "Brien (2018) noted this."
    assert _count_distinct_intext_citations(text) == 1


def test_name_char_cls_has_three_distinct_apostrophes():
    # Locks in the fix for the glyph-normalization defect: the class must carry
    # straight + both curly apostrophes, not three straight ones.
    from bibr.utils.text import NAME_CHAR_CLS

    cps = {ord(c) for c in NAME_CHAR_CLS}
    assert {0x27, 0x2018, 0x2019} <= cps
