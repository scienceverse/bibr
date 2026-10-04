"""The outline guard: the frozen v1 rules, on records and on a layer's own text."""

from __future__ import annotations

import re

import pytest

from bibr.document import outline_guard
from bibr.document.harvest import build_document_layer
from bibr.document.model import OutlineEntry
from bibr.document.outline_guard import PageText, judge
from tests.document import _linked
from tests.document._guard_cases import CASES
from tests.document.test_layer import _BUDGET


class _Recorded:
    """A record's own grounding verdicts, standing in for the document's text."""

    def __init__(self, rec: dict) -> None:
        self._rec = rec

    @property
    def chars(self) -> int:
        return self._rec["text_chars"]

    def grounded(self, entry: OutlineEntry) -> bool:
        item = self._rec["items"][entry.idx]
        return item["g_page"] if item["page_no"] is not None else item["g_doc"]


def _entries(rec: dict) -> list[OutlineEntry]:
    return [
        OutlineEntry(
            idx=idx,
            parent=None,
            level=item["level"],
            title=item["title"],
            page=None if item["page_no"] is None else item["page_no"] - 1,
            x=None,
            y=None,
            dest_name=None,
        )
        for idx, item in enumerate(rec["items"])
    ]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_guard_gives_the_verdict_the_frozen_guard_gave(name):
    case = CASES[name]
    rec = case["rec"]
    verdict = judge(
        _entries(rec),
        meta_title=rec["meta_title"],
        n_pages=rec["n_pages"],
        text=_Recorded(rec),
    )

    assert verdict.passed is case["pass"]
    assert verdict.reject == case["reject"]
    titles = [item["title"] for item in rec["items"]]
    assert [[rule, titles[idx]] for idx, rule in verdict.dropped] == case["dropped"]
    kept = [title for idx, title in enumerate(titles) if idx not in {d[0] for d in verdict.dropped}]
    assert kept == case["kept"]
    assert verdict.decided.component == "outline_guard"
    assert verdict.decided.version == "outline_guard/1"
    assert verdict.decided.calibrated is False


def test_a_verdict_that_looked_at_the_text_carries_its_share_and_the_misses():
    rec = CASES["half_grounded_passes"]["rec"]
    verdict = judge(_entries(rec), meta_title=None, n_pages=10, text=_Recorded(rec))

    assert verdict.passed
    assert verdict.decided.score == 0.5
    assert verdict.decided.evidence == ("ol0", "ol3")


def test_a_verdict_that_did_not_look_at_the_text_has_no_score():
    for name in ("no_outline", "no_pages", "all_on_one_page", "ungrounded_below_the_text_gate"):
        rec = CASES[name]["rec"]
        verdict = judge(_entries(rec), meta_title=None, n_pages=10, text=_Recorded(rec))
        assert verdict.decided.score is None, name
        assert verdict.decided.evidence == (), name


def test_without_text_the_grounding_rule_is_not_applied():
    rec = CASES["ungrounded"]["rec"]
    verdict = judge(_entries(rec), meta_title=None, n_pages=10, text=None)

    assert verdict.passed and verdict.reject is None


def test_the_letters_and_digits_of_every_code_point_are_those_of_str_isalnum():
    every = "".join(chr(code) for code in range(0x110000))
    expected = "".join(char for char in every if char.isalnum())

    assert re.sub(r"[\W_]+", "", every) == expected
    assert outline_guard.alnum(None) == ""


def test_alnum_folds_to_the_evaluations_keys():
    # NFKC then case folding: the ligature, the sharp s and the circled digit fold away.
    assert outline_guard.alnum("Straße ﬁnal ① 12") == "strassefinal112"
    assert outline_guard.alnum("  --  ") == ""


# --- The guard on a layer's text ---------------------------------------------------


def _layer(**kwargs):
    return build_document_layer(
        _linked.linked_paper(**kwargs), range(_linked.N_PAGES), budget=_BUDGET
    )


def _bookmarks(*entries: tuple[str, int]) -> list[_linked.Bookmark]:
    return [_linked.Bookmark(title, 0, ("array", page, "/Fit")) for title, page in entries]


def test_the_fixture_outline_passes_and_every_entry_is_grounded():
    layer = _layer()
    guard = layer.outline_guard

    assert guard.passed and guard.reject is None
    assert guard.dropped == _linked.OUTLINE_DROPPED
    # R3 looked: the text layer holds over 2,000 letters and digits.
    assert guard.decided.score == 1.0
    assert guard.decided.evidence == ()
    assert layer.presence.outline_guard_pass is True


def test_titles_printed_nowhere_near_their_target_are_rejected_as_ungrounded():
    outline = _bookmarks(("Zebra", 0), ("Quokka", 1), ("Yak", 2), ("Walrus", 3))
    guard = _layer(outline=outline).outline_guard

    assert not guard.passed and guard.reject == "R3_ungrounded"
    assert guard.decided.score == 0.0
    assert guard.decided.evidence == ("ol0", "ol1", "ol2", "ol3")


def test_a_title_is_grounded_on_the_page_either_side_of_its_target_and_not_further():
    # "References" is printed on page 4.
    for page, grounded in ((0, False), (2, False), (3, True), (4, True), (5, True)):
        outline = _bookmarks(
            ("References", page), ("Abstract", 0), ("1 Introduction", 1), ("3 Results", 3)
        )
        guard = _layer(outline=outline).outline_guard
        assert ("ol0" in guard.decided.evidence) is (not grounded), page


def test_an_entry_without_a_page_is_grounded_by_the_title_anywhere():
    outline = [
        _linked.Bookmark("Abstract", 0, ("array", 0, "/Fit")),
        _linked.Bookmark("References", 0, ("number", 99)),
        _linked.Bookmark("Walrus", 0, ("number", 99)),
        _linked.Bookmark("1 Introduction", 0, ("array", 1, "/Fit")),
        _linked.Bookmark("3 Results", 0, ("array", 3, "/Fit")),
    ]
    guard = _layer(outline=outline).outline_guard

    assert guard.passed
    assert guard.decided.evidence == ("ol2",)


def test_every_entry_on_one_page_is_rejected_on_its_targets():
    outline = _bookmarks(("Abstract", 1), ("Methods", 1), ("Results", 1), ("References", 1))
    guard = _layer(outline=outline).outline_guard

    assert not guard.passed and guard.reject == "R2_targets"
    assert guard.decided.score is None


def test_a_paper_without_an_outline_is_rejected_as_too_short():
    layer = _layer(outline=None)

    assert layer.outline == []
    assert layer.outline_guard.reject == "R1_too_few"
    assert layer.presence.has_outline is False
    assert layer.presence.outline_guard_pass is False


def test_a_page_range_grounds_titles_in_its_own_pages_only():
    outline = _bookmarks(("Zebra", 0), ("Quokka", 1), ("Yak", 2), ("Walrus", 3))
    layer = build_document_layer(_linked.linked_paper(outline=outline), [0, 1], budget=_BUDGET)

    # The two pages hold under 2,000 letters and digits: the grounding rule is not applied.
    assert layer.outline_guard.passed
    assert layer.outline_guard.decided.score is None


def test_page_text_is_the_records_folded_to_letters_and_digits():
    layer = _layer()
    text = PageText(layer.pages, _linked.N_PAGES)

    assert text.folded[0].startswith("linkedpaperfixture2026abstract")
    assert text.folded[5] == "appendixsupplementarymaterial"
    assert text.chars == sum(len(page) for page in text.folded)
    assert text.chars >= 2000
    assert text.whole == "".join(text.folded)


def test_a_page_the_layer_lacks_has_no_text():
    layer = build_document_layer(_linked.linked_paper(), [5], budget=_BUDGET)
    text = PageText(layer.pages, _linked.N_PAGES)

    assert text.folded == [""] * 5 + ["appendixsupplementarymaterial"]
