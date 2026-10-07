"""Citation linker: linear regexes and bookkeeping, and receipt offsets that index the text."""

import random
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.export.json_export import export_paper_to_json
from bibr.input.file import InputFile, InputFormat
from bibr.models import PaperMetadata
from bibr.paper import Paper, PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    CitationCandidate,
    CitationLinkingReceipt,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.structure import citation_linker
from tests.export.conftest import extraction_block


def _ref(bib_id, authors=None, year=2020, text_id=None):
    return PaperReference(
        bib_id=bib_id,
        title=f"Title {bib_id}",
        first_page=None,
        volume=None,
        authors=authors,
        year=year,
        container=None,
        text_id=text_id,
    )


def _sections():
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Intro",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        ),
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]


def _sent(text_id, text, section_id=1):
    return PaperSentence(text_id=text_id, text=text, section_id=section_id, paragraph_id=1)


def _elapsed(call):
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


def _random_strings(alphabet, count, max_len, seed):
    rng = random.Random(seed)  # noqa: S311 - deterministic test fixture, not crypto
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len))) for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# Flattened superscripts: anchored at the start of a word run
# ---------------------------------------------------------------------------

_FLATTENED_BEFORE = re.compile(
    r"([a-zA-Z][a-zA-Z\)\]]*[.,;:]?)(\d{1,3}(?:[,–-]\d{1,3})*)(?=[\s.,;:)\]]|$)"
)


def _flattened_matches(pattern, text):
    return [(m.group(1), m.group(2), m.start(1), m.end(2), m.end()) for m in pattern.finditer(text)]


def test_flattened_superscript_matches_are_unchanged():
    shapes = [
        "traits1 and characteristics2,3 here.",
        "rehabilitation,1 effectiveness.7,10,11 end",
        "1)bar2 and a)b]c3 and ))x4.",
        "ACGT ACGT1 x2y3 Fig.3 COVID19 (BERT12) [ab5]",
        "",
    ]
    shapes += _random_strings("aZb)].,;: 12-–x([", 3000, 24, seed=7)
    pattern = citation_linker.FLATTENED_SUP_CITE_RE
    for text in shapes:
        assert _flattened_matches(pattern, text) == _flattened_matches(_FLATTENED_BEFORE, text)


@pytest.mark.parametrize("run", ["ACGT" * 10_000, "a)" * 20_000])
def test_flattened_superscript_scan_is_linear_in_a_long_word_run(run):
    # Tried from every letter, the run was rescanned to its end each time:
    # tens of seconds for these 40,000 characters.
    elapsed, matches = _elapsed(
        lambda: list(citation_linker.FLATTENED_SUP_CITE_RE.finditer(f"was {run} as reported"))
    )

    assert matches == []
    assert elapsed < 2.0


async def test_long_sequence_sentence_links_without_stalling():
    sentence = _sent(1, "The sequence was " + "ACGT" * 10_000 + " as reported [1].")

    started = time.perf_counter()
    xrefs, _receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        [sentence], _sections(), [_ref(1)]
    )

    assert time.perf_counter() - started < 2.0
    assert [(x.xref_id, x.contents) for x in xrefs] == [(1, "[1]")]


# ---------------------------------------------------------------------------
# Parenthetical numbers: label guards read only the label
# ---------------------------------------------------------------------------

_ADJACENT_LABEL_BEFORE = re.compile(r"(?:[A-Za-z]{2,}[)\]]?\d+|[A-Z][A-Z0-9-]{2,}\s+\d+)\s*$")


def _paren_reasons(text):
    candidates, _score = citation_linker._parenthetical_candidates(
        [_sent(1, text)], set(range(1, 50)), frozenset(), set(range(1, 50))
    )
    return [(c.raw, c.rejection_reasons) for c in candidates]


def test_adjacent_label_guard_matches_the_forward_pattern():
    reversed_label = citation_linker._ADJACENT_NUMERIC_LABEL_REVERSED_RE
    befores = _random_strings("AB c1-2 )]\n", 4000, 16, seed=11)
    befores += ["A" + "-" * 300 + " 5 ", "IPCAT" + " " * 300 + "213 ", "x" + "1" * 300]
    for before in befores:
        assert bool(reversed_label.match(before[::-1])) == bool(
            _ADJACENT_LABEL_BEFORE.search(before)
        ), before


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Cohort GLASS63 (7) grew.", True),
        ("The IPCAT 213 (11) panel.", True),
        # Labels longer than any fixed tail are still read in full.
        ("The A" + "-" * 300 + " 5 (5) panel.", True),
        ("The IPCAT" + " " * 300 + "213 (11) panel.", True),
        ("As reported (7) before.", False),
        ("In x 12 (3) cases.", False),
    ],
)
def test_adjacent_label_guard_still_applies(text, label):
    [(_raw, reasons)] = _paren_reasons(text)

    assert ("adjacent_numeric_label" in reasons) is label


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Then in step \n (2) we rinsed.", "list_enumeration"),
        ("Each criterion   (4) applied.", "list_enumeration"),
        (
            "The total number of participants in the subdomain  (12) was small.",
            "parenthetical_count",
        ),
    ],
)
def test_label_words_before_trailing_whitespace_are_still_found(text, reason):
    [(_raw, reasons)] = _paren_reasons(text)

    assert reason in reasons


def test_parenthetical_guards_are_linear_in_the_sentence():
    # A 20,000-letter run read in full once per candidate: 84 s before.
    long_run = "Sequence " + "ACGT" * 5_000 + " end" + " see (1)" * 10 + "."
    # 4,000 candidates in one sentence, each reading the whole prefix and
    # renumbering every candidate: about 30 s before.
    many = "See " + "x (1) " * 4_000 + "."

    elapsed, results = _elapsed(lambda: [_paren_reasons(long_run), _paren_reasons(many)])

    assert [len(result) for result in results] == [10, 4_000]
    assert elapsed < 2.0


def test_parenthetical_guards_read_the_sentence_in_place():
    # Each candidate copied and stripped the whole text before it: 8,000
    # candidates after a 4,000,000-letter run took about 48 s.
    text = "Sequence " + "ACGT" * 1_000_000 + " (1)" * 8_000 + "."

    elapsed, results = _elapsed(lambda: _paren_reasons(text))

    assert len(results) == 8_000
    assert elapsed < 2.0


class _CountingCandidate(CitationCandidate):
    """A candidate that counts reads of its ``text_id`` and ``rejection_reasons``."""

    reads: Counter = Counter()

    def __getattribute__(self, name):
        if name in ("text_id", "rejection_reasons"):
            _CountingCandidate.reads[name] += 1
        return super().__getattribute__(name)


@pytest.fixture
def counting_candidates(monkeypatch):
    _CountingCandidate.reads = Counter()
    monkeypatch.setattr(citation_linker, "CitationCandidate", _CountingCandidate)
    return _CountingCandidate.reads


def test_recurring_sentence_count_is_computed_once(counting_candidates):
    sentences = [
        _sent(index, "Prior studies reported this effect (12) and replicated it (7, 8).")
        for index in range(1_000)
    ]

    candidates, _score = citation_linker._parenthetical_candidates(
        sentences, set(range(1, 50)), frozenset(), set(range(1, 50))
    )

    assert len(candidates) == 2_000
    assert {c.evidence[1] for c in candidates} == {"recurring_sentences:1000"}
    # Recounted per candidate it read every candidate again: 2,000 x 2,000.
    assert counting_candidates["rejection_reasons"] < 10 * len(candidates)


# ---------------------------------------------------------------------------
# Tier-3 (LLM path) patterns and bookkeeping
# ---------------------------------------------------------------------------

_NARRATIVE_BEFORE = re.compile(
    r"([A-Z][a-zA-ZÀ-ɏ'\-]+"
    r"(?:\s+(?:&|and)\s+[A-Z][a-zA-ZÀ-ɏ'\-]+)?"
    r"(?:\s+et\s+al\.?)?)"
    r"\s*\((\d{4}[a-z]?)\)"
)
_NON_CITATION_BRACKET_BEFORE = re.compile(
    r"^\s*(?:"
    r"[-+]?\s*\.?\d+(?:\.\d+)?\s*[,;–\-]\s*[-+]?\s*\.?\d+(?:\.\d+)?\s*"
    r"|item\s+\d+"
    r"|(?-i:[A-Z]{2,5})"
    r"|(?-i:[a-z])[A-Za-z\s\-']*"
    r"|[A-Za-z][A-Za-z\-']*\s[A-Za-z\s\-']+"
    r")\s*$",
    re.IGNORECASE,
)


def test_narrative_author_year_matches_real_names_as_before():
    texts = [
        "Smith (2020) and Smith and Jones (2019) and Smith et al. (2018a) agree.",
        "O'Brien-Smith & Lee (2001), deVries (2020), d'Alembert (1750), Müller (1999).",
        "Wolfeschlegelsteinhausenbergerdorff (1990) wrote this.",
    ]
    texts += _random_strings("Ab é'-& and et al. (2020)", 2000, 30, seed=5)
    pattern = citation_linker.NARRATIVE_AUTHOR_YEAR_RE
    for text in texts:
        assert [m.span() for m in pattern.finditer(text)] == [
            m.span() for m in _NARRATIVE_BEFORE.finditer(text)
        ], text


def test_non_citation_bracket_shapes_are_unchanged():
    contents = [".59, .72", "-.12, .04", "Item 5", "IRR", "sic", "Emphasis Added", "Smith", "a1"]
    contents += _random_strings(" \t\n12.,;-–+aZ'x", 4000, 14, seed=13)
    contents += _random_strings(["item", "IRR", " ", "1", ",", "a", "b"], 2000, 8, seed=17)
    for content in contents:
        assert citation_linker._looks_like_non_citation_bracket(content) == bool(
            _NON_CITATION_BRACKET_BEFORE.match(content)
        ), content


def test_bracket_spans_are_unchanged():
    texts = ["[1] [a [b] c] []] [x", "[[[ ]] [sic] [Ab1; Cd2]"]
    texts += _random_strings("[]a ;", 3000, 20, seed=19)
    for text in texts:
        assert [m.span() for m in citation_linker._bracket_spans(text)] == [
            m.span() for m in re.finditer(r"\[([^\]]+)\]", text)
        ], text


@pytest.mark.parametrize(
    ("label", "call"),
    [
        (
            "narrative",
            lambda: list(citation_linker.NARRATIVE_AUTHOR_YEAR_RE.finditer("A" * 30_000)),
        ),
        (
            "bracket shape",
            lambda: citation_linker._looks_like_non_citation_bracket("a" + " " * 60_000 + "1"),
        ),
        (
            "interval shape",
            lambda: citation_linker._looks_like_non_citation_bracket("1, " + " " * 60_000 + "x"),
        ),
        ("unclosed brackets", lambda: list(citation_linker._bracket_spans("[" * 80_000))),
    ],
)
def test_tier3_patterns_are_linear(label, call):
    # Each took 20 s or more on these inputs before.
    elapsed, _result = _elapsed(call)

    assert elapsed < 2.0, label


class _FakeLlm:
    def __init__(self, picks):
        self.picks = picks
        self.asked = []

    async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash):
        self.asked.extend(ambiguous_citations)
        return [
            SimpleNamespace(bib_id=bib_id, citation_text=text, text_id=text_id)
            for text_id, text in ambiguous_citations
            for bib_id in self.picks(text)
        ]


async def test_tier3_bookkeeping_keeps_its_results():
    refs = [
        _ref(1, "Smith, A.", 2020),
        _ref(2, "Smith, B.", 2020),
        _ref(3, "Jones, C.", 2019),
        _ref(4, "Brown, D.", 2018),
        _ref(5, "Kim, E.", 2017),
    ]
    texts = [
        "As shown by Smith (2020) and [Ab1], effects hold.",
        "Earlier work [Ab1; Cd2] and (Brown & Lee, 2018; Kim, 2017) agreed.",
        "Reported in Jones and Park (2019) with Miller et al. (2016).",
        "See [Zz9] and [Zz9] again, also (Garcia, 2015).",
        "Prior [see Smith (2020)] noted (Brown, 2018; Zed, 2014).",
    ]
    picks = {
        "Smith (2020)": [1],
        "[Ab1]": [3],
        "[Ab1; Cd2]": [3, 4],
        "Jones and Park (2019)": [3],
        "Miller et al. (2016)": [5],
        "[Zz9]": [4, 5],
        "(Garcia, 2015)": [2],
        "[see Smith (2020)]": [2],
        "Zed, 2014": [5],
    }
    llm = _FakeLlm(lambda text: picks.get(text, []))

    xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        [_sent(index, text) for index, text in enumerate(texts, start=1)],
        _sections(),
        refs,
        llm_client=llm,
    )

    # "Smith (2020)" inside "[see Smith (2020)]" is nested, so not asked about.
    assert sorted(llm.asked) == [
        (1, "Smith (2020)"),
        (1, "[Ab1]"),
        (2, "[Ab1; Cd2]"),
        (3, "Miller et al. (2016)"),
        (4, "(Garcia, 2015)"),
        (4, "[Zz9]"),
        (5, "Zed, 2014"),
        (5, "[see Smith (2020)]"),
    ]
    assert [(x.text_id, x.xref_id, x.tier, x.start, x.end) for x in xrefs] == [
        (2, 4, "author-year", 29, 46),
        (2, 5, "author-year", 48, 57),
        (3, 3, "author-year", 12, 33),
        (5, 4, "author-year", 32, 43),
        (1, 1, "llm", 12, 24),
        (3, 5, "llm", 39, 59),
        (4, 2, "llm", 32, 46),
        (5, 5, "llm", 45, 54),
        (1, 3, "llm", 29, 34),
        (2, 3, "llm", 13, 23),
        (4, 4, "llm", 4, 9),
        (4, 5, "llm", 4, 9),
        (5, 2, "llm", 6, 24),
    ]
    assert [
        (c.text_id, c.style, c.start, c.end, c.bib_ids, c.accepted, c.rejection_reasons)
        for c in receipt.candidates
        if c.style in ("llm", "author-year")
    ] == [
        (1, "author-year", 12, 24, (1,), True, ()),
        (2, "author-year", 29, 46, (4,), True, ()),
        (2, "author-year", 48, 57, (5,), True, ()),
        (3, "author-year", 12, 33, (3,), True, ()),
        (3, "author-year", 39, 59, (5,), True, ()),
        (4, "author-year", 32, 46, (2,), True, ()),
        (5, "author-year", 32, 43, (4,), True, ()),
        (5, "author-year", 45, 54, (5,), True, ()),
        (5, "author-year", 11, 23, (1, 2), False, ("ambiguous_same_surname_year",)),
        (1, "llm", 29, 34, (3,), True, ()),
        (2, "llm", 13, 23, (3, 4), True, ()),
        (4, "llm", 4, 9, (4, 5), True, ()),
        (4, "llm", 14, 19, (4, 5), True, ()),
        (5, "llm", 6, 24, (2,), True, ()),
    ]


async def test_tier3_bookkeeping_is_linear_in_the_candidates(counting_candidates):
    sentence = _sent(1, " ".join(f"[Ab{number}]" for number in range(1_000)))
    llm = _FakeLlm(lambda text: [1])

    xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        [sentence], _sections(), [_ref(1, "Smith, A.")], llm_client=llm
    )

    assert len(llm.asked) == 1_000
    assert len(xrefs) == 1
    assert sum(c.accepted for c in receipt.candidates) == 1_000
    # Each ambiguous citation scanned every candidate, twice: 1,000 x 1,000.
    assert counting_candidates["text_id"] < 20 * len(receipt.candidates)


# ---------------------------------------------------------------------------
# Receipt offsets index the exported sentence
# ---------------------------------------------------------------------------


def _numbered(texts, count=7):
    """Body *texts* with a printed, numbered reference list of *count* rows."""
    refs = [_ref(n, text_id=10_000 + n) for n in range(1, count + 1)]
    rows = [
        _sent(10_000 + n, f"[{n}] Smith AB. Printed reference {n}.", section_id=2)
        for n in range(1, count + 1)
    ]
    return [_sent(index, text) for index, text in enumerate(texts, start=1)] + rows, refs


def _offsets(receipt, text_id, text):
    return sorted(
        (
            (c.raw, c.start, c.end, text[c.start : c.end])
            for c in receipt.candidates
            if c.text_id == text_id
        ),
        key=lambda row: row[1:3],
    )


def _assert_indexes(rows, text):
    """Each candidate prints its raw text at its span, or has an empty span."""
    for raw, start, end, printed in rows:
        assert 0 <= start <= end <= len(text)
        assert printed in (" ".join(raw.split()), ""), (raw, start, end)


async def test_stripping_superscripts_moves_the_receipt_offsets():
    sentences, refs = _numbered(
        ["Prior work^{3} showed this and later work^{4,5} agreed, see also [6]."]
    )
    _xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        sentences, _sections(), refs
    )

    receipt = citation_linker.strip_citation_superscripts(sentences, _sections(), receipt)

    text = sentences[0].text
    assert text == "Prior work showed this and later work agreed, see also [6]."
    # A removed marker keeps an empty span where it stood.
    assert _offsets(receipt, 1, text) == [
        ("^{3}", 10, 10, ""),
        ("^{4,5}", 37, 37, ""),
        ("[6]", 55, 58, "[6]"),
    ]


async def test_receipt_follows_late_text_cleaning():
    sentences, refs = _numbered(
        ["  Effects were $\\alpha$ large $ ^{2} $ and  [6]  and [6] again^{9}."]
    )
    contents = PaperContents(
        sentences=sentences, sections=_sections(), tables=[], links=[], sections_text={}
    )
    _xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        sentences, _sections(), refs
    )
    receipt = citation_linker.strip_citation_superscripts(sentences, _sections(), receipt)
    contents.finalize_text()

    receipt = citation_linker.reanchor_citation_receipt(
        receipt, {sentence.text_id: sentence.text for sentence in sentences}
    )

    text = sentences[0].text
    assert text == "Effects were α large and [6] and [6] again9."
    rows = _offsets(receipt, 1, text)
    _assert_indexes(rows, text)
    # Each print of "[6]" goes to its own candidate. "$ ^{2} $" was removed
    # by the stripping and "^{9}" (rejected) rewritten to "9" by the
    # cleaning: both get an empty span, in the order of the text.
    assert [row[0] for row in rows] == ["$ ^{2} $", "[6]", "[6]", "^{9}"]
    assert [row[1:3] for row in rows if row[3]] == [(25, 28), (33, 36)]
    assert rows[0][1] <= 25 and rows[-1][1] >= 36


def test_reanchoring_takes_the_nearest_print_of_bare_digits():
    text = "In 2020 the traits2 grew 2.5 fold."
    candidate = CitationCandidate(
        text_id=1,
        start=21,
        end=22,
        raw="2",
        style="flattened-superscript",
        bib_ids=(2,),
        evidence=(),
        confidence=0.0,
        accepted=False,
        rejection_reasons=(),
    )
    receipt = CitationLinkingReceipt(
        style_scores={},
        candidates=(candidate,),
        resolved_candidate_fraction=None,
        unique_linked_bib_fraction=None,
    )

    [moved] = citation_linker.reanchor_citation_receipt(receipt, {1: text}).candidates

    assert (moved.start, moved.end) == (18, 19)


async def _linked_and_cleaned(texts, refs=None):
    """Link, strip and clean *texts*; return the final texts and the re-anchored receipt."""
    sentences, numbered_refs = _numbered(texts)
    contents = PaperContents(
        sentences=sentences, sections=_sections(), tables=[], links=[], sections_text={}
    )
    _xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        sentences, _sections(), refs or numbered_refs
    )
    receipt = citation_linker.strip_citation_superscripts(sentences, _sections(), receipt)
    contents.finalize_text()
    texts = {sentence.text_id: sentence.text for sentence in sentences}
    return texts, citation_linker.reanchor_citation_receipt(receipt, texts)


@pytest.mark.parametrize(
    "text",
    [
        "In $\\alpha$-treated mice, CD4 [4] cells expanded.",
        "The $\\alpha$ subunit of cells2 [2] was expressed.",
        "$\\beta$see$ ^{2} $$\\beta$,see2.5\n[2, 4][2, 4]cells3and",
    ],
    ids=["CD4", "cells2", "repeated"],
)
async def test_glued_digits_do_not_take_the_print_of_a_later_citation(text):
    # The digits glued to "CD4" were re-anchored onto the "4" inside "[4]"
    # and moved the search for "[4]" past its print: an accepted citation
    # the sentence still prints got an empty span.
    texts, receipt = await _linked_and_cleaned([text])

    final = texts[1]
    rows = _offsets(receipt, 1, final)
    _assert_indexes(rows, final)
    citations = [c for c in receipt.candidates if c.text_id == 1 and c.raw.startswith("[")]
    assert citations and all(c.accepted for c in citations)
    assert all(final[c.start : c.end] == c.raw for c in citations)
    assert sorted((c.start, c.end) for c in citations) == sorted(
        {(m.start(), m.end()) for m in re.finditer(r"\[[^\]]+\]", final)}
    )


async def test_glued_digits_point_at_the_digits_after_their_word():
    texts, receipt = await _linked_and_cleaned(
        ["In $\\alpha$-treated mice, CD4 [4] cells expanded."]
    )

    assert texts[1] == "In α-treated mice, CD4 [4] cells expanded."
    assert _offsets(receipt, 1, texts[1]) == [("4", 21, 22, "4"), ("[4]", 23, 26, "[4]")]


async def test_repeated_citation_keeps_its_own_print():
    # Cleaning deleted exactly as much before the first "[6]" as separates
    # the two: its old offset held the second print, which it took, and the
    # second "[6]" was left with an empty span.
    texts, receipt = await _linked_and_cleaned(["Both $\\alpha$ studies [6] or [6] agreed."])

    assert texts[1] == "Both α studies [6] or [6] agreed."
    assert _offsets(receipt, 1, texts[1]) == [("[6]", 15, 18, "[6]"), ("[6]", 22, 25, "[6]")]


async def test_one_span_linked_twice_keeps_one_print():
    # Tier 2 keeps a candidate per year of "(Smith, 2010, 2012)", all with
    # the same span.
    refs = [_ref(1, authors="Smith, A.", year=2010), _ref(2, authors="Smith, A.", year=2012)]
    texts, receipt = await _linked_and_cleaned(
        ["Both $\\alpha$ reviews (Smith, 2010, 2012) agree."], refs=refs
    )

    rows = _offsets(receipt, 1, texts[1])
    assert len(rows) == 2
    assert {row[1:] for row in rows} == {(15, 34, "(Smith, 2010, 2012)")}


async def test_citations_after_a_long_formula_keep_their_print():
    # Flattening the formula deleted more than the first look-back: every
    # citation after it got an empty span.
    texts, receipt = await _linked_and_cleaned(
        ["See [3] $" + "\\mathrm{a}" * 80 + "$ and cells2 [4] then [6]."]
    )

    final = texts[1]
    assert final == "See [3] " + "a" * 80 + " and cells2 [4] then [6]."
    assert [row[0] for row in _offsets(receipt, 1, final)] == ["[3]", "2", "[4]", "[6]"]
    assert [row[3] for row in _offsets(receipt, 1, final)] == ["[3]", "2", "[4]", "[6]"]


async def test_a_repeated_citation_after_a_long_formula_keeps_its_own_print():
    # The formula's cleaning moved "[4]" further back than the first search
    # looks, and its second print sat inside that window: the first "[4]"
    # took the second print, leaving "[5]" and the second "[4]" empty.
    texts, receipt = await _linked_and_cleaned(
        [
            "Given $"
            + "\\mathrm{a}" * 60
            + "$ as in [4], the model of [5] and the later work [4] agree."
        ]
    )

    final = texts[1]
    assert (
        final == "Given " + "a" * 60 + " as in [4], the model of [5] and the later work [4] agree."
    )
    rows = _offsets(receipt, 1, final)
    assert [row[3] for row in rows] == ["[4]", "[5]", "[4]"]
    assert [row[1] for row in rows] == [73, 91, 114]


@pytest.mark.parametrize(
    "text",
    [
        "In $\\alpha$-treated mice, CD4 [4] cells expanded and CD4 [4] again.",
        # More deleted between two citations than the search first looks back.
        "See [3] $"
        + "\\mathrm{a}" * 80
        + "$ cells2 [3] and [1,2] and [1,2] cells2 [2]. x"
        + " " * 700
        + "[2] and type2 [6].",
        "Then cells3 x^{2}^{4,5},   \\alphaIL6    $"
        + "\\mathrm{a}" * 55
        + "$[2, 4]^{3}and[4] x^{2}[2, 4]and[4][2]",
    ],
    ids=["glued-digits", "long-cleaning", "long-formula-repeats"],
)
async def test_reanchoring_twice_changes_nothing(text):
    # The exporter re-anchors the receipt post_parse re-anchored already.
    texts, receipt = await _linked_and_cleaned([text])

    assert citation_linker.reanchor_citation_receipt(receipt, texts) == receipt
    _assert_indexes(_offsets(receipt, 1, texts[1]), texts[1])


def _input_file():
    return InputFile(
        path=Path("/tmp/test.pdf"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf", detected_mime_type="application/pdf", file_type="pdf"
        ),
    )


async def test_exported_receipt_offsets_index_the_exported_text():
    from bibr.export import PaperExport

    sentences, refs = _numbered(["  Prior work^{3} showed $\\alpha$ effects  [6] and [6] again."])
    _xrefs, receipt = await citation_linker.detect_bib_xrefs_with_receipt(
        sentences, _sections(), refs
    )
    citation_linker.strip_citation_superscripts(sentences, _sections(), receipt)
    # The receipt as linking left it, before the text was stripped and
    # cleaned: the exporter checks the offsets itself.
    contents = PaperContents(
        sentences=sentences,
        sections=_sections(),
        tables=[],
        links=[],
        sections_text={},
        citation_receipt=receipt,
    )
    contents.finalize_text()
    paper = Paper(
        input_file=_input_file(),
        metadata=PaperMetadata(doi="10.1234/test", title="Test Paper"),
        contents=contents,
    )
    paper.extraction = extraction_block()

    out = export_paper_to_json(paper, validate=False)

    text = next(row["text"] for row in out["text"] if row["text_id"] == 1)
    assert text == "Prior work showed α effects [6] and [6] again."
    exported = CitationLinkingReceipt(
        style_scores={},
        candidates=tuple(
            CitationCandidate(**row)
            for row in out["extraction"]["diagnostics"]["citation_linking"]["candidates"]
        ),
        resolved_candidate_fraction=None,
        unique_linked_bib_fraction=None,
    )
    rows = _offsets(exported, 1, text)
    _assert_indexes(rows, text)
    assert [row[0] for row in rows] == ["^{3}", "[6]", "[6]"]
    assert [row[1:3] for row in rows if row[3]] == [(28, 31), (36, 39)]
    assert PaperExport.model_validate(out).extraction.diagnostics.citation_linking is not None


async def test_post_parse_receipt_indexes_the_final_sentences(monkeypatch):
    from bibr.paper_contents import PaperXref
    from bibr.pipeline.stages.post_parse import post_parse

    monkeypatch.setattr("bibr.config.Settings.EQUATION_EXTRACTION", False)
    text = "  Prior work^{3} showed $\\alpha$ effects  [6] and [6] again."
    contents = PaperContents(
        sentences=[PaperSentence(text_id=0, text=text, section_id=0, paragraph_id=1)],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: text},
    )

    def candidate(raw, start, *, superscript=False):
        return CitationCandidate(
            text_id=0,
            start=start,
            end=start + len(raw),
            raw=raw,
            style="numeric",
            bib_ids=(3,) if superscript else (6,),
            evidence=("superscript_marker",) if superscript else ("bracket_marker",),
            confidence=1.0,
            accepted=True,
            rejection_reasons=(),
        )

    receipt = CitationLinkingReceipt(
        style_scores={"numeric": 1.0},
        candidates=(
            candidate("^{3}", text.index("^{3}"), superscript=True),
            candidate("[6]", text.index("[6]")),
            candidate("[6]", text.rindex("[6]")),
        ),
        resolved_candidate_fraction=1.0,
        unique_linked_bib_fraction=1.0,
    )

    async def fake_detect(*, receipt_sink, **kwargs):
        receipt_sink.append(receipt)
        return [PaperXref(xref_id=6, xref_type="bib", contents="[6]", text_id=0)]

    extractor = MagicMock()
    extractor.extract_all_metadata = AsyncMock(return_value=PaperMetadata(doi="", title="T"))
    with (
        patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor),
        patch(
            "bibr.structure.implicit_sections.detect_implicit_sections",
            AsyncMock(return_value=None),
        ),
        patch("bibr.extract.research_integrity.extract_structured_integrity", AsyncMock()),
        patch("bibr.structure.citation_linker.detect_bib_xrefs", fake_detect),
    ):
        paper = await post_parse(
            contents=contents, file_name="x.pdf", file_hash="deadbeef", llm_client=MagicMock()
        )

    final = paper.contents.sentences[0].text
    assert final == "Prior work showed α effects [6] and [6] again."
    rows = _offsets(paper.contents.citation_receipt, 0, final)
    _assert_indexes(rows, final)
    assert [row[0] for row in rows] == ["^{3}", "[6]", "[6]"]
    assert [row[1:3] for row in rows if row[3]] == [(28, 31), (36, 39)]
