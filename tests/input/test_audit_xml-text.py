"""Text and XML helpers that were quadratic on crafted input.

Each helper keeps its old output and gets an input on which the old code takes
tens of seconds: a run of named entities in one element, thousands of ext-links
in one paragraph, a long junk run after a DOI, STX marks in one long token, and
a punctuation run inside a bookmark title.
"""

from __future__ import annotations

import random
import re
import time
import unicodedata

import pytest
from lxml import etree

from bibr.input import jats_native, pdf_outline
from bibr.input.consolidate_text import _resolve_stx_marks
from bibr.input.mathml_whitespace import FlatText
from bibr.input.pdf_outline import (
    HeadingRef,
    OutlineItem,
    _normalize_title,
    _walk_pdfium_outline,
    match_outline_to_headings,
)
from bibr.input.xml_entities import parse_xml
from bibr.utils import text as text_utils
from bibr.utils.text import CollapsedLength, collapse_ws, normalize_doi

_DOCTYPE = b'<?xml version="1.0"?><!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96)//EN" "x.dtd">'


def _elapsed(fn, *args):
    started = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - started


# --------------------------------------------------------------------------- entities


def test_a_long_run_of_entities_resolves_in_linear_time():
    data = _DOCTYPE + b"<article><p>" + b"&alpha;" * 100_000 + b"</p></article>"

    root, elapsed = _elapsed(parse_xml, data)

    assert root.find("p").text == "\u03b1" * 100_000
    # Appending to the paragraph text entity by entity took ~45 s here.
    assert elapsed < 2.0


def test_entity_runs_merge_into_the_text_or_tail_before_them():
    data = (
        _DOCTYPE + b"<article><p>A&alpha;&beta;B<i>x&gamma;</i>&delta;C<!--c-->"
        b"&epsilon;&nosuch;D<b/>E</p></article>"
    )

    p = parse_xml(data).find("p")

    assert list(p.iter(etree.Entity)) == []
    assert p.text == "A\u03b1\u03b2B"
    i, comment, b = list(p)
    assert (i.text, i.tail) == ("x\u03b3", "\u03b4C")
    assert comment.tail == "\u03b5&nosuch;D"
    assert b.tail == "E"


# --------------------------------------------------------------------------- JATS ext-links

_ARTICLE = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:mml="http://www.w3.org/1998/Math/MathML"><front><article-meta><title-group>'
    "<article-title>T</article-title></title-group></article-meta></front>"
    "<body><sec><title>H</title><p>{}</p></sec></body></article>"
)


def _article(body: str) -> bytes:
    return _ARTICLE.format(body).encode()


def _link(text: str = "", n: int = 0) -> str:
    return f'<ext-link ext-link-type="uri" xlink:href="https://e.org/{n}">{text}</ext-link>'


def _math(inner: str) -> str:
    return f"<mml:math><mml:mrow>{inner}</mml:mrow></mml:math>"


_PROSE_LINKS = "".join(f"word {_link('l', i)} " for i in range(4_000))
_LINKS_AMONG_DIGITS = _math("<mml:mi>x</mml:mi> " + f"<mml:mn>1</mml:mn>{_link()}" * 4_000)
_LINKS_AMONG_GAPS = _math("<mml:mi>x</mml:mi>" + f"<!--c--> {_link()}" * 4_000)


class _Work:
    """Counts the pieces :meth:`FlatText.join` walks and the characters
    :mod:`bibr.utils.text` normalizes."""

    def __init__(self, monkeypatch) -> None:
        self.pieces = self.chars = 0
        join = FlatText.join

        def counted_join(flat, start=0, stop=None):
            self.pieces += (len(flat.parts) if stop is None else stop) - start
            return join(flat, start, stop) if start or stop is not None else join(flat)

        monkeypatch.setattr(FlatText, "join", counted_join)
        monkeypatch.setattr(text_utils, "unicodedata", self)

    def normalize(self, form, text):
        self.chars += len(text)
        return unicodedata.normalize(form, text)

    def __getattr__(self, name):
        return getattr(unicodedata, name)


@pytest.mark.parametrize(
    "body",
    [_PROSE_LINKS, _LINKS_AMONG_DIGITS, _LINKS_AMONG_GAPS],
    ids=["prose", "digits after a gap", "gaps"],
)
def test_ext_link_offsets_take_linear_work(monkeypatch, body):
    parser = jats_native.JatsParser(_article(body))
    work = _Work(monkeypatch)

    parser.parse()

    assert len(parser._pending_url_links) == 4_000
    # Joining and collapsing the paragraph again walked thousands of pieces and
    # normalized thousands of characters per link; a few dozen remain, with
    # room for the open MathML rest a link joins again.
    assert work.pieces < 100 * 4_000
    assert work.chars < 100 * 4_000


def test_ext_link_offsets_are_where_the_link_text_starts():
    parser = jats_native.JatsParser(_article(_PROSE_LINKS))
    parser.parse()

    (entry,) = [e for e in parser.assembler.entries if e.text.startswith("word")]
    offsets = [offset for *_, offset in parser._pending_url_links]
    assert offsets == [7 * i + 5 for i in range(4_000)]
    assert {entry.text[offset] for offset in offsets} == {"l"}


def test_many_ext_links_parse_and_attach_in_linear_time():
    n = 12_000
    body = "".join(f"Sentence {i} has {_link('a link', i)}. " for i in range(n))

    def run():
        parser = jats_native.JatsParser(_article(body))
        contents = parser.parse()
        texts = [e.text for e in parser.assembler.entries if e.needs_segmentation]
        segments = [[s for s in re.split(r"(?<=\.) ", t) if s] for t in texts]
        parser.apply_segmentation(contents, segments)
        return contents

    contents, elapsed = _elapsed(run)

    assert len(contents.links) == n
    by_id = {s.text_id: s.text for s in contents.sentences}
    assert by_id[contents.links[-1].text_id] == f"Sentence {n - 1} has a link."
    # ~60 s before: the offsets re-collapsed the paragraph per link, and each
    # link searched the paragraph's sentences from the first.
    assert elapsed < 2.0


def test_sentence_starts_are_found_once_per_entry(monkeypatch):
    body = "".join(f"Sentence {i} has {_link('a link', i)}. " for i in range(50))
    parser = jats_native.JatsParser(_article(body))
    contents = parser.parse()
    texts = [e.text for e in parser.assembler.entries if e.needs_segmentation]
    calls = []
    starts = jats_native._sentence_starts

    def counted(candidates, entry_text):
        calls.append(len(candidates))
        return starts(candidates, entry_text)

    monkeypatch.setattr(jats_native, "_sentence_starts", counted)
    parser.apply_segmentation(contents, [re.split(r"(?<=\.) ", t) for t in texts])

    assert calls == [50]
    assert [link.text_id for link in contents.links] == [
        s.text_id for s in contents.sentences if "a link" in s.text
    ]


_LINK_CASES = {
    "plain": f"See {_link('the data')} and {_link('code')}. More {_link(' lead')}.",
    "odd spaces": f"A \u00a0{_link('x')}\u2000 b\n\t{_link('y')}\u3000",
    "a mark composing across a link": "caf\u00e9 " + _link("\u0301x") + "e" + _link("\u0301"),
    "Hangul jamo across links": "\u1100" + _link() + "\u1161" + _link() + "\u11a8" + _link("x"),
    "math gap before a link": (
        "<mml:math><mml:mi>x</mml:mi> <mml:mo>=</mml:mo> <mml:mn>1</mml:mn> </mml:math>"
        + _link("r")
        + " tail"
    ),
    "gap turned into a space after a link": (
        "<mml:math><mml:mi>a</mml:mi> "
        + _link("<break/>")
        + "<mml:mi>b</mml:mi> "
        + _link("c")
        + "</mml:math> end"
    ),
    "links inside math": _math(
        "<mml:mi>l</mml:mi> "
        + _link("<mml:mi>n</mml:mi>")
        + " <mml:mi>d</mml:mi> "
        + _link()
        + " "
        + _link()
        + " <mml:mi>e</mml:mi>"
    ),
    # The gap before "a" reads "b", added after the first link, as a second letter.
    "a gap read again after a link": _math(
        "<mml:mi>x</mml:mi> <mml:mi>a</mml:mi>"
        + _link()
        + "<mml:mi>b</mml:mi>"
        + _link()
        + "<mml:mi>c</mml:mi>"
        + _link()
    ),
    "a long rest after a gap": _math(
        "<mml:mi>x</mml:mi> " + f"<mml:mn>1</mml:mn>{_link()}" * 100 + "<mml:mi>y</mml:mi>"
    ),
    "nested and empty links": f"x {_link('a ' + _link('b') + ' c')} {_link()} y",
    "long CJK run": "\u4e2d" * 1500 + _link("\u6587") + "\u4e2d" * 1500 + _link("x"),
    "block in the paragraph": (
        f"before {_link('a')} <disp-formula><mml:math><mml:mi>z</mml:mi></mml:math>"
        f"</disp-formula> after {_link('b')}"
    ),
}


@pytest.mark.parametrize("body", list(_LINK_CASES.values()), ids=list(_LINK_CASES))
def test_ext_link_prefix_matches_a_full_recollapse(monkeypatch, body):
    measured = []
    incremental = jats_native._Walker._pre_link

    def checked(walker):
        got = incremental(walker)
        pre_raw = walker.flat.join()
        assert got == (len(collapse_ws(pre_raw)), pre_raw[-1:].isspace()), repr(pre_raw)
        measured.append(got)
        return got

    monkeypatch.setattr(jats_native._Walker, "_pre_link", checked)
    jats_native.JatsParser(_article(body)).parse()

    assert measured


def test_flat_text_settles_what_later_pieces_cannot_change():
    flat = FlatText()
    flat.add("Let ")
    flat.add_math("x", "mi", 1)
    assert (flat.settled, flat.text_end) == (2, 2)
    flat.add_math(" ", None, 1)
    flat.add_math("a", "mi", 1)
    # The gap reads the tokens after it: it is open until prose follows.
    assert (flat.settled, flat.text_end) == (2, 4)
    assert flat.join() == "Let xa"
    flat.add_math("b", "mi", 1)
    assert flat.join() == "Let x ab"
    flat.add_math("\n ", None, 1)
    assert (flat.settled, flat.text_end) == (2, 5)
    flat.add("here")
    assert (flat.settled, flat.text_end) == (7, 7)

    assert flat.join(0, 2) + flat.join(2, 5) + flat.join(5) == flat.join() == "Let x ab here"
    assert flat.join(5, 5) == ""

    flat.add_math(" ", None, 2)
    assert (flat.settled, flat.text_end) == (7, 7)
    flat.separate()  # the trailing gap becomes the separator
    assert (flat.settled, flat.text_end) == (8, 8)
    assert flat.join(7) == " "


def test_collapsed_length_matches_collapse_ws_of_every_prefix():
    pieces = [
        "a", "word ", " ", "  ", "\n", "\t", "\u00a0", "\u2000", "\u3000", "\x1c", "\x85",
        "e\u0301", "\u0301", "\u0316", "\u0328", "\u212b", "\u0f71\u0f72", "\u0344", "\u1100",
        "\u1161", "\u11a8", "\uac00", "\u0b47", "\u0b3e", "\u0bc6", "\u0bbe", "\u3099",
        "\u304b", "\u4e2d", "\u03b1", "\u1f71", "\u0345", ".",
    ]  # fmt: skip
    rng = random.Random(7)  # noqa: S311 - deterministic test fixture, not crypto
    for _ in range(2_000):
        collapsed = CollapsedLength()
        text = ""
        for _ in range(rng.randint(1, 8)):
            tail = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 3)))
            assert collapsed.length(tail) == len(collapse_ws(text + tail)), repr(text + tail)
            piece = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 5)))
            text += piece
            collapsed.add(piece)
            assert collapsed.length() == len(collapse_ws(text)), repr(text)
            assert collapsed.last == text[-1:]


def test_collapsed_length_is_exact_across_every_composition():
    # Every canonical pair NFC composes, added in separate pieces, and Hangul's.
    pairs = [("\u1100", "\u1161"), ("\uac00", "\u11a8")]
    for cp in range(0x110000):
        decomposition = unicodedata.decomposition(chr(cp))
        if decomposition and not decomposition.startswith("<"):
            parts = decomposition.split()
            if len(parts) == 2:
                pairs.append((chr(int(parts[0], 16)), chr(int(parts[1], 16))))
    for first, second in pairs:
        for split in ((first, second), ("a " + first, second + "b"), (first + second, "\u0301")):
            collapsed = CollapsedLength()
            for piece in split:
                collapsed.add(piece)
            assert collapsed.length() == len(collapse_ws("".join(split))), repr(split)


# --------------------------------------------------------------------------- DOI


def test_normalize_doi_strips_trailing_junk_in_linear_time():
    doi = "10.1234/" + "." * 80_000 + "a"

    result, elapsed = _elapsed(normalize_doi, doi)

    assert result == doi
    # The junk pattern backtracked from every dot: ~45 s.
    assert elapsed < 1.0


def test_normalize_doi_supplement_path_before_a_newline_in_linear_time():
    doi = "10.1234/x" + "/-/DC1" * 32_000 + "\nx"

    result, elapsed = _elapsed(normalize_doi, doi)

    assert result is None
    # ".*$" ran to the newline from every "/-/DC1": ~20 s.
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1234/abc.", "10.1234/abc"),
        ("10.1234/abc.);>", "10.1234/abc"),
        ("https://doi.org/10.1234/a.b.c,", "10.1234/a.b.c"),
        ("10.1234/a..b.", "10.1234/a..b"),
        ("10.1234/x.www.example.org/y", "10.1234/x"),
        ("10.1073/pnas.2501823122/-/DCSupplemental.", "10.1073/pnas.2501823122"),
        ("10.1234/...", None),
    ],
)
def test_normalize_doi_trailing_junk_is_unchanged(raw, expected):
    assert normalize_doi(raw) == expected


# --------------------------------------------------------------------------- STX marks


def test_stx_marks_in_one_long_token_resolve_in_linear_time():
    _resolve_stx_marks("off\x02line")  # load the lexicons outside the timing

    result, elapsed = _elapsed(_resolve_stx_marks, "ab\x02" * 20_000)

    assert result in ("ab" + "-ab" * 19_999, "ab" * 20_000)
    # Every mark scanned the token back to its start: ~60 s.
    assert elapsed < 2.0


def test_a_long_letter_run_before_a_mark_resolves_in_linear_time():
    _resolve_stx_marks("off\x02line")

    result, elapsed = _elapsed(_resolve_stx_marks, "a" * 80_000 + "1b\x02c")

    assert result in ("a" * 80_000 + "1b-c", "a" * 80_000 + "1bc")
    # "[^\W\d_]+$" backtracked from every letter of the run: ~60 s.
    assert elapsed < 1.0


def test_stx_url_context_holds_across_marks_and_ends_with_the_token():
    text = "see https://x.org/a\x02b\x02c and off\x02line 10.1234/ab\x02cd off\x02line"

    assert _resolve_stx_marks(text) == "see https://x.org/a-b-c and offline 10.1234/ab-cd offline"


def test_stx_url_context_found_after_an_earlier_mark_in_the_token():
    assert _resolve_stx_marks("ab\x02https://x.org/c\x02d").endswith("https://x.org/c-d")


# --------------------------------------------------------------------------- PDF outline


def test_outer_punctuation_strip_is_linear():
    title = "a" + "!" * 80_000 + "a"

    result, elapsed = _elapsed(_normalize_title, title)

    assert result == title
    # The trailing alternative backtracked from every "!": ~50 s.
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  1. Introduction:  ", "1. introduction"),
        ("--Methods--", "methods"),
        ("(a) Results!!", "a) results"),
        ("_x_", "x"),
        ("!!!", ""),
        ("Ab\u00b2!", "ab\u00b2"),
    ],
)
def test_outer_punctuation_strip_is_unchanged(raw, expected):
    assert _normalize_title(raw) == expected


class _Bookmark:
    level = 0

    def get_title(self):
        return "Section"

    def get_dest(self):
        return None


class _HugeOutlineDoc:
    def __init__(self, entries: int) -> None:
        self.entries = entries
        self.read = 0

    def get_toc(self):
        for _ in range(self.entries):
            self.read += 1
            yield _Bookmark()

    def __len__(self):
        return 10


def test_outline_read_is_capped():
    doc = _HugeOutlineDoc(50_000)

    items = _walk_pdfium_outline(doc)

    assert len(items) == pdf_outline._MAX_ENTRIES == 5_000
    assert doc.read == 5_001


def test_outline_at_the_cap_is_read_whole():
    assert len(_walk_pdfium_outline(_HugeOutlineDoc(5_000))) == 5_000


def test_matcher_normalizes_each_title_once(monkeypatch):
    calls = []
    normalize = pdf_outline._normalize_title

    def counted(text):
        calls.append(text)
        return normalize(text)

    monkeypatch.setattr(pdf_outline, "_normalize_title", counted)
    outline = [OutlineItem(title=f"Heading {i}", level=i % 3, page_no=1) for i in range(20)]
    headings = [HeadingRef(text=f"Heading {i}", page_no=1) for i in range(30)]

    matched = match_outline_to_headings(outline, headings)

    assert matched == {i: i % 3 + 1 for i in range(20)}
    # Two forms (with and without the numbering marker) per title, where every
    # bookmark-heading pair normalized both titles again.
    assert len(calls) == 2 * (len(outline) + len(headings))
