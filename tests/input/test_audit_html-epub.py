"""HTML and ePub input findings from the 2026-10-06 audit (§1.7, §1.8).

The native HTML parser recursed once per nesting level, re-walked a block
from its start for every link in it and scanned every section for every
block. inspect_html parsed input of any size and sniffed for tags in the
first 4 KB of raw bytes, which rejected UTF-16 HTML; ePub chapters were all
decoded as UTF-8.
"""

from __future__ import annotations

import codecs
import io
import random
import unicodedata
import zipfile

import pytest
from bs4 import BeautifulSoup, CData, NavigableString, Tag

from bibr.exceptions import InputValidationError
from bibr.input import html_native
from bibr.input.epub_native import EpubParser
from bibr.input.html_native import HtmlParser, inspect_html
from bibr.input.validate import validate_input_file
from bibr.paper_contents import CanonicalSection
from bibr.utils.text import collapse_ws


def _body_text(parser) -> str:
    return " ".join(entry[0] for entry in parser._deferred_texts)


def _parse(html: bytes) -> HtmlParser:
    ok, soup = inspect_html(html)
    assert ok
    parser = HtmlParser(html, parsed_soup=soup)
    parser._contents = parser.parse()
    return parser


class TestDeepNesting:
    @pytest.mark.parametrize("tag", ["div", "section"])
    def test_deeply_nested_containers_parse(self, tag):
        """1,200 nested containers (12 KB) raised RecursionError."""
        depth = 1500
        html = (
            f"<html><body><article><h1>T</h1>{f'<{tag}>' * depth}"
            f"<p>Deep text.</p>{f'</{tag}>' * depth}</article></body></html>"
        ).encode()

        parser = _parse(html)

        assert "Deep text." in _body_text(parser)

    def test_unclosed_spans_parse(self):
        """Legacy markup with unclosed inline tags nests every later element
        one level deeper."""
        html = (
            "<html><body><article><h1>T</h1><p>First.</p>"
            + "<span>x" * 1500
            + "<p>Last paragraph.</p></article></body></html>"
        ).encode()

        parser = _parse(html)

        assert "Last paragraph." in _body_text(parser)

    def test_deep_and_flat_markup_read_alike(self):
        flat = b"<article><h1>T</h1><h2>Methods</h2><p>A.</p><ul><li>B</li></ul></article>"
        deep = (
            b"<article><h1>T</h1>"
            + b"<div>" * 1100
            + b"<h2>Methods</h2><p>A.</p><ul><li>B</li></ul>"
            + b"</div>" * 1100
            + b"</article>"
        )

        flat_parser, deep_parser = _parse(flat), _parse(deep)

        assert deep_parser._deferred_texts == flat_parser._deferred_texts
        assert [(s.header, s.section_type) for s in deep_parser.sections] == [
            (s.header, s.section_type) for s in flat_parser.sections
        ]


def _reference_offset(block: Tag, anchor: Tag, display: str) -> int:
    """The per-anchor computation the one-pass walk replaced."""
    parts: list[str] = []
    for node in block.descendants:
        if node is anchor:
            break
        if isinstance(node, (NavigableString, CData)):
            parts.append(str(node))
    pre_raw = "".join(parts)
    pre = collapse_ws(pre_raw)
    if not pre or not display:
        return len(pre)
    first_inside = next(
        (str(s) for s in anchor.descendants if isinstance(s, (NavigableString, CData))),
        "",
    )
    return len(pre) + (1 if pre_raw[-1:].isspace() or first_inside[:1].isspace() else 0)


def _assert_offsets_match_reference(markup: str) -> None:
    block = BeautifulSoup(markup, "html5lib").find("p")
    assert block is not None
    anchors = [(a, html_native._text(a)) for a in block.find_all("a", href=True)]

    expected = [_reference_offset(block, a, display) for a, display in anchors]

    assert html_native._anchor_offsets(block, anchors) == expected


class TestAnchorOffsets:
    @pytest.mark.parametrize(
        "markup",
        [
            '<p>Intro text with <a href="u">data link</a>.</p>',
            '<p>  <a href="1">lead</a> x<a href="2"> y</a>z <a href="3"></a> <a href="4">w</a></p>',
            '<p>See <b>bold <a href="1">one</a></b>and<a href="2"><span> </span>two</a></p>',
            '<p>a<!-- a comment --> <a href="1">x</a><![CDATA[c]]><a href="2">y</a></p>',
            '<p>text <a href="1"><img src="i.png"></a> more <a href="2">z</a></p>',
            # A combining mark that composes with the text before the anchor.
            '<p>cafe<a href="1">\u0301 link</a> e<a href="2">\u0301</a>\u0301<a href="3">q</a></p>',
            # Marks reordered across the anchor boundary.
            '<p>o\u0302<a href="1">\u0323</a> a\u0328<a href="2">\u0301x</a></p>',
            # Hangul conjoining jamo composing across anchors.
            '<p>\u1100<a href="1">\u1161\u11a8</a> \u1100\u1161<a href="2">\u11a8</a></p>',
            # A two-part Indic vowel and a character NFC expands.
            '<p>\u0b15\u0b47<a href="1">\u0b3e</a> \u0958<a href="2">x</a></p>',
            # No ASCII at all between the anchors.
            '<p>\u4e2d\u6587<a href="1">\u6587\u732e</a>\u4e2d<a href="2">\u4e2d</a>\u3002<a href="3">\u300c\u5f15\u7528\u300d</a></p>',
            # Unicode whitespace runs.
            '<p>\u00a0<a href="1">x</a>\u2003\u2000<a href="2"> y</a>\n\t<a href="3">z</a></p>',
            '<p>   <a href="1">only space before</a></p>',
        ],
    )
    def test_offsets_match_the_per_anchor_computation(self, markup):
        _assert_offsets_match_reference(markup)

    def test_random_blocks_match_the_per_anchor_computation(self):
        pieces = [
            "word",
            " ",
            "  ",
            "\n",
            "\u00a0",
            "e",
            "\u0301",
            "\u0323",
            "\u0302",
            "o",
            "\u1100",
            "\u1161",
            "\u11a8",
            "\u0b47",
            "\u0b3e",
            "\u4e2d",
            "\u0958",
            "\u212b",
            ".",
            "<b>",
            "</b>",
            "<!--c-->",
        ]
        rng = random.Random(20261006)  # noqa: S311 - deterministic test fixture, not crypto
        for _ in range(300):
            parts = ["<p>"]
            for _ in range(rng.randint(1, 30)):
                if rng.random() < 0.25:
                    inner = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 3)))
                    parts.append(f'<a href="h">{inner}</a>')
                else:
                    parts.append(rng.choice(pieces))
            parts.append("</p>")
            _assert_offsets_match_reference("".join(parts))

    @pytest.mark.parametrize(
        "marks",
        ["\u0301\u0334", "\u0301\u0300\u0323\u0302\u0308", "\u0313\u0300\u0345"],
        ids=["two-classes", "latin", "greek"],
    )
    def test_links_inside_a_long_run_of_combining_marks(self, marks):
        """No point NFC cannot reach across falls inside a run of marks, so
        the text measured for each link grew with the run; the marks NFC
        cannot compose are now counted instead of kept."""
        rng = random.Random(marks)  # noqa: S311 - deterministic test fixture, not crypto
        body = "".join(
            f'<a href="{i}">{rng.choice(["", marks[0]])}</a>'
            + "".join(rng.choice(marks) for _ in range(rng.randint(1, 12)))
            for i in range(200)
        )

        _assert_offsets_match_reference(f"<p>A e{body} end</p>")
        _assert_offsets_match_reference(f"<p>\u03b1{body}</p>")

    def test_links_in_a_run_of_marks_normalise_bounded_text(self, monkeypatch):
        """2,000 links in one 4,000-mark run: collapsing every prefix from the
        block start normalised about 4 million characters."""
        links = 2000
        html = (
            '<html><head><meta charset="utf-8"></head><body><article><h1>T</h1><p>e'
            + "".join(f'<a href="https://e.org/{i}"></a>\u0301\u0334' for i in range(links))
            + "</p></article></body></html>"
        ).encode()
        soup = BeautifulSoup(html, "html5lib")
        normalised = [0]
        real = unicodedata.normalize

        def counting(form, text):
            normalised[0] += len(text)
            return real(form, text)

        monkeypatch.setattr(unicodedata, "normalize", counting)

        parser = HtmlParser(html, parsed_soup=soup)
        parser.parse()

        offsets = [link[4] for link in parser._pending_url_links]
        assert offsets[:4] == [1, 2, 4, 6]
        assert offsets[-1] == 2 * links - 2
        assert normalised[0] < 500 * links

    def test_a_block_is_walked_once_however_many_links_it_holds(self, monkeypatch):
        """~4,000 links in 150 KB took 8.5 s: each anchor re-walked the
        block from its start."""
        links = 1500
        html = (
            "<html><body><article><h1>T</h1><p>"
            + "".join(f'word <a href="https://e.org/{i}">link {i}</a> ' for i in range(links))
            + "</p></article></body></html>"
        ).encode()
        soup = BeautifulSoup(html, "html5lib")
        nodes = sum(1 for _ in soup.descendants)
        visits = [0]
        original = Tag.descendants

        def counting(self):
            for node in original.fget(self):
                visits[0] += 1
                yield node

        monkeypatch.setattr(Tag, "descendants", property(counting))

        parser = HtmlParser(html, parsed_soup=soup)
        parser.parse()

        assert len(parser._pending_url_links) == links
        # A handful of whole-document walks (noise removal, metadata, the
        # anchor search and the offsets); the per-anchor walk made ~links/2.
        assert visits[0] < 20 * nodes


class _CountingList(list):
    def __init__(self) -> None:
        super().__init__()
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


class TestReferenceSectionLookup:
    def test_blocks_do_not_scan_the_section_list(self):
        """Every block scanned all sections: 20k <h2><p> pairs took 7 s."""
        html = (
            b"<article><h1>T</h1>"
            + b"".join(b"<h2>Part %d</h2><p>Text %d.</p>" % (i, i) for i in range(300))
            + b"<h2>References</h2><p>Smith J. A study. 2020.</p><ol><li>Doe J. 2021.</li></ol>"
            + b"</article>"
        )
        ok, soup = inspect_html(html)
        assert ok
        parser = HtmlParser(html, parsed_soup=soup)
        parser.sections = _CountingList()

        contents = parser.parse()

        assert parser.sections.iterations == 0
        assert contents.native_ref_strings == ["Smith J. A study. 2020.", "Doe J. 2021."]
        assert contents.sections[-1].section_type == CanonicalSection.REFERENCES

    def test_a_reference_list_opens_the_references_section(self):
        html = (
            b"<article><h1>T</h1><h2>Intro</h2><p>Body.</p>"
            b'<ol class="references"><li>Doe J. 2021.</li></ol><p>Roe R. 2019.</p></article>'
        )

        contents = _parse(html)._contents

        assert contents.native_ref_strings == ["Doe J. 2021.", "Roe R. 2019."]


class TestInspectSizeCap:
    def test_html_over_the_cap_is_refused_before_parsing(self, monkeypatch):
        """Validation hands its parsed DOM to the parser, which skips its own
        size check for a supplied DOM, so the cap never applied."""
        monkeypatch.setattr(html_native, "_MAX_HTML_BYTES", 1024)
        calls: list[int] = []
        real = html_native.BeautifulSoup

        def recording(markup, *args, **kwargs):
            calls.append(len(markup))
            return real(markup, *args, **kwargs)

        monkeypatch.setattr(html_native, "BeautifulSoup", recording)
        html = b"<html><body><p>" + b"word " * 1000 + b"</p></body></html>"

        with pytest.raises(InputValidationError, match="1024-byte parse limit"):
            inspect_html(html)
        assert calls == []

    def test_validation_rejects_html_over_the_cap_as_invalid_input(self, monkeypatch):
        monkeypatch.setattr(html_native, "_MAX_HTML_BYTES", 1024)
        html = b"<html><body><p>" + b"word " * 1000 + b"</p></body></html>"

        with pytest.raises(InputValidationError, match=rf"\({len(html)} bytes\)"):
            validate_input_file("paper.html", html)

    def test_html_at_the_cap_is_read(self, monkeypatch):
        html = b"<html><body><p>Small article.</p></body></html>"
        monkeypatch.setattr(html_native, "_MAX_HTML_BYTES", len(html))

        ok, soup = inspect_html(html)

        assert ok
        assert soup is not None
        assert validate_input_file("paper.html", html).is_valid


_ARTICLE = (
    "<html><head><title>T</title></head><body><article><h1>Título</h1>"
    "<p>Die Temperatur betrug 37°C bei München.</p></article></body></html>"
)


class TestInspectEncodings:
    @pytest.mark.parametrize(
        "data",
        [
            _ARTICLE.encode("utf-16"),  # Word's "Save as Unicode": BOM + UTF-16LE
            codecs.BOM_UTF16_BE + _ARTICLE.encode("utf-16-be"),
            codecs.BOM_UTF8 + _ARTICLE.encode(),
        ],
        ids=["utf-16le-bom", "utf-16be-bom", "utf-8-bom"],
    )
    def test_html_with_a_bom_is_read(self, data):
        parser = _parse(data)

        assert "37°C bei München" in _body_text(parser)
        assert validate_input_file("paper.html", data).is_valid

    def test_a_long_comment_before_the_first_element(self):
        article = _ARTICLE.replace("<head>", '<head><meta charset="utf-8">')
        html = b"<!--" + b" licence text" * 2000 + b" -->\n" + article.encode()

        assert "37°C bei München" in _body_text(_parse(html))

    def test_a_long_head_without_html_or_body_tags(self):
        html = (
            b"<!doctype html><meta charset=utf-8><title>T</title><style>"
            + b".c{color:red}" * 1000
            + b"</style><p>Body text after the styles.</p>"
        )

        assert "Body text after the styles." in _body_text(_parse(html))

    @pytest.mark.parametrize(
        "data",
        [
            b"just some text\n" * 20,
            # Gzip header, then bytes that happen to spell a tag.
            b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03<p>text</p>",
            # UTF-32 is not an encoding html5lib reads.
            "<html><body><p>x</p></body></html>".encode("utf-32"),
        ],
        ids=["plain-text", "binary-before-tag", "utf-32"],
    )
    def test_non_html_is_still_refused(self, data):
        assert inspect_html(data) == (False, None)


def _epub_with_chapter(chapter: bytes) -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/ch1.xhtml", chapter)
    return buf.getvalue()


def _chapter(encoding_name: str) -> str:
    return (
        f'<?xml version="1.0" encoding="{encoding_name}"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><section><h1>Kapitel</h1>'
        "<p>Die Temperatur betrug 37°C bei München.</p></section></body></html>"
    )


class TestEpubChapterEncodings:
    @pytest.mark.parametrize(
        "chapter",
        [
            _chapter("UTF-16").encode("utf-16"),
            codecs.BOM_UTF16_BE + _chapter("UTF-16").encode("utf-16-be"),
            _chapter("UTF-16").encode("utf-16-le"),
            _chapter("UTF-16").encode("utf-16-be"),
            _chapter("ISO-8859-1").encode("latin-1"),
            _chapter("windows-1252").encode("cp1252"),
        ],
        ids=["utf-16-bom", "utf-16be-bom", "utf-16le", "utf-16be", "latin-1", "cp1252"],
    )
    def test_a_chapter_is_read_in_its_encoding(self, chapter):
        parser = EpubParser(_epub_with_chapter(chapter))
        parser.parse()

        assert "37°C bei München" in _body_text(parser)

    @pytest.mark.parametrize("declared", ["UTF-16", "punycode", "no-such-charset"])
    def test_a_declaration_that_cannot_hold_is_read_as_utf8(self, declared):
        """An ASCII-readable declaration naming UTF-16 is not UTF-16, and a
        name that is not a web text encoding is not looked up in Python's
        codec registry (which would run, say, the punycode decoder)."""
        parser = EpubParser(_epub_with_chapter(_chapter(declared).encode()))
        parser.parse()

        assert "37°C bei München" in _body_text(parser)
