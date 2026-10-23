"""HTML and ePub text, headings and reference lists that no element marks.

Found in the non-PDF sweep after #215 (a Word manuscript with hand-bolded
headings had no reference list):

- Text sitting directly in a ``<div>`` (ScienceDirect writes its paragraphs
  so), in a ``<dd>`` or in citeproc's ``<div class="csl-entry">`` was dropped:
  a page written that way failed to parse, and a pandoc or Quarto page or
  ePub lost its whole reference list.
- HTML that Word or Google Docs exports from such a manuscript has no heading
  elements, only bold paragraphs, so it had the same problem as the .docx.
- A UTF-8 page without an encoding declaration was read as windows-1252.
- Each ePub chapter's ``<title>`` would have become body text.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.input.epub_native import EpubParser
from bibr.input.html_native import HtmlParser, inspect_html
from bibr.paper_contents import CanonicalSection


def _parse(html: str | bytes):
    data = html.encode("utf-8") if isinstance(html, str) else html
    ok, soup = inspect_html(data)
    assert ok
    parser = HtmlParser(data, parsed_soup=soup)
    contents = parser.parse()
    _segment(parser, contents)
    return contents


def _segment(parser, contents) -> None:
    parser.apply_segmentation(
        contents, [[entry.text] for entry in parser.assembler.entries if entry.needs_segmentation]
    )


def _page(body: str, head: str = "<title>A study</title>") -> str:
    return (
        f'<!doctype html><html><head><meta charset="utf-8">{head}</head><body>{body}</body></html>'
    )


def _headers(contents) -> list[str]:
    return [section.header for section in contents.sections[1:]]


def _texts(contents, header: str | None = None) -> list[str]:
    if header is None:
        return [s.text for s in contents.sentences]
    ids = {s.section_id for s in contents.sections if s.header == header}
    return [s.text for s in contents.sentences if s.section_id in ids]


REFS = [
    "Smith, J. (2020). A paper. Journal of Things, 1, 1-2.",
    "Jones, K. (2019). Another paper. Journal of Things, 2, 3-4.",
]


class TestTextOutsideParagraphs:
    def test_a_page_of_div_paragraphs_parses(self):
        """ScienceDirect's body paragraphs are ``<div>``s; such a page failed
        with "HTML input did not contain parseable article content"."""
        contents = _parse(
            _page(
                "<article><h1>A study</h1><section><h2>1. Introduction</h2>"
                '<div class="u-margin-s-bottom">Body text <a href="#b1">[1]</a> here.</div>'
                "<div><span>A span <b>in bold</b></span> too.</div></section></article>"
            )
        )

        assert _texts(contents, "1. Introduction") == [
            "Body text [1] here.",
            "A span in bold too.",
        ]

    def test_text_around_blocks_keeps_its_order(self):
        contents = _parse(
            _page(
                "<h1>A study</h1><div>Lead text <p>A paragraph.</p>tail text</div>"
                '<a href="https://example.org"><div>A block inside a link</div></a>'
            )
        )

        assert _texts(contents) == [
            "Lead text",
            "A paragraph.",
            "tail text",
            "A block inside a link",
        ]

    def test_a_definition_list_of_references_reads_each_entry_with_its_label(self):
        contents = _parse(
            _page(
                "<h1>A study</h1><h2>Intro</h2><p>Text [1].</p><h2>References</h2>"
                f"<dl><dt>[1]</dt><dd>{REFS[0]}</dd><dt>[2]</dt><dd>{REFS[1]}</dd></dl>"
            )
        )

        assert contents.native_ref_strings == [f"[1] {REFS[0]}", f"[2] {REFS[1]}"]


class TestCiteprocReferences:
    """pandoc and Quarto write the list as ``<div id="refs">`` of
    ``<div class="csl-entry">``, under a heading or under none."""

    _REFS_DIV = (
        '<div id="refs" class="references csl-bib-body hanging-indent" role="list">'
        + "".join(
            f'<div id="ref-{i}" class="csl-entry" role="listitem">{ref}</div>'
            for i, ref in enumerate(REFS)
        )
        + "</div>"
    )

    @pytest.mark.parametrize("heading", ["", '<h1 class="unnumbered">References</h1>'])
    def test_the_list_is_read(self, heading):
        contents = _parse(
            _page(f"<h1>Introduction</h1><p>Text (Smith 2020).</p>{heading}{self._REFS_DIV}")
        )

        assert _headers(contents) == ["Introduction", "References"]
        assert contents.native_ref_strings == REFS
        assert contents.sections[-1].section_type == CanonicalSection.REFERENCES

    def test_a_pandoc_epub_keeps_its_list_and_drops_chapter_titles(self):
        def chapter(title: str, body: str) -> str:
            return (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml"><head><meta charset="utf-8"/>'
                f"<title>{title}</title></head><body>{body}</body></html>"
            )

        chapters = {
            "ch001.xhtml": chapter(
                "Introduction", "<section><h1>Introduction</h1><p>Text café.</p></section>"
            ),
            "ch002.xhtml": chapter(
                "References",
                f'<section><h1 class="unnumbered">References</h1>{self._REFS_DIV}</section>',
            ),
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr(
                "META-INF/container.xml",
                '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                'version="1.0"><rootfiles><rootfile full-path="EPUB/content.opf" '
                'media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "EPUB/content.opf",
                '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                "<dc:title>A study</dc:title></metadata><manifest>"
                + "".join(
                    f'<item id="c{i}" href="{name}" media-type="application/xhtml+xml"/>'
                    for i, name in enumerate(chapters)
                )
                + "</manifest><spine>"
                + "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
                + "</spine></package>",
            )
            for name, text in chapters.items():
                zf.writestr(f"EPUB/{name}", text)
        parser = EpubParser(buf.getvalue())
        contents = parser.parse()
        _segment(parser, contents)

        assert _headers(contents) == ["Introduction", "References"]
        assert _texts(contents) == ["Text café.", *REFS]
        assert contents.native_ref_strings == REFS


class TestWordProcessorExports:
    def test_word_html_with_bold_headings_and_a_bibliography(self):
        """Word's "Save as Web Page" of the #215 manuscript: Normal paragraphs
        in bold for headings, the list in the Bibliography style."""
        contents = _parse(
            _page(
                '<div class="WordSection1">'
                '<p class="MsoNormal"><b><span lang="EN-US">Method</span></b></p>'
                '<p class="MsoNormal"><span lang="EN-US">We did it (Smith, 2020).</span></p>'
                '<p class="MsoNormal"><b>Discussion</b></p>'
                '<p class="MsoNormal">It worked (Jones, 2019).</p>'
                + "".join(f'<p class="MsoBibliography">{ref}</p>' for ref in REFS)
                + '<p class="MsoNormal">Table 1</p>'
                "</div>"
            )
        )

        assert _headers(contents) == ["Method", "Discussion", "References"]
        assert contents.native_ref_strings == REFS
        assert _texts(contents, "Discussion") == ["It worked (Jones, 2019).", "Table 1"]
        assert RefLocator(contents).collect_reference_rows()["text"].tolist() == REFS

    def test_google_docs_html_bolds_with_font_weight(self):
        contents = _parse(
            _page(
                '<p class="c1"><span class="c3" style="font-weight:700">Method</span></p>'
                '<p class="c1"><span>We did it (Smith, 2020).</span></p>'
                '<p class="c1"><span style="font-weight: bold">References</span></p>'
                + "".join(f'<p class="c1"><span>{ref}</span></p>' for ref in REFS)
            )
        )

        assert _headers(contents) == ["Method", "References"]
        assert contents.native_ref_strings == REFS

    def test_bold_paragraphs_are_body_text_on_a_page_with_headings(self):
        contents = _parse(
            _page("<h1>A study</h1><h2>Intro</h2><p><b>An emphasised remark</b></p><p>Text.</p>")
        )

        assert _headers(contents) == ["A study", "Intro"]

    def test_a_references_paragraph_heads_the_list_on_a_page_with_headings(self):
        contents = _parse(
            _page(
                "<h1>A study</h1><h2>Intro</h2><p>Text.</p><p><strong>References</strong></p>"
                + "".join(f"<p>{ref}</p>" for ref in REFS)
            )
        )

        assert _headers(contents) == ["A study", "Intro", "References"]
        assert [s.level for s in contents.sections[1:]] == [1, 2, 2]
        assert contents.native_ref_strings == REFS


class TestReferenceHeadings:
    @pytest.mark.parametrize("heading", ["Literature Cited", "Reference list", "Literatur"])
    def test_other_reference_headings_type_the_section(self, heading):
        contents = _parse(
            _page(
                f"<h1>A study</h1><h2>Intro</h2><p>Text.</p><h2>{heading}</h2>"
                f"<ol><li>{REFS[0]}</li></ol>"
            )
        )

        assert contents.sections[-1].section_type == CanonicalSection.REFERENCES
        assert contents.native_ref_strings == [REFS[0]]

    def test_a_first_heading_reading_references_is_not_the_title(self):
        """A page without a title takes its first heading as one, which turned
        a lone "References" heading into the title and its list into body text."""
        contents = _parse(_page(f"<h2>References</h2><ol><li>{REFS[0]}</li></ol>", head=""))

        assert contents.sections[-1].section_type == CanonicalSection.REFERENCES
        assert contents.native_ref_strings == [REFS[0]]

    def test_what_follows_a_reference_list_is_not_a_reference(self):
        contents = _parse(
            _page(
                "<h1>A study</h1><h2>Intro</h2><p>Text.</p><h2>References</h2>"
                f'<ol><li>{REFS[0]}</li></ol><p><a href="#">Download references</a></p>'
            )
        )

        assert contents.native_ref_strings == [REFS[0]]
        assert _texts(contents, "Intro") == ["Text.", "Download references"]


class TestEncoding:
    def test_undeclared_utf8_is_read_as_utf8(self):
        html = "<html><body><h1>T</h1><p>“Café” – naïve</p></body></html>".encode()

        assert _texts(_parse(html)) == ["“Café” – naïve"]

    def test_a_declared_encoding_still_wins(self):
        html = (
            '<html><head><meta charset="windows-1252"></head>'
            "<body><h1>T</h1><p>caf\xe9</p></body></html>"
        ).encode("cp1252")

        assert _texts(_parse(html)) == ["café"]
