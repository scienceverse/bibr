"""Tests for native HTML/ePub ingestion."""

from __future__ import annotations

import io
import zipfile

from bibr.input.epub_native import EpubParser
from bibr.input.html_native import HtmlParser
from bibr.paper_contents import CanonicalSection

ARTICLE_HTML = b"""<!doctype html>
<html>
  <head>
    <title>Fallback Title</title>
    <meta name="citation_title" content="Native HTML Paper">
    <meta name="citation_doi" content="10.1234/html.1">
    <meta name="citation_author" content="Jane Smith">
    <meta name="citation_author" content="John Doe">
    <meta name="citation_journal_title" content="Journal of Markup">
    <meta name="citation_volume" content="12">
    <meta name="citation_issue" content="3">
    <meta name="citation_firstpage" content="55">
    <meta name="citation_lastpage" content="63">
    <meta name="citation_publication_date" content="2025/04/09">
    <meta name="citation_keywords" content="html; native ingestion">
    <meta name="citation_abstract" content="This abstract came from metadata.">
    <meta name="dc.publisher" content="Markup Press">
  </head>
  <body>
    <nav><p>Navigation noise should be dropped.</p></nav>
    <script>window.bad = true;</script>
    <article>
      <h1>Native HTML Paper</h1>
      <h2>Abstract</h2>
      <p>Abstract body text.</p>
      <h2>Introduction</h2>
      <p>Intro text with <a href="https://example.org/data">data link</a>.</p>
      <blockquote>Quoted material belongs in text.</blockquote>
      <h2>Methods</h2>
      <ul><li>First list item.</li><li>Second list item.</li></ul>
      <figure>
        <img src="fig1.png" alt="Fallback figure alt">
        <figcaption>Figure 1. HTML figure caption.</figcaption>
      </figure>
      <table>
        <caption>Table 1. Small table.</caption>
        <thead><tr><th>A</th><th>B</th></tr></thead>
        <tbody><tr><td>1</td><td>2</td></tr></tbody>
      </table>
      <h2>References</h2>
      <ol class="references">
        <li>Smith, J. (2020). First reference.</li>
        <li>Doe, J. (2021). Second reference.</li>
      </ol>
    </article>
  </body>
</html>"""


def _segment(parser: HtmlParser):
    contents = parser._contents
    segs = [[e.text] for e in parser.assembler.entries if e.needs_segmentation]
    parser.apply_segmentation(contents, segs)
    parser.create_content_sections(contents)
    return contents


def _make_epub_bytes() -> bytes:
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         version="3.0">
  <metadata>
    <dc:title>Book Article</dc:title>
    <dc:creator>Jane Smith</dc:creator>
    <dc:identifier>10.5678/epub.1</dc:identifier>
    <dc:publisher>ePub Press</dc:publisher>
    <dc:date>2024-02-03</dc:date>
    <dc:subject>ebooks</dc:subject>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>"""
    chapter1 = b"""<html xmlns="http://www.w3.org/1999/xhtml">
  <body><section><h1>Chapter One</h1><p>Chapter one text.</p></section></body>
</html>"""
    chapter2 = b"""<html xmlns="http://www.w3.org/1999/xhtml">
  <body><section><h1>References</h1><p>Smith, J. (2020). ePub ref.</p></section></body>
</html>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chapter1.xhtml", chapter1)
        zf.writestr("OEBPS/chapter2.xhtml", chapter2)
    return buf.getvalue()


def test_html_metadata_sections_tables_figures_refs_and_noise_filtering():
    parser = HtmlParser(ARTICLE_HTML)
    contents = parser.parse()
    parser._contents = contents

    meta = contents.preparsed_metadata
    assert meta is not None
    assert meta.title == "Native HTML Paper"
    assert meta.doi == "10.1234/html.1"
    assert meta.abstract == "This abstract came from metadata."
    assert meta.keywords == ["html", "native ingestion"]
    assert meta.journal == "Journal of Markup"
    assert meta.volume == "12"
    assert meta.issue == "3"
    assert meta.first_page == "55"
    assert meta.last_page == "63"
    assert meta.published == "2025-04-09"
    assert meta.publisher == "Markup Press"
    assert [a.family for a in meta.authors] == ["Smith", "Doe"]

    headers = {s.header: s for s in contents.sections}
    assert headers["Introduction"].section_type == CanonicalSection.INTRODUCTION
    assert headers["Methods"].section_type == CanonicalSection.METHODS
    assert headers["References"].section_type == CanonicalSection.REFERENCES
    assert contents.native_ref_strings == [
        "Smith, J. (2020). First reference.",
        "Doe, J. (2021). Second reference.",
    ]
    assert len(contents.tables) == 1
    assert contents.tables[0].caption == "Table 1. Small table."
    assert contents.tables[0].contents == [["A", "B"], ["1", "2"]]
    assert len(contents.figures) == 1
    assert contents.figures[0].caption == "Figure 1. HTML figure caption."

    deferred = " ".join(t[0] for t in parser._deferred_texts)
    assert "Intro text with data link." in deferred
    assert "Navigation noise" not in deferred
    assert "window.bad" not in deferred


def test_html_segmentation_populates_links_and_content_sections():
    parser = HtmlParser(ARTICLE_HTML)
    contents = parser.parse()
    parser._contents = contents

    contents = _segment(parser)

    assert any(link.url == "https://example.org/data" for link in contents.links)
    assert any(link.link_text == "data link" for link in contents.links)
    assert any(s.header == "Figure 1" for s in contents.sections)
    assert any(s.header == "Table 1" for s in contents.sections)
    assert contents.sections_text


def test_epub_spine_metadata_and_references_delegate_to_html_parser():
    parser = EpubParser(_make_epub_bytes())
    contents = parser.parse()

    meta = contents.preparsed_metadata
    assert meta is not None
    assert meta.title == "Book Article"
    assert meta.doi == "10.5678/epub.1"
    assert meta.publisher == "ePub Press"
    assert meta.published == "2024-02-03"
    assert meta.keywords == ["ebooks"]
    assert [a.family for a in meta.authors] == ["Smith"]
    assert "Chapter one text." in " ".join(t[0] for t in parser._deferred_texts)
    assert contents.native_ref_strings == ["Smith, J. (2020). ePub ref."]


def test_html_parser_rejects_oversized_input(monkeypatch):
    """Bound html5lib's pure-Python parse cost: HTML above the cap is refused
    rather than parsed (audit L9)."""
    import pytest

    from bibr.exceptions import ProcessingError
    from bibr.input import html_native

    monkeypatch.setattr(html_native, "_MAX_HTML_BYTES", 1024)
    parser = HtmlParser(b"<html><body>" + b"x" * 4096 + b"</body></html>")
    with pytest.raises(ProcessingError):
        parser.parse()


def test_html_parser_accepts_input_under_cap(monkeypatch):
    from bibr.input import html_native

    monkeypatch.setattr(html_native, "_MAX_HTML_BYTES", 1 << 20)
    parser = HtmlParser(b"<html><body><p>Small article.</p></body></html>")
    result = parser.parse()
    assert result is not None
