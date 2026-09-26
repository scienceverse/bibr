"""Tests for native HTML/ePub ingestion."""

from __future__ import annotations

import io
import zipfile

import pytest

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
    assert contents.tables[0].label == "1"
    assert contents.tables[0].contents == [["A", "B"], ["1", "2"]]
    assert len(contents.figures) == 1
    assert contents.figures[0].caption == "Figure 1. HTML figure caption."
    assert contents.figures[0].label == "1"

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


def test_html_anchor_in_first_sentence_is_recorded_once_there():
    html = (
        b"<!doctype html><html><body><article><h2>Intro</h2>"
        b'<p>Data are at <a href="https://example.org/data">https://example.org/data</a>. '
        b"Second sentence follows here.</p>"
        b"</article></body></html>"
    )
    parser = HtmlParser(html)
    contents = parser.parse()
    parser._contents = contents
    parser.apply_segmentation(
        contents,
        [["Data are at https://example.org/data.", "Second sentence follows here."]],
    )
    assert [(link.url, link.text_id) for link in contents.links] == [
        ("https://example.org/data", 1)
    ]


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


def _make_epub_with_entity_metadata() -> bytes:
    """An EPUB 2-style OPF carrying a DOCTYPE and named character entities."""
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = b"""<?xml version="1.0"?>
<!DOCTYPE package SYSTEM "http://www.idpf.org/dtds/2007/opf.dtd">
<package xmlns="http://www.idpf.org/2007/opf"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         version="2.0">
  <metadata>
    <dc:title>The &alpha;-Helix at 37&deg;C</dc:title>
    <dc:creator>Jos&eacute; M&uuml;ller</dc:creator>
    <dc:publisher>Verlag M&uuml;nchen</dc:publisher>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    chapter1 = b"""<html xmlns="http://www.w3.org/1999/xhtml">
  <body><section><h1>Chapter One</h1><p>Chapter one text.</p></section></body>
</html>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chapter1.xhtml", chapter1)
    return buf.getvalue()


def test_epub_opf_metadata_resolves_named_entities():
    """The OPF is parsed with entity expansion off, so named entities would
    otherwise ship as literal "&alpha;" markup in the title and author names."""
    meta = EpubParser(_make_epub_with_entity_metadata()).parse().preparsed_metadata

    assert meta.title == "The α-Helix at 37°C"
    assert [a.family for a in meta.authors] == ["Müller"]
    assert meta.publisher == "Verlag München"


def _make_epub_utf8(title: str, creator: str, body: str, extra: str = "") -> bytes:
    """An ePub whose OPF and chapter carry literal UTF-8, no entities.

    *extra* is markup placed after the chapter's paragraph."""
    container_xml = b"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         version="3.0">
  <metadata><dc:title>{title}</dc:title><dc:creator>{creator}</dc:creator></metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/></spine>
</package>""".encode()
    chapter1 = f"""<html xmlns="http://www.w3.org/1999/xhtml">
  <body><section><h1>Kapitel</h1><p>{body}</p>{extra}</section></body>
</html>""".encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chapter1.xhtml", chapter1)
    return buf.getvalue()


def test_epub_non_ascii_survives_the_synthesized_html():
    """The spine is re-emitted as one HTML document for HtmlParser. Without a
    charset declaration html5lib decodes those UTF-8 bytes as windows-1252 and
    mojibakes every non-ASCII character in the metadata and the body."""
    epub = _make_epub_utf8(
        "Der α-Helix Effekt", "José Müller", "Die Temperatur betrug 37°C bei München."
    )
    parser = EpubParser(epub)
    meta = parser.parse().preparsed_metadata

    assert meta.title == "Der α-Helix Effekt"
    assert [(a.given, a.family) for a in meta.authors] == [("José", "Müller")]
    body = " ".join(t[0] for t in parser._deferred_texts)
    assert "37°C bei München" in body


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


def test_declared_identifiers_and_language_reach_the_metadata():
    html = b"""<html lang="de"><head>
      <meta name="citation_title" content="Identified">
      <meta name="citation_pmid" content="31234567">
      <meta name="citation_arxiv_id" content="2101.12345">
    </head><body><article><h1>Intro</h1><p>Text.</p></article></body></html>"""
    meta = HtmlParser(html).parse().preparsed_metadata
    assert meta is not None
    assert (meta.pmid, meta.arxiv, meta.language) == ("31234567", "2101.12345", "de")


def test_html_figcaption_label_resolves_mentions():
    html = b"""<!doctype html><html><body><article>
      <h1>Labelled</h1>
      <p>Figure S3 and Figure 2 agree.</p>
      <figure><img src="a.png"><figcaption>Figure 2. Main.</figcaption></figure>
      <figure><img src="b.png"><figcaption>Supplementary Figure 3. Extra.</figcaption></figure>
      <figure><img src="c.png" alt="Figure 4 as alt text"></figure>
    </article></body></html>"""
    parser = HtmlParser(html)
    parser._contents = parser.parse()
    contents = _segment(parser)

    assert [figure.label for figure in contents.figures] == ["2", "S3", "4"]
    assert [(x.xref_type, x.xref_id, x.tier) for x in contents.xrefs] == [
        ("figure", 2, "label"),
        ("figure", 1, "label"),
    ]


def test_html_sentences_are_not_ocr_text_and_keep_their_prose():
    """Late cleanup assumed OCR input and fused "a 2 x 2 design" into "a2x2"
    and dropped the underscores from ``age_group`` and email addresses."""
    prose = [
        "Participants completed a 2 x 2 x 3 mixed design.",
        "Items 1 2 3 and 4 were reverse scored; age_group was coded.",
        "Data are available from john_smith@uni.edu on request.",
    ]
    html = (
        "<html><body><article><h2>Method</h2><p>" + " ".join(prose) + "</p></article></body></html>"
    ).encode()
    parser = HtmlParser(html)
    parser._contents = parser.parse()
    contents = parser._contents
    parser.apply_segmentation(
        contents, [list(prose) for e in parser.assembler.entries if e.needs_segmentation]
    )
    parser.create_content_sections(contents)

    contents.finalize_text()
    assert [s.text for s in contents.sentences] == prose
    assert all(sentence.from_ocr is False for sentence in contents.sentences)


def test_inline_markup_stays_attached_to_its_word():
    """``get_text(" ")`` put a space around every element: "H 2 S", "m 6 A",
    "( Figure 1 )". Only block-level elements separate words."""
    html = (
        b"<html><body><article><h2>Method</h2>"
        b"<p>Dissolved H<sub>2</sub>S and m<sup>6</sup>A (<a href='#f1'>Figure 1</a>) in "
        b"<i>Mus musculus</i><!-- note --> cells, by the <i>p</i>-value.<br>Next line.</p>"
        b"<ul><li>One</li><li>Two</li></ul></article></body></html>"
    )
    parser = HtmlParser(html)
    parser.parse()

    assert [entry.text for entry in parser.assembler.entries] == [
        "Dissolved H2S and m6A (Figure 1) in Mus musculus cells, by the p-value. Next line.",
        "One",
        "Two",
    ]


@pytest.mark.parametrize(
    "body",
    [
        "<p>" + "".join(f"<span>w{i} " for i in range(2000)) + "</p>",
        "".join(f"<p><font size={i % 7 + 1}>para {i}." for i in range(2000)),
    ],
    ids=["unclosed-spans", "font-soup"],
)
def test_deeply_nested_legacy_markup_still_parses(body):
    """html5lib nests every element after an unclosed ``<span>`` or
    ``<font>`` one level deeper; a recursive walk hit the recursion limit and
    failed the whole document."""
    html = ("<html><body><article><h2>Method</h2>" + body + "</article></body></html>").encode()
    parser = HtmlParser(html)
    parser.parse()

    texts = [entry.text for entry in parser.assembler.entries]
    assert texts
    assert texts[0].startswith(("w0 w1 ", "para 0."))


def test_html_captions_are_not_ocr_text():
    """Captions are document text too: the late cleanup's spaced-run collapse
    turned "items 1 2 3" into "items 123"."""
    figure_caption = "Figure 1. Scores on items 1 2 3 by age_group."
    table_caption = "Table 1. Items 1 2 3 by age_group."
    html = (
        "<html><body><article><h2>Results</h2><p>Body text.</p>"
        f'<figure><img src="a.png"><figcaption>{figure_caption}</figcaption></figure>'
        f"<table><caption>{table_caption}</caption><tr><th>A</th></tr><tr><td>1</td></tr>"
        "</table></article></body></html>"
    ).encode()
    parser = HtmlParser(html)
    parser._contents = parser.parse()
    contents = _segment(parser)

    contents.finalize_text()
    texts = [s.text for s in contents.sentences]
    assert figure_caption in texts
    assert table_caption in texts
    assert all(sentence.from_ocr is False for sentence in contents.sentences)


@pytest.mark.parametrize(
    ("math", "expected"),
    [
        (
            "<mi>β</mi> <mo>∼</mo> <mtext>Cauchy</mtext> <mo>(</mo> <mn>0</mn> <mo>,</mo> "
            "<mn>2</mn> <mo>.</mo> <mn>5</mn> <mo>)</mo>",
            "β∼Cauchy(0,2.5)",
        ),
        ("<msub><mi>t</mi> <mrow><mi>i</mi> <mo>-</mo> <mn>1</mn></mrow></msub>", "ti-1"),
        ("<mi>ln</mi> <mi>dbh</mi> <mo>+</mo> <mn>0.93</mn> <mtext>GeV</mtext>", "ln dbh+0.93 GeV"),
        ("<mi>M</mi> <mo>=</mo> <mfrac><mn>1</mn> <mn>2</mn></mfrac>", "M=1 2"),
        (
            "<mi>J</mi> <mo>=</mo> <mrow><mo>[</mo> <mtable><mtr><mtd><mn>0</mn></mtd> "
            "<mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>10</mn></mtd> <mtd><mn>20</mn></mtd>"
            "</mtr></mtable> <mo>]</mo></mrow>",
            "J=[ 0 1 10 20 ]",
        ),
        (
            "<msub><mi>E</mi> <mrow><mi>t</mi> <mo>-</mo> <mn>1</mn></mrow></msub> "
            '<mspace width="8pt"/> <mn>0</mn> <mo>&lt;</mo> <mi>λ</mi>',
            "Et-1 0<λ",
        ),
    ],
    ids=["decimal", "index", "words", "fraction", "matrix", "mspace"],
)
def test_whitespace_between_mathml_elements_is_dropped_unless_it_separates_words(math, expected):
    """eLife's HTML pretty-prints MathML like its JATS; kept as text the
    whitespace split "2.5" into "2. 5" (see the JATS parser's tests). Two
    numbers, matrix cells and ``<mspace>`` stay apart."""
    html = (
        '<html><head><meta charset="utf-8"></head><body><article><h2>Method</h2>'
        f"<p>We used <math>{math}</math> here.</p></article></body></html>"
    )
    parser = HtmlParser(html.encode())
    parser.parse()

    assert [entry.text for entry in parser.assembler.entries] == [f"We used {expected} here."]


def test_mathml_fraction_in_a_table_cell_keeps_its_numbers_apart():
    html = (
        b"<html><body><article><h2>Results</h2><p>Body text.</p>"
        b"<table><caption>Table 1. Shares.</caption><tr><th>share</th><th>n</th></tr>"
        b"<tr><td><math><mfrac><mn>3</mn> <mn>4</mn></mfrac></math></td><td>12</td></tr>"
        b"</table></article></body></html>"
    )
    parser = HtmlParser(html)
    parser._contents = parser.parse()
    contents = _segment(parser)

    assert contents.tables[0].contents == [["share", "n"], ["3 4", "12"]]


# ``pandas.read_html`` inferred column types: "0.050" came back as 0.05, "007"
# as 7, a decimal comma "1,5" as 15, and an empty cell as the string "nan".
_NUMERIC_TABLE = (
    "<table><caption>Table 2. Results.</caption>"
    "<thead><tr><th>N</th><th>Code</th><th>M</th><th>p</th></tr></thead>"
    "<tbody><tr><td>12</td><td>007</td><td>1,234</td><td>0.050</td></tr>"
    "<tr><td></td><td>010</td><td>1,5</td><td>0.10</td></tr></tbody></table>"
)
_NUMERIC_CONTENTS = [
    ["N", "Code", "M", "p"],
    ["12", "007", "1,234", "0.050"],
    ["", "010", "1,5", "0.10"],
]


def _article(body: str) -> bytes:
    return (
        f"<html><body><article><h1>Results</h1><p>Text.</p>{body}</article></body></html>".encode()
    )


def test_html_table_cells_are_exported_as_printed():
    from pathlib import Path

    from bibr.export import export_paper_to_json
    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper

    contents = HtmlParser(_article(_NUMERIC_TABLE)).parse()

    assert contents.tables[0].contents == _NUMERIC_CONTENTS
    paper = Paper(
        input_file=InputFile(
            path=Path("/tmp/paper.html"),
            file_hash="hash",
            input_format=InputFormat(
                file_extension=".html", detected_mime_type="text/html", file_type="html"
            ),
        ),
        metadata=PaperMetadata(title="Paper", doi=""),
        contents=contents,
    )
    exported = export_paper_to_json(paper, validate=True)["table"]
    assert [table["contents"] for table in exported] == [_NUMERIC_CONTENTS]


def test_epub_table_cells_keep_printed_text():
    parser = EpubParser(_make_epub_utf8("Tabelle", "Jane Smith", "Text.", _NUMERIC_TABLE))

    contents = parser.parse()

    assert [table.contents for table in contents.tables] == [_NUMERIC_CONTENTS]


def test_captioned_table_without_cell_text_is_kept():
    """A table with no cell grid (an image-only table) was dropped with its
    caption; its caption and markup now survive with empty contents."""
    contents = HtmlParser(
        _article(
            '<table><caption>Table 3. Scanned values.</caption><tr><td><img src="t3.png">'
            "</td></tr></table>"
            '<table><tr><td><img src="spacer.png"></td></tr></table>'
        )
    ).parse()

    assert len(contents.tables) == 1
    table = contents.tables[0]
    assert table.caption == "Table 3. Scanned values."
    assert table.label == "3"
    assert table.contents == []
    assert 'src="t3.png"' in table.tbl_html


@pytest.mark.parametrize(
    "caption", ["Figure 1. A layout-table figure.", "Scanned values without a label."]
)
def test_gridless_table_without_a_table_label_is_dropped(caption):
    """Only a caption that prints a table label makes a <table> with no cell
    text a table: a layout table holding a figure, or one whose caption names
    no table, is not one."""
    contents = HtmlParser(
        _article(f'<table><caption>{caption}</caption><tr><td><img src="f1.png"></td></tr></table>')
    ).parse()

    assert contents.tables == []


def test_hidden_table_is_dropped_and_mentions_resolve_to_the_visible_one():
    """A display:none copy of a table (print-only or responsive markup) is not
    part of the page; kept, it came out as an empty "Table 1" ahead of the
    visible one and the mention no longer resolved."""
    html = b"""<!doctype html><html><body><article>
      <h1>Results</h1>
      <p>Baseline characteristics are in Table 1.</p>
      <table style="display: none"><caption>Table 1. Baseline.</caption>
        <tr><th>Group</th><th>N</th></tr><tr><td>Control</td><td>120</td></tr></table>
      <table><caption>Table 1. Baseline.</caption>
        <tr><th>Group</th><th>N</th></tr><tr><td>Control</td><td>120</td></tr></table>
    </article></body></html>"""
    parser = HtmlParser(html)
    parser._contents = parser.parse()
    contents = _segment(parser)

    assert [(t.label, t.contents) for t in contents.tables] == [
        ("1", [["Group", "N"], ["Control", "120"]])
    ]
    assert [(x.xref_type, x.xref_id) for x in contents.xrefs] == [("table", 1)]
