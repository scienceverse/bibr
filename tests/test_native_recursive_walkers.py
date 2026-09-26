"""Regression tests for audit action 5 (native recursive walkers).

Each finding below failed on base (see FINDINGS.md): the test pins the fixed
behaviour, and the guard next to it pins the boundary the fix must not cross.
"""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("docx")

from docx import Document  # noqa: E402

from bibr.input.docx_native import DocxParser  # noqa: E402
from bibr.input.html_native import HtmlParser  # noqa: E402

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _html_texts(data: bytes) -> list[str]:
    parser = HtmlParser(data)
    parser.parse()
    return [entry.text for entry in parser.assembler.entries]


def _docx_texts(docx_bytes: bytes) -> tuple[DocxParser, list[str]]:
    parser = DocxParser(docx_bytes)
    parser.parse()
    return parser, [entry.text for entry in parser.assembler.entries]


def _save(doc: Document) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _raw_paragraph(doc: Document, inner_xml: str):
    """Append a paragraph whose content is the given raw w: XML."""
    from lxml import etree

    para = doc.add_paragraph()
    for run in list(para._element.findall(f"{{{_W}}}r")):
        para._element.remove(run)
    frag = etree.fromstring(f'<w:p xmlns:w="{_W}" xmlns:m="{_M}">{inner_xml}</w:p>')
    for child in list(frag):
        para._element.append(child)
    return para


class TestDivDirectText:
    def test_div_paragraphs_are_kept_with_inline_children(self):
        texts = _html_texts(
            b"""<html><body><article><h2>Introduction</h2>"""
            b"""<div class="para">Direct div text is common. <a href="#bib1">[1]</a>.</div>"""
            b"""<p>Paragraph kept.</p></article></body></html>"""
        )
        assert texts[0] == "Direct div text is common. [1]."
        assert "Paragraph kept." in texts

    def test_bare_section_text_is_kept_in_order(self):
        texts = _html_texts(
            b"""<html><body><section><h1>Chapter</h1>Bare section text here."""
            b"""<p>Kept p text.</p></section></body></html>"""
        )
        assert texts == ["Bare section text here.", "Kept p text."]

    def test_nested_div_paragraphs_stay_separate_entries(self):
        texts = _html_texts(
            b"""<html><body><article><h2>Intro</h2>"""
            b"""<div><div class="para">First div para.</div>"""
            b"""<div class="para">Second div para.</div></div>"""
            b"""</article></body></html>"""
        )
        assert texts == ["First div para.", "Second div para."]

    def test_div_with_block_children_recurses(self):
        texts = _html_texts(
            b"""<html><body><article><h2>Intro</h2>"""
            b"""<div><p>One.</p><p>Two.</p></div></article></body></html>"""
        )
        assert texts == ["One.", "Two."]

    def test_noise_tags_still_dropped(self):
        texts = _html_texts(
            b"""<html><body><article><h2>Intro</h2>"""
            b"""<div>Kept.<script>var x = 1;</script><style>.a{}</style></div>"""
            b"""</article></body></html>"""
        )
        assert texts == ["Kept."]

    def test_inline_wrapper_around_blocks_recurses(self):
        # An unclosed <b> nests whole sections inside an inline element;
        # the block structure wins instead of flattening into body text.
        parser = HtmlParser(
            b"""<html><body><article><h2>Intro</h2><p>Body.</p>"""
            b"""<b><section><h2>Footer head</h2><p>Footer text.</p></section></b>"""
            b"""</article></body></html>"""
        )
        contents = parser.parse()
        assert [s.header for s in contents.sections] == ["Root", "Intro", "Footer head"]
        assert [e.text for e in parser.assembler.entries] == ["Body.", "Footer text."]


class TestHtmlCharset:
    _BODY = "<html><body><article><h2>%s</h2><p>%s</p></article></body></html>"

    def test_utf8_without_meta_charset(self):
        data = (self._BODY % ("Über Größe", "Café naïve résumé")).encode("utf-8")
        assert _html_texts(data) == ["Café naïve résumé"]

    def test_xml_declaration_encoding_is_honoured(self):
        data = (
            '<?xml version="1.0" encoding="UTF-8"?>' + self._BODY % ("Über Größe", "Café")
        ).encode("utf-8")
        parser = HtmlParser(data)
        contents = parser.parse()
        assert contents.detected_title == "Über Größe"
        assert _html_texts(data) == ["Café"]

    def test_declared_windows_1252_still_decodes(self):
        data = (
            '<html><head><meta charset="windows-1252"></head>'
            "<body><article><h2>Intro</h2><p>Caf\xe9</p></article></body></html>"
        ).encode("latin-1")
        assert _html_texts(data) == ["Café"]


class TestFigureWrappedTable:
    def test_figure_wrapping_a_table_becomes_a_table(self):
        parser = HtmlParser(
            b"""<html><body><article><h2>Results</h2>"""
            b"""<figure class="table"><figcaption>Table 1. Demographics</figcaption>"""
            b"""<table><tr><th>A</th></tr><tr><td>1</td></tr></table></figure>"""
            b"""<p>Body.</p></article></body></html>"""
        )
        contents = parser.parse()
        assert contents.figures == []
        (table,) = contents.tables
        assert table.caption == "Table 1. Demographics"
        assert table.label == "1"

    def test_figure_with_an_image_and_a_layout_table_stays_a_figure(self):
        parser = HtmlParser(
            b"""<html><body><article><h2>Results</h2>"""
            b"""<figure><img src="a.png"><figcaption>Figure 1. Main.</figcaption>"""
            b"""<table><tr><td>x</td></tr></table></figure>"""
            b"""</article></body></html>"""
        )
        contents = parser.parse()
        assert [fig.caption for fig in contents.figures] == ["Figure 1. Main."]
        assert contents.tables == []


class TestHtmlAuthorMetadata:
    _HEAD = "<html><head>%s</head><body><article><h2>Intro</h2><p>Body.</p></article></body></html>"

    def _meta(self, head: str):
        parser = HtmlParser((self._HEAD % head).encode())
        parser.parse()
        return parser._metadata

    def test_first_source_wins_and_institutions_pair_up(self):
        meta = self._meta(
            '<meta name="citation_author" content="Jane Smith">'
            '<meta name="citation_author" content="John Doe">'
            '<meta name="dc.creator" content="Jane Smith">'
            '<meta name="dc.creator" content="John Doe">'
            '<meta name="citation_author_institution" content="Utrecht University">'
        )
        assert [(a.given, a.family, a.affiliation) for a in meta.authors] == [
            ("Jane", "Smith", "Utrecht University"),
            ("John", "Doe", ""),
        ]

    def test_non_doi_identifier_is_not_a_doi(self):
        meta = self._meta(
            '<meta name="citation_title" content="T">'
            '<meta name="citation_author" content="Jane Smith">'
            '<meta name="dc.identifier"'
            ' content="https://journal.example.org/article/view/123">'
        )
        assert meta.doi == ""

    def test_bare_doi_identifier_normalizes(self):
        meta = self._meta(
            '<meta name="citation_title" content="T">'
            '<meta name="citation_author" content="Jane Smith">'
            '<meta name="dc.identifier" content="https://doi.org/10.1234/abc.1">'
        )
        assert meta.doi == "10.1234/abc.1"

    def test_generic_author_list_splits_but_family_given_stays(self):
        meta = self._meta('<meta name="author" content="Jane Smith, John Doe">')
        assert [(a.given, a.family) for a in meta.authors] == [
            ("Jane", "Smith"),
            ("John", "Doe"),
        ]
        single = self._meta('<meta name="author" content="Smith, Jane">')
        assert [(a.given, a.family) for a in single.authors] == [("Jane", "Smith")]

    def test_citation_orcid_pairs_up(self):
        meta = self._meta(
            '<meta name="citation_author" content="Jane Smith">'
            '<meta name="citation_author_orcid" content="0000-0002-1825-0097">'
        )
        assert meta.authors[0].orcid == "https://orcid.org/0000-0002-1825-0097"


class TestPreparsedGating:
    _SEO = (
        b"""<html><head><title>Sleep and memory | Journal of Sleep | Example Press</title>"""
        b"""<meta name="description" content="Read the latest research from Example Press.">"""
        b"""<meta name="author" content="Jane Smith, John Doe"></head>"""
        b"""<body><article><h1>Sleep and memory</h1><h2>Abstract</h2>"""
        b"""<p>We tested 80 people.</p></article></body></html>"""
    )
    _CITED = (
        b"""<html><head><meta name="citation_title" content="Cited Paper">"""
        b"""<meta name="citation_author" content="Jane Smith">"""
        b"""<meta name="citation_abstract" content="Author-written abstract."></head>"""
        b"""<body><article><h1>Cited Paper</h1><h2>Abstract</h2>"""
        b"""<p>We tested 80 people.</p></article></body></html>"""
    )

    def test_seo_only_page_returns_no_preparsed_metadata(self):
        assert HtmlParser(self._SEO).parse().preparsed_metadata is None

    def test_seo_description_is_not_the_abstract(self):
        parser = HtmlParser(self._SEO)
        parser.parse()
        assert parser._metadata.abstract == ""

    def test_site_suffix_strips_when_h1_matches(self):
        parser = HtmlParser(self._SEO)
        parser.parse()
        assert parser._metadata.title == "Sleep and memory"

    def test_hyphenated_title_without_suffix_spacing_is_untouched(self):
        parser = HtmlParser(
            b"""<html><head><title>Sleep-memory study</title></head>"""
            b"""<body><article><h1>Sleep</h1><p>Body.</p></article></body></html>"""
        )
        parser.parse()
        assert parser._metadata.title == "Sleep-memory study"

    def test_citation_front_matter_still_preparsed(self):
        contents = HtmlParser(self._CITED).parse()
        assert contents.preparsed_metadata is not None
        assert contents.preparsed_metadata.abstract == "Author-written abstract."


class TestDocxInlineWalker:
    def test_inline_sdt_citation_is_kept(self):
        doc = Document()
        _raw_paragraph(
            doc,
            "<w:r><w:t>As shown previously </w:t></w:r>"
            '<w:sdt><w:sdtPr><w:alias w:val="citation"/></w:sdtPr>'
            "<w:sdtContent><w:r><w:t>(Smith et al., 2020)</w:t></w:r></w:sdtContent></w:sdt>"
            "<w:r><w:t>, effects were large.</w:t></w:r>",
        )
        _, texts = _docx_texts(_save(doc))
        assert texts == ["As shown previously (Smith et al., 2020), effects were large."]

    def test_smart_tag_fldsimple_sym_and_hyphen(self):
        doc = Document()
        _raw_paragraph(
            doc,
            "<w:r><w:t>Data collected in </w:t></w:r>"
            "<w:smartTag><w:r><w:t>Amsterdam</w:t></w:r></w:smartTag>"
            "<w:r><w:t> in 2019. See </w:t></w:r>"
            '<w:fldSimple w:instr="REF _Ref1"><w:r><w:t>Table 2</w:t></w:r></w:fldSimple>'
            "<w:r><w:t> for details. Level </w:t></w:r>"
            '<w:r><w:sym w:font="Symbol" w:char="F061"/></w:r>'
            "<w:r><w:t> = .05 for COVID</w:t></w:r>"
            "<w:r><w:noBreakHyphen/><w:t>19 cases.</w:t></w:r>",
        )
        _, texts = _docx_texts(_save(doc))
        assert texts == [
            "Data collected in Amsterdam in 2019. "
            "See Table 2 for details. Level \u03b1 = .05 for COVID-19 cases."
        ]

    def test_tracked_move_destination_is_kept(self):
        doc = Document()
        _raw_paragraph(
            doc,
            "<w:moveTo><w:r><w:t>Moved sentence that should remain.</w:t></w:r></w:moveTo>",
        )
        _, texts = _docx_texts(_save(doc))
        assert texts == ["Moved sentence that should remain."]

    def test_deletions_move_sources_and_field_code_stay_skipped(self):
        doc = Document()
        _raw_paragraph(
            doc,
            "<w:r><w:t>Kept</w:t></w:r>"
            "<w:del><w:r><w:delText>gone</w:delText></w:r></w:del>"
            "<w:moveFrom><w:r><w:t>gone</w:t></w:r></w:moveFrom>"
            "<w:r><w:instrText>PAGE</w:instrText></w:r>"
            "<w:r><w:t> here.</w:t></w:r>",
        )
        _, texts = _docx_texts(_save(doc))
        assert texts == ["Kept here."]

    def test_complex_field_result_runs_are_kept(self):
        doc = Document()
        _raw_paragraph(
            doc,
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            "<w:r><w:instrText> CITATION Smi20 </w:instrText></w:r>"
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            "<w:r><w:t>(Smith, 2020)</w:t></w:r>"
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>',
        )
        _, texts = _docx_texts(_save(doc))
        assert texts == ["(Smith, 2020)"]


class TestDocxHeadingFootnotes:
    def _footnote_doc(self):
        import zipfile

        from lxml import etree

        doc = Document()
        title = doc.add_paragraph("Cognitive load and recall")
        title.style = doc.styles["Title"]
        doc.add_heading("Intro heading", level=1)
        doc.add_paragraph("Body text.")
        data = _save(doc)

        parts = {}
        with zipfile.ZipFile(io.BytesIO(data)) as zin:
            for name in zin.namelist():
                parts[name] = zin.read(name)
        root = etree.fromstring(parts["word/document.xml"])
        ns = {"w": _W}
        paras = root.findall(".//w:p", ns)
        for index, note_id in enumerate(["1", "2"]):
            run = paras[index].find("w:r", ns)
            ref = etree.SubElement(run, f"{{{_W}}}footnoteReference")
            ref.set(f"{{{_W}}}id", note_id)
        parts["word/document.xml"] = etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )
        parts["word/footnotes.xml"] = (
            f'<w:footnotes xmlns:w="{_W}">'
            '<w:footnote w:id="1"><w:p><w:r><w:t>Author note funded.</w:t></w:r></w:p></w:footnote>'
            '<w:footnote w:id="2"><w:p><w:r><w:t>Preregistered note.</w:t></w:r></w:p></w:footnote>'
            "</w:footnotes>"
        ).encode()
        cts = parts["[Content_Types].xml"].decode()
        if "footnotes" not in cts:
            cts = cts.replace(
                "</Types>",
                '<Override PartName="/word/footnotes.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument'
                '.wordprocessingml.footnotes+xml"/></Types>',
            )
            parts["[Content_Types].xml"] = cts.encode()
        rels = parts["word/_rels/document.xml.rels"].decode()
        if "footnotes" not in rels:
            rels = rels.replace(
                "</Relationships>",
                '<Relationship Id="rFoot1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006'
                '/relationships/footnotes" Target="footnotes.xml"/></Relationships>',
            )
            parts["word/_rels/document.xml.rels"] = rels.encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, blob in parts.items():
                zout.writestr(name, blob)
        return buf.getvalue()

    def test_heading_anchored_footnotes_are_emitted(self):
        parser = DocxParser(self._footnote_doc())
        contents = parser.parse()
        parser.apply_segmentation(
            contents,
            [[entry.text] for entry in parser.assembler.entries if entry.needs_segmentation],
        )
        parser.create_content_sections(contents)
        headers = [section.header for section in contents.sections]
        assert "Footnote 1" in headers
        assert "Footnote 2" in headers
        text = " ".join(sentence.text for sentence in contents.sentences)
        assert "Author note funded." in text
        assert "Preregistered note." in text


class TestDocxCustomHeadings:
    def test_style_based_on_heading_is_a_heading(self):
        doc = Document()
        based = doc.styles.add_style("APA Heading 1", 1)
        based.base_style = doc.styles["Heading 1"]
        doc.add_paragraph("Method", style="APA Heading 1")
        doc.add_paragraph("We recruited 80 participants.")
        doc.add_paragraph("Results", style="APA Heading 1")
        doc.add_paragraph("Effects were large.")
        contents = DocxParser(_save(doc)).parse()
        assert [s.header for s in contents.sections] == ["Root", "Method", "Results"]

    def test_outline_level_marks_a_heading(self):
        from lxml import etree

        doc = Document()
        para = doc.add_paragraph("Discussion")
        ppr = para._element.get_or_add_pPr()
        lvl = etree.SubElement(ppr, f"{{{_W}}}outlineLvl")
        lvl.set(f"{{{_W}}}val", "1")
        doc.add_paragraph("Body.")
        contents = DocxParser(_save(doc)).parse()
        assert [s.header for s in contents.sections] == ["Root", "Discussion"]

    def test_normal_style_stays_body_text(self):
        doc = Document()
        doc.add_paragraph("Just a sentence.")
        contents = DocxParser(_save(doc)).parse()
        assert [s.header for s in contents.sections] == ["Root"]


class TestDocxDetectedTitle:
    def test_hand_formatted_title_does_not_promote_introduction(self):
        doc = Document()
        lead = doc.add_paragraph("My Hand-Formatted Title")
        for run in lead.runs:
            run.bold = True
        doc.add_heading("Introduction", level=1)
        doc.add_paragraph("Intro body.")
        doc.add_heading("Method", level=1)
        doc.add_paragraph("Method body.")
        contents = DocxParser(_save(doc)).parse()
        assert contents.detected_title != "Introduction"

    def test_title_style_still_detected(self):
        doc = Document()
        title = doc.add_paragraph("Real Title")
        title.style = doc.styles["Title"]
        doc.add_heading("Introduction", level=1)
        contents = DocxParser(_save(doc)).parse()
        assert contents.detected_title == "Real Title"

    def test_first_heading_one_that_is_not_a_section_stays_title(self):
        doc = Document()
        doc.add_heading("Paper About X", level=1)
        doc.add_paragraph("Body text.")
        contents = DocxParser(_save(doc)).parse()
        assert contents.detected_title == "Paper About X"


class TestDocxFigureCaptions:
    def _picture_doc(self, build):
        doc = Document()
        build(doc)
        return _save(doc)

    def _add_pic(self, doc):
        from docx.shared import Pt

        doc.add_paragraph().add_run().add_picture(io.BytesIO(_PNG), width=Pt(10))

    def test_icon_does_not_shift_later_captions(self):
        def build(doc):
            doc.add_paragraph("Intro text.")
            self._add_pic(doc)
            self._add_pic(doc)
            doc.add_paragraph("Figure 1. Main result.", style="Caption")
            self._add_pic(doc)
            doc.add_paragraph("Figure 2. Second result.", style="Caption")

        contents = DocxParser(self._picture_doc(build)).parse()
        assert [(fig.caption, fig.label) for fig in contents.figures] == [
            (None, None),
            ("Figure 1. Main result.", "1"),
            ("Figure 2. Second result.", "2"),
        ]

    def test_caption_above_the_figure_pairs(self):
        def build(doc):
            doc.add_paragraph("Figure 1. Above.", style="Caption")
            self._add_pic(doc)

        contents = DocxParser(self._picture_doc(build)).parse()
        assert [(fig.caption, fig.label) for fig in contents.figures] == [("Figure 1. Above.", "1")]

    def test_body_text_between_picture_and_caption_breaks_pairing(self):
        def build(doc):
            self._add_pic(doc)
            doc.add_paragraph("Some body text.")
            doc.add_paragraph("Figure 1. Orphan.", style="Caption")

        contents = DocxParser(self._picture_doc(build)).parse()
        assert [fig.caption for fig in contents.figures] == [None]
