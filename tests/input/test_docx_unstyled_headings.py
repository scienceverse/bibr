"""DOCX headings and reference lists that no heading style marks (#215).

A manuscript whose author bolded its headings by hand ("Abstract", "Method",
"References" as Normal paragraphs in bold) was read without a single section,
so no reference list was found and ``bib`` came out empty, although a reference
manager (EndNote) had generated the list. These tests cover the signs the
parser now reads: outline levels, bold standalone paragraphs in a document
without heading styles, a paragraph reading "References" in any document, and
the paragraph styles and fields reference managers generate their lists with.
"""

from __future__ import annotations

import io
import zipfile
from html import escape

import pytest

pytest.importorskip("docx")

from bibr.extract.ref_locator import RefLocator
from bibr.input.docx_native import DocxParser

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_WML = "application/vnd.openxmlformats-officedocument.wordprocessingml"

_STYLES = (
    f'<w:styles xmlns:w="{W}">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="TOCHeading"><w:name w:val="TOC Heading"/>'
    '<w:basedOn w:val="Heading1"/><w:pPr><w:outlineLvl w:val="9"/></w:pPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="MySection"><w:name w:val="My Section"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="BoldPara"><w:name w:val="Bold Para"/>'
    '<w:basedOn w:val="Normal"/><w:rPr><w:b/></w:rPr></w:style>'
    '<w:style w:type="character" w:styleId="Strong"><w:name w:val="Strong"/>'
    "<w:rPr><w:b/></w:rPr></w:style>"
    '<w:style w:type="paragraph" w:styleId="EndNoteBibliography">'
    '<w:name w:val="EndNote Bibliography"/><w:basedOn w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/>'
    '<w:basedOn w:val="Normal"/></w:style>'
    "</w:styles>"
)

REFS = [
    "American Psychological Association. (2009). Report of the task force on appropriate "
    "therapeutic responses to sexual orientation. APA.",
    "Anderson, J. R., Darke, N., & Smith, K. (2021). Conversion ideology and mental health. "
    "Journal of Homosexuality, 68(4), 123-145.",
    "Bishop, A. (2019). Harmful treatment: The global reach of conversion therapy. OutRight.",
    "Meyer, I. H. (2003). Prejudice, social stress, and mental health in lesbian, gay, and "
    "bisexual populations. Psychological Bulletin, 129(5), 674-697.",
]


def _run(text: str, *, bold: bool = False, char_style: str | None = None) -> str:
    props = ""
    if char_style:
        props += f'<w:rStyle w:val="{char_style}"/>'
    if bold:
        props += "<w:b/>"
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _para(*runs: str, style: str | None = None, outline: int | None = None) -> str:
    props = f'<w:pStyle w:val="{style}"/>' if style else ""
    if outline is not None:
        props += f'<w:outlineLvl w:val="{outline}"/>'
    ppr = f"<w:pPr>{props}</w:pPr>" if props else ""
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def _bold(text: str) -> str:
    return _para(_run(text, bold=True))


def _plain(text: str, **kwargs) -> str:
    return _para(_run(text), **kwargs)


def _field_char(kind: str) -> str:
    return f'<w:r><w:fldChar w:fldCharType="{kind}"/></w:r>'


def _instr(text: str) -> str:
    return f'<w:r><w:instrText xml:space="preserve">{escape(text)}</w:instrText></w:r>'


def _field_list(entries: list[str], instruction: str, style: str | None = None) -> str:
    """Entries inside one complex field spanning their paragraphs, as EndNote
    writes "ADDIN EN.REFLIST": begin and instruction in the first, end in the last."""
    out = []
    for i, entry in enumerate(entries):
        runs = []
        if i == 0:
            runs += [_field_char("begin"), _instr(instruction), _field_char("separate")]
        runs.append(_run(entry))
        if i == len(entries) - 1:
            runs.append(_field_char("end"))
        out.append(_para(*runs, style=style))
    return "".join(out)


def _docx(body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/word/document.xml" ContentType="{_WML}.document.main+xml"/>'
            f'<Override PartName="/word/styles.xml" ContentType="{_WML}.styles+xml"/>'
            "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{_RELS}"><Relationship Id="rId1" '
            f'Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        z.writestr(
            "word/_rels/document.xml.rels",
            f'<Relationships xmlns="{_RELS}"><Relationship Id="rIdStyles" '
            f'Type="{R}/styles" Target="styles.xml"/></Relationships>',
        )
        z.writestr("word/styles.xml", _STYLES)
        z.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>',
        )
    return buf.getvalue()


def _parse(body: str):
    parser = DocxParser(_docx(body))
    contents = parser.parse()
    parser.apply_segmentation(contents, [[entry.text] for entry in parser.assembler.entries])
    return contents


def _headers(contents) -> list[str]:
    return [section.header for section in contents.sections[1:]]


def _section_texts(contents, header: str) -> list[str]:
    ids = {s.section_id for s in contents.sections if s.header == header}
    return [s.text for s in contents.sentences if s.section_id in ids]


def _reference_rows(contents) -> list[str]:
    return RefLocator(contents).collect_reference_rows()["text"].tolist()


def _manuscript(references: str) -> str:
    """A manuscript with every heading a bold Normal paragraph, as in #215."""
    return (
        _bold("Exposure to Conversion Ideology Scale: A multidimensional measure")
        + _plain("Jane Doe, John Roe")
        + _bold("Abstract")
        + _plain("Conversion practices harm LGBTQA+ people (Meyer, 2003).")
        + _bold("Method")
        + _plain("We recruited 500 participants online (Anderson et al., 2021).")
        + _bold("Discussion")
        + _plain("Our findings extend earlier work (Bishop, 2019).")
        + references
    )


def test_bold_headings_and_an_endnote_list_give_sections_and_references():
    """The shape of the PsyArXiv preprint in #215."""
    body = _manuscript(
        _bold("References") + _field_list(REFS, " ADDIN EN.REFLIST ", style="EndNoteBibliography")
    )
    contents = _parse(body)

    assert _headers(contents) == ["Abstract", "Method", "Discussion", "References"]
    assert _section_texts(contents, "Method") == [
        "We recruited 500 participants online (Anderson et al., 2021)."
    ]
    assert _reference_rows(contents) == REFS


@pytest.mark.parametrize(
    "references",
    [
        pytest.param(_field_list(REFS, " ADDIN EN.REFLIST "), id="endnote-field"),
        pytest.param(
            _field_list(REFS, ' ADDIN ZOTERO_BIBL {"uncited":[]} CSL_BIBLIOGRAPHY '),
            id="zotero-field",
        ),
        pytest.param(
            _field_list(REFS, " ADDIN Mendeley Bibliography CSL_BIBLIOGRAPHY "),
            id="mendeley-field",
        ),
        pytest.param(_field_list(REFS, " BIBLIOGRAPHY "), id="word-field"),
        pytest.param("".join(_plain(r, style="EndNoteBibliography") for r in REFS), id="endnote"),
        pytest.param("".join(_plain(r, style="Bibliography") for r in REFS), id="bibliography"),
    ],
)
def test_a_generated_list_under_no_heading_gets_a_references_section(references):
    contents = _parse(_manuscript(references))

    assert _headers(contents)[-1] == "References"
    assert _section_texts(contents, "Discussion") == [
        "Our findings extend earlier work (Bishop, 2019)."
    ]
    assert _reference_rows(contents) == REFS


def test_a_generated_list_under_a_references_heading_gets_no_second_one():
    body = (
        _plain("Introduction", style="Heading1")
        + _plain("Body text.")
        + _plain("Literature Cited", style="Heading1")
        + _field_list(REFS, " ADDIN EN.REFLIST ")
    )
    contents = _parse(body)

    assert _headers(contents) == ["Introduction", "Literature Cited"]
    assert _section_texts(contents, "Literature Cited") == REFS


def test_what_follows_a_generated_list_is_not_read_as_references():
    """A manuscript's tables come after its list, under the same heading."""
    body = _manuscript(
        _bold("References")
        + _field_list(REFS, " ADDIN EN.REFLIST ")
        + _bold("Table 1")
        + _plain("Descriptive statistics for the study variables")
    )
    contents = _parse(body)

    assert _reference_rows(contents) == REFS
    assert _section_texts(contents, "Discussion")[-2:] == [
        "Table 1",
        "Descriptive statistics for the study variables",
    ]


def test_a_dated_entry_typed_after_the_generated_ones_stays_in_the_list():
    typed = "Zhang, L. (2024). An entry typed by hand. Journal of Things, 1, 1-2."
    body = _manuscript(
        _bold("References")
        + "".join(_plain(r, style="EndNoteBibliography") for r in REFS)
        + _plain(typed)
    )

    assert _reference_rows(_parse(body)) == [*REFS, typed]


def test_a_citation_field_naming_a_bibliography_marks_nothing():
    """A citation field's instruction carries the cited record, any words included."""
    citation = (
        _field_char("begin")
        + _instr(" ADDIN EN.CITE <EndNote><Cite><record><title>Bibliography of X</title>")
        + _field_char("separate")
        + _run("(Meyer, 2003)")
        + _field_char("end")
    )
    body = _plain("Introduction", style="Heading1") + _para(_run("As shown "), citation)
    contents = _parse(body)

    assert _headers(contents) == ["Introduction"]


def test_a_references_paragraph_is_a_heading_in_a_styled_document_too():
    body = (
        _plain("Introduction", style="Heading1")
        + _plain("Body text (Meyer, 2003).")
        + _plain("6. References")
        + "".join(_plain(r) for r in REFS)
    )
    contents = _parse(body)

    assert _headers(contents) == ["Introduction", "6. References"]
    assert _reference_rows(contents) == REFS


@pytest.mark.parametrize(
    "text",
    ["References show that this works.", "Referees", "Reference standards and values"],
)
def test_a_paragraph_only_starting_with_references_is_body_text(text):
    body = _plain("Introduction", style="Heading1") + _plain(text)

    assert _headers(_parse(body)) == ["Introduction"]


def test_bold_paragraphs_are_body_text_when_the_document_has_heading_styles():
    body = (
        _plain("Introduction", style="Heading1")
        + _bold("An emphasised remark")
        + _plain("Body text.")
        + _plain("Method", style="Heading1")
        + _plain("Discussion", style="Heading1")
    )

    assert _headers(_parse(body)) == ["Introduction", "Method", "Discussion"]


def test_one_styled_heading_leaves_the_bold_ones_to_find_the_sections():
    """A PsyArXiv manuscript styled only its "Abstract" as Heading 1 and
    bolded the rest by hand; it came out with two sections."""
    body = (
        _plain("Abstract", style="Heading1")
        + _plain("We asked a question (Smith, 2020).")
        + _bold("Method")
        + _plain("We recruited students.")
        + _bold("Discussion")
        + _plain("It worked.")
    )

    assert _headers(_parse(body)) == ["Abstract", "Method", "Discussion"]


def test_outline_levels_make_headings():
    body = (
        _plain("Methods", outline=0)
        + _plain("Participants", style="MySection")
        + _plain("We recruited students.")
        + _plain("Contents", style="TOCHeading")
    )
    contents = _parse(body)

    assert [(s.header, s.level) for s in contents.sections[1:]] == [
        ("Methods", 1),
        ("Participants", 2),
    ]
    assert _section_texts(contents, "Participants") == ["We recruited students.", "Contents"]


def test_bold_from_a_paragraph_or_character_style_counts():
    body = (
        _para(_run("Method"), style="BoldPara")
        + _plain("Text one.")
        + _para(_run("Results", char_style="Strong"))
        + _plain("Text two.")
    )

    assert _headers(_parse(body)) == ["Method", "Results"]


@pytest.mark.parametrize(
    "paragraph",
    [
        pytest.param(_para(_run("Participants.", bold=True), _run(" We recruited")), id="run-in"),
        pytest.param(_bold("Keywords:"), id="lead-in"),
        pytest.param(_bold("Figure 1"), id="figure-label"),
        pytest.param(_bold("Table 2. Means and standard deviations"), id="table-label"),
        pytest.param(_bold("This sentence is bold for emphasis and ends like one."), id="sentence"),
        pytest.param(_bold(" ".join(["word"] * 25)), id="long"),
        pytest.param(_bold("R Core Team (2023). R: A language and environment"), id="reference"),
        pytest.param(_bold("Smith J, Doe K. A paper. J Things. 2020;12:45-67"), id="vancouver"),
        pytest.param(_bold("Materials at https://osf.io/abcde"), id="url"),
        pytest.param(_bold("Table S1") + _bold("Descriptive statistics"), id="apa-table-title"),
        pytest.param(_plain("Figure 2") + _bold("Study design"), id="apa-figure-title"),
    ],
)
def test_bold_paragraphs_that_are_not_headings(paragraph):
    body = _bold("Method") + _plain("Text.") + paragraph

    assert _headers(_parse(body)) == ["Method"]


def test_the_bold_title_page_above_the_abstract_is_no_section():
    """The title over two bold lines and the bold author line came out as
    sections; with the LLM on, a title section competes with the real one."""
    body = (
        _bold("Short-term Cultural Evolution in Response to COVID-19")
        + _bold("Integrating Quantitative and Thematic Analysis")
        + _bold("Jane Doe, John Roe")
        + _plain("University of Somewhere")
        + _bold("Abstract")
        + _plain("We studied it.")
        + _bold("Method")
        + _plain("We did it.")
    )

    assert _headers(_parse(body)) == ["Abstract", "Method"]


def test_reference_entries_with_an_outline_level_are_not_headings():
    """A harvested PsyArXiv manuscript had a stray outline level 0 on three
    reference entries. They became headings, each starting a front-matter
    record, and counted as styled headings, so the bold ones went unread."""
    body = (
        _bold("Introduction")
        + _plain("Text (Kesebir & Kesebir, 2012).")
        + _bold("Discussion")
        + _plain("More text.")
        + _bold("References")
        + "".join(_plain(ref, outline=0) for ref in REFS)
        + _plain(
            "Micheletti, E. B. (2022). What is cultural evolution? doi.org/10.1093/x", outline=0
        )
    )
    contents = _parse(body)

    assert _headers(contents) == ["Introduction", "Discussion", "References"]
    assert len(_reference_rows(contents)) == len(REFS) + 1


def test_a_caption_in_the_title_style_is_no_heading():
    body = (
        _plain("A study of games", style="Title")
        + _plain("Introduction", style="Heading1")
        + _plain("Text.")
        + _plain("Table S1: Top video games by year (2020-2024).", style="Title")
    )
    contents = _parse(body)

    assert _headers(contents) == ["A study of games", "Introduction"]
    assert _section_texts(contents, "Introduction")[-1].startswith("Table S1")


def test_bold_entries_in_the_references_are_not_headings():
    """A harvested PsyArXiv manuscript bolded some entries of its list; each
    became a section, which split the paper into several front-matter records
    under the LLM."""
    body = (
        _bold("Method")
        + _plain("Text (Smith, 2020).")
        + _bold("References")
        + _plain(REFS[0])
        + _bold("Doe, K. 2021 Meta-analysis")
        + _plain(REFS[1])
        + _bold("Appendix A")
        + _plain("Items.")
    )

    assert _headers(_parse(body)) == ["Method", "References", "Appendix A"]
