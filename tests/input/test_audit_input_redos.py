import time

import pytest
from lxml import etree

from bibr.input.docx_references import opens_an_entry


@pytest.mark.parametrize("unit", ["A-", "Ab-", "A-1"])
def test_hyphenated_capitals_do_not_backtrack_exponentially(unit):
    start = time.perf_counter()
    opens_an_entry(unit * 26 + "!")
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    "text",
    [
        "Smith-Jones, A. (1999)",
        "Anderson-Smith-Jones A (2010)",
        "Smith-van der Berg, E. (2011)",
        "Петров-Водкин К. (1920)",
        "Jean-Pierre Dupont (2003).",
    ],
)
def test_hyphenated_bylines_still_open_an_entry(text):
    assert opens_an_entry(text)


def _epub_chapter_entries(head: str, body: str) -> list[str]:
    from bibr.input.epub_native import EpubParser
    from tests.input.test_validate_html_epub import _make_epub_with_chapter

    xhtml = (
        '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head>{head}</head><body>{body}</body></html>"
    )
    parser = EpubParser(_make_epub_with_chapter(xhtml))
    parser.parse()
    return [entry.text for entry in parser.assembler.entries]


_PARAGRAPHS = (
    "<h1>Chapter One</h1><p>First real paragraph is long enough to be text.</p>"
    "<p>Second paragraph is also here and long.</p>"
)


@pytest.mark.parametrize(
    "head",
    ["<title/>", '<script src="a.js"/>', "<style/>", '<title/><link rel="x" href="a.css"/>'],
)
def test_epub_self_closing_head_elements_do_not_swallow_the_chapter(head):
    entries = _epub_chapter_entries(head, _PARAGRAPHS)
    assert entries == [
        "First real paragraph is long enough to be text.",
        "Second paragraph is also here and long.",
    ]


@pytest.mark.parametrize("empty", ["<h2/>", "<ul/>", '<a id="x"/>', "<noscript/>", "<textarea/>"])
def test_epub_empty_elements_in_the_body_keep_the_text(empty):
    entries = _epub_chapter_entries("", empty + _PARAGRAPHS)
    assert entries[-2:] == [
        "First real paragraph is long enough to be text.",
        "Second paragraph is also here and long.",
    ]


@pytest.mark.parametrize(
    ("wrapped", "plain"),
    [
        (r"$$\text{hello {world} foo}$$", "hello {world} foo"),
        (r"$$\text{\textbf{Introduction}}$$", r"\textbf{Introduction}"),
        (r"$$\text{a}\text{b {c}}$$", "a b {c}"),
        (r"$$ \text{a {b}} \text{c} $$", "a {b} c"),
    ],
)
def test_wrapped_latex_text_with_nested_braces_is_unwrapped(wrapped, plain):
    from bibr.input.consolidate_text import unwrap_latex_text

    assert unwrap_latex_text(wrapped) == plain


@pytest.mark.parametrize(
    "text",
    [r"$$\text{a {b}} + x$$", r"$$\text{a {b}$$x", r"$$x \text{a {b}}$$", r"$$\text{a {b}$$$$"],
)
def test_other_display_math_is_not_unwrapped(text):
    from bibr.input.consolidate_text import unwrap_latex_text

    assert unwrap_latex_text(text) == text


def _text_box_entries(box: str) -> list[str]:
    import importlib

    from bibr.input.docx_native import DocxParser

    notes = importlib.import_module("tests.input.test_audit_final-docx-notes-links")
    body = (
        "<w:p><w:r><w:t>Before. </w:t></w:r><w:r><w:drawing>"
        f"<w:txbxContent>{box}</w:txbxContent></w:drawing></w:r></w:p>"
    )
    parser = DocxParser(notes._docx(body))
    parser.parse()
    return [entry.text for entry in parser.assembler.entries]


def test_a_word_split_across_runs_in_a_text_box_stays_whole():
    box = "<w:p><w:r><w:t>Unfor</w:t></w:r><w:r><w:t>tunately, a pull </w:t></w:r><w:r><w:t>quote.</w:t></w:r></w:p>"
    assert _text_box_entries(box) == ["Before. Unfortunately, a pull quote."]


def test_paragraphs_of_a_text_box_are_separated():
    box = "<w:p><w:r><w:t>One.</w:t></w:r></w:p><w:p><w:r><w:t>Two.</w:t></w:r></w:p>"
    assert _text_box_entries(box) == ["Before. One. Two."]


def test_every_ref_list_in_a_body_without_back_matter_is_read():
    from bibr.input.jats_native import JatsParser

    def refs(*numbers: int) -> str:
        return "<ref-list>" + "".join(
            f'<ref id="r{n}"><mixed-citation>{n} Author A. Title {n}. J. 20{n}.'
            "</mixed-citation></ref>"
            for n in numbers
        )

    paragraph = "<p>" + "A body sentence of the article. " * 3 + "</p>"
    xml = (
        "<article><front><article-meta><title-group><article-title>T</article-title>"
        "</title-group></article-meta></front><body>"
        f"<sec><title>Intro</title>{paragraph}</sec>"
        f"<sec><title>References A</title>{refs(10, 11)}</ref-list></sec>"
        f"<sec><title>References B</title>{refs(20, 21)}</ref-list></sec>"
        "</body></article>"
    )
    contents = JatsParser(xml.encode()).parse()

    assert [ref.split()[0] for ref in contents.native_ref_strings] == ["10", "11", "20", "21"]


def _jats_table(rows: str):
    from bibr.input.jats_native import JatsParser

    table = JatsParser._table_to_df(etree.fromstring(f"<table>{rows}</table>"))
    assert table is not None
    return table


def test_jats_colspan_and_rowspan_copy_the_cell_into_every_slot_it_covers():
    table = _jats_table(
        "<thead><tr><th rowspan='2'>Group</th><th colspan='2'>Score</th></tr></thead>"
        "<tbody><tr><td>A</td><td>1</td></tr>"
        "<tr><td rowspan='2'>B</td><td>2</td><td>3</td></tr>"
        "<tr><td>4</td><td>5</td></tr></tbody>"
    )

    assert list(table.columns) == ["Group", "Score", "Score"]
    assert table.values.tolist() == [["Group", "A", "1"], ["B", "2", "3"], ["B", "4", "5"]]


_MC_NS = (
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:v="urn:schemas-microsoft-com:vml"'
)
_BOX = "<w:txbxContent><w:p><w:r><w:t>Boxed pull quote.</w:t></w:r></w:p></w:txbxContent>"


def _paragraph_entries(paragraph: str) -> list[str]:
    import importlib

    from bibr.input.docx_native import DocxParser

    notes = importlib.import_module("tests.input.test_audit_final-docx-notes-links")
    parser = DocxParser(notes._docx(paragraph))
    parser.parse()
    return [entry.text for entry in parser.assembler.entries]


def test_a_word_2010_text_box_is_read_once():
    """Word stores a text box in mc:AlternateContent: the drawing in mc:Choice,
    the same box as VML in mc:Fallback."""
    paragraph = (
        f"<w:p {_MC_NS}><w:r><w:t>Before. </w:t></w:r><w:r><mc:AlternateContent>"
        f"<mc:Choice Requires='wps'><w:drawing><wp:anchor><wps:txbx>{_BOX}</wps:txbx>"
        f"</wp:anchor></w:drawing></mc:Choice>"
        f"<mc:Fallback><w:pict><v:shape><v:textbox>{_BOX}</v:textbox></v:shape></w:pict>"
        "</mc:Fallback></mc:AlternateContent></w:r></w:p>"
    )
    assert _paragraph_entries(paragraph) == ["Before. Boxed pull quote."]


def test_a_vml_text_box_is_read():
    paragraph = (
        f"<w:p {_MC_NS}><w:r><w:t>Before. </w:t></w:r><w:r><w:pict><v:shape>"
        f"<v:textbox>{_BOX}</v:textbox></v:shape></w:pict></w:r></w:p>"
    )
    assert _paragraph_entries(paragraph) == ["Before. Boxed pull quote."]
