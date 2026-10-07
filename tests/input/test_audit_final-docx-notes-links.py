"""DOCX notes, links and lookups that grew with crafted input (final review).

- A footnote or endnote referenced N times was queued N times, text and all:
  one note referenced 20,000 times in a 34 KB file made 1 GB of sentences.
  Notes nested in notes, hyperlinks in hyperlinks and text boxes in text boxes
  likewise kept one copy of the innermost text per level.
- Each hyperlink scanned the sentences for its paragraph, each heading scanned
  the sections for its parent, and python-docx searched the styles part for
  every paragraph's style: quadratic, up to a minute on files of 6-142 KB.

The scale tests count the work done rather than time it.
"""

from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("docx")

from docx import Document
from docx.oxml.styles import CT_Styles

from bibr.input.docx_footnotes import load_footnotes
from bibr.input.docx_native import DocxParser

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_WML = "application/vnd.openxmlformats-officedocument.wordprocessingml"
URL = "https://example.org/x"


def _docx(body: str, **parts: str) -> bytes:
    """A package whose ``w:body`` holds *body*, with the given ``footnotes``,
    ``endnotes`` or ``styles`` parts and a hyperlink relationship ``rIdLink``."""
    overrides = "".join(
        f'<Override PartName="/word/{name}.xml" ContentType="{_WML}.{name}+xml"/>' for name in parts
    )
    rels = "".join(
        f'<Relationship Id="rId{name}" Type="{R}/{name}" Target="{name}.xml"/>' for name in parts
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/word/document.xml" ContentType="{_WML}.document.main+xml"/>'
            f"{overrides}</Types>",
        )
        z.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{_RELS}"><Relationship Id="rId1" '
            f'Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        z.writestr(
            "word/_rels/document.xml.rels",
            f'<Relationships xmlns="{_RELS}">{rels}<Relationship Id="rIdLink" '
            f'Type="{R}/hyperlink" Target="{URL}" TargetMode="External"/></Relationships>',
        )
        z.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>',
        )
        for name, xml in parts.items():
            z.writestr(f"word/{name}.xml", xml)
    return buf.getvalue()


def _notes(kind: str, notes: dict[str, str]) -> str:
    entries = "".join(
        f'<w:{kind} w:id="{note_id}"><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:{kind}>'
        for note_id, text in notes.items()
    )
    return f'<w:{kind}s xmlns:w="{W}">{entries}</w:{kind}s>'


def _para(text: str, extra: str = "", style: str | None = None) -> str:
    props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{props}<w:r><w:t>{text}</w:t></w:r>{extra}</w:p>"


def _ref(note_id: str, kind: str = "footnote") -> str:
    return f'<w:r><w:{kind}Reference w:id="{note_id}"/></w:r>'


def _run(data: bytes):
    parser = DocxParser(data)
    contents = parser.parse()
    parser.apply_segmentation(contents, [[text] for text in parser.assembler.segmentable_texts])
    parser.create_content_sections(contents)
    return parser, contents


def _foot_xrefs(contents) -> list[tuple[str, str | None, str]]:
    """Each note reference as (note text, printed number, citing sentence)."""
    texts = {s.text_id: s.text for s in contents.sentences}
    return [
        (texts[x.xref_id], x.contents, texts[x.text_id])
        for x in contents.xrefs
        if x.xref_type == "foot"
    ]


class _CountingList(list):
    """A list that counts the items every iteration over it visits."""

    visits = 0

    def __iter__(self):
        for item in list.__iter__(self):
            self.visits += 1
            yield item

    def __reversed__(self):
        for item in list.__reversed__(self):
            self.visits += 1
            yield item


# ----- Notes -----


@pytest.mark.parametrize("kind", ["footnote", "endnote"])
def test_a_note_referenced_many_times_is_read_once(kind):
    """Each reference queued the note's whole text again: 20,000 references
    to a 50,000-character note in 34 KB made 1 GB of sentences."""
    body = _para("Body sentence.", _ref("1", kind) * 2000)
    _, contents = _run(_docx(body, **{f"{kind}s": _notes(kind, {"1": "The note."})}))

    notes = [s for s in contents.sections if s.synthetic_kind == "footnote"]
    assert [s.header for s in notes] == [f"{kind.title()} 1"]
    assert [s.text for s in contents.sentences].count("The note.") == 1
    assert _foot_xrefs(contents) == [("The note.", "1", "Body sentence.")]


def test_a_note_referenced_again_links_each_paragraph_to_the_one_note():
    body = (
        _para("First cites it.", _ref("1"))
        + _para("Second cites another.", _ref("2"))
        + _para("Third cites the first again.", _ref("1") + _ref("1"))
    )
    notes = _notes("footnote", {"1": "Note one.", "2": "Note two."})
    _, contents = _run(_docx(body, footnotes=notes))

    headers = [s.header for s in contents.sections if s.synthetic_kind == "footnote"]
    assert headers == ["Footnote 1", "Footnote 2"]
    assert _foot_xrefs(contents) == [
        ("Note one.", "1", "First cites it."),
        ("Note two.", "2", "Second cites another."),
        ("Note one.", "1", "Third cites the first again."),
    ]


def test_footnotes_and_endnotes_referenced_once_keep_their_order_and_numbers():
    body = (
        _para("A.", _ref("1", "endnote"))
        + _para("B.", _ref("1"))
        + _para("C.", _ref("2", "endnote"))
    )
    _, contents = _run(
        _docx(
            body,
            footnotes=_notes("footnote", {"1": "Foot one."}),
            endnotes=_notes("endnote", {"1": "End one.", "2": "End two."}),
        )
    )

    headers = [s.header for s in contents.sections if s.synthetic_kind == "footnote"]
    assert headers == ["Endnote 1", "Footnote 1", "Endnote 2"]
    assert _foot_xrefs(contents) == [
        ("End one.", "1", "A."),
        ("Foot one.", "1", "B."),
        ("End two.", "2", "C."),
    ]


def test_notes_nested_in_a_note_are_read_once():
    """Every note in the tree was read, each with the text of the notes inside
    it: 250 levels made 250 copies of the innermost text."""
    depth = 50
    notes = (
        f'<w:footnotes xmlns:w="{W}">'
        + "".join(f'<w:footnote w:id="{i}">' for i in range(depth))
        + "<w:p><w:r><w:t>Inner text.</w:t></w:r></w:p>"
        + "</w:footnote>" * depth
        + "</w:footnotes>"
    )
    doc = Document(io.BytesIO(_docx(_para("Body."), footnotes=notes)))

    assert load_footnotes(doc) == {"0": "Inner text."}


# ----- Nested hyperlinks and text boxes -----


def test_a_hyperlink_nested_in_hyperlinks_is_read_once():
    """Each level of the nest kept its own copy of the text under it: 250
    levels around a 1 MB paragraph made 250 MB of link text."""
    depth = 50
    body = (
        "<w:p>"
        + '<w:hyperlink r:id="rIdLink">' * depth
        + "<w:r><w:t>the data</w:t></w:r>"
        + "</w:hyperlink>" * depth
        + "</w:p>"
    )
    _, contents = _run(_docx(body))

    assert [(link.url, link.link_text) for link in contents.links] == [(URL, "the data")]
    assert [s.text for s in contents.sentences] == ["the data"]


def test_a_hyperlink_inside_an_anchor_link_keeps_its_url():
    body = (
        '<w:p><w:hyperlink w:anchor="methods"><w:r><w:t>See </w:t></w:r>'
        '<w:hyperlink r:id="rIdLink"><w:r><w:t>the data</w:t></w:r></w:hyperlink>'
        "</w:hyperlink></w:p>"
    )
    _, contents = _run(_docx(body))

    assert [(link.url, link.link_text) for link in contents.links] == [(URL, "the data")]
    assert [s.text for s in contents.sentences] == ["See the data"]


def test_a_text_box_nested_in_text_boxes_is_read_once():
    """Every text box in the drawing was read, each with the boxes inside it:
    240 levels made 240 MB of body text from 1 MB."""
    depth = 50
    body = (
        "<w:p><w:r><w:t>Before. </w:t></w:r><w:r><w:drawing>"
        + "<w:txbxContent>" * depth
        + "<w:p><w:r><w:t>Boxed text.</w:t></w:r></w:p>"
        + "</w:txbxContent>" * depth
        + "</w:drawing></w:r></w:p>"
    )
    parser = DocxParser(_docx(body))
    parser.parse()

    assert [entry.text for entry in parser.assembler.entries] == ["Before. Boxed text."]


# ----- Linear lookups -----


def test_hyperlinks_resolve_without_a_scan_per_link(monkeypatch):
    """Each link scanned the sentences for its paragraph: 100,000 links after
    20,000 paragraphs took 53 s."""
    paragraphs, links = 300, 2000
    link = '<w:hyperlink r:id="rIdLink"><w:r><w:t>x</w:t></w:r></w:hyperlink>'
    body = _para("Body.") * paragraphs + _para("Last.", link * links)
    parser = DocxParser(_docx(body))
    contents = parser.parse()
    emit = parser.assembler.emit
    emitted: list[_CountingList] = []

    def counting_emit(*args, **kwargs):
        sentences, *counters = emit(*args, **kwargs)
        emitted.append(_CountingList(sentences))
        return (emitted[-1], *counters)

    monkeypatch.setattr(parser.assembler, "emit", counting_emit)
    parser.apply_segmentation(contents, [[text] for text in parser.assembler.segmentable_texts])

    last = contents.sentences[-1]
    assert len(contents.links) == links
    assert {(lk.text_id, lk.paragraph_id) for lk in contents.links} == {
        (last.text_id, last.paragraph_id)
    }
    # A scan per link visited 2,000 x 301 sentences.
    assert emitted[0].visits < 20 * len(contents.sentences)


def _heading_styles() -> str:
    styles = "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/></w:style>'
        for n in (1, 2, 3)
    )
    return (
        f'<w:styles xmlns:w="{W}"><w:style w:type="paragraph" w:default="1" '
        f'w:styleId="Normal"><w:name w:val="Normal"/></w:style>{styles}</w:styles>'
    )


def test_heading_parents_are_found_without_a_scan_per_heading():
    """Each heading searched every section before it for its parent: 20,000
    Heading 1 paragraphs took 6 s."""
    count = 300
    body = "".join(_para(f"Part {i}", style="Heading1") for i in range(count))
    parser = DocxParser(_docx(body, styles=_heading_styles()))
    parser.sections = _CountingList()
    contents = parser.parse()

    assert len(contents.sections) == count + 1
    assert {s.parent_section_id for s in contents.sections[1:]} == {0}
    assert parser.sections.visits <= count  # a scan per heading visited 45,000


def test_heading_parents_are_the_latest_shallower_heading():
    levels = [2, 1, 2, 3, 3, 2, 1, 3, 2, 3, 1, 1]
    body = "".join(_para(f"H{i}", style=f"Heading{n}") for i, n in enumerate(levels))
    _, contents = _run(_docx(body, styles=_heading_styles()))

    sections = contents.sections[1:]
    expected = []
    for i, section in enumerate(sections):
        parents = [s.section_id for s in sections[:i] if s.level < section.level]
        expected.append(parents[-1] if parents else 0)
    assert [s.level for s in sections] == levels
    assert [s.parent_section_id for s in sections] == expected


def test_paragraph_styles_are_looked_up_once_per_document(monkeypatch):
    """python-docx searched the styles part for every paragraph's style: 10,000
    paragraphs and 10,000 styles in 56 KB took 52 s."""
    lookups = []
    for name in ("get_by_id", "default_for"):
        original = getattr(CT_Styles, name)
        monkeypatch.setattr(
            CT_Styles,
            name,
            lambda self, *args, _original=original: lookups.append(1) or _original(self, *args),
        )
    count = 300
    styles = "".join(
        f'<w:style w:type="paragraph" w:styleId="S{i}"><w:name w:val="s{i}"/></w:style>'
        for i in range(count)
    )
    body = "".join(_para("Body.", style=f"S{i}") for i in range(count)) + _para("Plain.") * count
    _, contents = _run(_docx(body, styles=f'<w:styles xmlns:w="{W}">{styles}</w:styles>'))

    assert len(contents.sentences) == 2 * count
    assert len(lookups) < 10  # one or two per paragraph before


def test_styles_resolve_as_python_docx_resolves_them():
    """The first style with an id wins; a character style or unknown id reads
    as the last default paragraph style; Caption-based styles are captions."""
    styles = (
        f'<w:styles xmlns:w="{W}">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Dup"><w:name w:val="heading 2"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Dup"><w:name w:val="Normal"/></w:style>'
        '<w:style w:type="character" w:styleId="CharHead"><w:name w:val="heading 1"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/>'
        '<w:basedOn w:val="Caption"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="LoopA"><w:name w:val="Loop A"/>'
        '<w:basedOn w:val="LoopB"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="LoopB"><w:name w:val="Loop B"/>'
        '<w:basedOn w:val="LoopA"/></w:style>'
        "</w:styles>"
    )
    table = (
        "<w:tbl><w:tblPr/><w:tblGrid/><w:tr><w:tc><w:p><w:r><w:t>Cell</w:t></w:r></w:p>"
        "</w:tc></w:tr></w:tbl>"
    )
    body = (
        _para("Methods", style="Dup")
        + _para("Not a heading.", style="CharHead")
        + _para("Unknown style.", style="Nope")
        + _para("Looping style.", style="LoopA")
        + table
        + _para("Results by group.", style="TableCaption")
    )
    _, contents = _run(_docx(body, styles=styles))

    assert [(s.header, s.level) for s in contents.sections[1:3]] == [("Methods", 2), ("Table 1", 1)]
    body_texts = [s.text for s in contents.sentences if s.section_id == 1]
    assert body_texts == ["Not a heading.", "Unknown style.", "Looping style."]
    assert [t.caption for t in contents.tables] == ["Results by group."]
