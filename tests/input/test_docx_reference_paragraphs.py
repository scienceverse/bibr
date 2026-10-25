"""A .docx reference list's own paragraphs as the reference segmentation.

Each paragraph of a typed or generated reference list is one entry, so when
the paragraphs pass the structure check (:mod:`bibr.input.docx_references`)
they reach the reference parser as they are and no LLM call segments them.
Lists whose paragraphs are not entries (hard-wrapped entries, floats and notes
under the references heading) are left to the LLM segmenter as before.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from html import escape
from unittest import mock

import pytest

pytest.importorskip("docx")

from bibr.extract.ref_extractor import ReferenceExtractor
from bibr.extract.ref_locator import RefLocator
from bibr.input.docx_native import DocxParser
from bibr.input.docx_references import ReferenceParagraph, reference_entries
from bibr.paper import PaperReference

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
_WML = "application/vnd.openxmlformats-officedocument.wordprocessingml"
_STYLES = (
    f'<w:styles xmlns:w="{W}">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>'
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
    "van der Berg, P., & Ng, T. (in press). Minority stress in later life. Ageing & Society.",
]


def _para(*lines: str, style: str | None = None) -> str:
    """One paragraph; several *lines* are joined by line breaks (Shift+Enter)."""
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = "<w:r><w:br/></w:r>".join(
        f'<w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r>' for line in lines
    )
    return f"<w:p>{ppr}{runs}</w:p>"


def _docx(*paragraphs: str, heading: str = "References") -> bytes:
    body = (
        _para("Method", style="Heading1")
        + _para("We asked people about things (Meyer, 2003).")
        + _para(heading, style="Heading1")
        + "".join(paragraphs)
    )
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


def _paragraphs(*texts: str) -> list[ReferenceParagraph]:
    return [ReferenceParagraph(text, (text,)) for text in texts]


def test_one_entry_per_paragraph_is_the_segmentation():
    assert reference_entries(_paragraphs(*REFS)) == (REFS, "ok")


@pytest.mark.parametrize(
    "end",
    [
        "Table 1",
        "Figure 2. Mean recall by condition.",
        "Appendix A",
        "Supplementary Information",
        "Extended Data Fig. 1 | Recall by condition.",
    ],
)
def test_the_list_ends_at_a_float_or_an_end_of_list_heading(end):
    paragraphs = _paragraphs(*REFS, end, "Condition means and standard deviations", *REFS[:2])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_supplementary_reference_heading_does_not_end_the_list():
    paragraphs = _paragraphs(*REFS[:3], "Supplemental References", *REFS[3:])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_later_references_section_starts_a_new_run_of_entries():
    # Tables follow the first list; an appendix's own list follows them.
    paragraphs = [ReferenceParagraph(text, (text,), 1) for text in (*REFS[:3], "Table 1", "M")]
    paragraphs += [ReferenceParagraph(text, (text,), 2) for text in REFS[3:]]
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_references_typed_as_a_paragraph_after_a_supplement_start_its_list():
    # A supplement typed under the main references heading, then its own
    # "References" as a plain paragraph: the floats end the first list only.
    paragraphs = _paragraphs(
        *REFS[:3], "Figure S1. Network of all variables.", "Note. Edges are partial correlations."
    )
    paragraphs += _paragraphs("References", *REFS[3:])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_supplementary_heading_over_references_does_not_end_the_list():
    paragraphs = _paragraphs(*REFS[:3], "Supplementary Materials", *REFS[3:])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_hard_wrapped_entries_are_left_to_the_segmenter():
    # Each entry broken into two paragraphs: the second halves carry no year.
    wrapped = []
    for ref in REFS[1:]:
        cut = ref.index(". ", ref.index(")") + 2) + 2
        wrapped += [ref[:cut], ref[cut:]]
    entries, reason = reference_entries(_paragraphs(*wrapped))
    assert entries is None and reason == "year_coverage"


def test_a_short_list_is_left_to_the_segmenter():
    assert reference_entries(_paragraphs(*REFS[:2])) == (None, "too_few")


def test_entries_joined_by_line_breaks_are_split_at_them():
    joined = ReferenceParagraph(" ".join(REFS[1:4]), tuple(REFS[1:4]))
    paragraphs = [*_paragraphs(REFS[0]), joined, *_paragraphs(REFS[4])]
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_line_break_inside_one_entry_does_not_split_it():
    head, doi = REFS[1], "https://doi.org/10.1080/00918369.2020.1804264"
    paragraphs = [*_paragraphs(REFS[0]), ReferenceParagraph(f"{head} {doi}", (head, doi))]
    paragraphs += _paragraphs(*REFS[2:])
    entries, reason = reference_entries(paragraphs)
    assert reason == "ok" and entries[1] == f"{head} {doi}" and len(entries) == len(REFS)


def test_parser_hands_the_paragraphs_on_as_native_reference_strings():
    contents = DocxParser(_docx(*(_para(ref) for ref in REFS), _para("Table 1"))).parse()

    assert contents.native_ref_strings == REFS
    assert contents.native_ref_strings_authoritative is False


def test_parser_reads_line_broken_entries_as_separate_entries():
    contents = DocxParser(_docx(_para(REFS[0]), _para(*REFS[1:4]), _para(REFS[4]))).parse()

    assert contents.native_ref_strings == REFS


def test_parser_leaves_hard_wrapped_lists_to_the_segmenter():
    wrapped = []
    for ref in REFS[1:]:
        cut = ref.index(". ", ref.index(")") + 2) + 2
        wrapped += [_para(ref[:cut]), _para(ref[cut:])]
    assert DocxParser(_docx(*wrapped)).parse().native_ref_strings is None


def test_a_long_heading_that_mentions_references_heads_no_list():
    # A heading that only mentions references ("... includes references to
    # ...") does not make the quotes under it reference entries.
    heading = _para(
        "Emotional Regulation: this category includes references to practical strategies",
        style="Heading1",
    )
    quotes = [_para(f'"...it helps me feel lighter in {y}..." (Hans)') for y in (2001, 2002, 2003)]
    contents = DocxParser(_docx(*(_para(ref) for ref in REFS), heading, *quotes)).parse()

    assert contents.native_ref_strings == REFS


def test_paragraphs_under_another_heading_are_no_reference_list():
    contents = DocxParser(_docx(*(_para(ref) for ref in REFS), heading="Discussion")).parse()

    assert contents.native_ref_strings is None


def test_extraction_parses_the_paragraphs_without_an_llm_call():
    parser = DocxParser(_docx(*(_para(ref) for ref in REFS)))
    contents = parser.parse()
    segments = [[entry.text] for entry in parser.assembler.entries if entry.needs_segmentation]
    parser.apply_segmentation(contents, segments)
    seen: list[str] = []

    def parse(self, segments):
        seen.extend(segments)
        return [
            PaperReference(
                bib_id=i,
                title=s,
                authors=None,
                year=None,
                container=None,
                volume=None,
                first_page=None,
            )
            for i, s in enumerate(segments, start=1)
        ]

    llm = mock.Mock()
    extractor = ReferenceExtractor(contents, llm_client=llm, parse_strategy="ner")
    with mock.patch.object(ReferenceExtractor, "_parse_references_ner_aligned", parse):
        asyncio.run(extractor.extract(RefLocator(contents).collect_reference_rows()))

    assert seen == REFS
    assert llm.mock_calls == []
