"""A .docx reference list's own paragraphs as the reference segmentation.

Each paragraph of a typed or generated reference list is one entry, so when
the paragraphs pass the structure check (:mod:`bibr.input.docx_references`)
they reach the reference parser as they are and no LLM call segments them.
Lists whose paragraphs are not entries (hard-wrapped entries, floats and notes
under the references heading) are left to the configured segmenter as before.
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
from bibr.input.docx_references import (
    ReferenceParagraph,
    is_reference_list_heading,
    reference_entries,
)
from bibr.paper import PaperReference

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
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
# Five more, so that one undated entry among them keeps the list at 10 of 11 dated.
MORE_REFS = [
    f"Author{i}, A. B. ({2010 + i}). Study {i} of recall. Memory, {i}(2), 10-20." for i in range(5)
]


def _run(line: str) -> str:
    """One run of text; a tab in *line* is a ``w:tab``."""
    texts = (f'<w:t xml:space="preserve">{escape(part)}</w:t>' for part in line.split("\t"))
    return f"<w:r>{'<w:tab/>'.join(texts)}</w:r>"


def _para(*lines: str, style: str | None = None, line_break: str = "<w:br/>") -> str:
    """One paragraph; several *lines* are joined by line breaks (Shift+Enter)."""
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = f"<w:r>{line_break}</w:r>".join(_run(line) for line in lines)
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
        "Extended Data Fig. 2 | Recall by condition and age group in the three samples, with the "
        "means of each condition marked by a line and the sample sizes above the bars.",
    ],
)
def test_the_list_ends_at_a_float_or_an_end_of_list_heading(end):
    paragraphs = _paragraphs(*REFS, end, "Condition means and standard deviations", *REFS[:2])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_float_before_the_first_entry_does_not_end_the_list():
    # A table anchored at the top of the references page: the list after it
    # is still read, and the label is left to the junk filter.
    entries, reason = reference_entries(_paragraphs("Table 1", *REFS, *MORE_REFS))
    assert reason == "ok" and entries[1:] == REFS + MORE_REFS


def test_a_supplementary_reference_heading_does_not_end_the_list():
    paragraphs = _paragraphs(*REFS[:3], "Supplemental References", *REFS[3:])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_later_references_section_starts_a_new_run_of_entries():
    # Tables follow the first list; an appendix's own list follows them.
    paragraphs = [ReferenceParagraph(text, (text,), 1) for text in (*REFS[:3], "Table 1", "M")]
    paragraphs += [ReferenceParagraph(text, (text,), 2) for text in REFS[3:]]
    assert reference_entries(paragraphs) == (REFS, "ok")


def _name_of_length(chars: int) -> str:
    name = "References (" + "x" * (chars - len("References ()")) + ")"
    assert len(name) == chars
    return name


@pytest.mark.parametrize(
    "name",
    [
        "References",
        "Supplementary References",
        "Appendix B: References",
        "References (Level 1 Heading)",
        _name_of_length(60),
    ],
)
def test_references_typed_as_a_paragraph_after_a_supplement_start_its_list(name):
    # A supplement typed under the main references heading, then its own
    # "References" as a plain paragraph: the floats end the first list only.
    paragraphs = _paragraphs(
        *REFS[:3], "Figure S1. Network of all variables.", "Note. Edges are partial correlations."
    )
    paragraphs += _paragraphs(name, *REFS[3:])
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_a_paragraph_over_60_characters_is_no_reference_list_name():
    paragraphs = _paragraphs(*REFS[:3], "Figure S1. Network of all variables.")
    paragraphs += _paragraphs(_name_of_length(61), *REFS[3:])
    assert reference_entries(paragraphs) == (REFS[:3], "ok")


@pytest.mark.parametrize(
    "name",
    [
        "References",
        "REFERENCES",
        "References:",
        "7. References",
        "7\tReferences",
        "4.2 References",
        "VII. References",
        "VII References",
        "References and Notes",
        "Reference List",
        "Bibliography",
        "Works Cited",
        "Literature Cited",
        "Cited Literature",
        "References Cited",
        "Literaturverzeichnis",
        "Supplementary References",
        "SI References",
        "Supplementary Materials References",
        "Appendix B. References",
        "9. References (data and method sources)",
        _name_of_length(60),
    ],
)
def test_a_reference_list_name_heads_a_list(name):
    assert is_reference_list_heading(name)


@pytest.mark.parametrize(
    "text",
    [
        "Self-References in Memory",
        "Reference List Searching",
        "References to the self",
        "A Bibliography of Memory",
        "Smith, J. (2019). A bibliography of memory. Wiley.",
        "Emotional Regulation: this category includes references to practical strategies",
        _name_of_length(61),
    ],
)
def test_text_that_only_holds_a_reference_list_word_heads_no_list(text):
    assert not is_reference_list_heading(text)


def test_a_short_entry_holding_a_reference_list_word_stays_in_the_list():
    short = "Smith, J. (2019). A bibliography of memory. Wiley."
    paragraphs = _paragraphs(*REFS[:3], short, *REFS[3:])
    assert reference_entries(paragraphs) == ([*REFS[:3], short, *REFS[3:]], "ok")


def test_a_supplementary_heading_over_references_does_not_end_the_list():
    paragraphs = _paragraphs(*REFS[:3], "Supplementary Materials", *REFS[3:])
    assert reference_entries(paragraphs) == (REFS, "ok")


@pytest.mark.parametrize(
    "following",
    [
        "Participants completed the 2019 version of the survey.",  # dated, no entry opening
        "Note. Items were rated on a seven-point scale.",  # an entry opening, no year
    ],
)
def test_a_supplementary_heading_over_anything_but_an_entry_ends_the_list(following):
    paragraphs = _paragraphs(*REFS[:3], "Supplementary Materials", following, *REFS[3:])
    assert reference_entries(paragraphs) == (REFS[:3], "ok")


@pytest.mark.parametrize(
    "entry",
    [
        "Supplementary Nutrition Assistance Program. (2020). Annual report. USDA.",
        "Supporting Information Services. (2016). Annual review of library services. SIS.",
        "Appendix Press. (2017). Collected essays on memory. Appendix Press.",
        "Annex, K., & Lee, P. Memory in context. Unpublished manuscript.",
    ],
)
def test_an_entry_that_opens_with_an_end_of_list_word_stays_in_the_list(entry):
    refs = [*REFS, *MORE_REFS]
    paragraphs = _paragraphs(*refs[:3], entry, *refs[3:])
    assert reference_entries(paragraphs) == ([*refs[:3], entry, *refs[3:]], "ok")


@pytest.mark.parametrize(
    "entry",
    [
        "World Health Organization. International classification of diseases, annex tables.",
        "Nakamura, K. (2020). Notes on extended data in memory research. Memory, 3, 1-9.",
    ],
)
def test_an_end_of_list_word_after_an_entrys_start_does_not_end_the_list(entry):
    refs = [*REFS, *MORE_REFS]
    paragraphs = _paragraphs(*refs[:3], entry, *refs[3:])
    assert reference_entries(paragraphs) == ([*refs[:3], entry, *refs[3:]], "ok")


def test_an_end_of_list_heading_is_at_most_20_words():
    words = "Appendix A: the wording of every item that participants answered in the three "
    words += "studies reported in this manuscript and its supplement"
    assert len(words.split()) == 21
    heading = " ".join(words.split()[:20])
    refs = [*REFS, *MORE_REFS]

    ended = reference_entries(_paragraphs(*refs[:5], heading, *refs[5:]))
    kept = reference_entries(_paragraphs(*refs[:5], words, *refs[5:]))

    assert ended == (refs[:5], "ok")
    assert kept == ([*refs[:5], words, *refs[5:]], "ok")


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


def test_three_entries_are_enough():
    assert reference_entries(_paragraphs(*REFS[:3])) == (REFS[:3], "ok")


def test_nine_in_ten_entries_with_a_year_are_enough():
    refs = [*REFS, *MORE_REFS[:4]]
    undated = ["Lee, K. Memory and the self. Unpublished manuscript.", "Kim, S. Untitled notes."]

    assert reference_entries(_paragraphs(*refs, undated[0])) == ([*refs, undated[0]], "ok")
    assert reference_entries(_paragraphs(*refs[:8], *undated)) == (None, "year_coverage")


def test_entries_joined_by_line_breaks_are_split_at_them():
    joined = ReferenceParagraph(" ".join(REFS[1:4]), tuple(REFS[1:4]))
    paragraphs = [*_paragraphs(REFS[0]), joined, *_paragraphs(REFS[4])]
    assert reference_entries(paragraphs) == (REFS, "ok")


def test_lines_that_do_not_rejoin_to_the_paragraphs_text_are_not_split():
    # Text the lines do not hold (a text box, an equation): the line breaks
    # cannot be placed in the paragraph's text.
    joined = ReferenceParagraph(" ".join([*REFS[1:4], "[Text box]"]), tuple(REFS[1:4]))
    paragraphs = [*_paragraphs(REFS[0]), joined, *_paragraphs(REFS[4])]
    assert reference_entries(paragraphs) == ([REFS[0], joined.text, REFS[4]], "ok")


def test_numbered_entries_joined_by_line_breaks_are_split_at_them():
    numbered = [f"[{i}] {ref}" for i, ref in enumerate(REFS, start=1)]
    joined = ReferenceParagraph(" ".join(numbered[1:4]), tuple(numbered[1:4]))
    paragraphs = [*_paragraphs(numbered[0]), joined, *_paragraphs(numbered[4])]
    assert reference_entries(paragraphs) == (numbered, "ok")


def test_a_line_break_inside_one_entry_does_not_split_it():
    head, doi = REFS[1], "https://doi.org/10.1080/00918369.2020.1804264"
    paragraphs = [*_paragraphs(REFS[0]), ReferenceParagraph(f"{head} {doi}", (head, doi))]
    paragraphs += _paragraphs(*REFS[2:])
    entries, reason = reference_entries(paragraphs)
    assert reason == "ok" and entries[1] == f"{head} {doi}" and len(entries) == len(REFS)


# An entry with its journal line after a line break: the journal line opens
# like a byline, and its pages fall in the years' range.
ENTRY_HEAD = "Zhang, L., & Ortiz, M. (2020). Attention and recall in older adults."
JOURNAL = "Psychological Science, 31(12), 1601-1612."


def test_a_line_break_before_a_journal_line_does_not_split_the_entry():
    joined = ReferenceParagraph(f"{ENTRY_HEAD} {JOURNAL}", (ENTRY_HEAD, JOURNAL))
    paragraphs = [*_paragraphs(REFS[0]), joined, *_paragraphs(*REFS[1:])]
    assert reference_entries(paragraphs) == ([REFS[0], f"{ENTRY_HEAD} {JOURNAL}", *REFS[1:]], "ok")


def test_a_doi_on_a_journal_line_gives_no_date():
    # The year inside a DOI is followed by a digit: the line is no entry of
    # its own.
    journal = "Cognition, 195, 104036. https://doi.org/10.1016/j.cognition.2019.104036"
    joined = ReferenceParagraph(f"{ENTRY_HEAD} {journal}", (ENTRY_HEAD, journal))
    paragraphs = [*_paragraphs(REFS[0]), joined, *_paragraphs(*REFS[1:])]
    assert reference_entries(paragraphs) == ([REFS[0], joined.text, *REFS[1:]], "ok")


@pytest.mark.parametrize(
    "entry",
    [
        "Kim, S. (2020, March 12). Memory on social media [Blog post]. Psych Blog.",
        "Kim, S. (2020/2021). Annual report on memory. Psych Society.",
        "World Health Organization. (n.d.). Depression fact sheet. WHO.",
        "Ortiz, M. (in press). Recall and age. Memory.",
        "Ortiz, M. (forthcoming). Recall and age. Memory.",
        "Lee, K.; Park, J. Memory and sleep. J. Sleep Res. 2019, 28, 112-120.",
        "Lee, K. Memory and sleep. Oxford University Press, 2019.",
        "Smith J, Lee K. Recall in older adults. J Gerontol. 2019;12:34-56.",
        "Smith J, Lee K. Recall in older adults. J Gerontol. 2019:12(3):34-56.",
    ],
)
def test_each_form_of_an_entrys_date_splits_a_paragraph_at_its_line_breaks(entry):
    # The entry under test is the only date on its line.
    lines = (REFS[1], entry, REFS[2])
    joined = ReferenceParagraph(" ".join(lines), lines)
    paragraphs = [*_paragraphs(REFS[0]), joined, *_paragraphs(*REFS[3:])]
    assert reference_entries(paragraphs) == ([REFS[0], *lines, *REFS[3:]], "ok")


def test_parser_keeps_a_journal_line_after_a_line_break_in_its_entry():
    contents = DocxParser(_docx(_para(REFS[0]), _para(ENTRY_HEAD, JOURNAL), *map(_para, REFS[1:])))
    native = contents.parse().native_ref_strings

    assert native == [REFS[0], f"{ENTRY_HEAD} {JOURNAL}", *REFS[1:]]


def test_parser_hands_the_paragraphs_on_as_native_reference_strings():
    contents = DocxParser(_docx(*(_para(ref) for ref in REFS), _para("Table 1"))).parse()

    assert contents.native_ref_strings == REFS
    assert contents.native_ref_strings_authoritative is False


def test_parser_reads_line_broken_entries_as_separate_entries():
    contents = DocxParser(_docx(_para(REFS[0]), _para(*REFS[1:4]), _para(REFS[4]))).parse()

    assert contents.native_ref_strings == REFS


def test_parser_splits_entries_at_legacy_carriage_returns():
    paragraphs = [_para(REFS[0]), _para(*REFS[1:4], line_break="<w:cr/>"), _para(REFS[4])]
    contents = DocxParser(_docx(*paragraphs)).parse()

    assert contents.native_ref_strings == REFS


def test_parser_does_not_split_a_paragraph_at_a_page_break():
    # A page break is layout, not a line break: the paragraph stays one entry
    # for the merged-reference splitter.
    paragraphs = [
        _para(REFS[0]),
        _para(*REFS[1:3], line_break='<w:br w:type="page"/>'),
        *map(_para, REFS[3:]),
    ]
    contents = DocxParser(_docx(*paragraphs)).parse()

    assert contents.native_ref_strings == [REFS[0], " ".join(REFS[1:3]), *REFS[3:]]


def test_parser_leaves_a_display_equation_out_of_its_reference_entry():
    # A display equation is an entry of its own in the parser's text.
    math = f'<m:oMathPara xmlns:m="{M}"><m:oMath><m:r><m:t>x+y</m:t></m:r></m:oMath></m:oMathPara>'
    with_math = _para(REFS[1]).replace("</w:p>", f"{math}</w:p>")
    contents = DocxParser(_docx(_para(REFS[0]), with_math, *map(_para, REFS[2:]))).parse()

    assert contents.native_ref_strings == REFS


def test_parser_reads_a_tab_after_an_entry_number_as_a_space():
    numbered = [f"[{i}]\t{ref}" for i, ref in enumerate(REFS, start=1)]
    paragraphs = [_para(numbered[0]), _para(*numbered[1:4]), _para(numbered[4])]
    contents = DocxParser(_docx(*paragraphs)).parse()

    assert contents.native_ref_strings == [f"[{i}] {ref}" for i, ref in enumerate(REFS, start=1)]


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


PROSE = [
    "Participants were asked to recall events that referred to themselves (Rogers et al., "
    "1977; Symons & Johnson, 1997), and to rate them in 2019.",
    "Self-reference effects were stronger in older adults (Gutchess et al., 2007), "
    "replicating Klein (2012) and extending Sui and Humphreys (2015).",
    "The effect was robust across samples collected in 2018 and 2019 (Cunningham et al., 2008).",
]


@pytest.mark.parametrize("heading", ["Self-References in Memory", "Reference List Searching"])
def test_a_short_heading_that_holds_a_reference_list_word_heads_no_list(heading):
    # A body subsection whose heading only holds a reference-list word: its
    # dated prose would pass the check together with the real list.
    body = [_para(heading, style="Heading1"), *(_para(text) for text in PROSE)]
    body += [_para("References", style="Heading1"), *(_para(ref) for ref in REFS)]
    contents = DocxParser(_docx(*body, heading="Introduction")).parse()

    assert contents.native_ref_strings == REFS


@pytest.mark.parametrize(
    "heading", ["7. References", "VII. Literature Cited", "Bibliography", "References:"]
)
def test_a_numbered_or_other_reference_list_name_heads_a_list(heading):
    contents = DocxParser(_docx(*(_para(ref) for ref in REFS), heading=heading)).parse()

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
