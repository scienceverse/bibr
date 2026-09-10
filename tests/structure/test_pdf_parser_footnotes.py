"""Tests for footnote handling in PDFParser."""


def test_footnote_xref_uses_ordinal_not_section_id():
    """A footnote xref should reference the footnote *number*, not the
    arbitrary section_id assigned by the parser counter."""
    from bibr.paper_contents import PaperContents
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result=[])
    contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
    )
    # Seed two pending footnotes so we can verify ordinals 1 and 2.
    parser._footnotes.record(
        text="first footnote text", page_number=1, body_section_id=0, deferred_text_index=0
    )
    parser._footnotes.record(
        text="second footnote text", page_number=1, body_section_id=0, deferred_text_index=1
    )
    parser._deferred_last_text_id = [None]
    parser._section_counter = 5
    parser._sentence_counter = 100
    parser._paragraph_counter = 50

    parser.create_content_sections(contents)
    foot_xrefs = [x for x in contents.xrefs if x.xref_type == "foot"]
    assert [x.xref_id for x in foot_xrefs] == [1, 2], (
        f"expected ordinals [1, 2], got {[x.xref_id for x in foot_xrefs]}"
    )


def test_footnote_xref_skips_display_formula_anchor():
    """Foot xrefs must not anchor on display-formula sentences.

    Display formulas export as "[equation]" placeholders, so an anchor
    there points consumers at text they never see. Mined from v12 econ
    outputs (48 xrefs / 25 papers): the last body sentence before a
    page-bottom footnote block is often the display formula itself
    ("The model is described as follows:" → formula → footnote block).
    """
    from bibr.paper_contents import PaperContents, PaperSentence
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result=[])
    contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
    )
    parser.sentences = [
        PaperSentence(
            text_id=10,
            text="The staggered DID model is described as follows:",
            section_id=1,
            paragraph_id=1,
        ),
        PaperSentence(
            text_id=11,
            text=r"y_{it} = \beta D_{it} + \epsilon_{it}",
            section_id=1,
            paragraph_id=2,
            is_display_formula=True,
        ),
    ]
    parser._footnotes.record(
        text="a real prose footnote", page_number=1, body_section_id=1, deferred_text_index=2
    )
    parser._deferred_last_text_id = [10, 11]

    parser.create_content_sections(contents)
    foot_xrefs = [x for x in contents.xrefs if x.xref_type == "foot"]
    assert len(foot_xrefs) == 1
    assert foot_xrefs[0].text_id == 10, (
        f"anchor should skip the formula sentence (11) for the prose one (10), "
        f"got {foot_xrefs[0].text_id}"
    )


def test_footnote_xref_keeps_formula_anchor_when_nothing_else_precedes():
    """If only formulas precede the footnote, the nearest formula anchor is
    still better than jumping to the document-start fallback."""
    from bibr.paper_contents import PaperContents, PaperSentence
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result=[])
    contents = PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
    )
    parser.sentences = [
        PaperSentence(
            text_id=11,
            text=r"y = X\beta + \epsilon",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        ),
    ]
    parser._footnotes.record(
        text="footnote after a leading formula",
        page_number=1,
        body_section_id=1,
        deferred_text_index=1,
    )
    parser._deferred_last_text_id = [11]

    parser.create_content_sections(contents)
    foot_xrefs = [x for x in contents.xrefs if x.xref_type == "foot"]
    assert len(foot_xrefs) == 1
    assert foot_xrefs[0].text_id == 11


def test_paper_figure_has_body_section_id_field():
    from dataclasses import fields

    from bibr.paper_contents import PaperFigure, PaperTable

    fig_fields = {f.name for f in fields(PaperFigure)}
    tbl_fields = {f.name for f in fields(PaperTable)}
    assert "_body_section_id" in fig_fields
    assert "_body_section_id" in tbl_fields
