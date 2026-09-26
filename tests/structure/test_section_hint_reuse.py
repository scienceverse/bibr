"""Section-hint reuse must match on canonical class, not literal strings.

A printed "Literature Cited" / "5 References" / "Bibliography" heading must
own the reference hint's entries instead of standing empty beside a synthetic
"References" section (and likewise "Abstract:" / "Summary" for abstracts).
"""

from bibr.structure.pdf_parser import PDFParser


def _parse(pages):
    parser = PDFParser(pages)
    contents = parser.parse()
    return parser, contents


def _printed_sections(parser):
    return [s for s in parser.sections if s.section_id != 0]


def test_printed_alias_heading_owns_reference_entries():
    """Heading-first: alias headings reuse the hint section (no synthetic)."""
    for heading in ("Literature Cited", "5 References", "Bibliography", "References:"):
        parser, _ = _parse(
            [
                [
                    {
                        "label": "paragraph_title",
                        "content": heading,
                        "bbox_2d": [100, 80, 900, 110],
                    },
                    {
                        "label": "reference_content",
                        "content": "Smith, J. (2020). A paper.",
                        "bbox_2d": [100, 150, 900, 200],
                    },
                ]
            ]
        )
        sections = _printed_sections(parser)
        assert len(sections) == 1, (heading, sections)
        assert sections[0].header == heading
        assert sections[0].header_is_synthetic is False
        entries = [e for e in parser.assembler.entries if "Smith" in e.text]
        assert len(entries) == 1
        assert entries[0].section_id == sections[0].section_id


def test_hint_first_alias_heading_reuses_hint_section():
    """Hint-first: a later alias heading joins the hint's section."""
    parser, _ = _parse(
        [
            [
                {
                    "label": "reference_content",
                    "content": "Smith, J. (2020). A paper.",
                    "bbox_2d": [100, 150, 900, 200],
                },
                {
                    "label": "paragraph_title",
                    "content": "Literature Cited",
                    "bbox_2d": [100, 80, 900, 110],
                },
                {
                    "label": "reference_content",
                    "content": "Jones, A. (2021). Other.",
                    "bbox_2d": [100, 210, 900, 260],
                },
            ]
        ]
    )
    sections = _printed_sections(parser)
    assert len(sections) == 1
    assert sections[0].header_is_synthetic is False
    assert {e.section_id for e in parser.assembler.entries} == {sections[0].section_id}


def test_printed_alias_heading_owns_abstract_entries():
    """Heading-first for abstracts: 'Abstract:' and 'Summary' own the hint."""
    for heading in ("Abstract:", "Summary"):
        parser, _ = _parse(
            [
                [
                    {
                        "label": "paragraph_title",
                        "content": heading,
                        "bbox_2d": [100, 80, 900, 110],
                    },
                    {
                        "label": "abstract",
                        "content": "We study things here.",
                        "bbox_2d": [100, 150, 900, 200],
                    },
                ]
            ]
        )
        sections = _printed_sections(parser)
        assert len(sections) == 1, (heading, sections)
        assert sections[0].header_is_synthetic is False


def test_unrelated_heading_does_not_capture_hint():
    """Guard: a hint still opens its own section past unrelated headings."""
    parser, _ = _parse(
        [
            [
                {
                    "label": "paragraph_title",
                    "content": "Discussion",
                    "bbox_2d": [100, 80, 900, 110],
                },
                {
                    "label": "reference_content",
                    "content": "Smith, J. (2020). A paper.",
                    "bbox_2d": [100, 150, 900, 200],
                },
            ]
        ]
    )
    sections = _printed_sections(parser)
    assert [(s.header, s.header_is_synthetic) for s in sections] == [
        ("Discussion", False),
        ("References", True),
    ]
    entries = [e for e in parser.assembler.entries if "Smith" in e.text]
    assert entries[0].section_id == sections[1].section_id
