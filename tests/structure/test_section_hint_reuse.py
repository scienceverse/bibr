"""Section-hint reuse matches printed aliases, not just the literal name.

A printed "Literature Cited" / "5 References" / "Bibliography" heading must
own the reference hint's entries instead of standing empty beside a synthetic
"References" section (and likewise "Abstract:" / a front-page "Summary" for
abstracts). The alias must be adjacent to the hint region: a later
"Summary", "Author summary" or "Supplementary references" is its own section.
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
    for heading in (
        "Literature Cited",
        "5 References",
        "Bibliography",
        "References:",
        "Reference List",
    ):
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


def _region(label, content, y):
    return {"label": label, "content": content, "bbox_2d": [100, y, 900, y + 30]}


def _headers(parser):
    return [s.header for s in _printed_sections(parser)]


def _entry_headers(parser):
    header = {s.section_id: s.header for s in parser.sections}
    return [(header[e.section_id], e.text) for e in parser.assembler.entries]


def test_discussion_summary_subsection_stays_out_of_the_abstract():
    """BMJ-style 'Discussion > Summary' after an abstract hint region."""
    parser, _ = _parse(
        [
            [
                _region("doc_title", "A Study of Things", 60),
                _region("abstract", "We studied things. They were interesting.", 100),
                _region("paragraph_title", "Introduction", 220),
                _region("text", "Things matter.", 250),
            ],
            [
                _region("paragraph_title", "Discussion", 60),
                _region("text", "We found effects.", 90),
                _region("paragraph_title", "Summary", 130),
                _region("text", "In summary, the main finding replicates.", 160),
            ],
        ]
    )
    assert "Summary" in _headers(parser)
    assert ("Summary", "In summary, the main finding replicates.") in _entry_headers(parser)


def test_author_summary_after_abstract_hint_is_its_own_section():
    """PLOS 'Author summary' directly after the abstract region."""
    parser, _ = _parse(
        [
            [
                _region("doc_title", "A Study of Things", 60),
                _region("abstract", "We studied things. They were interesting.", 100),
                _region("paragraph_title", "Author summary", 200),
                _region("text", "Plain-language: things are interesting to everyone.", 230),
                _region("paragraph_title", "Introduction", 300),
            ]
        ]
    )
    assert _headers(parser)[1:3] == ["Abstract", "Author summary"]
    assert (
        "Author summary",
        "Plain-language: things are interesting to everyone.",
    ) in _entry_headers(parser)


def test_summary_heading_after_abstract_hint_is_not_folded_in():
    """Hint-first: only a literal or 'abstract' alias reuses the abstract hint."""
    parser, _ = _parse(
        [
            [
                _region("doc_title", "A Study of Things", 60),
                _region("abstract", "We studied things.", 100),
                _region("paragraph_title", "Summary", 200),
                _region("text", "A lay summary follows.", 230),
            ]
        ]
    )
    assert "Summary" in _headers(parser)
    assert ("Summary", "A lay summary follows.") in _entry_headers(parser)


def test_supplementary_references_are_not_folded_into_references():
    """Hint-first: 'Supplementary references' right after the list stays apart."""
    parser, _ = _parse(
        [
            [
                _region("paragraph_title", "References", 60),
                _region("reference_content", "Smith, J. (2020). A paper.", 100),
                _region("paragraph_title", "Supplementary references", 200),
                _region("text", "Jones, A. (2021). Other.", 240),
            ]
        ]
    )
    assert _headers(parser) == ["References", "Supplementary references"]
    assert ("Supplementary references", "Jones, A. (2021). Other.") in _entry_headers(parser)


def test_non_adjacent_alias_heading_does_not_capture_a_later_hint():
    """Heading-first reuse needs the alias heading to be the current section."""
    parser, _ = _parse(
        [
            [
                _region("paragraph_title", "Literature Cited", 60),
                _region("text", "This list was compiled in 2020.", 100),
                _region("paragraph_title", "Appendix A", 200),
                _region("reference_content", "Smith, J. (2020). A paper.", 240),
            ]
        ]
    )
    assert [(s.header, s.header_is_synthetic) for s in _printed_sections(parser)] == [
        ("Literature Cited", False),
        ("Appendix A", False),
        ("References", True),
    ]


def test_bilingual_resumen_does_not_take_the_english_abstract():
    """'Resumen' is not an alias: the English 'Abstract' heading keeps its text."""
    parser, _ = _parse(
        [
            [
                _region("doc_title", "Un estudio", 60),
                _region("paragraph_title", "Resumen", 100),
                _region("text", "Estudiamos cosas.", 130),
                _region("paragraph_title", "Abstract", 200),
                _region("abstract", "We studied things.", 230),
            ]
        ]
    )
    assert ("Abstract", "We studied things.") in _entry_headers(parser)
    assert ("Resumen", "Estudiamos cosas.") in _entry_headers(parser)


def test_alias_heading_does_not_reuse_a_hint_section_that_is_not_current():
    """Hint-first reuse needs the matching hint section itself to be current."""
    parser, _ = _parse(
        [
            [
                _region("reference_content", "Smith, J. (2020). A paper.", 60),
                _region("abstract", "We studied things.", 200),
                _region("paragraph_title", "Literature Cited", 300),
                _region("text", "Jones, A. (2021). Other.", 340),
            ]
        ]
    )
    assert ("Literature Cited", "Jones, A. (2021). Other.") in _entry_headers(parser)


def test_summary_heading_before_a_later_page_abstract_region_is_not_reused():
    """'Summary' names the abstract only on the front page."""
    parser, _ = _parse(
        [
            [_region("doc_title", "A Study of Things", 60), _region("text", "Body.", 100)],
            [
                _region("paragraph_title", "Summary", 60),
                _region("abstract", "We studied things again.", 100),
            ],
        ]
    )
    assert [(s.header, s.header_is_synthetic) for s in _printed_sections(parser)][1:] == [
        ("Summary", False),
        ("Abstract", True),
    ]


def test_non_english_reference_heading_owns_the_reference_entries():
    """A printed "BIBLIOGRAFIA" holds the entries instead of standing empty (#108, #124)."""
    for heading in ("BIBLIOGRAFIA", "Literaturverzeichnis", "Riferimenti bibliografici"):
        parser, _ = _parse(
            [
                [
                    _region("paragraph_title", heading, 60),
                    _region("reference_content", "Rossi M. Adenomatosi polmonare. 1978.", 100),
                ]
            ]
        )
        assert [(s.header, s.header_is_synthetic) for s in _printed_sections(parser)] == [
            (heading, False)
        ], heading


def test_front_page_abstract_heading_in_another_language_leaves_the_region_to_the_hint():
    """ "RIASSUNTO" does not take the abstract region: the synthetic twin holds
    it, and post-parse dedup keeps the populated section."""
    parser, _ = _parse(
        [
            [
                _region("doc_title", "Causa inconsueta di lesioni", 40),
                _region("paragraph_title", "RIASSUNTO", 100),
                _region("abstract", "In questa nota gli Autori descrivono un caso.", 130),
            ]
        ]
    )
    assert [(s.header, s.header_is_synthetic) for s in _printed_sections(parser)][1:] == [
        ("RIASSUNTO", False),
        ("Abstract", True),
    ]


def _lancet_back_matter(contributors_text):
    return [
        [
            _region("paragraph_title", "Discussion", 60),
            _region("text", "The trial shows a benefit of radiotherapy.", 100),
            _region("paragraph_title", "Contributors", 700),
            _region("reference", contributors_text, 720),
        ],
        [
            _region("text", "(MP, MW, and DCW).", 60),
            _region("paragraph_title", "Declaration of interests", 200),
            _region("text", "MDJ has received consultancy fees.", 230),
            _region("paragraph_title", "References", 400),
            _region("reference_content", "1 Smith J, Jones A. A trial. Lancet 2020; 1: 1-9.", 430),
            _region("reference_content", "2 Brown K, et al. Another trial. BMJ 2021; 2: 3-4.", 470),
        ],
    ]


def test_back_matter_paragraph_labelled_reference_stays_in_its_section():
    """A "Contributors" paragraph the layout model labels ``reference`` (#124).

    It must not open the References section: the paragraph would become a
    bibliography entry and the reference list would move ahead of the
    remaining back matter.
    """
    parser, _ = _parse(
        _lancet_back_matter("The writing group members are listed in the byline. MDJ was the chief")
    )
    sections = _printed_sections(parser)
    assert [(s.header, s.header_is_synthetic) for s in sections] == [
        ("Discussion", False),
        ("Contributors", False),
        ("Declaration of interests", False),
        ("References", False),
    ]
    entries = _entry_headers(parser)
    assert ("Contributors", "(MP, MW, and DCW).") in entries
    assert any(h == "Contributors" and t.startswith("The writing group") for h, t in entries)
    assert [t for h, t in entries if h == "References"][0].startswith("1 Smith J")


def test_reference_region_with_entry_signals_still_opens_references():
    """A ``reference`` region that reads like entries keeps opening the References section."""
    parser, _ = _parse(_lancet_back_matter("1 Smith J, et al. An early trial. Lancet 2019; 1: 1."))
    headers = [(s.header, s.header_is_synthetic) for s in _printed_sections(parser)]
    assert ("Contributors", False) in headers
    entries = _entry_headers(parser)
    assert not any(h == "Contributors" and t.startswith("1 Smith") for h, t in entries)


def test_reference_region_without_a_later_printed_heading_opens_references():
    parser, _ = _parse(
        [
            [
                _region("paragraph_title", "Discussion", 60),
                _region("reference_content", "Smith J, Jones A. A paper on trials", 100),
            ]
        ]
    )
    assert [(s.header, s.header_is_synthetic) for s in _printed_sections(parser)] == [
        ("Discussion", False),
        ("References", True),
    ]


def test_empty_reference_region_after_a_singular_reference_heading_opens_references():
    """A printed "Reference" heading names the list even though it classifies as nothing.

    osf_p736e: the list runs from a singular "Reference" heading over five pages,
    an empty ``reference`` region opens each later page, and the supplement prints
    its own "References" much later. The region must open the References section
    instead of being read as prose of "Reference"; otherwise the list stays untyped
    and only the supplement's references are parsed.
    """
    parser, _ = _parse(
        [
            [
                _region("paragraph_title", "Acknowledgments", 60),
                _region("text", "The authors thank the participants.", 100),
                _region("paragraph_title", "Reference", 200),
                _region(
                    "text",
                    "Areh, I., & Umek, P. (2007). Predicting recall. Studia Psychologica.",
                    240,
                ),
            ],
            [
                _region("reference", "", 40),
                _region("text", "Evans, J. R., & Fisher, R. P. (2011). Eyewitness memory.", 60),
                _region("paragraph_title", "Data Availability Statement", 300),
                _region("text", "This research is part of a larger project.", 340),
            ],
            [
                _region("paragraph_title", "Supplementary Material", 60),
                _region("text", "Additional analyses follow.", 100),
                _region("paragraph_title", "References", 400),
                _region("text", "Ashton, M. C., & Lee, K. (2009). The HEXACO-60.", 440),
            ],
        ]
    )
    # The supplement's printed "References" takes over the section the region opened.
    assert _headers(parser) == [
        "Acknowledgments",
        "Reference",
        "References",
        "Data Availability Statement",
        "Supplementary Material",
    ]
    entries = _entry_headers(parser)
    assert ("References", "Evans, J. R., & Fisher, R. P. (2011). Eyewitness memory.") in entries
    assert not any(h == "Reference" and t.startswith("Evans") for h, t in entries)
