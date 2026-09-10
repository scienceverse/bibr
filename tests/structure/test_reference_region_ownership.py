"""Reference layout ownership regressions using synthetic page fragments."""

from __future__ import annotations


def _region(index: int, label: str, content: str, bbox: list[int]) -> dict:
    return {
        "index": index,
        "native_label": label,
        "label": label,
        "content": content,
        "bbox_2d": bbox,
    }


def _parse(pages: list[list[dict]]):
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(pages)
    contents = parser.parse()
    parser.apply_segmentation(contents, [[text] for text in parser.assembler.segmentable_texts])
    return contents


def _section_texts(contents, header: str) -> list[str]:
    section_ids = {section.section_id for section in contents.sections if section.header == header}
    return [sentence.text for sentence in contents.sentences if sentence.section_id in section_ids]


def test_endnotes_keep_reference_children_and_bibliography_drops_page_envelopes():
    pages = [
        [
            _region(0, "paragraph_title", "Endnotes", [100, 100, 300, 130]),
            _region(1, "reference", "1 First note.\n2 Second note.", [100, 150, 900, 240]),
            _region(2, "reference_content", "1 First note.", [100, 150, 900, 190]),
            _region(3, "reference_content", "2 Second note.", [100, 195, 900, 240]),
        ],
        [
            _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
            _region(
                1,
                "reference",
                "Lau, Ernst 1927. A.\nLipmann, Otto 1907. B.",
                [100, 150, 900, 260],
            ),
            _region(
                2,
                "reference_content",
                "Lau, Ernst 1927. A.",
                [100, 150, 900, 200],
            ),
            _region(
                3,
                "reference_content",
                "Lipmann, Otto 1907. B.",
                [100, 205, 900, 260],
            ),
        ],
    ]

    contents = _parse(pages)

    assert _section_texts(contents, "Endnotes") == ["1 First note.", "2 Second note."]
    assert _section_texts(contents, "References") == [
        "Lau, Ernst 1927. A.",
        "Lipmann, Otto 1907. B.",
    ]
    assert len(contents.region_summaries) == 8
    assert [summary.label for summary in contents.region_summaries].count("reference") == 2


def test_reference_envelope_without_children_remains_text_source():
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(
                    1,
                    "reference",
                    "Lau, Ernst 1927. The sole detected reference.",
                    [100, 150, 900, 240],
                ),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [
        "Lau, Ernst 1927. The sole detected reference."
    ]


def test_reference_envelope_with_one_child_is_not_shadowed():
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(
                    1,
                    "reference",
                    "Lau, Ernst 1927. The aggregate reference.",
                    [100, 150, 900, 240],
                ),
                _region(
                    2,
                    "reference_content",
                    "Lau, Ernst 1927. The child reference.",
                    [100, 150, 900, 200],
                ),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [
        "Lau, Ernst 1927. The aggregate reference.",
        "Lau, Ernst 1927. The child reference.",
    ]


def test_noncontained_reference_children_do_not_shadow_envelope():
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference", "Aggregate.", [100, 150, 400, 260]),
                _region(2, "reference_content", "Child one.", [500, 150, 900, 200]),
                _region(3, "reference_content", "Child two.", [500, 205, 900, 260]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [
        "Aggregate.",
        "Child one.",
        "Child two.",
    ]


def test_publisher_note_ends_reference_section_before_affiliation_tail():
    publisher_note = (
        "Publisher’s Note Springer Nature remains neutral with regard to jurisdictional "
        "claims in published maps and institutional affiliations."
    )
    affiliation = "Chair of Romance Cultural Studies Saarland University 66123 Saarbrücken Germany."
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(
                    1, "reference_content", "Bühler, Charlotte 1922. First.", [100, 150, 900, 190]
                ),
                _region(
                    2, "reference_content", "Bühring, Gerald 2007. Second.", [100, 195, 900, 235]
                ),
                _region(3, "reference_content", publisher_note, [100, 800, 900, 850]),
            ],
            [
                _region(0, "reference_content", affiliation, [100, 100, 900, 150]),
            ],
        ]
    )

    assert _section_texts(contents, "References") == [
        "Bühler, Charlotte 1922. First.",
        "Bühring, Gerald 2007. Second.",
    ]
    assert _section_texts(contents, "Publisher's Note") == [publisher_note, affiliation]


def test_bare_publisher_note_text_does_not_end_references():
    title = "Publisher's Note"
    later_ref = "Doe, John 2022. A later source."
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference_content", "Smith, Jane 2021. First.", [100, 150, 900, 190]),
                _region(2, "text", title, [100, 195, 900, 235]),
                _region(3, "reference_content", later_ref, [100, 240, 900, 280]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [
        "Smith, Jane 2021. First.",
        title,
        later_ref,
    ]
    assert _section_texts(contents, "Publisher's Note") == []


def test_reference_title_mentioning_publisher_note_does_not_end_section():
    title = "Publisher's note on economic institutions in transition economies. (2022)."
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference_content", "Smith, Jane 2021. First.", [100, 150, 900, 190]),
                _region(2, "text", title, [100, 195, 900, 235]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [
        "Smith, Jane 2021. First.",
        title,
    ]
    assert _section_texts(contents, "Publisher's Note") == []
