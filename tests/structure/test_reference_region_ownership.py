"""Reference layout ownership regressions using synthetic page fragments."""

from __future__ import annotations

import pytest


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


def test_reference_envelope_and_one_child_with_different_texts_are_both_kept():
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


def test_reference_envelope_with_one_child_holding_its_text_is_emitted_once():
    entry = "Lau, Ernst 1927. The sole detected reference."
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference", entry, [100, 150, 900, 240]),
                _region(2, "reference_content", entry, [100, 150, 900, 200]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [entry]


_ENTRIES = [
    "Adams, A. (2001). One. J, 1.",
    "Baker, B. (2002). Two. J, 2.",
    "Clark, C. (2003). Three. J, 3.",
    "Dunn, D. (2004). Four. J, 4.",
]


def test_reference_envelope_with_entry_boxes_for_only_some_entries_keeps_every_entry():
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference", "\n".join(_ENTRIES), [100, 150, 900, 500]),
                _region(2, "reference_content", _ENTRIES[0], [100, 150, 900, 190]),
                _region(3, "reference_content", _ENTRIES[1], [100, 195, 900, 235]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == ["\n".join(_ENTRIES)]
    # The shadowed entry boxes still feed the layout-anchor tiers.
    assert [(summary.label, summary.content) for summary in contents.region_summaries[1:]] == [
        ("reference", "\n".join(_ENTRIES)),
        ("reference_content", _ENTRIES[0]),
        ("reference_content", _ENTRIES[1]),
    ]


def test_reference_envelope_keeps_entry_boxes_it_does_not_hold():
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                # truncated aggregate read: holds the first two entries only
                _region(1, "reference", "\n".join(_ENTRIES[:2]), [100, 150, 900, 500]),
                _region(2, "reference_content", _ENTRIES[1], [100, 195, 900, 235]),
                _region(3, "reference_content", _ENTRIES[2], [100, 240, 900, 280]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [
        "\n".join(_ENTRIES[:2]),
        _ENTRIES[2],
    ]


def _noisy(text: str) -> str:
    """A second OCR read of *text*: "rn" for "m", which makes it one char longer."""
    return text.replace("m", "rn", 1)


@pytest.mark.parametrize(
    ("envelope", "children"),
    [
        ("\n".join(_noisy(e) for e in _ENTRIES), _ENTRIES),
        ("\n".join(_ENTRIES), [_noisy(e) for e in _ENTRIES]),
    ],
    ids=["envelope-read-longer", "entry-reads-longer"],
)
def test_reference_envelope_read_apart_from_its_entry_boxes_is_emitted_once(envelope, children):
    # A scanned page OCRs every box on its own, so the aggregate read and the
    # entry reads differ by a character here and there, in either direction.
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference", envelope, [100, 150, 900, 500]),
                *(
                    _region(
                        2 + i, "reference_content", child, [100, 150 + 45 * i, 900, 190 + 45 * i]
                    )
                    for i, child in enumerate(children)
                ),
            ]
        ]
    )

    assert _section_texts(contents, "References") == children


def test_entry_box_outside_the_envelope_is_not_judged_by_its_text():
    # "PubMed" is an entry of its own in the other column. The aggregate box's
    # text contains the word, but only the entry boxes inside it are its copies.
    envelope = "Adams, A. (2001). One. J, 1.\nDoe, J. (2020). Searching PubMed. J, 2, 3-4."
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 100, 300, 130]),
                _region(1, "reference", envelope, [100, 150, 480, 400]),
                _region(2, "reference_content", _ENTRIES[0], [100, 150, 480, 190]),
                _region(3, "reference_content", "PubMed", [520, 150, 900, 190]),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [envelope, "PubMed"]


def test_long_reference_envelope_keeps_the_one_entry_without_a_box():
    from tests.reference_fixtures import REFERENCE_LIST, read_again

    # Eleven of twelve entries have a box. The aggregate read, noisier than the
    # entry reads, holds the twelfth, so it stays and the entry boxes it
    # repeats are hidden: every entry is emitted once.
    envelope = "\n".join(read_again(entry) for entry in REFERENCE_LIST)
    contents = _parse(
        [
            [
                _region(0, "paragraph_title", "References", [100, 60, 300, 90]),
                _region(1, "reference", envelope, [100, 100, 900, 100 + 45 * len(REFERENCE_LIST)]),
                *(
                    _region(
                        2 + i, "reference_content", entry, [100, 100 + 45 * i, 900, 140 + 45 * i]
                    )
                    for i, entry in enumerate(REFERENCE_LIST)
                    if i != 2
                ),
            ]
        ]
    )

    assert _section_texts(contents, "References") == [envelope]


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
