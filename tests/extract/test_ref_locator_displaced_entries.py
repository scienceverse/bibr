"""Reference entries the reading order filed under a statement after the heading.

Frontiers prints Funding, Conflict of interest, Generative AI statement and
Publisher's note in a column above a full-width References heading. The
layout reading order can put that heading first, so it heads one statement or
nothing and every entry lands under "Publisher's note": 8 of 34 Frontiers in
Psychology papers in jats_bs_v1 came out with no references.
"""

from __future__ import annotations

import pytest

from bibr.extract.ref_locator import RefLocator
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)

ENTRIES = [
    "Adhikari, K., and Panda, R. K. (2018). Users' information privacy concerns. "
    "J. Glob. Mark. 31, 96-110.",
    "Akay, O. (2026). Children's privacy, autonomy, and digital identity. Front. Psychol. 17:1.",
    "Amon, M. J., Kartvelishvili, N., and Kapadia, A. (2022). Sharenting and children's "
    "privacy. Proc. ACM Hum.-Comput. Interact. 6, 1-30.",
    "Barnes, R., and Potter, A. (2021). Sharenting and parents' digital literacy. "
    "Commun. Res. Pract. 7, 6-20.",
    "Benamati, J. H., Ozdemir, Z. D., and Smith, H. J. (2017). An empirical test. "
    "J. Inf. Sci. 43, 583-600.",
    "Berg, V., Arabiat, D., and Robinson, S. (2024). Young children and the creation of a "
    "digital identity. JMIR Pediatr. Parent. 7:e54414.",
]
FUNDING = (
    "The author(s) declare(s) that financial support was received from Emerson College "
    "for the research, authorship, and/or publication of this article."
)
NOTE = (
    "All claims expressed in this article are solely those of the authors and do not "
    "necessarily represent those of their affiliated organizations."
)


def _section(section_id, header, section_type):
    return PaperSection(
        section_id,
        header,
        1 if section_id else 0,
        None if section_id == 0 else 0,
        section_type,
        1.0,
        "exact_alias",
    )


def _contents(sections, rows):
    sentences = [
        PaperSentence(
            text_id=text_id,
            text=text,
            section_id=section_id,
            paragraph_id=text_id,
            page_number=20,
            provenance=[Provenance(page_no=20, bbox=(0.0, 0.0, 1.0, 1.0))],
        )
        for text_id, (section_id, text) in enumerate(rows, start=1)
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={s.section_id: "" for s in sections},
    )


def _frontiers(references_rows):
    sections = [
        _section(0, "Root", CanonicalSection.UNKNOWN),
        _section(1, "Conclusion", CanonicalSection.DISCUSSION),
        _section(2, "Funding", CanonicalSection.FUNDING),
        _section(3, "References", CanonicalSection.REFERENCES),
        _section(4, "Conflict of interest", CanonicalSection.COI),
        _section(5, "Publisher's note", CanonicalSection.ENDNOTE),
    ]
    rows = [
        (1, "In closing, this study is the first to apply all components of the theory."),
        *[(3, text) for text in references_rows],
        (4, "The author(s) declared that this work was conducted in the absence of conflicts."),
        (5, NOTE),
        *[(5, entry) for entry in ENTRIES],
    ]
    return _contents(sections, rows)


@pytest.mark.parametrize("references_rows", [[], [FUNDING]], ids=["empty", "one-statement"])
def test_entries_under_publisher_note_become_the_list(references_rows):
    contents = _frontiers(references_rows)
    ref_df = RefLocator(contents).collect_reference_rows()

    assert ref_df["text"].tolist() == ENTRIES
    assert "displaced_reference_entries" in contents.reference_boundary_reason_flags
    types = {s.header: s.section_type for s in contents.sections}
    assert types["Publisher's note"] == CanonicalSection.REFERENCES


def test_a_populated_references_section_is_left_alone():
    contents = _frontiers(ENTRIES[:3])
    ref_df = RefLocator(contents).collect_reference_rows()

    assert ref_df["text"].tolist() == ENTRIES[:3]
    assert "displaced_reference_entries" not in contents.reference_boundary_reason_flags


def test_a_body_section_after_the_heading_stops_the_search():
    sections = [
        _section(0, "Root", CanonicalSection.UNKNOWN),
        _section(1, "References", CanonicalSection.REFERENCES),
        _section(2, "Appendix A", CanonicalSection.APPENDIX),
        _section(3, "Publisher's note", CanonicalSection.ENDNOTE),
    ]
    rows = [(1, FUNDING), (2, "Table A1 lists the items."), *[(3, e) for e in ENTRIES]]
    contents = _contents(sections, rows)
    ref_df = RefLocator(contents).collect_reference_rows()

    assert ref_df["text"].tolist() == [FUNDING]


def test_too_few_entries_do_not_count():
    contents = _frontiers([])
    contents.sentences = [
        s for s in contents.sentences if s.text not in ENTRIES[2:]
    ]  # two entries under the note: below the floor of five
    with pytest.raises(ValueError):
        RefLocator(contents).collect_reference_rows()
