"""RefLocator takes a section for the bibliography only on a whole references heading.

Two regressions: the header-text fallback accepted any heading that merely
contained "reference" ("Revealed Preferences", "Reference standard"), and the
printed-heading override left the body section the classifier had typed
REFERENCES with that type, so its sentences were read as bibliography.
"""

from __future__ import annotations

import pytest

from bibr.extract.front_role import FrontRolePredictions, RoleScores
from bibr.extract.ref_extractor import _map_bib_text_ids
from bibr.extract.ref_locator import RefLocator
from bibr.paper import PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    RegionSummary,
)


def _contents(sections, rows, *, region_summaries=None, predictions=None) -> PaperContents:
    """PaperContents from ``(section, type[, score, source])`` specs and ``(section_id, text)`` rows."""
    paper_sections = [PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN)]
    for section_id, spec in enumerate(sections, start=1):
        header, section_type, *rest = spec
        score, source = rest if rest else (0.0, None)
        paper_sections.append(PaperSection(section_id, header, 1, 0, section_type, score, source))
    sentences = [
        PaperSentence(
            text_id=text_id,
            text=text,
            section_id=section_id,
            paragraph_id=text_id,
            page_number=1,
        )
        for text_id, (section_id, text) in enumerate(rows, start=1)
    ]
    return PaperContents(
        sentences=sentences,
        sections=paper_sections,
        tables=[],
        links=[],
        sections_text={s.section_id: "" for s in paper_sections},
        region_summaries=region_summaries or [],
        front_role_predictions=predictions,
    )


def _types(contents: PaperContents) -> list[tuple[str, CanonicalSection, float, str | None]]:
    return [
        (s.header, s.section_type, s.classification_score, s.classification_source)
        for s in contents.sections[1:]
    ]


@pytest.mark.parametrize(
    "heading",
    ["Revealed Preferences", "Reference standard", "Reference values", "Self-reference effects"],
)
def test_heading_that_only_contains_reference_is_not_the_bibliography(heading):
    contents = _contents(
        [
            ("Introduction", CanonicalSection.INTRODUCTION, 1.0, "exact_alias"),
            (heading, CanonicalSection.UNKNOWN),
            ("Sources", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "We study choices."),
            (2, "Consumers reveal their preferences through purchases in 2019 markets."),
            (3, "Smith, J. (2019). Choice. Journal of Econ, 1, 1-2."),
        ],
    )

    with pytest.raises(ValueError, match="No reference section found"):
        RefLocator(contents).collect_reference_rows()

    assert contents.sections[2].section_type == CanonicalSection.UNKNOWN


@pytest.mark.parametrize("heading", ["Selected References", "Appendix B. References"])
def test_header_text_fallback_accepts_a_reference_word_after_a_prefix(heading):
    contents = _contents(
        [
            ("Discussion", CanonicalSection.DISCUSSION, 1.0, "exact_alias"),
            (heading, CanonicalSection.UNKNOWN),
        ],
        [
            (1, "We found an effect."),
            (2, "Kahneman, D. (1973). Attention and effort. Prentice-Hall."),
            (2, "Posner, M. I. (1980). Orienting of attention. QJEP, 32, 3-25."),
        ],
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text_id"]) == [2, 3]
    assert contents.sections[2].section_type == CanonicalSection.REFERENCES


def test_header_text_fallback_accepts_a_whole_non_english_heading():
    contents = _contents(
        [
            ("Einleitung", CanonicalSection.UNKNOWN),
            ("Literaturverzeichnis", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "Wir untersuchen Entscheidungen."),
            (2, "Müller, A. (2019). Lehrbuch. Stuttgart: Thieme."),
            (2, "Weber, C. (2017). Pharmakologie. München: Elsevier."),
        ],
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text_id"]) == [2, 3]
    assert contents.sections[2].section_type == CanonicalSection.REFERENCES


def test_header_alias_override_retypes_the_misclassified_body_section():
    contents = _contents(
        [
            ("Introduction", CanonicalSection.INTRODUCTION, 1.0, "exact_alias"),
            # exp #2: the classifier typed body sections REFERENCES ...
            ("General Discussion", CanonicalSection.REFERENCES, 0.6, "model"),
            ("Limitations of the study", CanonicalSection.REFERENCES, 0.55, "model"),
            # ... and left the printed references heading UNKNOWN.
            ("References", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "We study attention."),
            (2, "Our results extend Attention and effort (Kahneman, 1973) to new tasks."),
            (3, "Attention and effort (Kahneman, 1973) did not test older adults."),
            (4, "Kahneman, D. (1973). Attention and effort. Prentice-Hall."),
            (4, "Posner, M. I. (1980). Orienting of attention. QJEP, 32, 3-25."),
        ],
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text_id"]) == [4, 5]
    assert _types(contents) == [
        ("Introduction", CanonicalSection.INTRODUCTION, 1.0, "exact_alias"),
        ("General Discussion", CanonicalSection.DISCUSSION, 1.0, "exact_alias"),
        ("Limitations of the study", CanonicalSection.DISCUSSION, 0.95, "substring_alias"),
        ("References", CanonicalSection.REFERENCES, 0.0, None),
    ]
    assert contents.reference_boundary_reason_flags == ["classifier_references_demoted"]

    ref = PaperReference(
        bib_id=1,
        title="Attention and effort",
        authors="Kahneman, D.",
        year=1973,
        first_page=None,
        volume=None,
        container=None,
    )
    _map_bib_text_ids([ref], contents)
    assert ref.text_id == 4


def test_header_alias_override_keeps_a_second_reference_list_typed():
    contents = _contents(
        [
            ("Supplementary References", CanonicalSection.REFERENCES, 0.95, "substring_alias"),
            ("References", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "Adams, A. (2001). Supplement. J, 1."),
            (2, "Baker, B. (2002). Main list. J, 2."),
        ],
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text_id"]) == [2]
    assert _types(contents) == [
        ("Supplementary References", CanonicalSection.REFERENCES, 0.95, "substring_alias"),
        ("References", CanonicalSection.REFERENCES, 0.0, None),
    ]
    assert contents.reference_boundary_reason_flags == []


@pytest.mark.parametrize(
    ("heading", "entries"),
    [
        # a second list whose heading names no references, read from its rows,
        # one of them split over two rows
        (
            "Studies Included in the Meta-Analysis",
            [
                "Adams, A. (2001). One. J, 1.",
                "Baker, B. (2002). Two studies of attention in older adults.",
                "Journal of Aging, 2, 3-4.",
            ],
        ),
        # a second list headed in another language, with rows that open on a
        # bare surname and initial (no comma, so they do not read as entries)
        (
            "Piśmiennictwo",
            ["Kowalski J. Psychologia. Warszawa: PWN; 2019.", "Nowak A. Pamięć. Kraków: UJ; 2018."],
        ),
        # a heading that looks up to REFERENCES without naming them
        (
            "Citations",
            ["Kowalski J. Psychologia. Warszawa: PWN; 2019.", "Nowak A. Pamięć. Kraków: UJ; 2018."],
        ),
    ],
)
def test_header_alias_override_keeps_a_second_list_typed(heading, entries):
    contents = _contents(
        [
            ("Discussion", CanonicalSection.DISCUSSION, 1.0, "exact_alias"),
            (heading, CanonicalSection.REFERENCES, 0.6, "model"),
            ("References", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "We found an effect."),
            *((2, entry) for entry in entries),
            (3, "Kahneman, D. (1973). Attention and effort. Prentice-Hall."),
        ],
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text_id"]) == [len(entries) + 2]
    assert _types(contents)[1] == (heading, CanonicalSection.REFERENCES, 0.6, "model")
    assert contents.reference_boundary_reason_flags == []


def test_front_role_override_retypes_the_misclassified_body_section():
    probs = {"ref_header": 0.92, "heading": 0.08}
    predictions = FrontRolePredictions(
        {(1, 0): RoleScores(probs=probs, top="ref_header", confidence=0.92)},
        model_version="t",
    )
    contents = _contents(
        [
            ("Введение", CanonicalSection.INTRODUCTION, 1.0, "exact_alias"),
            ("Flipped preferences", CanonicalSection.REFERENCES, 0.6, "model"),
            ("Список литературы", CanonicalSection.UNKNOWN),
        ],
        [
            (1, "Текст введения."),
            (2, "Мы обсуждаем результаты."),
            (3, "Иванов И. И. (2019). Рост микроводорослей. Биология, 12(3), 45–52."),
        ],
        region_summaries=[
            RegionSummary(page=1, index=0, label="paragraph_title", bbox=None, section_id=3)
        ],
        predictions=predictions,
    )

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text_id"]) == [3]
    assert _types(contents) == [
        ("Введение", CanonicalSection.INTRODUCTION, 1.0, "exact_alias"),
        ("Flipped preferences", CanonicalSection.UNKNOWN, 0.0, None),
        ("Список литературы", CanonicalSection.REFERENCES, 0.0, None),
    ]
    assert contents.reference_boundary_reason_flags == [
        "front_role_ref_header",
        "classifier_references_demoted",
    ]
