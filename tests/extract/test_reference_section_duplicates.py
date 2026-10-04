"""One references section per paper (#108, #124).

A printed non-English reference heading ("BIBLIOGRAFIA") could stand empty
next to the layout hint's synthetic "References" that holds every entry.
IMRaD dedup kept the empty printed one by source trust, and the reference
locator then re-promoted the populated one without touching its
diagnostics, so the export carried two ``references`` sections, one of
them with source ``imrad_dedup``.
"""

from __future__ import annotations

from bibr.extract.ref_locator import RefLocator
from bibr.paper import enforce_imrad_order
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)

REFS = [
    "1. Liebow AA. Bronchiolo-alveolar carcinoma. Adv Intern Med 1960; 10: 329-358.",
    "2. Rossi M, Bianchi L. Adenomatosi polmonare. Minerva Med 1978; 69: 1-9.",
]


def _section(section_id, header, section_type, source, *, synthetic=False):
    return PaperSection(
        section_id,
        header,
        1 if section_id else 0,
        None if section_id == 0 else 0,
        section_type,
        1.0 if source == "exact_alias" else 0.0,
        source,
        header_is_synthetic=synthetic,
    )


def _sentence(text_id, text, section_id, page):
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=text_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=(0.0, 10.0 * text_id, 10.0, 10.0 * text_id + 5))],
    )


def _italian_case_report():
    sections = [
        _section(0, "Root", CanonicalSection.UNKNOWN, None),
        _section(1, "DISCUSSIONE E CONCLUSIONI", CanonicalSection.DISCUSSION, "exact_alias"),
        _section(2, "BIBLIOGRAFIA", CanonicalSection.REFERENCES, "exact_alias"),
        _section(3, "References", CanonicalSection.REFERENCES, "exact_alias", synthetic=True),
    ]
    sentences = [_sentence(1, "L'osservazione descritta si riferisce a un caso.", 1, 5)] + [
        _sentence(10 + i, ref, 3, 6) for i, ref in enumerate(REFS)
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={s.section_id: "" for s in sections},
    )


def _types(contents):
    return [
        (s.header, s.section_type, s.classification_source)
        for s in contents.sections
        if s.section_id
    ]


def test_dedup_keeps_the_populated_synthetic_references_section():
    contents = _italian_case_report()
    populated = {s.section_id for s in contents.sentences}

    enforce_imrad_order(contents.sections, populated)

    assert _types(contents)[1:] == [
        ("BIBLIOGRAFIA", CanonicalSection.UNKNOWN, "imrad_dedup"),
        ("References", CanonicalSection.REFERENCES, "exact_alias"),
    ]


def test_dedup_without_content_still_ranks_by_source_trust():
    contents = _italian_case_report()

    enforce_imrad_order(contents.sections)

    assert _types(contents)[1:] == [
        ("BIBLIOGRAFIA", CanonicalSection.REFERENCES, "exact_alias"),
        ("References", CanonicalSection.UNKNOWN, "imrad_dedup"),
    ]


def test_reference_locator_leaves_one_references_section_with_diagnostics():
    """The locator re-promotes the populated section and resets the empty twin."""
    contents = _italian_case_report()
    enforce_imrad_order(contents.sections)  # the pre-fix ranking: printed twin wins

    rows = RefLocator(contents).collect_reference_rows()

    assert list(rows["text"]) == REFS
    assert _types(contents)[1:] == [
        ("BIBLIOGRAFIA", CanonicalSection.UNKNOWN, "imrad_dedup"),
        ("References", CanonicalSection.REFERENCES, "exact_alias"),
    ]
    references = next(s for s in contents.sections if s.header == "References")
    assert references.classification_score == 1.0
