"""Structured-abstract subheadings printed as rows of their own (#125)."""

import asyncio

from bibr.paper import enforce_imrad_order
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
)
from bibr.pipeline.stages.post_parse import _classify_sections


def _contents(headers: list[tuple[str, int]], sentences_per_section: int = 2) -> PaperContents:
    sections = [PaperSection(section_id=0, header="", level=0, parent_section_id=None)]
    sentences = []
    text_id = 0
    for sid, (header, page) in enumerate(headers, start=1):
        sections.append(
            PaperSection(
                section_id=sid,
                header=header,
                level=2,
                parent_section_id=0,
                provenance=[Provenance(page_no=page, bbox=None)],
            )
        )
        for _ in range(sentences_per_section):
            text_id += 1
            sentences.append(
                PaperSentence(text_id=text_id, text="Text.", section_id=sid, paragraph_id=sid)
            )
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={},
        detected_title="A study of something",
    )


def _by_header(contents):
    return {s.header: s for s in contents.sections if s.level > 0}


def test_subheading_run_closed_by_keywords_joins_the_abstract():
    """An Open Research Europe layout: the subheads are rows of their own,
    with the peer-review sidebar printed between them."""
    contents = _contents(
        [
            ("A study of something", 1),
            ("Abstract", 1),
            ("Background", 1),
            ("Methods", 1),
            ("Open Peer Review", 1),
            ("Reviewer Status", 1),
            ("Results", 1),
            ("Conclusions", 2),
            ("Keywords", 2),
            ("Introduction", 2),
            ("Methods", 3),
            ("Results", 4),
            ("Discussion", 5),
        ]
    )
    asyncio.run(_classify_sections(contents, [], True, None))
    secs = [s for s in contents.sections if s.level > 0]
    abstract = secs[1]
    labels = [secs[i] for i in (2, 3, 6, 7)]
    assert all(s.section_type == CanonicalSection.ABSTRACT for s in labels)
    assert all(s.classification_source == "parent_context" for s in labels)
    assert all((s.level, s.parent_section_id) == (2, abstract.section_id) for s in labels)
    # The body's parts open their own level-1 sections.
    body = secs[9:]
    assert [s.section_type for s in body] == [
        CanonicalSection.INTRODUCTION,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
    ]
    assert all((s.level, s.parent_section_id) == (1, 0) for s in body)

    enforce_imrad_order(contents.sections)
    assert all(s.section_type == CanonicalSection.ABSTRACT for s in labels)
    assert abstract.section_type == CanonicalSection.ABSTRACT


def test_unclosed_run_after_the_abstract_stays_the_body():
    """Unnumbered body Methods and Results right after the abstract, with no
    Keywords or Introduction closing a run, are the body."""
    contents = _contents(
        [
            ("Abstract", 1),
            ("Methods", 1),
            ("Results", 2),
            ("Discussion", 3),
        ]
    )
    asyncio.run(_classify_sections(contents, [], True, None))
    by = _by_header(contents)
    assert by["Methods"].section_type == CanonicalSection.METHODS
    assert by["Results"].section_type == CanonicalSection.RESULTS


def test_long_label_sections_are_the_body():
    contents = _contents(
        [("Abstract", 1), ("Methods", 1), ("Results", 1), ("Keywords", 1)],
        sentences_per_section=12,
    )
    asyncio.run(_classify_sections(contents, [], True, None))
    by = _by_header(contents)
    assert by["Methods"].section_type == CanonicalSection.METHODS
