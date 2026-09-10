"""Regression test for report-only oversized-abstract handling."""

from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
)
from bibr.structure.implicit_sections import (
    _MAX_ABSTRACT_CHARS,
    _trim_bloated_abstract,
)


def test_first_sentence_over_limit_is_retained_verbatim(caplog):
    abs_sec = PaperSection(
        section_id=1,
        header="Abstract",
        level=1,
        parent_section_id=0,
        section_type=CanonicalSection.ABSTRACT,
    )
    huge = "x" * (_MAX_ABSTRACT_CHARS + 100)
    sentences = [
        PaperSentence(text_id=1, text=huge, section_id=1, paragraph_id=1, page_number=1),
        PaperSentence(text_id=2, text="trailing.", section_id=1, paragraph_id=2, page_number=1),
    ]
    pc = PaperContents(
        sentences=sentences,
        sections=[abs_sec],
        tables=[],
        links=[],
        sections_text={1: huge + " trailing."},
    )

    _trim_bloated_abstract(pc)

    abstract_sents = [s for s in pc.sentences if s.section_id == 1]
    assert [sentence.text for sentence in abstract_sents] == [huge, "trailing."]
    assert pc.sections_text[1] == huge + " trailing."
    assert "section_id=1" in caplog.text
    assert "text_ids=(1, 2)" in caplog.text
