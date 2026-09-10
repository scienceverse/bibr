"""Contract tests for the shared :class:`DocumentAssembler`.

These lock the invariants the PDF/DOCX parsers rely on after the deferred-text
machinery was extracted: typed entries with inline side-channels (replacing the
old parallel arrays + padding logic), segmentable counting/validation, and the
emission loop's text_id / paragraph_id / last_text_id bookkeeping.
"""

from __future__ import annotations

import pytest

from bibr.paper_contents import PaperSentence, Provenance
from bibr.structure.assembler import DeferredText, DocumentAssembler


def _factory(entry: DeferredText, text: str, text_id: int, paragraph_id: int) -> PaperSentence:
    """Minimal factory that carries the entry side-channels onto the sentence."""
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=entry.section_id,
        paragraph_id=paragraph_id,
        page_number=entry.page_number,
        is_display_formula=entry.is_formula,
        provenance=list(entry.provenance),
        region_meta=entry.region_meta,
    )


def test_append_returns_index_and_keeps_side_channels_inline():
    asm = DocumentAssembler()
    prov = [Provenance(page_no=1, bbox=(0, 0, 1, 1))]
    meta = {"font_size": 10.0}

    idx0 = asm.append("body", 1, 3, True, provenance=prov, region_meta=meta)
    idx1 = asm.append("$$x$$", 1, 3, False, is_formula=True)

    assert (idx0, idx1) == (0, 1)
    assert len(asm) == 2
    # Side-channels are carried on the entry itself — no parallel arrays.
    assert asm.entries[0].provenance == prov
    assert asm.entries[0].region_meta == meta
    # append() copies the provenance list so later caller mutation cannot leak.
    prov.append(Provenance(page_no=2))
    assert asm.entries[0].provenance != prov
    # Defaults for the formula entry.
    assert asm.entries[1].provenance == []
    assert asm.entries[1].region_meta is None
    assert asm.entries[1].is_formula is True


def test_segmentable_texts_and_count():
    asm = DocumentAssembler()
    asm.append("first body", None, 1, True)
    asm.append("$$formula$$", None, 1, False, is_formula=True)
    asm.append("second body", None, 2, True)

    assert asm.segmentable_count == 2
    assert asm.segmentable_texts == ["first body", "second body"]


def test_emit_validates_segment_count():
    asm = DocumentAssembler()
    asm.append("a", None, 1, True)
    asm.append("b", None, 1, True)

    with pytest.raises(ValueError, match="Expected 2 segment lists, got 1"):
        asm.emit(
            [["a."]],
            sentence_factory=_factory,
            sentence_counter=1,
            paragraph_counter=0,
        )


def test_emit_splits_segmented_passes_through_and_tracks_ids():
    asm = DocumentAssembler()
    asm.append("Body one. Body two.", 1, 5, True)
    asm.append("$$E=mc^2$$", 1, 5, False, is_formula=True)

    sentences, next_sc, next_pc = asm.emit(
        [["Body one.", "Body two."]],
        sentence_factory=_factory,
        sentence_counter=1,
        paragraph_counter=0,
    )

    assert [s.text for s in sentences] == ["Body one.", "Body two.", "$$E=mc^2$$"]
    assert [s.text_id for s in sentences] == [1, 2, 3]
    # Each deferred entry gets its own paragraph_id, even single-sentence ones.
    assert [s.paragraph_id for s in sentences] == [1, 1, 2]
    assert sentences[2].is_display_formula is True
    assert next_sc == 4
    assert next_pc == 2
    # last_text_id: one slot per entry, holding the last emitted text_id.
    assert asm.last_text_id == [2, 3]


def test_emit_skips_empty_segments_and_records_none_when_entry_emits_nothing():
    asm = DocumentAssembler()
    asm.append("all blank", None, 1, True)
    asm.append("kept", None, 1, True)

    sentences, next_sc, next_pc = asm.emit(
        [["  ", ""], ["kept."]],
        sentence_factory=_factory,
        sentence_counter=1,
        paragraph_counter=0,
    )

    assert [s.text for s in sentences] == ["kept."]
    # First entry produced no sentence -> None; paragraph counter still advanced.
    assert asm.last_text_id == [None, 1]
    assert next_sc == 2
    assert next_pc == 2


def test_emit_invokes_on_sentence_hook_for_every_sentence():
    asm = DocumentAssembler()
    asm.append("one. two.", None, 1, True)
    asm.append("$$f$$", None, 1, False, is_formula=True)
    seen: list[str] = []

    asm.emit(
        [["one.", "two."]],
        sentence_factory=_factory,
        sentence_counter=1,
        paragraph_counter=0,
        on_sentence=lambda s: seen.append(s.text),
    )

    assert seen == ["one.", "two.", "$$f$$"]


def test_last_text_id_initialised_empty_before_emit():
    # create_content_sections (footnote path) may consult last_text_id even
    # when emit never ran (nothing deferred). It must be a list, not unset.
    assert DocumentAssembler().last_text_id == []


def test_build_sections_text_groups_and_joins_by_section():
    sentences = [
        PaperSentence(text_id=1, text="a1", section_id=1, paragraph_id=1),
        PaperSentence(text_id=2, text="a2", section_id=1, paragraph_id=1),
        PaperSentence(text_id=3, text="b1", section_id=2, paragraph_id=2),
    ]
    assert DocumentAssembler.build_sections_text(sentences) == {1: "a1 a2", 2: "b1"}


def test_append_normalizes_unicode_to_nfc():
    assembler = DocumentAssembler()
    assembler.append("Garci\u0301a", None, 1, True)
    assert assembler.entries[0].text == "García"
