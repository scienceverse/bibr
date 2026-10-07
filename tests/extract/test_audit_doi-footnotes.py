"""Audit fixes: DOI provenance and note citation splitting.

The DOI provenance diff ran in time quadratic in a sentence's length and the
note citation split in time cubic in a note's length; a resolver host printed
in capitals hid the label in front of a DOI; casefolding a text-layer line
moved the end of its DOI.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from bibr.extract import doi_identity
from bibr.extract import footnote_citations as fc
from bibr.extract.pdf_doi_evidence import TextLayerLine
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence

# ---------------------------------------------------------------------------
# DOI provenance
# ---------------------------------------------------------------------------

_WORDS = ["the", "e\ufb00ect", "of", "data", "o\ufb03ce", "model", "study", "a\ufb00ord"]
_PROVENANCE = {
    "source_kind": "sentence",
    "page": 3,
    "section_id": 1,
    "section_type": None,
    "region_index": None,
    "region_type": None,
    "text_id": 1,
}


def _prose(words: int) -> str:
    return " ".join(_WORDS[(k * 7) % len(_WORDS)] for k in range(words))


def _diffed_whole(source: str, cleaned: str) -> list[tuple[int, int]]:
    """The mapping as one diff of the whole text computes it."""
    ranges = [(0, 0)] * len(cleaned)
    opcodes = SequenceMatcher(None, source, cleaned, autojunk=False).get_opcodes()
    for tag, source_start, source_end, clean_start, clean_end in opcodes:
        clean_length, source_length = clean_end - clean_start, source_end - source_start
        for offset in range(clean_length):
            if tag == "equal":
                ranges[clean_start + offset] = (source_start + offset, source_start + offset + 1)
            else:
                ranges[clean_start + offset] = (
                    source_start + (offset * source_length) // clean_length,
                    source_start
                    + ((offset + 1) * source_length + clean_length - 1) // clean_length,
                )
    return ranges


class _WindowedMatcher(SequenceMatcher):
    """A SequenceMatcher that refuses to diff more than a window of text."""

    def __init__(self, isjunk=None, a="", b="", autojunk=True):
        assert max(len(a), len(b)) <= 2_000, "diffed the whole text"
        super().__init__(isjunk, a, b, autojunk)


@pytest.mark.parametrize(
    "source",
    [
        "Cite as: https://doi.org/10. 1016/j.lanwpc.2023. 100933 (2023).",
        "An e\ufb00ect on o\ufb03ce work, doi: 10.1234/abc-\n123x and more.",
        "Received 1 May.\x07\x07 DOI 10.1037/xge00012\u00ad34.supp, Spring-\ner.",
        "https://www.exam-\r\nple.org/path then cafe\u0301 \uff0c doi:10.1007/s11192-019-03217-6",
        "Text\x00\x00\x00\x00 with   gaps https://doi.org/10.1234/abc-       123and.",
        f"{_prose(250)} doi: 10.1234/abc.2020.1 and https://doi.org/10. 1016/j.x.2023. 100933.",
    ],
    ids=["registrant-wrap", "ligatures", "controls-soft-hyphen", "nfc-crlf", "gaps", "long"],
)
def test_doi_provenance_maps_as_one_diff_of_the_whole_text_did(source):
    cleaned = doi_identity._repair_doi_text(source)

    assert cleaned != source
    assert doi_identity._cleaned_char_source_ranges(source, cleaned) == _diffed_whole(
        source, cleaned
    )


def test_doi_provenance_of_a_long_footnote_diffs_only_around_each_repair(monkeypatch):
    # A footnote of 15,000 characters with a ligature in a third of its words:
    # one diff of the whole text took seconds, on the event loop.
    text = f"{_prose(3000)} doi: 10.1234/abc.2020.1 and https://doi.org/10. 1016/j.x.2023. 100933."
    assert len(text) > 15_000
    monkeypatch.setattr(doi_identity, "SequenceMatcher", _WindowedMatcher)

    found = doi_identity._candidates_from_text(text, **_PROVENANCE)

    assert [(c.normalized, c.raw) for c in found] == [
        ("10.1234/abc.2020.1", "10.1234/abc.2020.1"),
        ("10.1016/j.x.2023.100933", "10. 1016/j.x.2023. 100933."),
    ]


def test_doi_provenance_crosses_a_long_run_of_control_characters(monkeypatch):
    monkeypatch.setattr(doi_identity, "SequenceMatcher", _WindowedMatcher)
    text = "Received 1 May." + "\x00" * 3000 + " doi: 10.1234/abc.2020.1 (2020)"

    [candidate] = doi_identity._candidates_from_text(text, **_PROVENANCE)

    assert candidate.raw == "10.1234/abc.2020.1"


def test_text_layer_doi_positions_of_a_long_line_are_diffed_in_windows(monkeypatch):
    monkeypatch.setattr(doi_identity, "SequenceMatcher", _WindowedMatcher)
    text = f"{_prose(1500)} https://doi.org/10.1234/own.2020.4"
    line = TextLayerLine(1, text, tuple((float(i), 500.0) for i in range(len(text))), "")

    [(candidate, tail)] = doi_identity._text_layer_candidates(line, [], {})

    assert candidate.raw == "10.1234/own.2020.4"
    assert tail == ""


# ---------------------------------------------------------------------------
# DOI markers and text-layer ends
# ---------------------------------------------------------------------------


def _contents(text: str, page: int) -> PaperContents:
    return PaperContents(
        sentences=[PaperSentence(1, text, 1, 1, page_number=page)],
        sections=[PaperSection(1, "Introduction", 1, None, CanonicalSection.INTRODUCTION)],
        tables=[],
        links=[],
        sections_text={},
    )


@pytest.mark.parametrize(
    ("text", "marker_kind", "tier"),
    [
        ("Article DOI: HTTPS://DOI.ORG/10.1234/self.1", "article_doi", 3),
        ("Article DOI: Https://Www.Doi.Org/10.1234/self.1", "article_doi", 3),
        ("Data DOI: HTTPS://DX.DOI.ORG/10.1234/data.1", "data_doi", 0),
        ("Journal DOI: HTTPS://DOI.ORG/10.1234/serial", "journal_doi", 1),
    ],
)
def test_a_resolver_host_in_capitals_keeps_the_label_before_it(text, marker_kind, tier):
    [candidate] = doi_identity.collect_doi_candidates(_contents(text, page=3))

    assert candidate.marker_kind == marker_kind
    assert candidate.selection_tier == tier


def test_a_text_layer_doi_ends_where_the_line_prints_it_after_letters_casefolding_lengthens():
    # "ß" casefolds to "ss": an offset found in the casefolded line fell two
    # characters past the DOI's end in the line itself.
    text = "Weiß, Straße: https://doi.org/10.1234/jex.2026.04.006 x."
    line = TextLayerLine(1, text, tuple((970.0, 500.0) for _ in text), "")

    [(candidate, tail)] = doi_identity._text_layer_candidates(line, [], {})

    assert candidate.normalized == "10.1234/jex.2026.04.006"
    assert tail == " x."


# ---------------------------------------------------------------------------
# Note citations
# ---------------------------------------------------------------------------

_FIRST = ["Anna", "John", "Maria", "Pierre", "Carlos", "Luisa", "Hans"]
_LAST = ["Karenina", "Milius", "Dupont", "Garcia", "Lopez", "Novak", "Rossi"]
_TOPICS = ["argument", "remarks", "court", "doctrine", "parties", "theory", "account"]


def _commentary_note(chars: int) -> str:
    """A long note of commentary naming people ("by Anna Karenina and others")
    in every sentence and citing nothing."""
    sentences: list[str] = []
    k = 0
    while sum(len(sentence) + 1 for sentence in sentences) < chars:
        sentences.append(
            f"The {_TOPICS[k % 7]} {k} was developed further by {_FIRST[k % 7]} "
            f"{_LAST[(k * 3) % 7]} and others in their {_TOPICS[(k * 5) % 7]}."
        )
        k += 1
    return "12. " + " ".join(sentences)


def test_a_long_note_of_commentary_is_split_without_rereading_it_at_every_break(monkeypatch):
    # Every break read each hand-over's whole remainder again: about 1,600
    # characters read per character of this note, 6.7 million in all.
    note = _commentary_note(4_000)
    budget = [200 * len(note)]
    looks_like_citation = fc.looks_like_citation

    def counted(clause: str) -> bool:
        budget[0] -= len(clause)
        assert budget[0] >= 0, "the split re-read the note"
        return looks_like_citation(clause)

    monkeypatch.setattr(fc, "looks_like_citation", counted)

    assert fc._split_citations(note) == [note.removeprefix("12. ")]


def test_a_hand_over_at_the_end_of_a_long_note_still_leads_to_its_citation():
    commentary = " ".join(
        f"The court held in case {k} that the doctrine applies where the parties agreed."
        for k in range(40)
    )
    note = (
        f"3. {commentary} Estos autores han demostrado, segun W. Stoczkowski, Aux "
        "origines de l'humanité, Paris, Le Pommier, 2001, p. 12."
    )
    assert len(note) > 3_000

    assert fc._split_citations(note) == [
        "W. Stoczkowski, Aux origines de l'humanité, Paris, Le Pommier, 2001, p. 12."
    ]
