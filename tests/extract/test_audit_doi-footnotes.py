"""Audit fixes: DOI provenance, note citation splitting and email affinity.

The DOI provenance diff ran in time quadratic in a sentence's length and the
note citation split in time cubic in a note's length; a resolver host printed
in capitals hid the label in front of a DOI; casefolding a text-layer line
moved the end of its DOI; an accented surname matched unrelated email
addresses through its ASCII fragments ("Šimić" -> "imi").
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

import pytest

from bibr.extract import author_email_harvester as harvester
from bibr.extract import doi_identity
from bibr.extract import footnote_citations as fc
from bibr.extract.pdf_doi_evidence import TextLayerLine
from bibr.models import PaperAuthor
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


_KOREAN_NFD = unicodedata.normalize(
    "NFD", "이 연구는 한국 사회의 변화와 정책 효과를 분석하였다 자료는 국가 통계에서 수집되었다"
)
_TWO_URLS = " https://doi.org/10. 1016/j.lanwpc.2023. 100933 and https://doi.org/10.1016/j.cell.2020.01.001."
_TWO_URL_RAWS = ["10. 1016/j.lanwpc.2023. 100933", "10.1016/j.cell.2020.01.001."]
_GLYPHS = "\x03\x04\x05\x06\x07\x08\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19"


@pytest.mark.parametrize(
    ("source", "raws"),
    [
        # Decomposed Korean (a macOS or HWP text layer), which NFC rewrites in
        # every syllable, before two DOIs under one registrant.
        (
            f"{_KOREAN_NFD} {_KOREAN_NFD} doi:10.1234/abc-\n123x; doi:10.1234/abc-123y",
            ["10.1234/abc-\n123x;", "10.1234/abc-123y"],
        ),
        (f"{_KOREAN_NFD} {_KOREAN_NFD}{_TWO_URLS}", _TWO_URL_RAWS),
        # Control characters filling the window, too few to be set aside.
        ("Intro su\u00adper text" + "\x00" * 60 + _TWO_URLS, _TWO_URL_RAWS),
        # Words of unmapped glyphs (PDFium's text of a Type 3 font).
        (
            " ".join(_GLYPHS[k % 7 : k % 7 + 3 + k % 5] for k in range(16)) + _TWO_URLS,
            _TWO_URL_RAWS,
        ),
        # A wrap's whitespace taken out, longer than the reach.
        (
            "See doi:10.1234/abc-" + " " * 600 + "123x and doi:10.1234/abc-123x here.",
            ["10.1234/abc-" + " " * 600 + "123x", "10.1234/abc-123x"],
        ),
        # A whitespace run taken out before a URL, and a later one kept.
        (
            "E\ufb00ect\n"
            + " " * 120
            + "https://doi.org/10.1016/j.cell.2020.01.001"
            + " " * 60
            + "and doi:10.1234/abc.1",
            ["10.1016/j.cell.2020.01.001", "10.1234/abc.1"],
        ),
        # A ligature just before a wrap whose whitespace was taken out, and
        # one before a long whitespace run that was kept.
        (
            "o\ufb00er\n" + " " * 600 + "https://doi.org/10.1016/j.cell.2020.01.001 and more.",
            ["10.1016/j.cell.2020.01.001"],
        ),
        (
            "\ufb01" + " " * 700 + "rst https://doi.org/10.1016/j.cell.2020.01.001 and more.",
            ["10.1016/j.cell.2020.01.001"],
        ),
        # Ligatures between control characters: only the controls are deleted.
        ("\ufb01\x07" * 1000 + _TWO_URLS, _TWO_URL_RAWS),
    ],
    ids=[
        "nfd-korean",
        "nfd-korean-urls",
        "nul-run",
        "glyph-words",
        "long-wrap",
        "spaces",
        "ligature-then-wrap",
        "ligature-then-kept-spaces",
        "lig-ctrl",
    ],
)
def test_doi_provenance_resyncs_where_the_two_texts_next_agree(source, raws):
    # Resyncing at the longest agreement in reach took a later repeat (the
    # second DOI's prefix, a run of spaces) and mapped the text before it as
    # one replacement: the first DOI's raw spelling came out of the second.
    cleaned = doi_identity._repair_doi_text(source)
    line = TextLayerLine(1, source, tuple((float(i), 500.0) for i in range(len(source))), "")

    assert doi_identity._cleaned_char_source_ranges(source, cleaned) == _diffed_whole(
        source, cleaned
    )
    assert [c.raw for c in doi_identity._candidates_from_text(source, **_PROVENANCE)] == raws
    assert [c.raw for c, _tail in doi_identity._text_layer_candidates(line, [], {})] == raws


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


# ---------------------------------------------------------------------------
# Email affinity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "family", "local", "names"),
    [
        # Accented names: their ASCII fragments ("imi", "dvo") name nobody.
        ("Ivan", "Šimić", "jimiller", False),
        ("Jan", "Dvořák", "dvoretsky", False),
        ("Jan", "Dvořák", "dvorak", True),
        ("Jiří", "Novák", "jirinovak", True),
        ("Xiaohong", "Lü", "luxh", True),
        # Letters NFKD leaves whole, spelled out as the address does.
        ("Ivar", "Bræin", "braein", True),
        ("Jon", "Þór", "thorj", True),
        # ASCII names as before.
        ("Xiang-Min", "Yang", "yxiangmind", True),
        ("Xiaohong", "Li", "lixh", True),
        ("Jane", "Doe", "editor", False),
    ],
)
def test_email_name_affinity_folds_accents_before_matching(given, family, local, names):
    assert harvester._email_name_affinity(given, family, local) is names


def test_given_name_affinity_folds_accents():
    assert harvester._given_name_affinity("Jiří", "jirisimic") == 2
    assert harvester._given_name_affinity("Šimon", "imonx") == 0


def test_an_accented_surname_does_not_take_an_unrelated_address():
    # The lab manager's "jimiller" holds "imi", the ASCII letters of "Šimić":
    # it was handed to Šimić and blocked his own address.
    from types import SimpleNamespace

    import pandas as pd

    lines = [
        "Ivan Šimić and John Roe",
        "Department of X",
        "Lab manager: jimiller@appliedthings.org",
        "Filler sentence one.",
        "Filler sentence two.",
        "Corresponding author: Ivan Šimić, ivan.simic@uni.hr",
    ]
    contents = SimpleNamespace(
        sentences=[
            SimpleNamespace(text=t, section_id=0, text_id=i, page_number=1)
            for i, t in enumerate(lines)
        ],
        sections=[SimpleNamespace(section_id=0, section_type="abstract", header="")],
        sentences_df=pd.DataFrame(
            [
                {"text_id": i, "text": t, "page_number": 1, "section_id": 0}
                for i, t in enumerate(lines)
            ]
        ),
        detected_headers=[],
        detected_footers=[],
    )
    authors = [
        PaperAuthor(author_id=1, given="Ivan", family="Šimić", affiliation="X", corresponding=True),
        PaperAuthor(author_id=2, given="John", family="Roe", affiliation="X"),
    ]

    harvester.AuthorEmailHarvester(contents).harvest(authors)

    assert authors[0].email == "ivan.simic@uni.hr"
    assert authors[0].corresponding is True
    assert authors[1].email in (None, "")
