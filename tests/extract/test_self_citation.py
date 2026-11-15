"""The paper's own printed citation as a second source of its metadata.

No model is loaded: the reference parser is a stand-in that returns fixed fields.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bibr.clients.skipped import SkippedLlmClient
from bibr.config import Settings
from bibr.extract.field_decisions import (
    FieldCandidate,
    FieldDecision,
    apply_decision,
    field_decisions_of,
    fill_empty,
)
from bibr.extract.self_citation import (
    PrintedCitation,
    apply_printed_citation,
    find_printed_citation,
    read_printed_citation,
    same_title,
)
from bibr.input.file import InputFile
from bibr.models import PaperAuthor, PaperMetadata, PaperReference
from bibr.paper import Paper
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.processing_warnings import WarningCode
from bibr.validation import IssueSeverity, ValidationIssue

TITLE = "Reading glacier retreat from archived survey photographs"


def _contents(*rows, sections=None) -> PaperContents:
    """Sentences from ``(text, page, paragraph_id[, section_id])`` rows, text ids from 1."""
    sentences = [
        PaperSentence(
            index,
            row[0],
            row[3] if len(row) > 3 else 1,
            row[2],
            page_number=row[1],
        )
        for index, row in enumerate(rows, start=1)
    ]
    return PaperContents(
        sentences=sentences,
        sections=sections
        or [
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "Front", 1, 0, CanonicalSection.TITLE),
            PaperSection(2, "References", 1, 0, CanonicalSection.REFERENCES),
        ],
        tables=[],
        links=[],
        sections_text={},
    )


# ── Finding the line ────────────────────────────────────────────────────


def test_a_citation_split_into_sentences_is_joined_up_to_its_doi():
    contents = _contents(
        (TITLE, 1, 1),
        ("Citation: Moreau L, Okafor N and Lind P (2021) Reading glacier retreat.", 1, 2),
        ("Front. Earth Sci. 9:61234.", 1, 2),
        ("doi: 10.5555/feart.2021.61234", 1, 2),
        ("Received: 2 March 2021", 1, 2),
    )
    line = find_printed_citation(contents)
    assert line is not None
    assert line.text_ids == (2, 3, 4)
    assert line.label == "Citation:"
    assert line.strong is False
    assert line.text.startswith("Moreau L, Okafor N")
    assert line.text.endswith("10.5555/feart.2021.61234")


def test_a_label_alone_on_its_row_takes_the_next_row():
    contents = _contents(
        ("Full citation:", 1, 1),
        (f"Moreau, L., & Okafor, N. (2017). {TITLE}. Polar Records, 9(2), 1-20.", 1, 2),
        ("This is the accepted manuscript.", 1, 3),
    )
    line = find_printed_citation(contents)
    assert line is not None
    assert line.strong is True
    assert line.text_ids == (1, 2)
    assert line.text.startswith("Moreau, L., & Okafor, N. (2017).")


def test_the_block_stops_at_another_label_and_at_the_paragraph_end():
    contents = _contents(
        ("To cite this article: Moreau L (2019) Glacier survey notes.", 1, 1),
        ("To link to this article: https://doi.org/10.5555/x", 1, 1),
    )
    assert find_printed_citation(contents).text_ids == (1,)
    contents = _contents(
        ("Cite this: Moreau L (2019) Glacier survey notes from the field.", 1, 1),
        ("The ice sheet lost mass in every year.", 1, 2),
    )
    assert find_printed_citation(contents).text_ids == (1,)


def test_a_label_that_names_this_article_wins_over_a_bare_citation():
    contents = _contents(
        ("Citation: Data Team (2020) The glacier photograph archive, version 3.", 1, 1),
        ("How to cite this article: Moreau L. Glacier notes. Polar Rec 2019;9:1.", 1, 2),
    )
    line = find_printed_citation(contents)
    assert line.strong is True
    assert line.text_ids == (2,)


@pytest.mark.parametrize(
    ("row", "strong"),
    [
        ("Pour citer cet article : Moreau L. Les glaciers alpins. Revue 12 (2019) 1-9.", True),
        ("Cómo citar este artículo: Moreau L. Los glaciares. Revista 12 (2019) 1-9.", True),
        ("Para citar: Moreau L. Los glaciares andinos. Revista 12 (2019) 1-9.", False),
        ("Zitierweise: Moreau L. Gletscher der Alpen. Zeitschrift 12 (2019) 1-9.", False),
        ("Citation Moreau L, Okafor N (2019) Glacier survey notes. J Ice 3: e12.", False),
        ("Cite this paper as: Moreau L. Glacier survey notes. J Ice 3 (2019) e12.", True),
    ],
)
def test_the_labels_bibr_knows(row, strong):
    line = find_printed_citation(_contents((row, 1, 1)))
    assert line is not None
    assert line.strong is strong


@pytest.mark.parametrize(
    "rows",
    [
        # Page 2 is not the first page.
        [(TITLE, 1, 1), ("To cite this article: Moreau L (2019) Glacier notes.", 2, 2)],
        # A referee report's own citation names another work.
        [("How to cite this report: Lind P. Referee report for: Glacier notes.", 1, 1)],
        # Prose that happens to open with the word.
        [("Citation analysis of glacier studies shows steady growth since 1990.", 1, 1)],
        # A reference entry on the first page.
        [("Citation: Moreau L (2019) Glacier survey notes from the field.", 1, 1, 2)],
        # Nothing after the label.
        [("To cite this article:", 1, 1)],
    ],
)
def test_no_line(rows):
    assert find_printed_citation(_contents(*rows)) is None


# ── Names and titles ────────────────────────────────────────────────────


def test_titles_match_across_case_accents_and_punctuation():
    assert same_title("Glacier Retreat: A Survey", "glacier retreat - a survey.")
    assert same_title("Évolution des glaciers", "Evolution des glaciers")
    assert not same_title("Glacier retreat", "Sea ice in the Arctic")
    assert not same_title("", "Glacier retreat")


# ── Parsing with the reference parser ───────────────────────────────────


def _parser_returning(fields):
    class _Parser:
        last_truncated_count = 0
        seen: list[str] = []

        def parse_batch(self, segments):
            self.seen.extend(segments)
            return [dict(fields) for _ in segments]

    parser = _Parser()
    return parser, patch("bibr.extract.ref_extractor._get_ner_parser", return_value=parser)


def test_the_line_goes_through_the_reference_parse():
    printed = "Moreau L, Okafor N (2019) Glacier survey notes. J Ice 3(2): 11-19."
    contents = _contents(
        (f"Citation: {printed}", 1, 1),
        ("doi: 10.5555/ice.3.11", 1, 1),
    )
    parser, patched = _parser_returning(
        {
            "authors": "Moreau L, Okafor N",
            "title": "Glacier survey notes",
            "container": "J Ice",
            "year": 2019,
            "volume": "3",
            "issue": "2",
            "first_page": "11",
            "last_page": "19",
        }
    )
    with patched:
        line = read_printed_citation(
            contents, file_hash="f", llm_client=SkippedLlmClient(), settings=Settings
        )
    assert parser.seen == [f"{printed} doi: 10.5555/ice.3.11"]
    assert contents.printed_citation is line
    ref = line.reference
    assert (ref.title, ref.container, ref.year, ref.volume, ref.issue) == (
        "Glacier survey notes",
        "J Ice",
        2019,
        "3",
        "2",
    )
    assert (ref.first_page, ref.last_page) == ("11", "19")
    # The list's finalize read the DOI the parser did not tag off the printed line.
    assert ref.doi == "10.5555/ice.3.11"


def test_a_parse_that_fails_leaves_the_line_unparsed():
    contents = _contents(("To cite this article: Moreau L (2019) Glacier survey notes.", 1, 1))

    class _Broken:
        def parse_batch(self, segments):
            raise RuntimeError("no session")

    with patch("bibr.extract.ref_extractor._get_ner_parser", return_value=_Broken()):
        line = read_printed_citation(
            contents, file_hash="f", llm_client=SkippedLlmClient(), settings=Settings
        )
    assert line.reference is None
    paper = _paper(contents, PaperMetadata(doi="", title=""))
    receipt = apply_printed_citation(paper)
    assert receipt["check"] == "not_parsed"
    assert receipt["accepted"] is False


# ── Deciding and filling ────────────────────────────────────────────────


def _reference(**fields) -> PaperReference:
    base = {
        "bib_id": 1,
        "title": TITLE,
        "authors": "Moreau, L., & Okafor, N.",
        "container": "Polar Records",
        "year": 2017,
        "volume": "9",
        "issue": "2",
        "first_page": "101",
        "last_page": "120",
        "doi": "10.5555/polar.2017.9",
    }
    base.update(fields)
    return PaperReference(**base)


def _paper(contents, metadata, *, issues=()) -> Paper:
    input_file = InputFile(path="paper.pdf")
    input_file.file_hash = "a" * 16
    return Paper(
        input_file=input_file,
        contents=contents,
        metadata=metadata,
        validation_issues=list(issues),
    )


def _with_line(reference, *, strong=True, label="To cite this article:") -> PaperContents:
    contents = _contents(("row", 1, 1))
    contents.printed_citation = PrintedCitation((4, 5), label, strong, "the line", reference)
    return contents


def _author(family, given="A.", position=1) -> PaperAuthor:
    return PaperAuthor(author_id=position, given=given, family=family, affiliation="")


def test_a_matching_doi_fills_only_the_empty_fields():
    metadata = PaperMetadata(
        doi="10.5555/POLAR.2017.9",
        title="A different printed title",
        authors=[_author("Moreau")],
        journal="Polar Rec.",
        volume="9",
    )
    paper = _paper(_with_line(_reference()), metadata)
    receipt = apply_printed_citation(paper)
    assert receipt["check"] == "doi_match"
    assert receipt["accepted"] is True
    assert receipt["filled"] == ["published", "issue", "first_page", "last_page"]
    assert (metadata.published, metadata.issue, metadata.first_page, metadata.last_page) == (
        "2017",
        "2",
        "101",
        "120",
    )
    # Never overwritten.
    assert metadata.title == "A different printed title"
    assert metadata.journal == "Polar Rec."
    assert receipt["agreement"] == {
        "doi": "agree",
        "title": "disagree",
        "first_author": "agree",
        "volume": "agree",
    }
    # The title and the authors were not filled, so nothing is said about them.
    assert paper.processing_warnings == []


def _fill_from(reference, text="the line") -> PaperMetadata:
    contents = _contents(("row", 1, 1))
    contents.printed_citation = PrintedCitation(
        (4, 5), "To cite this article:", True, text, reference
    )
    metadata = PaperMetadata(doi="10.5555/polar.2017.9", title="", journal="")
    apply_printed_citation(_paper(contents, metadata))
    return metadata


def test_filled_values_have_their_white_space_collapsed():
    metadata = _fill_from(
        _reference(
            title="  Ice \r\n  cores of\tthe   north ",
            container="Adv Sex Reprod \r\nHealth Res",
            volume="9 \n",
            issue="\r\n2",
            first_page="10 ",
            last_page=" 20",
            authors="Moreau,\r\nL., & Okafor,\nN.",
        )
    )
    assert metadata.title == "Ice cores of the north"
    assert metadata.journal == "Adv Sex Reprod Health Res"
    assert (metadata.volume, metadata.issue) == ("9", "2")
    assert (metadata.first_page, metadata.last_page) == ("10", "20")
    assert [(a.given, a.family) for a in metadata.authors] == [("L.", "Moreau"), ("N.", "Okafor")]


@pytest.mark.parametrize("journal", ["Neurol.\r\nInt", "Brain\r\nSci"])
def test_a_journal_name_broken_over_lines_is_one_line(journal):
    metadata = _fill_from(_reference(container=journal))
    assert metadata.journal == " ".join(journal.split())
    assert "\r" not in metadata.journal and "\n" not in metadata.journal


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        ("1262", "1275", ("1262", "1275")),
        ("60", "72", ("60", "72")),
        ("1262", "75", ("1262", "75")),
        ("e1", "e12", ("e1", "e12")),
        ("e00376", "18", ("e00376-18", None)),
        ("S163", "9", ("S163", "S169")),
        ("e120", "5", ("e120", "e125")),
        ("e120", "20", ("e120-20", None)),
    ],
)
def test_only_a_page_range_is_split_into_first_and_last_page(first, last, expected):
    metadata = _fill_from(_reference(first_page=first, last_page=last))
    assert (metadata.first_page, metadata.last_page) == expected


def test_an_article_number_keeps_the_printed_string():
    text = "mSphere 3:e00376-18. https://doi.org/10.1128/msphere.00376-18."
    metadata = _fill_from(
        _reference(volume="3", first_page="e00376", last_page="18", year=2018), text
    )
    assert metadata.first_page == "e00376-18"
    assert metadata.last_page is None
    assert metadata.volume == "3"


def test_a_different_doi_rejects_the_line_whatever_else_agrees():
    metadata = PaperMetadata(doi="10.5555/other", title=TITLE)
    paper = _paper(_with_line(_reference()), metadata)
    receipt = apply_printed_citation(paper)
    assert (receipt["check"], receipt["accepted"], receipt["filled"]) == (
        "doi_mismatch",
        False,
        [],
    )
    assert receipt["agreement"]["title"] == "agree"
    assert metadata.journal is None


def test_without_a_doi_comparison_the_title_decides():
    metadata = PaperMetadata(doi="", title=TITLE.upper())
    receipt = apply_printed_citation(_paper(_with_line(_reference(), strong=False), metadata))
    assert (receipt["check"], receipt["accepted"]) == ("title_match", True)
    assert metadata.journal == "Polar Records"

    metadata = PaperMetadata(doi="", title="Sea ice in the Arctic")
    receipt = apply_printed_citation(_paper(_with_line(_reference(), strong=True), metadata))
    assert (receipt["check"], receipt["accepted"]) == ("title_mismatch", False)
    assert metadata.journal is None


def test_without_a_title_the_first_author_decides_and_the_title_is_filled():
    metadata = PaperMetadata(doi="", title="", authors=[_author("Lind"), _author("Moreau")])
    paper = _paper(_with_line(_reference(doi=None), strong=False), metadata)
    receipt = apply_printed_citation(paper)
    assert (receipt["check"], receipt["accepted"]) == ("first_author_match", True)
    assert metadata.title == TITLE
    assert "title" in receipt["filled"] and "author" not in receipt["filled"]
    # The first author row is Lind, so the first authors disagree.
    assert receipt["agreement"]["first_author"] == "disagree"
    (warning,) = paper.processing_warnings
    assert warning.code == WarningCode.SELF_CITATION_FILLED
    assert warning.message.startswith("title read from the paper's printed citation")
    decision = field_decisions_of(metadata).get("title")
    assert (decision.source, decision.rule) == ("self_citation", "self_citation_fill")
    assert decision.selected.evidence_ids == ("text:4", "text:5")

    metadata = PaperMetadata(doi="", title="", authors=[_author("Lind")])
    receipt = apply_printed_citation(_paper(_with_line(_reference(doi=None)), metadata))
    assert (receipt["check"], receipt["accepted"]) == ("first_author_mismatch", False)
    assert metadata.title == ""


def test_with_nothing_to_compare_only_a_label_naming_this_article_is_taken():
    metadata = PaperMetadata(doi="", title="")
    paper = _paper(_with_line(_reference(doi=None), strong=True), metadata)
    receipt = apply_printed_citation(paper)
    assert (receipt["check"], receipt["accepted"]) == ("strong_cue", True)
    assert metadata.title == TITLE
    assert [(a.given, a.family) for a in metadata.authors] == [("L.", "Moreau"), ("N.", "Okafor")]
    assert receipt["filled"][:2] == ["title", "author"]
    (warning,) = paper.processing_warnings
    assert warning.message.startswith("title and author read from")
    assert field_decisions_of(metadata).get("author").rule == "self_citation_fill"

    metadata = PaperMetadata(doi="", title="")
    receipt = apply_printed_citation(
        _paper(_with_line(_reference(doi=None), strong=False, label="Citation:"), metadata)
    )
    assert (receipt["check"], receipt["accepted"], receipt["filled"]) == ("weak_cue", False, [])
    assert metadata.title == ""


def test_a_list_cut_short_by_et_al_fills_no_authors():
    metadata = PaperMetadata(doi="10.5555/polar.2017.9", title="")
    reference = _reference(authors="Moreau L, Okafor N, Lind P, et al.")
    receipt = apply_printed_citation(_paper(_with_line(reference), metadata))
    assert "title" in receipt["filled"]
    assert "author" not in receipt["filled"]
    assert metadata.authors == []


def test_no_locator_is_taken_from_a_line_that_disagrees_on_one():
    metadata = PaperMetadata(
        doi="10.5555/polar.2017.9", title=TITLE, volume="10", authors=[_author("Moreau")]
    )
    receipt = apply_printed_citation(_paper(_with_line(_reference()), metadata))
    assert receipt["agreement"]["volume"] == "disagree"
    assert receipt["filled"] == ["journal", "published"]
    assert (metadata.issue, metadata.first_page, metadata.last_page) == (None, None, None)


def test_an_abstained_front_matter_takes_nothing_from_the_line():
    abstention = ValidationIssue(
        code="VAL_METADATA_MULTI_ITEM",
        severity=IssueSeverity.WARNING,
        message="several records",
        origin_stage="metadata",
        blocking=True,
    )
    metadata = PaperMetadata(doi="", title="")
    receipt = apply_printed_citation(
        _paper(_with_line(_reference(doi=None)), metadata, issues=[abstention])
    )
    assert (receipt["check"], receipt["accepted"], receipt["filled"]) == (
        "front_matter_abstained",
        False,
        [],
    )
    assert metadata.title == ""


def test_the_year_agreement_reads_the_printed_date():
    metadata = PaperMetadata(doi="10.5555/polar.2017.9", title=TITLE, published="12 May 2016")
    receipt = apply_printed_citation(_paper(_with_line(_reference()), metadata))
    assert receipt["agreement"]["year"] == "disagree"
    assert "published" not in receipt["filled"]


def test_the_receipt_is_made_once():
    metadata = PaperMetadata(doi="", title="")
    paper = _paper(_with_line(_reference(doi=None)), metadata)
    first = apply_printed_citation(paper)
    again = apply_printed_citation(paper)
    assert again is first
    assert len(paper.processing_warnings) == 1


def test_no_line_no_receipt():
    metadata = PaperMetadata(doi="", title="")
    paper = _paper(_contents(("row", 1, 1)), metadata)
    assert apply_printed_citation(paper) is None
    assert paper.contents.printed_citation_receipt is None


# ── The decision ledger ─────────────────────────────────────────────────


def test_a_fill_keeps_the_decision_receipt_and_never_overwrites():
    metadata = PaperMetadata(doi="", title="")
    empty = FieldCandidate("title", "llm", "")
    apply_decision(metadata, FieldDecision("title", "", None, "none", (), producer="llm"))
    candidate = FieldCandidate("title", "self_citation", TITLE)
    assert fill_empty(metadata, candidate, "self_citation_fill", "printed citation")
    decision = field_decisions_of(metadata).get("title")
    assert decision.value == TITLE and decision.selected is candidate
    assert field_decisions_of(metadata).superseded == []
    assert not fill_empty(
        metadata, FieldCandidate("title", "self_citation", "Other"), "self_citation_fill", "x"
    )
    assert metadata.title == TITLE
    assert not fill_empty(PaperMetadata(doi="", title=""), empty, "self_citation_fill", "x")


# ── Where it runs ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("on", "no_llm", "strategy", "preparsed", "runs"),
    [
        (True, False, "ner", False, True),
        (False, False, "ner", False, False),
        (True, True, "ner", False, False),
        (True, False, "llm", False, False),
        (True, False, "off", False, False),
        (True, False, "ner", True, False),
    ],
)
async def test_post_parse_reads_the_line_only_with_the_local_parser(
    monkeypatch, on, no_llm, strategy, preparsed, runs
):
    from bibr.config import GlobalSettings
    from bibr.extract import self_citation
    from bibr.pipeline.stages.post_parse import _read_printed_citation

    calls = []
    monkeypatch.setattr(
        self_citation, "read_printed_citation", lambda contents, **kw: calls.append(kw)
    )
    contents = _contents(("row", 1, 1))
    if preparsed:
        contents.preparsed_metadata = PaperMetadata(doi="", title="")
    await _read_printed_citation(
        contents,
        "hash",
        SkippedLlmClient(),
        no_llm=no_llm,
        parse_strategy=strategy,
        settings=GlobalSettings(SELF_CITATION_METADATA=on),
        memory_mode=None,
    )
    assert bool(calls) is runs


def test_the_reader_is_on_by_default():
    from bibr.config import GlobalSettings

    assert GlobalSettings().SELF_CITATION_METADATA is True


async def test_a_reader_failure_never_fails_post_parse(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.extract import self_citation
    from bibr.pipeline.stages.post_parse import _read_printed_citation

    def boom(contents, **kw):
        raise RuntimeError("broken")

    monkeypatch.setattr(self_citation, "read_printed_citation", boom)
    await _read_printed_citation(
        _contents(("row", 1, 1)),
        "hash",
        SkippedLlmClient(),
        no_llm=False,
        parse_strategy="ner",
        settings=GlobalSettings(),
        memory_mode=None,
    )
