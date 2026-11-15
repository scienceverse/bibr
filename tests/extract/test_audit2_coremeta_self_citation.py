"""The printed citation is read off the first page the parse saw.

Page numbers stay absolute under page slicing (``--pages 5-12``, serve
``start_page``), so an article that starts on page 5 of a volume PDF prints its
"How to cite" line on page 5, and no row carries page 1.
"""

from __future__ import annotations

import pytest

from bibr.extract.self_citation import (
    PrintedCitation,
    apply_printed_citation,
    find_printed_citation,
)
from bibr.input.file import InputFile
from bibr.models import PaperMetadata, PaperReference
from bibr.paper import Paper
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence

_CITATION = (
    "Moreau, L., & Okafor, N. (2017). Reading glacier retreat from archived survey "
    "photographs. Polar Records, 9(2), 1-20. https://doi.org/10.5555/polar.2017.1"
)


def _contents(first_page: int) -> PaperContents:
    rows = [
        ("Reading glacier retreat from archived survey photographs", first_page, 1),
        (f"To cite this article: {_CITATION}", first_page, 2),
        ("Introduction text on the next page.", first_page + 1, 3),
    ]
    return PaperContents(
        sentences=[
            PaperSentence(index, text, 1, paragraph, page_number=page)
            for index, (text, page, paragraph) in enumerate(rows, start=1)
        ],
        sections=[
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "Front", 1, 0, CanonicalSection.TITLE),
        ],
        tables=[],
        links=[],
        sections_text={},
    )


def test_the_citation_line_is_found_on_page_one_of_an_unsliced_parse():
    line = find_printed_citation(_contents(1))
    assert line is not None and line.text == _CITATION


def test_the_citation_line_is_found_on_the_first_page_of_a_sliced_parse():
    line = find_printed_citation(_contents(5))
    assert line is not None
    assert line.text_ids == (2,)
    assert line.text == _CITATION


def test_an_input_without_pages_is_not_searched():
    contents = _contents(1)
    for sentence in contents.sentences:
        sentence.page_number = None
    assert find_printed_citation(contents) is None


def test_a_later_page_is_still_not_read():
    contents = _contents(5)
    contents.sentences[1].page_number = 6
    assert find_printed_citation(contents) is None


# ── Names read off the line ─────────────────────────────────────────────


def _filled_names(authors: str) -> list[tuple[str, str]]:
    reference = PaperReference(
        bib_id=1,
        title="Reading glacier retreat from archived survey photographs",
        authors=authors,
        container="Polar Records",
        year=2017,
        volume="9",
        issue="2",
        first_page="1",
        last_page="20",
        doi="10.5555/polar.2017.1",
    )
    contents = _contents(1)
    contents.printed_citation = PrintedCitation((2,), "To cite this article:", True, "x", reference)
    input_file = InputFile(path="paper.pdf")
    input_file.file_hash = "a" * 16
    metadata = PaperMetadata(doi="10.5555/polar.2017.1", title="")
    apply_printed_citation(Paper(input_file=input_file, contents=contents, metadata=metadata))
    return [(author.given, author.family) for author in metadata.authors]


@pytest.mark.parametrize(
    ("authors", "names"),
    [
        # A suffix is never the family name, nor the given name.
        ("John Smith Jr. and Jane Doe", [("John", "Smith Jr."), ("Jane", "Doe")]),
        (
            "Martin Luther King III, Coretta Scott King",
            [("Martin Luther", "King III"), ("Coretta Scott", "King")],
        ),
        ("King ML Jr, Doe A", [("ML", "King Jr"), ("A", "Doe")]),
        ("Smith, J., Jr., & Doe, A.", [("J.", "Smith Jr."), ("A.", "Doe")]),
        # What was read before is read the same.
        ("Moreau, L., & Okafor, N.", [("L.", "Moreau"), ("N.", "Okafor")]),
        (
            "Moreau L, Okafor NA and De la Cruz P",
            [("L", "Moreau"), ("NA", "Okafor"), ("P", "De la Cruz")],
        ),
        ("Lena Moreau, Nkem Okafor", [("Lena", "Moreau"), ("Nkem", "Okafor")]),
    ],
)
def test_printed_names_fill_the_authors_with_their_suffix_in_the_family_name(authors, names):
    assert _filled_names(authors) == names


@pytest.mark.parametrize(
    "authors",
    ["Moreau L, Okafor N, et al.", "Lena Moreau,a Nkem Okafor *b", "Open Science Collaboration"],
)
def test_a_list_read_only_in_part_fills_no_authors(authors):
    assert _filled_names(authors) == []
