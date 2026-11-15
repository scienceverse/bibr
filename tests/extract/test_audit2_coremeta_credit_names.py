"""Names read off a CRediT contribution statement keep a suffix in the family name.

The statement is the last author source, used when no byline reached the
extraction; "John Smith Jr.: Conceptualization" used to come back as a person
whose family name is "Jr.".
"""

import pytest

from bibr.extract.core_metadata import credit_statement_authors
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence


def _statement(*lines: str) -> PaperContents:
    return PaperContents(
        sentences=[
            PaperSentence(index, line, 1, index, page_number=9)
            for index, line in enumerate(lines, start=1)
        ],
        sections=[
            PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN),
            PaperSection(1, "Author contributions", 1, 0, CanonicalSection.AUTHOR_CONTRIBUTIONS),
        ],
        tables=[],
        links=[],
        sections_text={},
    )


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("John Smith Jr.: Conceptualization, Writing.", ("John", "Smith Jr.")),
        ("Martin Luther King III: Supervision.", ("Martin Luther", "King III")),
        ("Robert Downey Sr: Formal analysis.", ("Robert", "Downey Sr")),
        ("Ana de la Cruz Jr.: Methodology.", ("Ana", "de la Cruz Jr.")),
        # A two-token name is never read as name plus suffix.
        ("Kim II: Data curation.", ("Kim", "II")),
        ("Ana de la Cruz: Methodology.", ("Ana", "de la Cruz")),
    ],
)
def test_a_suffix_stays_with_the_family_name(line, name):
    assert credit_statement_authors(_statement(line)) == [name]
