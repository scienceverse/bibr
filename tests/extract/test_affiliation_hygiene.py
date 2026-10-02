"""Affiliation values after the reconcilers: cleaning, grounded re-join, folded dedup.

The author LLM returns one "; "-joined affiliation string per author, and the
numbered reconciler overwrites it with the printed definition. Both copy what
the page prints around the institution, and "; " is also a printed character.
The cases below are modelled on real tester and development papers (synthetic
institutions where the real text is not needed).
"""

import pandas as pd
import pytest

from bibr.extract.core_metadata import CoreMetadataExtractor, _clean_affiliation_value
from bibr.extract.research_integrity import affiliation_key, collect_affiliations
from bibr.paper import PaperAuthor


def _author(author_id: int, given: str, family: str, affiliation: str = "") -> PaperAuthor:
    return PaperAuthor(author_id=author_id, given=given, family=family, affiliation=affiliation)


def _frame(*rows: tuple[int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_id": list(range(len(rows))),
            "page_number": [page for page, _ in rows],
            "text": [text for _, text in rows],
        }
    )


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        # JNS/AMA house style: the list's connective before the next marker.
        (
            "Department of Neurological Surgery, Houston\r\nMethodist Hospital, Houston, "
            "Texas; and",
            "Department of Neurological Surgery, Houston Methodist Hospital, Houston, Texas",
        ),
        # A footnote definition ending in the corresponding author's address.
        (
            "Usher Institute, University of\r\nEdinburgh, Edinburgh, UK. E-mail: "
            "someone@example.ac.uk",
            "Usher Institute, University of Edinburgh, Edinburgh, UK",
        ),
        (
            "Department of Psychology, Example University, 00185 Rome, Italy; name@example.com",
            "Department of Psychology, Example University, 00185 Rome, Italy",
        ),
        (
            "Max Planck Institute for Example Research, Tel.: +49 89 1234",
            "Max Planck Institute for Example Research",
        ),
        (
            "Icahn School of Medicine \r\nat Mount Sinai, New York, NY, USA.",
            "Icahn School of Medicine at Mount Sinai, New York, NY, USA",
        ),
        # BMC / Springer pointers to the end-of-article list.
        (
            "Example University of Norway, Tromsø, Norway\r\nFull list of author information "
            "is available at the end of the article",
            "Example University of Norway, Tromsø, Norway",
        ),
        (
            "Department of Health, Example University; Extended author information available "
            "on the last page of the article",
            "Department of Health, Example University",
        ),
        (
            "School of Health Sciences, Example University, UK |",
            "School of Health Sciences, Example University, UK",
        ),
        ("Example Systems †", "Example Systems"),
        # Markers the LLM copied.
        (
            "1 Department of Radiology, Example Hospital",
            "Department of Radiology, Example Hospital",
        ),
        (
            "2. NIHR Example Research Centre, Bristol, UK.",
            "NIHR Example Research Centre, Bristol, UK",
        ),
        ("1Department of Physics, Example University", "Department of Physics, Example University"),
        ("a School of Chemistry, Example University", "School of Chemistry, Example University"),
        ("and", ""),
        # Must survive as printed.
        (
            "Tel Aviv-Yaffo Academic College, Yaffo, Israel",
            "Tel Aviv-Yaffo Academic College, Yaffo, Israel",
        ),
        ("Department of X & Y, University Z", "Department of X & Y, University Z"),
        ("3M Company, St Paul, MN, USA", "3M Company, St Paul, MN, USA"),
        ("Example University, Boston, MA, U.S.A.", "Example University, Boston, MA, U.S.A."),
    ],
)
def test_affiliation_values_lose_what_is_printed_around_the_institution(raw, clean):
    assert _clean_affiliation_value(raw) == clean


def test_a_printed_inner_semicolon_is_not_a_second_affiliation():
    """One printed affiliation "..., Cambridge, MA, USA; Basel, Switzerland": the
    "; " split made "Basel, Switzerland" a row of its own for 11 authors."""
    printed = (
        "1 Disease Area Research, Example Biomedical Research, Cambridge, MA, USA; "
        "Basel, Switzerland"
    )
    frame = _frame((1, "Ann Lee1, Bo Chen1"), (1, printed))
    copied = (
        "Disease Area Research, Example Biomedical Research, Cambridge, MA, USA; Basel, Switzerland"
    )
    authors = [_author(1, "Ann", "Lee", copied), _author(2, "Bo", "Chen", copied)]

    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    texts, author_ids = collect_affiliations(authors)
    assert texts == [
        "Disease Area Research, Example Biomedical Research, Cambridge, MA, USA, Basel, Switzerland"
    ]
    assert author_ids == [[1, 2]]


def test_separately_printed_affiliations_stay_separate():
    """ "Example Systems" is not an institution word, but it was printed on its
    own, so it is the author's second affiliation and not a fragment."""
    frame = _frame(
        (1, "Ann Lee1,2"),
        (1, "1 Institute of Example Models, Example City, USA"),
        (1, "2 Example Systems"),
    )
    authors = [
        _author(1, "Ann", "Lee", "Institute of Example Models, Example City, USA; Example Systems")
    ]

    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[0].affiliation == (
        "Institute of Example Models, Example City, USA; Example Systems"
    )


def test_connective_and_contact_tails_never_become_affiliations():
    """World Neurosurgery and EMBO Reports styles, through the numbered reconciler:
    the export had an affiliation "and", and an e-mail in another author's row."""
    frame = _frame(
        (1, "Ann Lee1 and Bo Chen1,2"),
        (
            1,
            "From the 1 Department of Neurological Surgery, Houston\r\nMethodist Hospital, "
            "Houston, Texas; and 2 Department of Neurological Surgery, Example College of "
            "Medicine, Houston, Texas.",
        ),
    )
    authors = [_author(1, "Ann", "Lee", "x"), _author(2, "Bo", "Chen", "y")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    texts, author_ids = collect_affiliations(authors)
    assert texts == [
        "Department of Neurological Surgery, Houston Methodist Hospital, Houston, Texas",
        "Department of Neurological Surgery, Example College of Medicine, Houston, Texas",
    ]
    assert author_ids == [[1, 2], [2]]

    frame = _frame(
        (1, "Ann Lee1 and Bo Chen2"),
        (
            1,
            "1 Department of Example Studies, Example University, Edinburgh, UK. "
            "2 Usher Institute, University of\r\nEdinburgh, Edinburgh, UK. E-mail: "
            "someone@example.ac.uk",
        ),
    )
    authors = [_author(1, "Ann", "Lee"), _author(2, "Bo", "Chen")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert (
        authors[0].affiliation == "Department of Example Studies, Example University, Edinburgh, UK"
    )
    assert authors[1].affiliation == "Usher Institute, University of Edinburgh, Edinburgh, UK"


def test_llm_markers_emails_and_marker_only_parts_are_dropped():
    frame = _frame((1, "Ann Lee1,2"))
    authors = [
        _author(
            1,
            "Ann",
            "Lee",
            "1 Department of Psychology, Example University, Rome, Italy; name@example.com; "
            "2; 3; 2 Department of Psychology, Example University, Rome, Italy.",
        )
    ]

    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[0].affiliation == "Department of Psychology, Example University, Rome, Italy"


def test_variants_of_one_institution_are_one_row():
    """Exact-string dedup listed the same institution twice: once as the
    reconciler copied it (line breaks, "LondonUK"), once as the LLM wrote it."""
    authors = [
        _author(1, "A", "B", "Department of Epidemiology, University College London,\r\nLondonUK"),
        _author(2, "C", "D", "Department of Epidemiology, University College London, London, UK"),
        _author(3, "E", "F", "Department of Public Health & Primary Care, Universität Bern"),
        _author(4, "G", "H", "Department of Public Health and Primary Care, Universitat Bern"),
    ]

    texts, author_ids = collect_affiliations(authors)

    assert texts == [
        "Department of Epidemiology, University College London, London, UK",
        "Department of Public Health & Primary Care, Universität Bern",
    ]
    assert author_ids == [[1, 2], [3, 4]]


def test_the_folded_key_keeps_different_institutions_and_other_scripts_apart():
    assert affiliation_key("Department of Nursing, University A") != affiliation_key(
        "Department of Nursing, University B"
    )
    assert affiliation_key("東京大学医学部") == "東京大学医学部"
