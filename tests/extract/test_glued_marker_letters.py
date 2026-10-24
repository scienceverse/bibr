"""A lowercase affiliation letter the model left glued to a family name.

Letter superscripts flatten into the names they mark ("Muhammet Ozera,
Amy L. Jonesa"), and the model strips all but one of them. Found on a PMC
PDF, a second PMC PDF and a PsyArXiv .docx, each time on the first author.
"""

from __future__ import annotations

import pandas as pd

from bibr.extract.core_metadata import CoreMetadataExtractor
from bibr.paper import PaperAuthor


def _author(author_id: int, given: str, family: str) -> PaperAuthor:
    return PaperAuthor(author_id=author_id, given=given, family=family, affiliation="")


def _extractor(printed: list[str] | None = None) -> CoreMetadataExtractor:
    extractor = object.__new__(CoreMetadataExtractor)
    extractor.validation_issues = []
    if printed is not None:
        extractor.sentences_df = pd.DataFrame({"text": printed})
    return extractor


def _families(authors: list[PaperAuthor]) -> list[str]:
    return [author.family for author in authors]


BYLINE = (
    "Muhammet Ozera, Suleyman Y. Goksub, Amy L. Jonesa and Nina Sanfordc\n"
    "aDepartment of Hematology and Oncology, Dallas, TX, USA; bDepartment of Family "
    "Medicine, Oak Park, IL, USA; cDepartment of Radiation Oncology, Dallas, TX, USA\n"
    "CONTACT Muhammet Ozer muhammet.ozer@example.org"
)


def test_the_letter_the_model_kept_goes_like_the_ones_it_stripped():
    authors = [
        _author(1, "Muhammet", "Ozera"),
        _author(2, "Suleyman Y.", "Goksu"),
        _author(3, "Amy L.", "Jones"),
        _author(4, "Nina", "Sanford"),
    ]

    kept, transforms = _extractor()._sanitize_authors(authors, BYLINE)

    assert _families(kept) == ["Ozer", "Goksu", "Jones", "Sanford"]
    assert transforms == ["glued_marker_letters_dropped"]


def test_the_bare_family_may_be_printed_outside_the_context():
    context = "Theresa Phama* and Lisa M. D. Archibalda\naCommunication Sciences and Disorders"
    authors = [_author(1, "Theresa", "Phama"), _author(2, "Lisa M. D.", "Archibald")]

    kept, _ = _extractor(["PHAM AND ARCHIBALD", "Results"])._sanitize_authors(authors, context)

    assert _families(kept) == ["Pham", "Archibald"]


def test_a_family_ending_in_a_marker_letter_stays_without_a_bare_printing():
    context = "Ana Silva, Amy L. Jonesa\naDepartment of Medicine, Lisbon, Portugal"
    authors = [_author(1, "Ana", "Silva"), _author(2, "Amy L.", "Jones")]

    kept, transforms = _extractor(["Silva et al."])._sanitize_authors(authors, context)

    assert _families(kept) == ["Silva", "Jones"]
    assert transforms == []


def test_a_family_printed_with_its_own_letter_glued_on_stays():
    context = "Ana Silvab, Amy L. Jonesa\nCorrespondence: Silv, Department of Medicine"
    authors = [_author(1, "Ana", "Silva"), _author(2, "Amy L.", "Jones")]

    kept, _ = _extractor()._sanitize_authors(authors, context)

    assert _families(kept) == ["Silva", "Jones"]


def test_no_other_author_printed_with_that_letter_leaves_the_family():
    context = "Muhammet Ozera, Amy L. Jonesb\nCONTACT Muhammet Ozer"
    authors = [_author(1, "Muhammet", "Ozera"), _author(2, "Amy L.", "Jones")]

    kept, _ = _extractor()._sanitize_authors(authors, context)

    assert _families(kept) == ["Ozera", "Jones"]


def test_a_byline_without_glued_letters_is_untouched():
    context = "Muhammet Ozer, Amy L. Jones\nCONTACT Muhammet Ozer"
    authors = [_author(1, "Muhammet", "Ozer"), _author(2, "Amy L.", "Jones")]

    kept, transforms = _extractor()._sanitize_authors(authors, context)

    assert _families(kept) == ["Ozer", "Jones"]
    assert transforms == []


SEVEN_AFFILIATIONS = "\n" + "; ".join(f"{m}School {m.upper()}, Wuhan, China" for m in "abcdefg")


def test_a_listed_family_is_not_a_stem():
    # "Wang" carries a real "g" marker, but "Cheng" minus its "g" is co-author "Chen".
    context = "Wei Cheng, Fang Chend, Hui Wangg" + SEVEN_AFFILIATIONS
    authors = [_author(1, "Wei", "Cheng"), _author(2, "Fang", "Chen"), _author(3, "Hui", "Wang")]

    kept, transforms = _extractor(["Chen et al. (2020)"])._sanitize_authors(authors, context)

    assert _families(kept) == ["Cheng", "Chen", "Wang"]
    assert transforms == []


def test_a_listed_family_is_not_another_authors_glued_letter():
    # "Yang" is "Yan" with a "g" after it, but it is an author's name, not a marker.
    context = "Min Yang, Jun Yanb, Wei Cheng" + SEVEN_AFFILIATIONS + "\nEditor: Hui Chen"
    authors = [_author(1, "Min", "Yang"), _author(2, "Jun", "Yan"), _author(3, "Wei", "Cheng")]

    kept, transforms = _extractor()._sanitize_authors(authors, context)

    assert _families(kept) == ["Yang", "Yan", "Cheng"]
    assert transforms == []


def test_a_letter_that_opens_no_affiliation_entry_is_kept():
    context = "Muhammet Ozera, Amy L. Jonesa\nUT Southwestern Medical Center\nCONTACT Muhammet Ozer"
    authors = [_author(1, "Muhammet", "Ozera"), _author(2, "Amy L.", "Jones")]

    kept, _ = _extractor()._sanitize_authors(authors, context)

    assert _families(kept) == ["Ozera", "Jones"]


def test_a_marker_left_at_the_end_of_the_line_above_its_entry_counts():
    context = (
        "Rabie Fadila Benjamin L. Walterb\r\n"
        "Dustin A. Heldmana a\r\n"
        "Great Lakes NeuroTechnologies Inc, Cleveland, OH, USA; bCenter for Neuro-Restoration"
    )
    authors = [
        _author(1, "Rabie", "Fadila"),
        _author(2, "Benjamin L.", "Walter"),
        _author(3, "Dustin A.", "Heldman"),
    ]

    kept, _ = _extractor(["Fadil et al.", "Correspondence: Rabie Fadil"])._sanitize_authors(
        authors, context
    )

    assert _families(kept) == ["Fadil", "Walter", "Heldman"]
