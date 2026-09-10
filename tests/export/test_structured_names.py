"""``bib[].author``/``editor`` (export schema 10.9): structured names derived
from the verbatim ``authors``/``editors`` strings, which stay canonical and
unchanged. Naming rule (CSL-borrowed): plural = verbatim printed string,
singular = derived structured array."""

import pytest
from pydantic import ValidationError

from bibr.export.json_export import (
    export_paper_to_json,
    validate_export,
)
from bibr.export.models import AuthorExport, PersonNameExport
from bibr.models import PaperMetadata, PaperReference
from tests.test_export_units import _minimal_paper


def _reference(**overrides) -> PaperReference:
    defaults = {
        "bib_id": 1,
        "title": "A title",
        "first_page": None,
        "volume": None,
        "authors": "Eagly, A. H., & Wood, W.",
        "year": 2020,
        "container": "J. Test",
        "editors": "Brown, A., & Green, B. (Eds.)",
    }
    defaults.update(overrides)
    return PaperReference(**defaults)


def _bib_row(**overrides) -> dict:
    metadata = PaperMetadata(doi="10.1234/test", title="T", references=[_reference(**overrides)])
    return export_paper_to_json(_minimal_paper(metadata=metadata))["bib"][0]


def test_bib_keeps_verbatim_strings_and_gains_structured_names():
    row = _bib_row()

    assert row["authors"] == "Eagly, A. H., & Wood, W."
    assert row["author"] == [
        {"family": "Eagly", "given": "A. H."},
        {"family": "Wood", "given": "W."},
    ]
    assert row["editors"] == "Brown, A., & Green, B. (Eds.)"
    assert row["editor"] == [
        {"family": "Brown", "given": "A."},
        {"family": "Green", "given": "B."},
    ]


@pytest.mark.parametrize(
    ("structured_key", "verbatim_key"), [("author", "authors"), ("editor", "editors")]
)
def test_structured_name_parts_are_substrings_of_the_verbatim(structured_key, verbatim_key):
    """The ground-truth invariant: the splitter never invents characters, so a
    consumer can always fall back to the verbatim string. Editor strings carry
    conventions author strings don't (a trailing "(Ed.)"), so both pairs are
    exercised."""
    row = _bib_row(
        authors="King, M. L., Jr.; World Health Organization",
        editors="Van Houten, K. (Ed.)",
    )

    verbatim = row[verbatim_key]
    assert row[structured_key]
    for person in row[structured_key]:
        for value in person.values():
            assert value in verbatim, (value, verbatim)


def test_corporate_and_suffixed_names_split_as_documented():
    row = _bib_row(authors="King, M. L., Jr.; World Health Organization")

    assert row["author"] == [
        {"family": "King", "given": "M. L.", "suffix": "Jr."},
        {"literal": "World Health Organization"},
    ]


def test_nothing_to_split_encodes_as_null_never_an_empty_list():
    """``bib[].author``/``editor`` are nullable like their verbatim siblings.
    ``null`` and ``[]`` encode to different column types in R, so the empty
    split must not leak out as a list."""
    row = _bib_row(authors=None, editors="   ")

    assert row["authors"] is None
    assert row["author"] is None
    assert row["editor"] is None


def test_paper_authors_accept_a_suffix_and_default_it_to_null():
    with_suffix = AuthorExport(
        author_id=1, given="M. L.", family="King", corresponding=False, suffix="Jr."
    )
    without = AuthorExport(author_id=2, given="A.", family="Author", corresponding=False)

    assert with_suffix.model_dump()["suffix"] == "Jr."
    assert without.model_dump()["suffix"] is None


def test_person_name_export_rejects_an_empty_identity():
    """A record identifying nobody (no family/given/literal) is a vacuous row,
    not a best-effort split, and must not validate clean."""
    with pytest.raises(ValidationError):
        PersonNameExport()

    with pytest.raises(ValidationError):
        # A bare suffix with no name to attach it to is still empty-identity.
        PersonNameExport(suffix="Jr.")


def test_person_name_export_omits_absent_parts():
    assert PersonNameExport(literal="ACME Inc").model_dump() == {"literal": "ACME Inc"}
    assert PersonNameExport(family="Wood", given="W.").model_dump() == {
        "family": "Wood",
        "given": "W.",
    }


def test_export_with_structured_names_validates_as_schema_10_9():
    metadata = PaperMetadata(doi="10.1234/test", title="T", references=[_reference()])
    payload = export_paper_to_json(_minimal_paper(metadata=metadata))

    assert payload["schema_version"] == "11.0"
    assert validate_export(payload) == []
