"""A title-less (RSC-style) reference keeps its journal as the container."""

from __future__ import annotations

from bibr.extract.ref_extractor import _finalize_reference_fields


def _fields(title, **extra):
    return {
        "title": title,
        "authors": "T. Angata and A. Varki",
        "year": 2002,
        "container": None,
    } | extra


def test_the_journal_parsed_as_title_moves_to_the_container():
    segment = "1 T. Angata and A. Varki, Chem. Rev., 2002, 102, 439–470."

    fields = _finalize_reference_fields(_fields("Chem. Rev.,"), segment)

    assert fields["title"] == ""
    assert fields["container"] == "Chem. Rev."
    assert (fields["volume"], fields["first_page"], fields["last_page"]) == ("102", "439", "470")


def test_parsed_numbers_are_kept():
    segment = "A. Said and M. Ramadan, Electr. Power Syst. Res., 2026, 252, 112438."

    fields = _finalize_reference_fields(
        _fields("Electr. Power Syst. Res.,", volume="252", first_page="112438"), segment
    )

    assert fields["container"] == "Electr. Power Syst. Res."
    assert (fields["volume"], fields["first_page"]) == ("252", "112438")


def test_a_real_title_is_left_alone():
    segment = "J. Doe, Sialic acids in health, Chem. Rev., 2002, 102, 439–470."

    fields = _finalize_reference_fields(_fields("Sialic acids in health"), segment)

    assert fields["title"] == "Sialic acids in health"
    assert fields["container"] is None


def test_a_parsed_container_is_left_alone():
    segment = "1 T. Angata and A. Varki, Chem. Rev., 2002, 102, 439–470."

    fields = _finalize_reference_fields(_fields("Chem. Rev.,", container="Chem. Rev."), segment)

    assert fields["title"] == "Chem. Rev.,"
