import pytest

from bibr.export.name_split import split_person_names


def test_empty_input_returns_empty_list():
    assert split_person_names(None) == []
    assert split_person_names("   ") == []


def test_semicolon_separated_family_given():
    assert split_person_names("Eagly, A. H.; Wood, W.") == [
        {"family": "Eagly", "given": "A. H."},
        {"family": "Wood", "given": "W."},
    ]


def test_apa_comma_ampersand_style():
    assert split_person_names("Smith, J., & Jones, K. L.") == [
        {"family": "Smith", "given": "J."},
        {"family": "Jones", "given": "K. L."},
    ]


def test_suffix_is_split_out():
    assert split_person_names("King, M. L., Jr.") == [
        {"family": "King", "given": "M. L.", "suffix": "Jr."},
    ]


def test_corporate_name_falls_back_to_literal():
    assert split_person_names("World Health Organization") == [
        {"literal": "World Health Organization"},
    ]


def test_unparseable_chunk_falls_back_to_literal_per_chunk():
    out = split_person_names("Smith, J.; ???")
    assert out[0] == {"family": "Smith", "given": "J."}
    assert out[1] == {"literal": "???"}


def test_multi_comma_institutional_name_falls_back_to_literal():
    """Corporate detection must run before comma-pairing fragments the string.

    Regression for a reviewer-found bug: an address-bearing institutional
    author ("Institute ..., City, State, Country") has more than two commas,
    which used to get chopped into a fabricated {"family": ..., "given": ...}
    pair instead of being recognized as one unsplittable corporate name.
    """
    assert split_person_names("National Institute of Mental Health, Bethesda, MD, USA") == [
        {"literal": "National Institute of Mental Health, Bethesda, MD, USA"},
    ]


def test_and_joined_institutional_name_falls_back_to_literal():
    """Corporate detection must run before "&"/"and" splitting fragments it.

    Regression for a reviewer-found bug: an institutional name whose defining
    keyword sits on one side of an "and" ("Department of Health and Human
    Services") used to get torn into a literal plus a fabricated person.
    """
    assert split_person_names("Department of Health and Human Services, Washington, DC") == [
        {"literal": "Department of Health and Human Services, Washington, DC"},
    ]


def test_and_joined_institutional_name_with_oxford_comma_falls_back_to_literal():
    assert split_person_names("National Academies of Sciences, Engineering, and Medicine") == [
        {"literal": "National Academies of Sciences, Engineering, and Medicine"},
    ]


def test_semicolon_list_mixing_person_and_org_keeps_person_structured():
    """Semicolons are the one legitimate person/org mixing point.

    A whole-string corporate check (run before splitting on ";") would wrongly
    swallow the person into the same literal as the org — corporate detection
    must run per semicolon-delimited chunk, not on the raw verbatim.
    """
    assert split_person_names("Smith, J.; World Health Organization") == [
        {"family": "Smith", "given": "J."},
        {"literal": "World Health Organization"},
    ]


def test_academ_surname_is_not_misclassified_as_corporate():
    """The "academy"/"academies" keyword must not match a surname stem.

    Regression for a reviewer-found false positive: the prior `academ\\w*`
    pattern matched any word starting with "academ", degrading a legitimate
    surname like "Academo" to a literal instead of a structured person.
    """
    assert split_person_names("Academo, R.; Smith, J.") == [
        {"family": "Academo", "given": "R."},
        {"family": "Smith", "given": "J."},
    ]


def test_single_editor_role_tag_is_stripped_not_leaked_into_given():
    """Regression: "(Ed.)" is a structural role marker, not part of the name.

    Editor strings (unlike author strings) carry this APA convention. Before
    the fix, it leaked verbatim into the given-name field ("R. (Ed.)").
    """
    assert split_person_names("Roe, R. (Ed.)") == [{"family": "Roe", "given": "R."}]


def test_multi_editor_role_tag_is_stripped_from_the_last_name():
    assert split_person_names("Smith, J., & Doe, A. (Eds.)") == [
        {"family": "Smith", "given": "J."},
        {"family": "Doe", "given": "A."},
    ]


def test_per_name_editor_role_tags_are_all_stripped():
    assert split_person_names("Roe, R. (Ed.), & Doe, A. (Ed.)") == [
        {"family": "Roe", "given": "R."},
        {"family": "Doe", "given": "A."},
    ]


def test_editor_role_tag_coexists_with_suffix():
    assert split_person_names("King, M. L., Jr. (Ed.)") == [
        {"family": "King", "given": "M. L.", "suffix": "Jr."},
    ]


def test_editor_role_tag_before_semicolon_is_stripped():
    """A role tag trailing a semicolon-bounded chunk is trailing too."""
    assert split_person_names("Smith, J. (Ed.); Doe, A.") == [
        {"family": "Smith", "given": "J."},
        {"family": "Doe", "given": "A."},
    ]


def test_editor_role_tag_spelled_out_is_also_stripped():
    """ "(Editor)"/"(Editors)" (unabbreviated) get the same treatment as
    "(Ed.)"/"(Eds.)" — handling one spelling but not the other would be
    arbitrary from the data's perspective."""
    assert split_person_names("Roe, R. (Editor)") == [{"family": "Roe", "given": "R."}]
    assert split_person_names("Smith, J., & Doe, A. (Editors)") == [
        {"family": "Smith", "given": "J."},
        {"family": "Doe", "given": "A."},
    ]


def test_mid_string_role_tag_is_not_stripped_and_stays_substring_safe():
    """A "(Eds)" that is NOT in trailing position (sitting inside what would
    otherwise be a single name) must be left alone rather than stripped.

    Regression for a reviewer-found bug: unconditionally deleting the tag
    wherever it appeared could glue two non-adjacent fragments together
    ("Van" + "Houten" -> "Van Houten", which never appears contiguously in
    the source) — a genuine violation of the "never invents characters"
    invariant. The splitter is not expected to produce a clean split here;
    it only has to never invent text, so the odd tag stays inline as part of
    the (structured-or-literal) name it landed in.
    """
    out = split_person_names("Van (Eds) Houten, K.")
    for person in out:
        for value in person.values():
            assert value in "Van (Eds) Houten, K."


@pytest.mark.parametrize(
    "verbatim",
    [
        "Eagly, A. H.; Wood, W.",
        "Smith, J., & Jones, K. L.",
        "King, M. L., Jr.",
        "World Health Organization",
        "van der Berg, P.",
        "Roe, R. (Ed.)",
        "Smith, J., & Doe, A. (Eds.)",
        "Roe, R. (Ed.), & Doe, A. (Ed.)",
        "King, M. L., Jr. (Ed.)",
        "Smith, J. (Ed.); Doe, A.",
        "Roe, R. (Editor)",
        "Smith, J., & Doe, A. (Editors)",
        "Van (Eds) Houten, K.",
    ],
)
def test_every_emitted_part_substring_matches_the_verbatim(verbatim):
    """Ground-truth invariant: the splitter never invents characters."""
    for person in split_person_names(verbatim):
        for value in person.values():
            assert value in verbatim, (value, verbatim)
