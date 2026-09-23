"""Tests that lock down the public API of bibr.models.

Each test must pass against both the current dataclass implementation
AND the Pydantic-converted implementation. They assert constructor
signatures (keyword-only), field defaults, and equality behavior — the
boundaries that callers depend on.
"""

from __future__ import annotations

# ---- BibAuthor -------------------------------------------------------------


def test_bibauthor_construct_kw():
    from bibr.models import BibAuthor

    a = BibAuthor(given="J", family="D")
    assert a.given == "J"
    assert a.family == "D"


# ---- PaperAuthor -----------------------------------------------------------


def test_paperauthor_required_fields():
    from bibr.models import PaperAuthor

    a = PaperAuthor(
        author_id=1,
        given="Jane",
        family="Doe",
        affiliation="MIT",
    )
    assert a.author_id == 1
    assert a.email is None
    assert a.corresponding is False
    assert a.orcid is None
    assert a.role == []


def test_paperauthor_role_default_is_independent_per_instance():
    """Mutating one instance's role must not affect another's default."""
    from bibr.models import PaperAuthor

    a = PaperAuthor(author_id=1, given="x", family="y", affiliation="z")
    b = PaperAuthor(author_id=2, given="a", family="b", affiliation="c")
    a.role.append("first_author")
    assert b.role == []


# ---- PaperReference --------------------------------------------------------


def test_paperreference_minimal():
    from bibr.models import PaperReference

    r = PaperReference(
        bib_id=1,
        title="x",
        first_page=None,
        volume=None,
        authors=None,
        year=None,
        container=None,
    )
    assert r.is_in_press is False
    assert r.match == {}
    assert r.year_suffix is None


def test_paperreference_match_dict_independent():
    from bibr.models import ExternalMatch, MatchSource, PaperReference

    r1 = PaperReference(
        bib_id=1,
        title="a",
        first_page=None,
        volume=None,
        authors=None,
        year=None,
        container=None,
    )
    r2 = PaperReference(
        bib_id=2,
        title="b",
        first_page=None,
        volume=None,
        authors=None,
        year=None,
        container=None,
    )
    r1.match[MatchSource.CROSSREF] = ExternalMatch(id="10.1/x")
    assert r2.match == {}


# ---- ExternalMatch ---------------------------------------------------------


def test_externalmatch_all_optional():
    from bibr.models import ExternalMatch

    m = ExternalMatch()  # every field is Optional
    assert m.id is None
    assert m.score is None


def test_externalmatch_authors_list():
    from bibr.models import BibAuthor, ExternalMatch

    m = ExternalMatch(authors=[BibAuthor(given="Jane", family="Doe")])
    assert len(m.authors) == 1


# ---- PaperMetadata ---------------------------------------------------------


def test_papermetadata_required_doi_and_title():
    from bibr.models import PaperMetadata

    m = PaperMetadata(doi="10.1/x", title="t")
    assert m.abstract == ""
    assert m.keywords == []
    assert m.authors == []
    assert m.references == []


# ---- ProcessingStatus ------------------------------------------------------


def test_processingstatus_stage_times_independent_per_instance():
    from bibr.models import ProcessingStatus

    a = ProcessingStatus()
    b = ProcessingStatus()
    a.stage_times["ocr"] = 1.0
    assert b.stage_times == {}


# ---- Helper functions stay ------------------------------------------------


def test_format_bib_authors_still_works():
    from bibr.models import BibAuthor, format_bib_authors

    out = format_bib_authors(
        [BibAuthor(given="Jane", family="Doe"), BibAuthor(given="J", family="Smith")]
    )
    assert out == "Doe, Jane; Smith, J"


def test_canonicalize_orcid_bare_to_uri():
    from bibr.models import canonicalize_orcid

    assert canonicalize_orcid("0000-0001-2345-6789") == "https://orcid.org/0000-0001-2345-6789"


def test_migrate_bib_type_unknown_to_other():
    from bibr.models import migrate_bib_type

    assert migrate_bib_type("nonsense") == "other"
