"""The per-reference check (``bib_check``): recorded Crossref and doi.org
responses end to end, then the comparators one rule at a time."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bibr.config import GlobalSettings
from bibr.enrich import check as check_mod
from bibr.enrich import lookup as lookup_mod
from bibr.enrich.check import check_bib, check_bibs
from bibr.enrich.lookup import ReferenceLookup
from bibr.paper import PaperReference
from tests.crossref_recorded import (
    CARNEY_DOI,
    COHEN_DOI,
    DATACITE_DOI,
    OSC_DOI,
    UNREGISTERED_DOI,
    WAKEFIELD_DOI,
    RecordedCrossref,
    RecordedDoiOrg,
)

CARNEY_AUTHORS = "Carney, D. R., Cuddy, A. J. C., & Yap, A. J."
CARNEY_TITLE = (
    "Power posing: Brief nonverbal displays affect neuroendocrine levels and risk tolerance"
)
WAKEFIELD_TITLE = (
    "Ileal-lymphoid-nodular hyperplasia, non-specific colitis, and pervasive developmental "
    "disorder in children"
)
WAKEFIELD_AUTHORS = (
    "Wakefield, A. J., Murch, S. H., Anthony, A., Linnell, J., Casson, D. M., Malik, M., "
    "Berelowitz, M., Dhillon, A. P., Thomson, M. A., Harvey, P., Valentine, A., "
    "Davies, S. E., & Walker-Smith, J. A."
)


def _ref(bib_id, title, **fields):
    for name in ("first_page", "volume", "container", "authors", "year", "doi"):
        fields.setdefault(name, None)
    return PaperReference(bib_id=bib_id, title=title, **fields)


@pytest.fixture
def doi_org(monkeypatch):
    fake = RecordedDoiOrg()
    real = lookup_mod.registration_agencies

    async def recorded(dois, *, user_agent=None, http=None):  # noqa: ARG001
        return await real(dois, user_agent=user_agent, http=fake)

    monkeypatch.setattr(lookup_mod, "registration_agencies", recorded)
    return fake


async def _checked(refs, client=None) -> dict[int, dict]:
    """Enrich *refs* from the recordings, export them, verify, check."""
    from bibr.enrich.references import enrich_references
    from bibr.enrich.verify import verify_bibs
    from bibr.export.json_export import _export_paper_payload
    from bibr.export.models import PaperExport
    from tests.export.conftest import _demo_paper

    await enrich_references(
        refs,
        crossref_client=client or RecordedCrossref(),
        settings=GlobalSettings(crossref={"doi_agency_lookup": True}),
    )
    paper = _demo_paper(with_refs=True)
    paper.metadata.references = refs
    paper.metadata.enrichment_complete = True
    payload = _export_paper_payload(paper)
    verify_bibs(payload)
    check_bibs(payload)
    PaperExport.model_validate(payload)
    return {row["bib_id"]: row for row in payload["bib_check"]}


async def test_recorded_cases(doi_org):
    refs = [
        # 1. Carney, Cuddy & Yap (2010), as printed in APA style.
        _ref(
            1,
            CARNEY_TITLE,
            authors=CARNEY_AUTHORS,
            year=2010,
            container="Psychological Science",
            volume="21",
            issue="10",
            first_page="1363",
            last_page="1368",
            doi=CARNEY_DOI,
        ),
        # 2. Wakefield et al. (1998), retracted in 2010.
        _ref(
            2,
            WAKEFIELD_TITLE,
            authors=WAKEFIELD_AUTHORS,
            year=1998,
            container="The Lancet",
            volume="351",
            issue="9103",
            first_page="637",
            last_page="641",
            doi=WAKEFIELD_DOI,
        ),
        # 3. Open Science Collaboration (2015): an organization author.
        _ref(
            3,
            "Estimating the reproducibility of psychological science",
            authors="Open Science Collaboration",
            year=2015,
            container="Science",
            volume="349",
            issue="6251",
            first_page="aac4716",
            doi=OSC_DOI,
        ),
        # 4. Cohen's book, printed 1988, registered as its 2013 edition.
        _ref(
            4,
            "Statistical power analysis for the behavioral sciences",
            authors="Cohen, J.",
            year=1988,
            publisher="Lawrence Erlbaum Associates",
            doi=COHEN_DOI,
        ),
        # 5. Carney et al. printed with the OSC DOI: a 2015 record for a 2010 work.
        _ref(5, CARNEY_TITLE, authors=CARNEY_AUTHORS, year=2010, doi=OSC_DOI),
        # 6. A DOI doi.org says does not exist.
        _ref(6, CARNEY_TITLE, authors=CARNEY_AUTHORS, year=2010, doi=UNREGISTERED_DOI),
        # 7. A DataCite DOI: Crossref has no record, doi.org knows it.
        _ref(7, "A dataset", authors="Doe, J.", year=2019, doi=DATACITE_DOI),
        # 8. Every search answered, none found it.
        _ref(8, "A title nobody ever wrote about sourdough", authors="Nobody, A.", year=2021),
        # 9. Nothing to look up.
        _ref(9, "Short"),
    ]
    rows = await _checked(refs)

    assert {bib_id: row["status"] for bib_id, row in rows.items()} == {
        1: "matched",
        2: "matched",
        3: "matched",
        4: "matched",
        5: "fields_disagree",
        6: "doi_unregistered",
        7: "unchecked",
        8: "not_found",
        9: "unchecked",
    }
    # The record title has no subtitle; the printed one does. The DOI
    # identifies the work, so "Power Posing" printed in full agrees.
    assert rows[1]["notes"] == [] and rows[1]["retraction"] is None
    assert rows[1]["service"] == "crossref"

    assert rows[2]["notes"] == []
    assert rows[2]["retraction"] == {
        "type": "retraction",
        "doi": "10.1016/s0140-6736(10)60175-4",
        "date": "2010-02-06",
        "source": "retraction-watch",
        "label": "Retraction",
    }
    assert rows[3]["notes"] == []
    # A book's DOI carries its edition's year: a minor note, not a disagreement.
    assert rows[4]["notes"] == [
        {"field": "year", "printed": "1988", "record": "2013", "severity": "minor"}
    ]
    assert rows[5]["notes"] == [
        {"field": "doi", "printed": OSC_DOI, "record": None, "severity": "major"}
    ]
    assert rows[6]["notes"] == [
        {"field": "doi", "printed": UNREGISTERED_DOI, "record": None, "severity": "major"}
    ]
    for bib_id in (7, 8, 9):
        assert rows[bib_id]["notes"] == [] and rows[bib_id]["service"] is None


async def test_a_printed_doi_of_another_work_disagrees(doi_org):
    """The OSC DOI printed for Carney et al. with a year the DOI's record
    accepts: the record is another work, so title and first author differ."""
    ref = _ref(1, CARNEY_TITLE, authors=CARNEY_AUTHORS, year=2015, doi=OSC_DOI)
    (row,) = (await _checked([ref])).values()

    assert row["status"] == "fields_disagree"
    assert [(n["field"], n["severity"]) for n in row["notes"]] == [
        ("title", "major"),
        ("author", "major"),
    ]
    assert row["notes"][1]["record"] == "Open Science Collaboration"


async def test_a_search_hit_is_checked_too(doi_org):
    ref = _ref(
        1,
        "Power posing",
        authors=CARNEY_AUTHORS,
        year=2010,
        container="Psychol. Sci.",
        volume="21",
        first_page="1363",
    )
    (row,) = (await _checked([ref])).values()
    assert (row["status"], row["notes"]) == ("matched", [])


async def test_timed_out_and_failed_lookups_are_unchecked(doi_org):
    timed_out = _ref(1, "A title long enough to search for", year=2020)
    failed = _ref(2, "Another title long enough to search for", year=2020)

    class Broken(RecordedCrossref):
        async def search(self, query, limit=3, **_kw):
            if "Another" in query:
                raise RuntimeError("connection reset")
            return await super().search(query, limit)

    from bibr.enrich.references import enrich_references
    from bibr.export.json_export import _export_paper_payload
    from tests.export.conftest import _demo_paper

    await enrich_references([failed], crossref_client=Broken(), settings=GlobalSettings())
    timed_out._lookup = ReferenceLookup(queried=True, timed_out=True, finished=True)
    paper = _demo_paper(with_refs=True)
    paper.metadata.references = [timed_out, failed]
    payload = _export_paper_payload(paper)
    assert [r["outcome"] for r in payload["extraction"]["enrichment"]["lookups"]] == [
        "timed_out",
        "error",
    ]
    check_bibs(payload)
    assert [r["status"] for r in payload["bib_check"]] == ["unchecked", "unchecked"]


def test_unregistered_doi_without_an_agency_answer_is_not_found():
    """doi.org did not answer (or was not asked): a Crossref 404 is all there is."""
    row = check_bib({"bib_id": 1, "doi": "10.1000/x"}, [], {"bib_id": 1, "outcome": "doi_404"})
    assert (row["status"], row["notes"]) == ("not_found", [])
    crossref = {"bib_id": 1, "outcome": "doi_404", "registration_agency": "crossref"}
    assert check_bib({"bib_id": 1}, [], crossref)["status"] == "not_found"


def test_a_reference_without_a_lookup_row_is_unchecked():
    assert check_bib({"bib_id": 1, "title": "x"}, [], None)["status"] == "unchecked"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _match(**fields) -> dict:
    return {"bib_id": 1, "service": "crossref", "score": 1.0, **fields}


def _notes(bib: dict, match: dict, reference: str = "", verification=None) -> list[tuple]:
    row = check_bib({"bib_id": 1, **bib}, [match], None, reference, verification)
    return [(n["field"], n["severity"]) for n in row["notes"]]


def test_a_short_record_title_printed_in_the_reference_needs_another_identifier():
    """ "Power Posing" (11 characters) inside a longer printed title is not enough
    on its own; with the volume and first page agreeing it is."""
    bib = {"title": "Power posing and the hormones of leaders"}
    record = _match(title="Power Posing")
    assert _notes(bib, record) == [("title", "major")]
    assert (
        _notes(
            {**bib, "volume": "21", "first_page": "1363"},
            {**record, "volume": "21", "first_page": "1363"},
        )
        == []
    )


def test_a_long_record_title_printed_in_the_reference_agrees():
    """The parser ran the title into the venue: the record title is printed."""
    record = _match(title="Estimating the reproducibility of psychological science")
    bib = {"title": "Estimating the reproducibility of psychological science. Science"}
    assert _notes(bib, record) == []


def test_title_severity_follows_the_similarity():
    record = _match(title="The effect of sleep on memory consolidation in adults")
    assert _notes({"title": "The effect of sleep on memory consolidation in adult"}, record) == []
    assert _notes({"title": "The effects of sleep on memory consolidation"}, record) == [
        ("title", "minor")
    ]
    assert _notes({"title": "A theory of reading"}, record) == [("title", "major")]


def test_a_retracted_record_title_is_compared_without_its_prefix():
    record = _match(title="RETRACTED: A study of things that were never measured")
    assert _notes({"title": "A study of things that were never measured"}, record) == []


def test_first_author_missing_is_major_and_co_authors_minor():
    record = _match(author=[{"family": "Smith"}, {"family": "Jones"}])
    assert _notes({"authors": "Brown, A., & Jones, B."}, record) == [("author", "major")]
    assert _notes({"authors": "Smith, A."}, record) == [("author", "minor")]
    assert _notes({"authors": "Smith, A., & Jones, B."}, record) == []


def test_et_al_excuses_missing_co_authors_only_of_a_longer_list():
    two = _match(author=[{"family": "Smith"}, {"family": "Jones"}])
    three = _match(author=[{"family": "Smith"}, {"family": "Jones"}, {"family": "Lee"}])
    assert _notes({"authors": "Smith et al."}, two) == [("author", "minor")]
    assert _notes({"authors": "Smith et al."}, three) == []


def test_a_name_without_latin_letters_is_not_judged():
    record = _match(author=[{"family": "王"}, {"family": "李"}])
    assert _notes({"authors": "Wang, X., & Li, Y."}, record) == []


def test_organization_authors_are_compared_by_their_literal():
    record = _match(author=[{"literal": "Open Science Collaboration"}])
    assert _notes({"authors": "Open Science Collaboration"}, record) == []
    assert _notes({"authors": "Nosek, B. A."}, record) == [("author", "major")]


def test_the_first_author_is_found_in_the_reference_string_when_unparsed():
    record = _match(author=[{"family": "Smith"}])
    assert _notes({}, record, reference="Smith, J. (2020). A title. Journal.") == []
    assert _notes({}, record, reference="Brown, J. (2020). A title. Journal.") == [
        ("author", "major")
    ]


@pytest.mark.parametrize(
    ("printed", "record", "expected"),
    [
        (2010, 2010, []),
        (2011, 2010, []),
        (2009, 2010, []),
        # Print issue two years after the online-first record.
        (2012, 2010, []),
        (2013, 2010, [("year", "major")]),
        (2008, 2010, [("year", "major")]),
    ],
)
def test_year_tolerance(printed, record, expected):
    assert _notes({"year": printed}, _match(year=record)) == expected


def test_a_book_year_difference_is_minor():
    assert _notes({"year": 1988}, _match(year=2013, bib_type="book")) == [("year", "minor")]
    assert _notes({"year": 1988, "bib_type": "book"}, _match(year=2013)) == [("year", "minor")]


@pytest.mark.parametrize(
    ("printed", "record", "coherent"),
    [
        ("Psychol. Sci.", "Psychological Science", True),
        ("J Pers Soc Psychol", "Journal of Personality and Social Psychology", True),
        ("Proc. Natl. Acad. Sci. U.S.A.", "Proceedings of the National Academy of Sciences", True),
        ("The Lancet", "Lancet", True),
        ("Nature", "Science", False),
        ("Psychological Bulletin", "Psychological Review", False),
    ],
)
def test_journal_abbreviations(printed, record, coherent):
    assert check_mod.journal_coherent(printed, record) is coherent
    expected = [] if coherent else [("container", "minor")]
    assert _notes({"container": printed}, _match(container=record)) == expected


def test_verification_decides_the_locator_notes():
    record = _match(volume="21", issue="10", first_page="1363", last_page="1368")
    bib = {"volume": "12", "issue": "3", "first_page": "1363", "last_page": "1368"}
    verification = {"bib_id": 1, "service": "crossref", "volume": "disagree", "issue": "disagree"}
    # Two minor notes, but an issue note does not count: still matched.
    row = check_bib({"bib_id": 1, **bib}, [record], None, "", verification)
    assert [(n["field"], n["severity"]) for n in row["notes"]] == [
        ("volume", "minor"),
        ("issue", "minor"),
    ]
    assert row["status"] == "matched"
    # Without a verification row, the values are compared directly.
    row = check_bib({"bib_id": 1, **bib, "first_page": "99"}, [record], None)
    assert [n["field"] for n in row["notes"]] == ["volume", "issue", "first_page"]
    assert row["status"] == "fields_disagree"


def test_a_verification_row_of_another_service_is_not_used():
    record = _match(volume="21")
    verification = {"bib_id": 1, "service": "openalex", "volume": "agree"}
    assert _notes({"volume": "12"}, record, verification=verification) == [("volume", "minor")]


def test_a_printed_doi_that_is_not_the_records_is_major():
    row = check_bib({"bib_id": 1, "doi": "10.1000/printed"}, [_match(doi="10.1000/other")], None)
    assert row["status"] == "fields_disagree"
    assert row["notes"] == [
        {
            "field": "doi",
            "printed": "10.1000/printed",
            "record": "10.1000/other",
            "severity": "major",
        }
    ]


def test_crossref_is_checked_before_another_service():
    openalex = {**_match(title="Another work entirely"), "service": "openalex", "score": 1.0}
    crossref = _match(title="A study of sleep and memory", score=0.85)
    row = check_bib(
        {"bib_id": 1, "title": "A study of sleep and memory"}, [openalex, crossref], None
    )
    assert (row["service"], row["status"]) == ("crossref", "matched")


def test_a_withdrawal_is_a_retraction_and_a_correction_is_not():
    corrected = _match(updated_by=[{"type": "correction", "doi": "10.1000/c"}])
    withdrawn = _match(updated_by=[{"type": "withdrawal", "doi": "10.1000/w"}])
    assert check_bib({"bib_id": 1}, [corrected], None)["retraction"] is None
    assert check_bib({"bib_id": 1}, [withdrawn], None)["retraction"]["type"] == "withdrawal"


def test_check_bibs_writes_one_row_per_bib_and_tallies():
    data = {
        "bib": [{"bib_id": 1, "title": "Power posing"}, {"bib_id": 2, "title": "Gone"}],
        "bib_match": [
            _match(
                title="Power Posing",
                updated_by=[{"type": "retraction", "doi": "10.1000/r"}],
            )
        ],
        "extraction": {"enrichment": {"lookups": [{"bib_id": 2, "outcome": "clean_miss"}]}},
    }
    report = check_bibs(data)
    assert [row["status"] for row in data["bib_check"]] == ["matched", "not_found"]
    assert report.statuses == {"matched": 1, "not_found": 1}
    assert report.retracted == 1


def test_the_wording_never_says_fabricated():
    """A reference Crossref does not know is "not found in Crossref"."""
    from bibr.export.schema_artifact import build_export_schema

    assert "fabricat" not in json.dumps(build_export_schema()).lower()
    assert "fabricat" not in Path(check_mod.__file__).read_text().lower()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def payload(demo_paper) -> dict:
    """A complete export of the demo paper (``completed_at`` included)."""
    from bibr.export.json_export import _export_paper_payload

    return _export_paper_payload(demo_paper)


def _ctx(*, enrich: bool, verify: bool = True):
    settings = GlobalSettings(crossref={"verify": verify, "consolidate": "off"})
    config = SimpleNamespace(consolidate=None, enrichment_enabled=lambda _s: enrich)
    return SimpleNamespace(settings=settings, config=config)


async def test_the_export_stage_checks_only_enriched_runs(payload):
    from bibr.pipeline.stages.export import _consolidate_payload

    assert payload["bib_check"] == []
    assert await _consolidate_payload(_ctx(enrich=False, verify=False), payload) is False
    assert payload["bib_check"] == []

    assert await _consolidate_payload(_ctx(enrich=True), payload) is True
    assert [row["bib_id"] for row in payload["bib_check"]] == [b["bib_id"] for b in payload["bib"]]
    from bibr.export.models import PaperExport

    PaperExport.model_validate(payload)


def test_result_exposes_bib_check(payload):
    from bibr.api import Result

    check_bibs(payload)
    result = Result(payload)
    assert [row["status"] for row in result.bib_check] == [
        row["status"] for row in payload["bib_check"]
    ]
    assert "bib_check" in dir(result)


def test_bib_check_and_lookups_are_parquet_tables(payload, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from bibr.export.tables import write_tables

    payload["extraction"]["enrichment"]["lookups"] = [
        {"bib_id": b["bib_id"], "outcome": "doi_hit"} for b in payload["bib"]
    ]
    check_bibs(payload)
    write_tables([payload], tmp_path)

    rows = pq.read_table(tmp_path / "bib_check.parquet").to_pylist()
    assert [row["status"] for row in rows] == [row["status"] for row in payload["bib_check"]]
    assert rows[0]["paper_id"] == payload["paper_id"]
    lookups = pq.read_table(tmp_path / "extraction_enrichment_lookups.parquet").to_pylist()
    assert {row["outcome"] for row in lookups} == {"doi_hit"}


def test_inspect_counts_the_statuses(capsys):
    from bibr.local.inspect import run_inspect

    run_inspect(str(Path(__file__).parents[1] / "fixtures" / "inspect_full_export.json"))
    assert "Check: 1 matched, 1 not_found" in capsys.readouterr().out
