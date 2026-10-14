"""Retraction and other update notices, and organization authors, from
recorded Crossref responses (tests/fixtures/crossref/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bibr.enrich.references import _best_title_candidate, _build_match, _TitleCandidate
from bibr.enrich.schemas import CrossrefWorkItem, strip_notice_prefix
from bibr.export.json_export import _export_paper_payload
from bibr.export.models import PaperExport
from bibr.models import MatchSource, format_bib_authors

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "crossref"


def _work(name: str) -> CrossrefWorkItem:
    return CrossrefWorkItem.from_raw(
        json.loads((FIXTURES / f"work_{name}.json").read_text())["message"]
    )


def test_updated_by_is_parsed_from_the_record():
    item = _work("wakefield")

    by_type = {u.type: u for u in item.updates}
    assert set(by_type) == {"retraction", "correction"}
    retraction = by_type["retraction"]
    assert retraction.doi == "10.1016/s0140-6736(10)60175-4"
    assert retraction.date == "2010-02-06"
    assert retraction.source == "retraction-watch"
    assert retraction.label == "Retraction"


def test_update_types_are_snake_case_and_repeats_dropped():
    item = CrossrefWorkItem.from_raw(
        {
            "DOI": "10.1/x",
            "updated-by": [
                {"type": "Expression-of-Concern", "DOI": "10.1000/EOC", "source": "publisher"},
                {"type": "expression_of_concern", "DOI": "10.1000/eoc", "source": "publisher"},
                {"type": "retraction", "DOI": "not a doi", "updated": {"date-parts": [[2020]]}},
                {"label": "no type"},
                "not an object",
            ],
        }
    )
    assert [(u.type, u.doi, u.date) for u in item.updates] == [
        ("expression_of_concern", "10.1000/eoc", None),
        ("retraction", None, "2020"),
    ]


def test_match_keeps_the_deposited_title_and_lists_the_notices():
    match = _build_match(_work("wakefield"), 100.0)

    assert match.title.startswith("RETRACTED: Ileal-lymphoid-nodular hyperplasia")
    assert [u.type for u in match.updated_by] == ["correction", "retraction"]


def test_organization_author_is_kept_whole():
    """Open Science Collaboration (2015) has one author, an organization; it
    used to be dropped, leaving the match with no authors."""
    match = _build_match(_work("osc"), 100.0)

    assert len(match.authors) == 1
    (author,) = match.authors
    assert (author.given, author.family, author.literal) == ("", "", "Open Science Collaboration")
    assert format_bib_authors(match.authors) == "Open Science Collaboration"


def test_people_keep_given_and_family():
    match = _build_match(_work("carney"), 100.0)
    assert [(a.family, a.literal) for a in match.authors] == [
        ("Carney", None),
        ("Cuddy", None),
        ("Yap", None),
    ]


@pytest.mark.parametrize(
    ("title", "stripped"),
    [
        ("RETRACTED: Ileal-lymphoid hyperplasia", "Ileal-lymphoid hyperplasia"),
        ("RETRACTED ARTICLE: Riemann integration", "Riemann integration"),
        ("WITHDRAWN: A study", "A study"),
        ("Expression of concern: A study", "A study"),
        ("Retraction: A study", "Retraction: A study"),
        ("Retracted", "Retracted"),
        ("RETRACTED:", "RETRACTED:"),
    ],
)
def test_notice_prefix(title, stripped):
    assert strip_notice_prefix(title) == stripped


def test_a_retracted_record_wins_on_its_own_title():
    """The printed title of a retracted work has no "RETRACTED:" prefix; the
    record's prefix must not cost it the match."""
    wakefield = _TitleCandidate.from_crossref(_work("wakefield"))
    title = (
        "Ileal-lymphoid-nodular hyperplasia, non-specific colitis, and pervasive "
        "developmental disorder in children"
    )
    best = _best_title_candidate(title, [wakefield], 1998, "Wakefield, A. J., Murch, S. H.")
    assert best is not None
    assert best[1] == 100.0


def test_a_notice_titled_like_the_work_keeps_its_prefix():
    """A notice about the work carries the same prefix but lists no updates
    itself, so it is scored on its whole title."""
    raw = json.loads((FIXTURES / "work_wakefield.json").read_text())["message"]
    notice = dict(raw, DOI="10.1016/notice")
    notice.pop("updated-by")
    candidate = _TitleCandidate.from_crossref(CrossrefWorkItem.from_raw(notice))
    assert candidate.scoring_title == candidate.title


def test_export_writes_update_notices_and_organization_authors(demo_paper):
    ref = demo_paper.metadata.references[0]
    ref.match = {MatchSource.CROSSREF: _build_match(_work("wakefield"), 100.0)}
    demo_paper.metadata.match = {MatchSource.CROSSREF: _build_match(_work("osc"), 100.0)}

    payload = _export_paper_payload(demo_paper)
    PaperExport.model_validate(payload)

    (row,) = payload["bib_match"]
    assert row["title"].startswith("RETRACTED: ")
    assert {
        "type": "retraction",
        "doi": "10.1016/s0140-6736(10)60175-4",
        "date": "2010-02-06",
        "source": "retraction-watch",
        "label": "Retraction",
    } in row["updated_by"]
    (meta,) = payload["metadata_match"]
    assert meta["author"] == [{"literal": "Open Science Collaboration"}]
    assert "updated_by" not in meta


def test_export_omits_updated_by_when_the_record_lists_none(export_payload):
    for row in export_payload["bib_match"] + export_payload["metadata_match"]:
        assert "updated_by" not in row
