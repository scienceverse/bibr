"""Tests for reference consolidation (bib_match → bib merge)."""

import pytest

from bibr.enrich.consolidate import consolidate_bibs


def _data(bib=None, bib_match=None):
    return {
        "bib": bib or [],
        "bib_match": bib_match or [],
        "extraction": {"warnings": [], "diagnostics": {}},
    }


def _taken(data) -> dict[int, list[str]]:
    """The consolidation receipt, ``{bib_id: fields}``; bib rows never carry it."""
    assert all("consolidated_fields" not in row for row in data["bib"])
    diagnostics = (data.get("extraction") or {}).get("diagnostics") or {}
    return {row["bib_id"]: row["fields"] for row in diagnostics.get("consolidation") or []}


def test_fill_fills_empty_fields_only():
    data = _data(
        bib=[{"bib_id": 1, "title": "Printed Title", "doi": None, "volume": ""}],
        bib_match=[
            {
                "bib_id": 1,
                "service": "crossref",
                "doi": "10.1/x",
                "title": "External Title",
                "volume": "12",
            }
        ],
    )
    n = consolidate_bibs(data, mode="fill")
    assert n == 1
    row = data["bib"][0]
    assert row["doi"] == "10.1/x"  # was None → filled
    assert row["volume"] == "12"  # was "" → filled
    assert row["title"] == "Printed Title"  # printed value kept
    assert _taken(data) == {1: ["doi", "volume"]}


def test_replace_overwrites_disagreeing_fields():
    data = _data(
        bib=[{"bib_id": 1, "title": "Attention is al you need", "doi": "10.1/x"}],
        bib_match=[
            {
                "bib_id": 1,
                "service": "crossref",
                "title": "Attention is all you need",
                "doi": "10.1/x",
            }
        ],
    )
    n = consolidate_bibs(data, mode="replace")
    assert n == 1
    assert data["bib"][0]["title"] == "Attention is all you need"
    # doi agreed — not listed as consolidated
    assert _taken(data) == {1: ["title"]}


def test_replace_keeps_printed_fields_against_a_search_match():
    # A bibliographic-search hit for another work (different DOI): it may fill
    # gaps, but must not rewrite what the paper printed — DOI included.
    data = _data(
        bib=[{"bib_id": 1, "doi": "10.1/printed", "volume": "12", "issue": None}],
        bib_match=[
            {
                "bib_id": 1,
                "service": "crossref",
                "score": 100.0,
                "doi": "10.1/other",
                "volume": "13",
                "issue": "2",
            }
        ],
    )
    n = consolidate_bibs(data, mode="replace")
    assert n == 1
    row = data["bib"][0]
    assert row["doi"] == "10.1/printed"
    assert row["volume"] == "12"
    assert row["issue"] == "2"
    assert _taken(data) == {1: ["issue"]}


def test_replace_only_fills_when_no_doi_was_printed():
    data = _data(
        bib=[{"bib_id": 1, "doi": None, "title": "Printed Title", "volume": "12"}],
        bib_match=[
            {
                "bib_id": 1,
                "service": "crossref",
                "doi": "10.1/x",
                "title": "External Title",
                "volume": "13",
            }
        ],
    )
    consolidate_bibs(data, mode="replace")
    row = data["bib"][0]
    assert row["doi"] == "10.1/x"
    assert row["title"] == "Printed Title"
    assert row["volume"] == "12"
    assert _taken(data) == {1: ["doi"]}


def test_replace_matches_the_printed_doi_case_and_prefix_insensitively():
    data = _data(
        bib=[{"bib_id": 1, "doi": "https://doi.org/10.1037/ABC.123", "volume": "1"}],
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1037/abc.123", "volume": "2"}],
    )
    consolidate_bibs(data, mode="replace")
    row = data["bib"][0]
    assert row["volume"] == "2"
    assert row["doi"] == "10.1037/abc.123"


def test_replace_overwrites_from_the_match_carrying_the_printed_doi():
    # The higher-precedence Crossref row is a search hit for another work; the
    # lower-precedence row carries the printed DOI and is the one to trust.
    data = _data(
        bib=[{"bib_id": 1, "doi": "10.1/printed", "volume": "12"}],
        bib_match=[
            {"bib_id": 1, "service": "crossref", "doi": "10.1/other", "volume": "99"},
            {"bib_id": 1, "service": "someday-openalex", "doi": "10.1/printed", "volume": "13"},
        ],
    )
    consolidate_bibs(data, mode="replace")
    assert data["bib"][0]["volume"] == "13"


def test_unmatched_rows_untouched_and_unmarked():
    data = _data(
        bib=[{"bib_id": 1, "doi": None}, {"bib_id": 2, "doi": None}],
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}],
    )
    n = consolidate_bibs(data, mode="fill")
    assert n == 1
    assert data["bib"][1]["doi"] is None
    assert _taken(data) == {1: ["doi"]}


def test_repeated_consolidation_unions_the_receipt():
    data = _data(
        bib=[{"bib_id": 1, "doi": None, "volume": None}],
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}],
    )
    consolidate_bibs(data, mode="fill")
    data["bib_match"].append({"bib_id": 1, "service": "crossref", "volume": "4"})
    consolidate_bibs(data, mode="fill")
    assert _taken(data) == {1: ["doi", "volume"]}


def test_consolidation_without_an_extraction_block_records_nothing():
    """A Paper exported outside the pipeline has nowhere to keep the receipt;
    the merge still happens and nothing leaks onto the bib rows."""
    data = {
        "bib": [{"bib_id": 1, "doi": None}],
        "bib_match": [{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}],
    }
    assert consolidate_bibs(data, mode="fill") == 1
    assert data["bib"][0] == {"bib_id": 1, "doi": "10.1/x"}
    assert "extraction" not in data


def test_authors_and_editors_never_consolidated():
    data = _data(
        bib=[{"bib_id": 1, "authors": None, "editors": None}],
        bib_match=[
            {
                "bib_id": 1,
                "service": "crossref",
                "authors": [{"given": "A", "family": "B"}],
                "editors": [{"given": "C", "family": "D"}],
            }
        ],
    )
    consolidate_bibs(data, mode="replace")
    assert data["bib"][0]["authors"] is None
    assert data["bib"][0]["editors"] is None


def test_service_precedence_prefers_crossref():
    data = _data(
        bib=[{"bib_id": 1, "doi": None}],
        bib_match=[
            {"bib_id": 1, "service": "someday-openalex", "doi": "10.1/other"},
            {"bib_id": 1, "service": "crossref", "doi": "10.1/crossref"},
        ],
    )
    consolidate_bibs(data, mode="fill")
    assert data["bib"][0]["doi"] == "10.1/crossref"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        consolidate_bibs(_data(), mode="merge")


def test_no_matches_is_noop():
    data = _data(bib=[{"bib_id": 1, "doi": None}])
    assert consolidate_bibs(data, mode="fill") == 0


def test_fill_missing_key_in_bib():
    data = _data(
        bib=[{"bib_id": 1}],  # no "doi" key at all
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}],
    )
    consolidate_bibs(data, mode="fill")
    assert data["bib"][0]["doi"] == "10.1/x"


def test_replace_year_reconciles_orphaned_year_suffix():
    # "Smith (2005a)" whose printed DOI names a 2007 work: once `year` is
    # overwritten the printed disambiguator no longer describes the row, so it
    # must not be left orphaned as a stale "a".
    data = _data(
        bib=[
            {"bib_id": 1, "year": 2005, "year_suffix": "a", "is_in_press": False, "doi": "10.1/x"}
        ],
        bib_match=[{"bib_id": 1, "service": "crossref", "year": 2007, "doi": "10.1/x"}],
    )
    consolidate_bibs(data, mode="replace")
    row = data["bib"][0]
    assert row["year"] == 2007
    assert row["year_suffix"] is None


def test_fill_year_clears_stale_in_press_flag():
    # In-press ref (year exported as None, is_in_press True) matched to a
    # now-published work: a concrete consolidated year means it is no longer
    # "in press", so the flag must reconcile rather than contradict the year.
    data = _data(
        bib=[{"bib_id": 1, "year": None, "year_suffix": None, "is_in_press": True}],
        bib_match=[{"bib_id": 1, "service": "crossref", "year": 2020}],
    )
    consolidate_bibs(data, mode="fill")
    row = data["bib"][0]
    assert row["year"] == 2020
    assert row["is_in_press"] is False


def test_year_companions_untouched_when_year_not_consolidated():
    # Guard: if `year` itself isn't taken from a match, the companions stay.
    data = _data(
        bib=[{"bib_id": 1, "year": 2005, "year_suffix": "a", "is_in_press": False, "doi": None}],
        bib_match=[{"bib_id": 1, "service": "crossref", "year": 2005, "doi": "10.1/x"}],
    )
    consolidate_bibs(data, mode="replace")
    row = data["bib"][0]
    assert row["year_suffix"] == "a"  # year agreed → suffix still valid
    assert row["doi"] == "10.1/x"


# --- setting / RunConfig / export hook ---------------------------------------

import asyncio
from pathlib import Path
from types import SimpleNamespace


def test_crossref_consolidate_setting():
    from pydantic import ValidationError

    from bibr.config import CrossrefOptions

    assert CrossrefOptions().consolidate == "off"
    assert CrossrefOptions(consolidate="FILL").consolidate == "fill"
    with pytest.raises(ValidationError):
        CrossrefOptions(consolidate="merge")


def test_local_pipeline_consolidate_kwarg():
    from bibr.local.pipeline import LocalPipeline

    pipe = LocalPipeline(llm_backend="cloud", consolidate="replace")
    assert pipe._config.consolidate == "replace"
    assert LocalPipeline(llm_backend="cloud")._config.consolidate is None


class _FakePaper:
    def __init__(self, data):
        self._data = data
        self.processing_warnings = []
        self.llm_usage_labels = {}
        self.llm_trace = []
        self.text_quality = None
        self.metadata = None
        self.extraction = None

    def export_to_json(self, *, include_regions=False, include_region_meta=False):
        return self._data


class _Prog:
    def stage_start(self, name):
        pass

    def stage_end(self, name):
        pass


def _run_export(data, config, *, settings=None):
    from bibr.config import GlobalSettings
    from bibr.pipeline.stages.export import ExportStage

    def _fail(*a, **k):
        raise AssertionError(f"export errored: {a} {k}")

    fs = SimpleNamespace(
        paper=_FakePaper(data),
        warnings=[],
        result_json=None,
        path=Path("x.pdf"),
        free_all=lambda: None,
        set_error=_fail,
    )
    ctx = SimpleNamespace(
        progress=_Prog(),
        config=config,
        settings=settings or GlobalSettings(),
        alive=lambda: [fs],
    )
    asyncio.run(ExportStage().run(ctx))
    return fs.result_json


def _exportable(bib_match=None):
    return {
        "bib": [{"bib_id": 1, "doi": None}],
        "bib_match": bib_match if bib_match is not None else [],
        "extraction": {"warnings": []},
    }


def test_export_hook_consolidates_when_runconfig_set():
    from bibr.pipeline.context import RunConfig

    data = _exportable(bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}])
    out = _run_export(data, RunConfig(consolidate="fill"))
    assert out["bib"][0]["doi"] == "10.1/x"
    assert _taken(out) == {1: ["doi"]}


def test_export_hook_falls_back_to_settings():
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import RunConfig

    settings = GlobalSettings()
    settings.crossref.consolidate = "fill"
    data = _exportable(bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}])
    out = _run_export(data, RunConfig(), settings=settings)  # consolidate=None → settings
    assert _taken(out) == {1: ["doi"]}


def test_export_hook_off_by_default():
    from bibr.pipeline.context import RunConfig

    data = _exportable(bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}])
    out = _run_export(data, RunConfig())
    assert out["bib"][0]["doi"] is None
    assert _taken(out) == {}


def test_export_hook_warns_when_crossref_disabled():
    from bibr.pipeline.context import RunConfig

    data = _exportable(bib_match=[])
    out = _run_export(data, RunConfig(consolidate="fill", crossref=False))
    assert out["extraction"]["warnings"] == [
        {
            "code": "CONSOLIDATE_WITHOUT_ENRICHMENT",
            "message": "consolidate enabled but Crossref enrichment is off — no matches to merge",
        }
    ]


def test_fill_takes_the_iso_date_and_keeps_the_printed_one():
    data = _data(
        bib=[{"bib_id": 1, "date": "Spring 2019", "published_date": None, "year": None}],
        bib_match=[
            {"bib_id": 1, "service": "crossref", "published_date": "2019-04-02", "year": 2019}
        ],
    )
    consolidate_bibs(data, mode="fill")
    row = data["bib"][0]
    assert row["date"] == "Spring 2019"  # printed, never overwritten
    assert row["published_date"] == "2019-04-02"
    assert row["year"] == 2019
    assert _taken(data) == {1: ["year", "published_date"]}


def test_a_taken_year_carries_published_date_with_it():
    # The record dates the work only by year: published_date follows the year
    # instead of keeping a printed date the year now contradicts.
    data = _data(
        bib=[{"bib_id": 1, "doi": "10.1234/x", "year": 2006, "published_date": "2006"}],
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1234/x", "year": 2007}],
    )
    consolidate_bibs(data, mode="replace")
    row = data["bib"][0]
    assert (row["year"], row["published_date"]) == (2007, "2007")
    assert _taken(data) == {1: ["year", "published_date"]}


def test_fill_never_rewrites_a_printed_date_for_a_taken_year():
    # The printed entry dates the work May 2019 but lost its year; filling the
    # year from the match must not overwrite what the entry printed.
    data = _data(
        bib=[{"bib_id": 1, "date": "May 2019", "published_date": "2019-05", "year": None}],
        bib_match=[
            {"bib_id": 1, "service": "crossref", "year": 2020, "published_date": "2020-01-15"}
        ],
    )
    consolidate_bibs(data, mode="fill")
    row = data["bib"][0]
    assert (row["year"], row["published_date"], row["date"]) == (2020, "2019-05", "May 2019")
    assert _taken(data) == {1: ["year"]}


def test_fill_gives_a_year_only_record_its_year_as_the_date():
    data = _data(
        bib=[{"bib_id": 1, "published_date": None, "year": None}],
        bib_match=[{"bib_id": 1, "service": "crossref", "year": 2020, "published_date": None}],
    )
    consolidate_bibs(data, mode="fill")
    row = data["bib"][0]
    assert (row["year"], row["published_date"]) == (2020, "2020")


def test_replace_keeps_a_printed_type_over_a_catch_all_match():
    """core-api-5: Crossref types outside the BibType vocabulary map to "other";
    a same-DOI match carrying it must not overwrite a printed "book"."""
    data = _data(
        bib=[{"bib_id": 1, "bib_type": "book", "doi": "10.1017/cbo9780511809071"}],
        bib_match=[
            {
                "bib_id": 1,
                "service": "crossref",
                "doi": "10.1017/cbo9780511809071",
                "bib_type": "other",
            }
        ],
    )
    assert consolidate_bibs(data, mode="replace") == 0
    assert data["bib"][0]["bib_type"] == "book"


def test_a_catch_all_type_still_fills_a_missing_one():
    data = _data(
        bib=[{"bib_id": 1, "bib_type": None, "doi": "10.1/x"}],
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x", "bib_type": "other"}],
    )
    assert consolidate_bibs(data, mode="fill") == 1
    assert data["bib"][0]["bib_type"] == "other"
