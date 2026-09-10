"""Tests for reference consolidation (bib_match → bib merge)."""

import pytest

from bibr.enrich.consolidate import consolidate_bibs


def _data(bib=None, bib_match=None):
    return {
        "bib": bib or [],
        "bib_match": bib_match or [],
        "processing_warnings": [],
    }


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
    assert row["consolidated_fields"] == "doi,volume"


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
    assert data["bib"][0]["consolidated_fields"] == "title"


def test_unmatched_rows_untouched_and_unmarked():
    data = _data(
        bib=[{"bib_id": 1, "doi": None}, {"bib_id": 2, "doi": None}],
        bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}],
    )
    n = consolidate_bibs(data, mode="fill")
    assert n == 1
    assert data["bib"][1]["doi"] is None
    assert "consolidated_fields" not in data["bib"][1]


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
    # "Smith (2005a)" matched to a different-year work: once `year` is
    # overwritten the printed disambiguator no longer describes the row, so it
    # must not be left orphaned as a stale "a".
    data = _data(
        bib=[{"bib_id": 1, "year": 2005, "year_suffix": "a", "is_in_press": False}],
        bib_match=[{"bib_id": 1, "service": "crossref", "year": 2007}],
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
    assert out["bib"][0]["consolidated_fields"] == "doi"


def test_export_hook_falls_back_to_settings():
    from bibr.config import GlobalSettings
    from bibr.pipeline.context import RunConfig

    settings = GlobalSettings()
    settings.crossref.consolidate = "fill"
    data = _exportable(bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}])
    out = _run_export(data, RunConfig(), settings=settings)  # consolidate=None → settings
    assert out["bib"][0]["consolidated_fields"] == "doi"


def test_export_hook_off_by_default():
    from bibr.pipeline.context import RunConfig

    data = _exportable(bib_match=[{"bib_id": 1, "service": "crossref", "doi": "10.1/x"}])
    out = _run_export(data, RunConfig())
    assert out["bib"][0]["doi"] is None
    assert "consolidated_fields" not in out["bib"][0]


def test_export_hook_warns_when_crossref_disabled():
    from bibr.pipeline.context import RunConfig

    data = _exportable(bib_match=[])
    out = _run_export(data, RunConfig(consolidate="fill", crossref=False))
    assert any("consolidate" in w for w in out["extraction"]["warnings"])
