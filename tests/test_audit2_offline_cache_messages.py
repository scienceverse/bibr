"""A model missing from the cache under HF_HUB_OFFLINE is reported as that, not as a download."""

from __future__ import annotations

import logging

from bibr.local.cli.dry_run import format_download_notice
from bibr.utils import ml_runtime

_PENDING = [("paper classifier", "scienceverse/bibr-paper-classifier", "~90 MB")]


def test_the_first_run_notice_does_not_promise_a_download_offline(monkeypatch):
    monkeypatch.setattr("bibr.utils.hf_cache.hub_offline", lambda: True)

    lines = format_download_notice(_PENDING)

    assert "downloading" not in lines[0]
    assert "HF_HUB_OFFLINE" in lines[0]
    assert "scienceverse/bibr-paper-classifier" in lines[1]


def test_the_first_run_notice_online_is_unchanged(monkeypatch):
    monkeypatch.setattr("bibr.utils.hf_cache.hub_offline", lambda: False)

    assert "First run: downloading" in format_download_notice(_PENDING)[0]


def _missing(*_args, **_kwargs):
    raise FileNotFoundError("onnx/model.onnx is not in the cache")


def test_a_bundle_missing_from_the_cache_offline_is_a_warning(monkeypatch, caplog):
    monkeypatch.setattr("bibr.utils.hf_cache.hub_offline", lambda: True)
    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", _missing)

    with caplog.at_level(logging.INFO, logger=ml_runtime.__name__):
        found = ml_runtime.find_onnx_bundle("org/model", "abc123", label="paper classifier")

    assert found is None
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "org/model@abc123" in record.getMessage()
    assert "HF_HUB_OFFLINE" in record.getMessage()


def test_a_bundle_that_is_not_published_stays_an_info_line_online(monkeypatch, caplog):
    monkeypatch.setattr("bibr.utils.hf_cache.hub_offline", lambda: False)
    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", _missing)

    with caplog.at_level(logging.INFO, logger=ml_runtime.__name__):
        assert ml_runtime.find_onnx_bundle("org/model", "abc123") is None

    (record,) = caplog.records
    assert record.levelno == logging.INFO
