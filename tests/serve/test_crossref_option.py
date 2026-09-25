"""Per-request ``crossref`` switch on the serve API.

Enrichment is opt-in (``CROSSREF_ENRICH`` off by default); a request can
force it on or off for itself. The decode → predict → RunConfig chain must
carry the tri-state through, and the response cache must never hand a
``crossref=true`` caller a cached unenriched result (or vice versa).
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("litserve")

from fastapi import HTTPException  # noqa: E402

from bibr.config import GlobalSettings  # noqa: E402
from bibr.extract.ref_extractor import _resolve_ref_strategies  # noqa: E402
from bibr.pipeline.context import RunConfig  # noqa: E402
from bibr.serve.deployments.pipeline import BibrPipelineAPI  # noqa: E402


class _MemoryCache:
    def __init__(self):
        self.values: dict[str, bytes] = {}
        self.set_calls = 0

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self.set_calls += 1
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


class _RecordingPipeline:
    def __init__(self, deployment_crossref: bool | None = None):
        self._config = RunConfig(crossref=deployment_crossref)
        self.configs: list[RunConfig] = []

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.configs.append(config)
        await asyncio.sleep(0)
        return {"paper_id": paper_id, "info": {"file_name": filename}}


def _api(tmp_path, *, setting: bool, deployment_crossref: bool | None = None):
    settings = GlobalSettings()
    settings.crossref.enrich = setting
    api = BibrPipelineAPI(upload_root=tmp_path, settings=settings)
    pipeline = _RecordingPipeline(deployment_crossref)
    cache = _MemoryCache()
    api._pipeline = pipeline
    api._cache = cache
    api._cache_inited = True
    api._inflight_sem = None
    return api, pipeline, cache


def _inputs(crossref: bool | None = None, content: bytes = b"same-pdf") -> dict:
    inputs = {
        "filename": "paper.pdf",
        "content": content,
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }
    if crossref is not None:
        inputs["crossref"] = crossref
    return inputs


# --- decode_request: tri-state parsing ---------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("1", True),
        ("YES", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", None),
        (None, None),
    ],
)
async def test_decode_request_parses_crossref_tri_state(tmp_path, monkeypatch, raw, expected):
    import bibr.serve.deployments.pipeline as mod

    monkeypatch.setattr(
        mod, "consume_upload_descriptor", lambda *_a, **_k: (b"%PDF-1.4 stub", "ab" * 32)
    )
    api = BibrPipelineAPI(upload_root=tmp_path)
    descriptor = {"upload_id": "u1", "filename": "paper.pdf", "size": 13, "sha256": "ab" * 32}
    if raw is not None:
        descriptor["crossref"] = raw

    decoded = await api.decode_request(descriptor)

    assert decoded["crossref"] is expected


async def test_decode_request_rejects_non_boolean_crossref(tmp_path, monkeypatch):
    import bibr.serve.deployments.pipeline as mod

    monkeypatch.setattr(
        mod, "consume_upload_descriptor", lambda *_a, **_k: (b"%PDF-1.4 stub", "ab" * 32)
    )
    api = BibrPipelineAPI(upload_root=tmp_path)
    descriptor = {
        "upload_id": "u1",
        "filename": "paper.pdf",
        "size": 13,
        "sha256": "ab" * 32,
        "crossref": "sometimes",
    }

    with pytest.raises(HTTPException) as excinfo:
        await api.decode_request(descriptor)
    assert excinfo.value.status_code == 400
    assert "crossref must be a boolean" in excinfo.value.detail


# --- RunConfig plumbing -------------------------------------------------------


async def test_request_crossref_overrides_run_config(tmp_path):
    api, pipeline, _ = _api(tmp_path, setting=False)

    await api.predict(_inputs(crossref=True))
    await api.predict(_inputs(crossref=False, content=b"other"))
    await api.predict(_inputs(content=b"third"))

    forced_on, forced_off, absent = pipeline.configs
    assert forced_on.crossref is True
    assert forced_off.crossref is False
    # Absent keeps the deployment's tri-state (None → CROSSREF_ENRICH).
    assert absent.crossref is None
    assert absent.enrichment_enabled(api._settings) is False


async def test_absent_request_value_keeps_deployment_level_switch(tmp_path):
    api, pipeline, _ = _api(tmp_path, setting=True, deployment_crossref=False)

    await api.predict(_inputs())

    (config,) = pipeline.configs
    assert config.crossref is False


# --- cache key separation -----------------------------------------------------


def _key(api: BibrPipelineAPI, inputs: dict, *, crossref: bool) -> str:
    import hashlib

    digest = hashlib.sha256(inputs["content"]).hexdigest()
    ref_seg, refs = _resolve_ref_strategies(None, None)
    return api._cache_key(
        digest,
        None,
        None,
        False,
        False,
        None,
        refs=refs,
        ref_seg=ref_seg,
        crossref=crossref,
        input_format=".pdf",
    )


def test_cache_key_separates_enriched_from_unenriched(tmp_path):
    api, _, _ = _api(tmp_path, setting=False)
    inputs = _inputs()

    assert _key(api, inputs, crossref=True) != _key(api, inputs, crossref=False)
    assert _key(api, inputs, crossref=True).endswith(":enrich:fmt:pdf")


async def test_crossref_true_never_reads_a_cached_unenriched_result(tmp_path):
    api, pipeline, cache = _api(tmp_path, setting=False)

    await api.predict(_inputs())  # unenriched (setting off, no override)
    await api.predict(_inputs(crossref=True))  # must miss the cache and run again

    assert len(pipeline.configs) == 2
    assert pipeline.configs[1].crossref is True
    assert set(cache.values) == {
        _key(api, _inputs(), crossref=False),
        _key(api, _inputs(), crossref=True),
    }


async def test_crossref_false_never_reads_a_cached_enriched_result(tmp_path):
    api, pipeline, cache = _api(tmp_path, setting=True)

    await api.predict(_inputs())  # enriched (setting on, no override)
    await api.predict(_inputs(crossref=False))  # must miss the cache and run again

    assert len(pipeline.configs) == 2
    assert pipeline.configs[1].crossref is False
    assert set(cache.values) == {
        _key(api, _inputs(), crossref=True),
        _key(api, _inputs(), crossref=False),
    }


async def test_explicit_value_matching_the_setting_shares_the_cache_entry(tmp_path):
    """``crossref=true`` on a deployment that already enriches is the same result."""
    api, pipeline, cache = _api(tmp_path, setting=True)

    await api.predict(_inputs())
    await api.predict(_inputs(crossref=True))

    assert len(pipeline.configs) == 1
    assert cache.set_calls == 1


def test_effective_crossref_resolution_order(tmp_path):
    api, _, _ = _api(tmp_path, setting=False, deployment_crossref=None)
    assert api._effective_crossref(None) is False
    assert api._effective_crossref(True) is True
    api._settings.crossref.enrich = True
    assert api._effective_crossref(None) is True
    assert api._effective_crossref(False) is False
    api._pipeline._config = dataclasses.replace(api._pipeline._config, crossref=False)
    assert api._effective_crossref(None) is False
