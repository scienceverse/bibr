"""Per-request ``include_region_meta`` switch on the serve API.

``extraction.text_regions`` (each text row's page, box, font and region type)
is opt-in. The decode → predict → RunConfig chain must carry the switch
through, and the response cache must never hand a caller that asked for the
regions a cached result without them (or the other way round).
"""

from __future__ import annotations

import asyncio
import hashlib

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

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


class _RecordingPipeline:
    def __init__(self):
        self._config = RunConfig()
        self.configs: list[RunConfig] = []

    async def process_file(self, filename, paper_id, content, config):  # noqa: ARG002
        self.configs.append(config)
        await asyncio.sleep(0)
        return {"paper_id": paper_id, "info": {"file_name": filename}}


def _api(tmp_path):
    api = BibrPipelineAPI(upload_root=tmp_path, settings=GlobalSettings())
    pipeline = _RecordingPipeline()
    cache = _MemoryCache()
    api._pipeline = pipeline
    api._cache = cache
    api._cache_inited = True
    api._inflight_sem = None
    return api, pipeline, cache


def _inputs(include_region_meta: bool | None = None, content: bytes = b"same-pdf") -> dict:
    inputs = {
        "filename": "paper.pdf",
        "content": content,
        "start_page": None,
        "end_page": None,
        "include_figures": False,
        "include_regions": False,
        "consolidate": None,
    }
    # Left out, the inputs look like those of an embedder that predates the field.
    if include_region_meta is not None:
        inputs["include_region_meta"] = include_region_meta
    return inputs


# --- decode_request -----------------------------------------------------------


def _descriptor(**fields: str) -> dict:
    return {"upload_id": "u1", "filename": "paper.pdf", "size": 13, "sha256": "ab" * 32, **fields}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("1", True),
        ("YES", True),
        ("false", False),
        ("0", False),
        ("no", False),
        (None, False),
    ],
)
async def test_decode_request_parses_include_region_meta(tmp_path, monkeypatch, raw, expected):
    import bibr.serve.deployments.pipeline as mod

    monkeypatch.setattr(
        mod, "consume_upload_descriptor", lambda *_a, **_k: (b"%PDF-1.4 stub", "ab" * 32)
    )
    api = BibrPipelineAPI(upload_root=tmp_path)
    fields = {} if raw is None else {"include_region_meta": raw}

    decoded = await api.decode_request(_descriptor(**fields))

    assert decoded["include_region_meta"] is expected


async def test_decode_request_rejects_non_boolean_include_region_meta(tmp_path, monkeypatch):
    import bibr.serve.deployments.pipeline as mod

    monkeypatch.setattr(
        mod, "consume_upload_descriptor", lambda *_a, **_k: (b"%PDF-1.4 stub", "ab" * 32)
    )
    api = BibrPipelineAPI(upload_root=tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await api.decode_request(_descriptor(include_region_meta="sometimes"))
    assert excinfo.value.status_code == 400
    assert "include_region_meta must be a boolean" in excinfo.value.detail


# --- RunConfig plumbing -------------------------------------------------------


async def test_request_include_region_meta_reaches_run_config(tmp_path):
    api, pipeline, _ = _api(tmp_path)

    await api.predict(_inputs(include_region_meta=True))
    await api.predict(_inputs(include_region_meta=False, content=b"other"))
    await api.predict(_inputs(content=b"third"))

    asked, declined, absent = pipeline.configs
    assert asked.include_region_meta is True
    assert declined.include_region_meta is False
    assert absent.include_region_meta is False


# --- cache key separation -----------------------------------------------------


def _key(api: BibrPipelineAPI, inputs: dict, **options: bool) -> str:
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
        input_format=".pdf",
        **options,
    )


def test_cache_key_separates_results_with_and_without_text_regions(tmp_path):
    api, _, _ = _api(tmp_path)
    inputs = _inputs()

    # Off adds nothing, so the keys of results already cached stay valid.
    assert _key(api, inputs, include_region_meta=False) == _key(api, inputs)
    assert _key(api, inputs, include_region_meta=True) != _key(api, inputs)
    assert ":rmeta" in _key(api, inputs, include_region_meta=True)


async def test_region_meta_request_never_reads_a_cached_result_without_it(tmp_path):
    api, pipeline, cache = _api(tmp_path)

    await api.predict(_inputs())  # cached without text_regions
    await api.predict(_inputs(include_region_meta=True))  # must miss and run again

    assert len(pipeline.configs) == 2
    assert pipeline.configs[1].include_region_meta is True
    assert set(cache.values) == {
        _key(api, _inputs()),
        _key(api, _inputs(), include_region_meta=True),
    }


async def test_request_without_region_meta_never_reads_a_cached_result_with_it(tmp_path):
    api, pipeline, _ = _api(tmp_path)

    await api.predict(_inputs(include_region_meta=True))
    await api.predict(_inputs(include_region_meta=False))

    assert len(pipeline.configs) == 2
    assert pipeline.configs[1].include_region_meta is False
