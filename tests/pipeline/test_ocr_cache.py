"""Disk cache for OCR stage output (bibr.pipeline.ocr_cache)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.config import Settings
from bibr.input.pdf_outline import OutlineItem
from bibr.ocr.types import OcrRegionResult
from bibr.pipeline import ocr_cache
from bibr.pipeline.context import PipelineContext, RunConfig, StageSignals
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.ocr import OcrStage
from bibr.pipeline.stages.render_ocr import InterleavedRenderOcrStage
from bibr.pipeline.state import FileState
from bibr.processing_warnings import ProcessingWarning, WarningCode

_PAGE_FAILED = ProcessingWarning(
    WarningCode.OCR_PAGE_FAILED,
    "OCR failed for a page; its text is missing (page 7): RuntimeError: temporary failure",
)


@pytest.fixture
def enabled_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings.cache, "ocr", True)
    monkeypatch.setattr(Settings.cache, "ocr_dir", str(tmp_path))
    return tmp_path


def _ctx(file_states, *, resources=None, config=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=resources or MagicMock(),
        config=config or RunConfig(),
        signals=StageSignals(any_needs_ocr=True, preloading_ocr=False),
    )


def _regions():
    return [
        [
            OcrRegionResult(
                index=0,
                native_label="text",
                label="text",
                content="Hello world",
                bbox_2d=[10.0, 20.0, 30.0, 40.0],
                font_size=12.0,
                font_bold=True,
                image_b64=None,
            ),
            OcrRegionResult(
                index=1,
                native_label="figure",
                label="figure",
                content="",
                bbox_2d=[0.0, 0.0, 100.0, 100.0],
                image_b64="ZmFrZS1wbmc=",
            ),
        ],
        [],
    ]


def _fs(name="paper.pdf", file_hash="abc123"):
    fs = FileState(path=Path(name))
    fs.file_hash = file_hash
    return fs


def _identity(**overrides):
    """A concrete resolved OCR runtime identity for cache-key contracts."""
    from bibr.ocr.profiles import OcrRuntimeIdentity

    values = {
        "backend": "serve-http",
        "model": "paddle-ocr-vl-1.6",
        "profile": "paddle",
        "normalizer_version": "paddle-canonical-v1",
    }
    values.update(overrides)
    return OcrRuntimeIdentity(**values)


# --- round-trip serialization ------------------------------------------------


def test_roundtrip_preserves_all_fields(enabled_cache):
    fs, cfg = _fs(), RunConfig(ocr_backend="glm-llama")
    identity = _identity(
        backend="glm-llama",
        model="THUDM/GLM-OCR",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    ocr_cache.store(fs, cfg, identity, _regions())

    loaded = ocr_cache.load(fs, cfg, identity)

    assert loaded is not None
    assert [[r.to_dict() for r in page] for page in loaded] == [
        [r.to_dict() for r in page] for page in _regions()
    ]
    # image_b64 (the only image-derived field) survives as its base64 string.
    assert loaded[0][1].image_b64 == "ZmFrZS1wbmc="


def test_roundtrip_preserves_native_rejection_alternate_sources(enabled_cache):
    fs, cfg = _fs(), RunConfig(ocr_backend="glm-llama")
    identity = _identity(
        backend="glm-llama",
        model="THUDM/GLM-OCR",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    regions = [
        [
            OcrRegionResult(
                index=0,
                native_label="text",
                label="text",
                content="Ernst Lau (1927).",
                raw_content="Ernst Lau (1927).",
                native_text_candidate="Ernst Lau ().",
                native_text_rejection_reason="private_use",
                bbox_2d=[109, 118, 833, 896],
            )
        ]
    ]

    ocr_cache.store(fs, cfg, identity, regions)
    loaded = ocr_cache.load(fs, cfg, identity)

    assert loaded == regions
    assert loaded[0][0].native_text_candidate == "Ernst Lau ()."
    assert loaded[0][0].raw_content == "Ernst Lau (1927)."


def test_bundle_roundtrip_restores_all_pre_parse_artifacts(enabled_cache):
    source, cfg = _fs(), RunConfig(ocr_backend="glm-llama", ref_seg_strategy="geom")
    source.native_metadata = {"title": "Embedded title", "doi": "10.1/example"}
    source.ref_line_geometry = [{"text": "Example reference", "page": 2}]
    source.pdf_outline = [OutlineItem(title="Methods", level=2, page_no=3, y_top=42.5)]
    identity = _identity(
        backend="glm-llama",
        model="THUDM/GLM-OCR",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    ocr_cache.store(source, cfg, identity, _regions())

    restored = _fs()

    assert ocr_cache.load_bundle(restored, cfg, identity) is True
    assert restored.ocr_regions is not None
    assert restored.ocr_regions[0][0].content == "Hello world"
    assert restored.native_metadata == source.native_metadata
    assert restored.ref_line_geometry == source.ref_line_geometry
    assert restored.pdf_outline == source.pdf_outline


def test_bundle_roundtrip_restores_detached_pdf_inspection(enabled_cache):
    from bibr.ocr.pdf_inspection import PdfInspection, PdfPageInspection

    source, cfg = _fs(), RunConfig(ocr_backend="glm-llama")
    source.pdf_inspection = PdfInspection(
        pages=(PdfPageInspection(0, 600.0, 800.0, (0.0, 0.0, 600.0, 800.0), 42),),
        layout_results=[],
        metadata={"title": "Embedded"},
        outline=[],
        reference_lines=[],
        component_errors={"outline": "unavailable"},
    )
    identity = _identity(
        backend="glm-llama",
        model="THUDM/GLM-OCR",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    ocr_cache.store(source, cfg, identity, _regions())
    restored = _fs()

    assert ocr_cache.load_bundle(restored, cfg, identity) is True
    assert restored.pdf_inspection is not None
    assert restored.pdf_inspection.pages == source.pdf_inspection.pages
    assert restored.pdf_inspection.component_errors == {"outline": "unavailable"}


def test_miss_when_not_yet_stored(enabled_cache):
    assert ocr_cache.load(_fs(), RunConfig(ocr_backend="glm-llama"), _identity()) is None


def test_no_file_hash_never_caches(enabled_cache):
    fs = _fs(file_hash=None)
    ocr_cache.store(fs, RunConfig(), _identity(), _regions())
    assert ocr_cache.load(fs, RunConfig(), _identity()) is None


# --- key composition ---------------------------------------------------------


def test_key_changes_with_page_range():
    fs = _fs()
    identity = _identity()
    base = ocr_cache._key(fs, RunConfig(ocr_backend="glm-llama"), identity)
    start = ocr_cache._key(fs, RunConfig(ocr_backend="glm-llama", start_page=2), identity)
    end = ocr_cache._key(fs, RunConfig(ocr_backend="glm-llama", end_page=5), identity)
    assert base != start != end and base != end


def test_key_changes_with_backend_and_model():
    fs = _fs()
    base = ocr_cache._key(fs, RunConfig(ocr_backend="glm-llama"), _identity())
    other_backend = ocr_cache._key(
        fs, RunConfig(ocr_backend="glm-mlx"), _identity(backend="glm-mlx")
    )
    other_model = ocr_cache._key(
        fs,
        RunConfig(ocr_backend="glm-llama", ocr_model="other/model"),
        _identity(model="other/model"),
    )
    assert base != other_backend
    assert base != other_model


def test_key_uses_all_concrete_runtime_identity_components():
    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_model="paddle-ocr-vl-1.6")
    paddle = _identity()
    glm = _identity(model="THUDM/GLM-OCR", profile="glm", normalizer_version="glm-canonical-v1")
    same_model_glm_profile = _identity(profile="glm", normalizer_version="glm-canonical-v1")
    newer_normalizer = _identity(normalizer_version="paddle-canonical-v2")

    assert ocr_cache._key(fs, cfg, paddle) != ocr_cache._key(fs, cfg, glm)
    assert ocr_cache._key(fs, cfg, paddle) != ocr_cache._key(fs, cfg, same_model_glm_profile)
    assert ocr_cache._key(fs, cfg, paddle) != ocr_cache._key(fs, cfg, newer_normalizer)


def test_key_distinguishes_concrete_paddle_fallback_identities():
    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http")
    managed = _identity(backend="paddle", model="PaddlePaddle/PaddleOCR-VL-1.6")
    served_fallback = _identity(backend="serve-http", model="paddle-ocr-vl-1.6")

    assert ocr_cache._key(fs, cfg, managed) != ocr_cache._key(fs, cfg, served_fallback)


def test_key_changes_with_effective_generation_max_tokens():
    from bibr.config import snapshot_settings

    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline = snapshot_settings()
    baseline.ocr.generation_max_tokens = None
    larger = baseline.model_copy(deep=True)
    larger.ocr.generation_max_tokens = 2048

    assert ocr_cache._key(fs, cfg, _identity(), baseline) != ocr_cache._key(
        fs, cfg, _identity(), larger
    )


def test_serve_paddle_key_changes_with_table_recovery_limit(monkeypatch):
    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline = ocr_cache._key(fs, cfg, _identity())

    monkeypatch.setattr(ocr_cache, "PADDLE_TABLE_RECOVERY_MAX_TOKENS", 16384)

    assert baseline != ocr_cache._key(fs, cfg, _identity())


def test_non_serve_paddle_key_ignores_serve_table_recovery_limit(monkeypatch):
    fs = _fs()
    cfg = RunConfig(ocr_backend="paddle-vllm", ocr_profile="paddle")
    identity = _identity(backend="paddle-vllm")
    baseline = ocr_cache._key(fs, cfg, identity)

    monkeypatch.setattr(ocr_cache, "PADDLE_TABLE_RECOVERY_MAX_TOKENS", 16384)

    assert baseline == ocr_cache._key(fs, cfg, identity)


def test_key_changes_with_effective_generation_temperature():
    from bibr.config import snapshot_settings

    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline = snapshot_settings()
    baseline.ocr.generation_temperature = None
    warmer = baseline.model_copy(deep=True)
    warmer.ocr.generation_temperature = 0.2

    assert ocr_cache._key(fs, cfg, _identity(), baseline) != ocr_cache._key(
        fs, cfg, _identity(), warmer
    )


def test_explicit_global_generation_limit_changes_paddle_task_budgets_and_cache_key():
    from bibr.config import snapshot_settings

    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    implicit = snapshot_settings()
    implicit.ocr.generation_max_tokens = None
    implicit.ocr.generation_temperature = None
    explicit_defaults = implicit.model_copy(deep=True)
    explicit_defaults.ocr.generation_max_tokens = 1024
    explicit_defaults.ocr.generation_temperature = 0.0

    assert ocr_cache._key(fs, cfg, _identity(), implicit) != ocr_cache._key(
        fs, cfg, _identity(), explicit_defaults
    )


def test_key_changes_with_file_hash():
    cfg = RunConfig(ocr_backend="glm-llama")
    identity = _identity()
    assert ocr_cache._key(_fs(file_hash="a"), cfg, identity) != ocr_cache._key(
        _fs(file_hash="b"), cfg, identity
    )


def test_key_changes_with_effective_figure_image_mode(enabled_cache):
    fs = _fs()
    with patch.object(Settings, "FIGURE_IMAGES", False):
        global_off = ocr_cache._key(fs, RunConfig(ocr_backend="glm-llama"), _identity())
        request_on = ocr_cache._key(
            fs, RunConfig(ocr_backend="glm-llama", include_figures=True), _identity()
        )
    with patch.object(Settings, "FIGURE_IMAGES", True):
        global_on = ocr_cache._key(fs, RunConfig(ocr_backend="glm-llama"), _identity())
        request_off = ocr_cache._key(
            fs, RunConfig(ocr_backend="glm-llama", include_figures=False), _identity()
        )

    assert global_off != request_on
    assert global_on == request_on
    assert global_off == request_off
    assert global_on != request_off


def test_key_changes_with_native_text_settings(monkeypatch):
    fs = _fs()
    cfg = RunConfig(ocr_backend="glm-llama")
    base = ocr_cache._key(fs, cfg, _identity())

    monkeypatch.setattr(Settings.ocr, "native_text_min_chars", 999)

    assert ocr_cache._key(fs, cfg, _identity()) != base


def test_key_changes_with_effective_reference_segmentation_strategy():
    fs = _fs()
    geom = ocr_cache._key(
        fs, RunConfig(ocr_backend="glm-llama", ref_seg_strategy="geom"), _identity()
    )
    llm = ocr_cache._key(
        fs, RunConfig(ocr_backend="glm-llama", ref_seg_strategy="llm"), _identity()
    )

    assert geom != llm


# --- corrupt entry treated as a miss -----------------------------------------


def test_corrupt_entry_is_miss_and_deleted(enabled_cache):
    fs, cfg = _fs(), RunConfig(ocr_backend="glm-llama")
    identity = _identity()
    path = ocr_cache._path(fs, cfg, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json", encoding="utf-8")

    assert ocr_cache.load(fs, cfg, identity) is None
    assert not path.exists()


def test_version_mismatch_is_miss(enabled_cache, monkeypatch):
    fs, cfg = _fs(), RunConfig(ocr_backend="glm-llama")
    identity = _identity()
    ocr_cache.store(fs, cfg, identity, _regions())
    monkeypatch.setattr(ocr_cache, "_CACHE_FORMAT_VERSION", 999)
    # New version resolves to a different path, so this is a plain miss — the
    # important guarantee is that an entry written under the old version is
    # never deserialized as the new schema.
    assert ocr_cache.load(fs, cfg, identity) is None


def test_format_six_payload_is_rejected_after_layout_key_removal(enabled_cache):
    fs, cfg, identity = _fs(), RunConfig(ocr_backend="glm-llama"), _identity()
    path = ocr_cache._path(fs, cfg, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 6, "regions": []}), encoding="utf-8")

    assert ocr_cache.load(fs, cfg, identity) is None
    assert not path.exists()


@pytest.mark.parametrize("bundle", [False, True])
def test_cache_restores_partial_page_evidence_and_warnings(enabled_cache, bundle):
    fs, cfg, identity = _fs(), RunConfig(ocr_backend="glm-http"), _identity()
    fs.ocr_pages_attempted = 10
    fs.ocr_pages_failed = 4
    fs.warnings = [_PAGE_FAILED]
    ocr_cache.store(fs, cfg, identity, _regions())

    incoming = _fs()
    incoming.warnings = [ProcessingWarning(WarningCode.OCR_REGION_FAILED, "from this run")]
    loader = ocr_cache.load_bundle if bundle else ocr_cache.load
    assert loader(incoming, cfg, identity)
    assert incoming.ocr_pages_attempted == 10
    assert incoming.ocr_pages_failed == 4
    assert incoming.warnings == [
        ProcessingWarning(WarningCode.OCR_REGION_FAILED, "from this run"),
        _PAGE_FAILED,
    ]
    assert loader(incoming, cfg, identity)
    assert incoming.warnings.count(fs.warnings[0]) == 1


@pytest.mark.parametrize("bundle", [False, True])
def test_bad_completion_evidence_is_a_miss_without_partial_restore(enabled_cache, bundle):
    fs, cfg, identity = _fs(), RunConfig(), _identity()
    ocr_cache.store(fs, cfg, identity, _regions())
    path = ocr_cache._path(fs, cfg, identity)
    payload = json.loads(path.read_text())
    payload["ocr_quality"]["pages_failed"] = -1
    path.write_text(json.dumps(payload))
    loader = ocr_cache.load_bundle if bundle else ocr_cache.load
    assert not loader(fs, cfg, identity)
    assert fs.ocr_regions is None and not fs.warnings
    assert not path.exists()


@pytest.mark.parametrize("bundle", [False, True])
def test_prose_warnings_are_a_miss(enabled_cache, bundle):
    """Format 9 stored warnings as prose; a coded reader must not load them."""
    fs, cfg, identity = _fs(), RunConfig(), _identity()
    ocr_cache.store(fs, cfg, identity, _regions())
    path = ocr_cache._path(fs, cfg, identity)
    payload = json.loads(path.read_text())
    payload["ocr_quality"]["warnings"] = ["OCR failed for page index 6: temporary failure"]
    path.write_text(json.dumps(payload))
    loader = ocr_cache.load_bundle if bundle else ocr_cache.load
    assert not loader(fs, cfg, identity)
    assert fs.ocr_regions is None and not fs.warnings
    assert not path.exists()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("threshold", [0.5, 0.9])
async def test_cache_hits_obey_current_ocr_success_threshold(
    enabled_cache, monkeypatch, fused, threshold
):
    from bibr.ocr.profiles import resolve_ocr_runtime_identity

    cfg = RunConfig(ocr_backend="glm-http")
    identity = resolve_ocr_runtime_identity(cfg, Settings)
    source = _fs()
    source.ocr_pages_attempted = 10
    source.ocr_pages_failed = 4
    source.warnings = [_PAGE_FAILED]
    regions = [_regions()[0] for _ in range(6)] + [[] for _ in range(4)]
    ocr_cache.store(source, cfg, identity, regions)
    monkeypatch.setattr(Settings.ocr, "min_success_rate", threshold)

    incoming = _fs()
    context = _ctx([incoming], resources=_ocr_rm(), config=cfg)
    with patch("bibr.pipeline.stages.ocr.ocr_page_regions", AsyncMock()) as recognize:
        if fused:
            assert await InterleavedRenderOcrStage()._probe_cache(context) == []
        else:
            await OcrStage().run(context)
    recognize.assert_not_called()
    assert incoming.warnings == source.warnings
    assert incoming.ocr_pages_failed == 4
    assert incoming.error_code == ("ocr_mostly_failed" if threshold == 0.9 else None)


# --- stage integration: cache hit skips OCR ----------------------------------


def _ocr_rm():
    rm = MagicMock()
    rm.ocr = MagicMock(recognize=AsyncMock(return_value="text"), loaded=True)
    rm.shutdown_ocr = AsyncMock(return_value=None)
    rm.await_ocr = AsyncMock(return_value=None)
    del rm.ocr.wait_for_server
    return rm


@pytest.mark.asyncio
async def test_stage_populates_cache_then_second_run_skips_ocr(enabled_cache):
    cfg = RunConfig(ocr_backend="glm-llama")

    # First run: OCR executes and writes the cache entry.
    fs1 = _fs()
    fs1.page_images = [MagicMock()]
    fs1.page_indices = [0]
    fs1.layout_results = [[]]
    rm1 = _ocr_rm()
    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hello"}]),
        ) as first_ocr,
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs1], resources=rm1, config=cfg))
    assert first_ocr.await_count == 1
    assert fs1.ocr_regions is not None

    # Second run: same file_hash + params → served from disk, no OCR at all.
    fs2 = _fs()
    fs2.page_images = [MagicMock()]
    fs2.page_indices = [0]
    fs2.layout_results = [[]]
    rm2 = _ocr_rm()
    with patch("bibr.pipeline.stages.ocr.ocr_page_regions", AsyncMock()) as second_ocr:
        await OcrStage().run(_ctx([fs2], resources=rm2, config=cfg))

    second_ocr.assert_not_called()
    rm2.await_ocr.assert_not_called()
    assert fs2.ocr_regions is not None
    assert fs2.ocr_regions[0][0].content == "hello"


@pytest.mark.asyncio
async def test_automatic_paddle_resolves_selected_identity_before_cache_lookup(enabled_cache):
    """An automatic chain cannot probe a cache under its unresolved alias."""
    cfg = RunConfig(ocr_backend="paddle")
    fs = _fs()
    # A region that will call the backend: the chain must resolve before the
    # lookup (a native-only file would need neither the engine nor its identity).
    fs.layout_results = [[{"label": "text", "task_type": "text"}]]
    selected = _identity(
        backend="glm-llama",
        model="glm/selected",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None

    async def select_runtime():
        rm.ocr_runtime_identity = selected

    rm.await_ocr = AsyncMock(side_effect=select_runtime)
    with patch("bibr.pipeline.ocr_cache.load", return_value=_regions()) as load:
        await OcrStage().run(_ctx([fs], resources=rm, config=cfg))

    rm.await_ocr.assert_awaited_once()
    assert load.call_args.args[2] == selected


@pytest.mark.asyncio
async def test_interleaved_automatic_paddle_resolves_identity_before_bundle_cache(enabled_cache):
    cfg = RunConfig(ocr_backend="paddle")
    selected = _identity(
        backend="glm-llama",
        model="glm/selected",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None

    async def select_runtime():
        rm.ocr_runtime_identity = selected

    rm.await_ocr = AsyncMock(side_effect=select_runtime)
    stage = InterleavedRenderOcrStage()

    def cache_hit(fs, *_args):
        fs.ocr_regions = []
        return True

    with (
        patch("bibr.pipeline.ocr_cache.is_enabled", return_value=True),
        patch("bibr.pipeline.ocr_cache.load_bundle", side_effect=cache_hit) as load,
    ):
        pending = await stage._probe_cache(_ctx([_fs()], resources=rm, config=cfg))

    assert pending == []
    rm.await_ocr.assert_awaited_once()
    assert load.call_args.args[2] == selected


@pytest.mark.asyncio
async def test_interleaved_automatic_paddle_defers_startup_when_cache_disabled(monkeypatch):
    """No bundle to look up → no need for a concrete identity before layout.

    The automatic chain used to start its engine here, before native text was
    known, purely to key a cache that is off by default. OcrStage now starts it
    after NativeTextStage, and only if a region still needs the backend.
    """
    monkeypatch.setattr(Settings.cache, "ocr", False)
    cfg = RunConfig(ocr_backend="paddle")
    rm = _ocr_rm()
    rm.ocr = None
    rm.ocr_runtime_identity = None
    rm.await_ocr = AsyncMock(side_effect=AssertionError("must not start before native text"))
    fs = _fs()
    ctx = _ctx([fs], resources=rm, config=cfg)

    pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == [fs]
    rm.await_ocr.assert_not_awaited()
    assert "ocr_runtime_identity" not in ctx.scratch


@pytest.mark.asyncio
async def test_interleaved_automatic_paddle_reuses_retained_runtime_identity(enabled_cache):
    cfg = RunConfig(ocr_backend="paddle")
    selected = _identity(
        backend="glm-llama",
        model="glm/selected",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    rm = _ocr_rm()
    rm.ocr_runtime_identity = selected
    stage = InterleavedRenderOcrStage()

    def cache_hit(fs, *_args):
        fs.ocr_regions = []
        return True

    with (
        patch("bibr.pipeline.ocr_cache.is_enabled", return_value=True),
        patch("bibr.pipeline.ocr_cache.load_bundle", side_effect=cache_hit) as load,
    ):
        pending = await stage._probe_cache(_ctx([_fs()], resources=rm, config=cfg))

    assert pending == []
    rm.await_ocr.assert_not_awaited()
    assert load.call_args.args[2] == selected


@pytest.mark.asyncio
async def test_interleaved_automatic_paddle_refreshes_unloaded_runtime_identity(enabled_cache):
    cfg = RunConfig(ocr_backend="paddle")
    stale = _identity(
        backend="glm-llama",
        model="glm/stale",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    replacement = _identity()
    rm = _ocr_rm()
    rm.ocr.loaded = False
    rm.ocr_runtime_identity = stale
    stage = InterleavedRenderOcrStage()

    async def select_replacement():
        rm.ocr.loaded = True
        rm.ocr_runtime_identity = replacement

    rm.await_ocr = AsyncMock(side_effect=select_replacement)

    def cache_hit(fs, *_args):
        fs.ocr_regions = []
        return True

    with (
        patch("bibr.pipeline.ocr_cache.is_enabled", return_value=True),
        patch("bibr.pipeline.ocr_cache.load_bundle", side_effect=cache_hit) as load,
    ):
        ctx = _ctx([_fs()], resources=rm, config=cfg)
        ctx.scratch["ocr_runtime_identity"] = stale
        pending = await stage._probe_cache(ctx)

    assert pending == []
    rm.await_ocr.assert_awaited_once()
    assert load.call_args.args[2] == replacement


# --- OCR startup failures must not abort the chunk ---------------------------


@pytest.mark.asyncio
async def test_interleaved_ocr_init_failure_fails_files_not_chunk(enabled_cache):
    """``run_stage`` has no ``except``: an escaping error would kill the chunk."""
    from bibr.exceptions import UpstreamServiceError

    cfg = RunConfig(ocr_backend="paddle")
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None
    rm.ocr.loaded = False
    boom = UpstreamServiceError("ocr", "No OCR startup candidate succeeded")
    rm.await_ocr = AsyncMock(side_effect=boom)

    fs = _fs()
    ctx = _ctx([fs], resources=rm, config=cfg)
    pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == []
    assert fs.error is not None
    assert fs.error_code == "ocr_failed"
    assert fs.failed_stage == "render_ocr"
    assert fs.original_error is boom
    assert ctx.signals.ocr_init_error is boom


@pytest.mark.asyncio
async def test_interleaved_missing_runtime_identity_fails_file_not_chunk(enabled_cache):
    """Startup that reports success without a concrete identity also fails soft."""
    cfg = RunConfig(ocr_backend="paddle")
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None
    rm.ocr.loaded = False
    rm._ocr = MagicMock(loaded=False)
    rm.await_ocr = AsyncMock(return_value=None)

    fs = _fs()
    ctx = _ctx([fs], resources=rm, config=cfg)
    pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == []
    assert fs.error_code == "ocr_failed"
    assert isinstance(ctx.signals.ocr_init_error, RuntimeError)


@pytest.mark.asyncio
async def test_interleaved_prior_init_error_skips_doomed_constructor(enabled_cache):
    """A later window must not re-run the slow constructor that already failed."""
    cfg = RunConfig(ocr_backend="paddle")
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None
    rm.ocr.loaded = False

    fs = _fs()
    ctx = _ctx([fs], resources=rm, config=cfg)
    ctx.signals.ocr_init_error = RuntimeError("earlier window")
    pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == []
    rm.await_ocr.assert_not_awaited()
    assert fs.error_code == "ocr_failed"


@pytest.mark.asyncio
async def test_interleaved_native_only_chunk_skips_ocr_startup(enabled_cache):
    """An all-DOCX chunk needs no OCR runtime, so it must not start (or fail on) one."""
    cfg = RunConfig(ocr_backend="paddle")
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None
    rm.ocr.loaded = False
    rm.await_ocr = AsyncMock(side_effect=RuntimeError("must not be called"))

    fs = _fs("paper.docx")
    fs.contents = MagicMock()
    ctx = _ctx([fs], resources=rm, config=cfg)
    pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == []
    assert fs.error is None
    rm.await_ocr.assert_not_awaited()


@pytest.mark.asyncio
async def test_interleaved_init_failure_spares_native_files_in_mixed_chunk(enabled_cache):
    """Only the OCR-needing files fail; a DOCX in the same chunk still proceeds."""
    cfg = RunConfig(ocr_backend="paddle")
    rm = _ocr_rm()
    rm.ocr_runtime_identity = None
    rm.ocr.loaded = False
    rm.await_ocr = AsyncMock(side_effect=RuntimeError("engine missing"))

    pdf = _fs("paper.pdf")
    docx = _fs("paper.docx", file_hash="def456")
    docx.contents = MagicMock()
    ctx = _ctx([pdf, docx], resources=rm, config=cfg)
    pending = await InterleavedRenderOcrStage()._probe_cache(ctx)

    assert pending == []
    assert pdf.error_code == "ocr_failed"
    assert docx.error is None


@pytest.mark.asyncio
async def test_disabled_cache_does_not_read_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings.cache, "ocr", False)
    monkeypatch.setattr(Settings.cache, "ocr_dir", str(tmp_path))
    cfg = RunConfig(ocr_backend="glm-llama")
    fs = _fs()
    fs.page_images = [MagicMock()]
    fs.page_indices = [0]
    fs.layout_results = [[]]
    rm = _ocr_rm()
    with (
        patch(
            "bibr.pipeline.stages.ocr.ocr_page_regions",
            AsyncMock(return_value=[{"content": "hi"}]),
        ),
        patch(
            "bibr.pipeline.stages.ocr._postprocess_ocr_regions",
            side_effect=lambda x, *_args, **_kwargs: x,
        ),
    ):
        await OcrStage().run(_ctx([fs], resources=rm, config=cfg))

    # Nothing written when off.
    assert not any(tmp_path.glob("*.json"))


def test_key_changes_with_profile_image_geometry(monkeypatch):
    """A geometry change must invalidate cached OCR, like any render knob.

    Geometry decides which pixels reach the model, so entries written under the
    old budget are not valid answers under the new one. Simulates the 2026-08-03
    correction by putting GLM's constants back on the paddle profile.
    """
    from dataclasses import replace

    from bibr.ocr import profiles as ocr_profiles

    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline = ocr_cache._key(fs, cfg, _identity())

    monkeypatch.setitem(
        ocr_profiles._PROFILES,
        "paddle",
        replace(ocr_profiles.PADDLE_PROFILE, image=ocr_profiles.GLM_IMAGE_GEOMETRY),
    )

    assert baseline != ocr_cache._key(fs, cfg, _identity())


# --- model pins ---------------------------------------------------------------
#
# A complete entry lets the pipeline skip layout detection and OCR inference
# outright, so the pins selecting those weights decide its contents. Serving a
# re-pinned model against a stale entry replays the *old* model's regions,
# which also makes an A/B evaluation of the two report no difference at all.


def _settings_pair():
    from bibr.config import snapshot_settings

    baseline = snapshot_settings()
    return baseline, baseline.model_copy(deep=True)


def test_key_changes_with_the_layout_model_revision():
    fs = _fs()
    cfg = RunConfig(ocr_backend="glm-llama")
    baseline, repinned = _settings_pair()
    repinned.layout.model_revision = "0" * 40

    assert ocr_cache._key(fs, cfg, _identity(), baseline) != ocr_cache._key(
        fs, cfg, _identity(), repinned
    )


def test_key_changes_with_the_layout_model_id():
    """PP-DocLayoutV3 → V4 is a checkpoint swap, not only a re-pin."""
    fs = _fs()
    cfg = RunConfig(ocr_backend="glm-llama")
    baseline, other = _settings_pair()
    other.layout.model_id = "PaddlePaddle/PP-DocLayoutV4_safetensors"

    assert ocr_cache._key(fs, cfg, _identity(), baseline) != ocr_cache._key(
        fs, cfg, _identity(), other
    )


def test_key_changes_with_the_layout_onnx_bundle():
    """The ONNX runtime loads its own bundle; moving only that pair must miss too."""
    fs = _fs()
    cfg = RunConfig(ocr_backend="glm-llama")
    baseline, repinned = _settings_pair()
    repinned.layout.onnx_revision = "0" * 40
    _, moved = _settings_pair()
    moved.layout.onnx_model_id = "/local/layout-v4-bundle"

    keys = {
        ocr_cache._key(fs, cfg, _identity(), settings) for settings in (baseline, repinned, moved)
    }
    assert len(keys) == 3


def test_key_changes_with_the_ml_runtime():
    """Torch and ONNX may be pinned to different layout models (say V4 and V3)."""
    fs = _fs()
    cfg = RunConfig(ocr_backend="glm-llama")
    keys = set()
    for runtime in ("auto", "torch", "onnx"):
        _, settings = _settings_pair()
        settings.ml.runtime = runtime
        keys.add(ocr_cache._key(fs, cfg, _identity(), settings))
    assert len(keys) == 3


def test_key_changes_with_the_paddle_model_revision():
    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline, repinned = _settings_pair()
    repinned.ocr.paddle_revision = "0" * 40

    assert ocr_cache._key(fs, cfg, _identity(), baseline) != ocr_cache._key(
        fs, cfg, _identity(), repinned
    )


def test_key_changes_with_the_paddle_model_id():
    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline, other = _settings_pair()
    other.ocr.paddle_model = "PaddlePaddle/PaddleOCR-VL-9.9"

    assert ocr_cache._key(fs, cfg, _identity(), baseline) != ocr_cache._key(
        fs, cfg, _identity(), other
    )


def test_the_served_alias_does_not_hide_a_repin():
    """``identity.model`` is the alias vLLM is launched with, not the pin.

    ``--served-model-name paddle-ocr-vl-1.6`` stays put while ``--revision``
    changes, so the identity alone cannot tell the two runs apart.
    """
    fs = _fs()
    cfg = RunConfig(ocr_backend="serve-http", ocr_profile="paddle")
    baseline, repinned = _settings_pair()
    repinned.ocr.paddle_revision = "0" * 40
    alias = _identity(model="paddle-ocr-vl-1.6")

    assert alias.model == _identity().model
    assert ocr_cache._key(fs, cfg, alias, baseline) != ocr_cache._key(fs, cfg, alias, repinned)


def test_key_changes_with_the_bibr_version(monkeypatch):
    import bibr

    fs = _fs()
    cfg = RunConfig(ocr_backend="glm-llama")
    base = ocr_cache._key(fs, cfg, _identity())
    monkeypatch.setattr(bibr, "__version__", "99.0.0")
    assert ocr_cache._key(fs, cfg, _identity()) != base
