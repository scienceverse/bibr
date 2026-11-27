"""Wave 4A tests: text-layer render DPI override, cache-key isolation, and RGB render parity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pypdfium2
import pytest

from bibr.config import GlobalSettings
from bibr.ocr.image_utils import iter_pdf_pages_with_index
from bibr.ocr.profiles import OcrRuntimeIdentity
from bibr.ocr.registry import TEXT_LAYER_BACKEND
from bibr.pipeline import ocr_cache
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.layout import LayoutStage, effective_render_dpi
from bibr.pipeline.state import FileState


def _ctx(
    file_states: list[FileState],
    *,
    config: RunConfig | None = None,
    settings: GlobalSettings | None = None,
    resources: object | None = None,
    scratch: dict | None = None,
) -> PipelineContext:
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=resources or MagicMock(),
        config=config or RunConfig(),
        settings=settings or GlobalSettings(),
        scratch=scratch or {},
    )


def test_effective_render_dpi_defaults_to_100_in_text_layer_mode_without_figures() -> None:
    settings = GlobalSettings()
    assert settings.layout.dpi == 200
    assert settings.layout.text_layer_dpi == 100

    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    assert (
        effective_render_dpi(
            _ctx([fs], config=RunConfig(ocr_backend="text-layer"), settings=settings)
        )
        == 100
    )
    assert (
        effective_render_dpi(
            _ctx([fs], config=RunConfig(ocr_backend="glm-llama"), settings=settings)
        )
        == 200
    )


def test_effective_render_dpi_preserves_full_dpi_when_figures_or_consensus_active() -> None:
    settings = GlobalSettings()
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")

    # Explicit include_figures=True keeps full LAYOUT_DPI so figure crops stay sharp.
    assert (
        effective_render_dpi(
            _ctx(
                [fs],
                config=RunConfig(ocr_backend="text-layer", include_figures=True),
                settings=settings,
            )
        )
        == 200
    )

    # Global FIGURE_IMAGES=True also keeps full LAYOUT_DPI when RunConfig leaves it unset.
    fig_settings = GlobalSettings()
    fig_settings.FIGURE_IMAGES = True
    assert (
        effective_render_dpi(
            _ctx([fs], config=RunConfig(ocr_backend="text-layer"), settings=fig_settings)
        )
        == 200
    )

    # Consensus backend or active scan_recognizers keeps full LAYOUT_DPI.
    cons_settings = GlobalSettings()
    cons_settings.ocr.consensus_backend = "glm-http"
    assert (
        effective_render_dpi(
            _ctx([fs], config=RunConfig(ocr_backend="text-layer"), settings=cons_settings)
        )
        == 200
    )
    assert (
        effective_render_dpi(
            _ctx(
                [fs],
                config=RunConfig(ocr_backend="text-layer"),
                settings=settings,
                scratch={"scan_recognizers": (object(), None)},
            )
        )
        == 200
    )


def test_effective_render_dpi_respects_null_and_caps_at_layout_dpi() -> None:
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")

    # text_layer_dpi=None disables the override.
    s_none = GlobalSettings()
    s_none.layout.text_layer_dpi = None
    assert (
        effective_render_dpi(
            _ctx([fs], config=RunConfig(ocr_backend="text-layer"), settings=s_none)
        )
        == 200
    )

    # Never upscales past layout.dpi when layout.dpi is lower than text_layer_dpi.
    s_low = GlobalSettings()
    s_low.layout.dpi = 72
    s_low.layout.text_layer_dpi = 100
    assert (
        effective_render_dpi(_ctx([fs], config=RunConfig(ocr_backend="text-layer"), settings=s_low))
        == 72
    )

    # Auto fallback to text-layer also uses text_layer_dpi.
    res = MagicMock()
    res.ocr_text_layer_fallback = "no local OCR runtime"
    assert (
        effective_render_dpi(
            _ctx([fs], config=RunConfig(), settings=GlobalSettings(), resources=res)
        )
        == 100
    )


@pytest.mark.asyncio
async def test_layout_stage_passes_text_layer_dpi_to_renderer() -> None:
    fs = FileState(path=Path("x.pdf"), pdf_bytes=b"%PDF")
    rm = MagicMock()
    rm.layout = MagicMock(detect_batch=AsyncMock(return_value=[[]]))
    seen_dpi: list[int] = []

    def fake_render(pdf_bytes, dpi, start_page, end_page, max_pixels, max_dimension, **_kw):
        seen_dpi.append(dpi)
        return iter([(0, MagicMock(width=827, height=1170))])

    with patch("bibr.pipeline.stages.layout._iter_pdf_pages", side_effect=fake_render):
        await LayoutStage().run(
            _ctx([fs], config=RunConfig(ocr_backend="text-layer"), resources=rm)
        )

    assert seen_dpi == [100]


def test_ocr_cache_key_distinguishes_text_layer_dpi_without_affecting_ocr_backends() -> None:
    fs = FileState(path=Path("paper.pdf"))
    fs.content_sha256 = "0" * 64
    tl_identity = OcrRuntimeIdentity(
        backend=TEXT_LAYER_BACKEND,
        model="text-layer",
        profile=None,
        normalizer_version="text-layer-v1",
    )
    ocr_identity = OcrRuntimeIdentity(
        backend="glm-llama",
        model="THUDM/GLM-OCR",
        profile="glm",
        normalizer_version="glm-canonical-v1",
    )
    s100 = GlobalSettings()
    s100.layout.text_layer_dpi = 100
    s200 = GlobalSettings()
    s200.layout.text_layer_dpi = 200

    # Text-layer without figures: cache key reflects text_layer_dpi.
    cfg_tl = RunConfig(ocr_backend=TEXT_LAYER_BACKEND, include_figures=False)
    assert ocr_cache._key(fs, cfg_tl, tl_identity, s100) != ocr_cache._key(
        fs, cfg_tl, tl_identity, s200
    )

    # Text-layer WITH figures: both use layout.dpi=200, so keys match.
    cfg_tl_fig = RunConfig(ocr_backend=TEXT_LAYER_BACKEND, include_figures=True)
    assert ocr_cache._key(fs, cfg_tl_fig, tl_identity, s100) == ocr_cache._key(
        fs, cfg_tl_fig, tl_identity, s200
    )

    # Non-text-layer OCR backends: text_layer_dpi has no effect on the key.
    cfg_ocr = RunConfig(ocr_backend="glm-llama", include_figures=False)
    assert ocr_cache._key(fs, cfg_ocr, ocr_identity, s100) == ocr_cache._key(
        fs, cfg_ocr, ocr_identity, s200
    )


def test_rev_byteorder_render_is_bit_identical_to_bgr_render() -> None:
    pdf_bytes = Path("tests/fixtures/native_text_sample.pdf").read_bytes()
    fast_imgs = [img for _, img in iter_pdf_pages_with_index(pdf_bytes, dpi=100)]

    doc = pypdfium2.PdfDocument(pdf_bytes)
    try:
        ref_imgs = []
        for idx in range(len(doc)):
            page = doc[idx]
            try:
                bm = page.render(scale=100 / 72, rev_byteorder=False)
                try:
                    ref_imgs.append(bm.to_pil())
                finally:
                    bm.close()
            finally:
                page.close()
    finally:
        doc.close()

    assert len(fast_imgs) == len(ref_imgs) and len(fast_imgs) > 0
    for fast, ref in zip(fast_imgs, ref_imgs, strict=True):
        assert fast.mode == ref.mode == "RGB"
        assert fast.size == ref.size
        assert np.array_equal(np.asarray(fast), np.asarray(ref))
