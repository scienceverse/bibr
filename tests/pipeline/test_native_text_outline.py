"""NativeTextStage — PDF-outline (bookmarks) harvest wiring.

The outline harvest is dark by default: only when
``Settings.pipeline.outline_headings`` is on does the stage read the PDF
outline and stash it on ``FileState.pdf_outline`` for the parser. These tests
mock ``extract_pdf_outline`` so they don't need a bookmarked PDF.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.config import Settings
from bibr.input.pdf_outline import OutlineItem
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.native_text import NativeTextStage
from bibr.pipeline.state import FileState


@pytest.fixture(autouse=True)
def _adapt_outline_mock(monkeypatch):
    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.ocr.pdf_inspection import PdfInspection

    def adapter(
        pdf_bytes,
        layout_results,
        *,
        page_indices,
        first_page_text,
        fill_native_text,
        include_outline,
        include_ref_geometry,
        min_chars,
        min_printable_ratio,
    ):
        import bibr.input.pdf_metadata as metadata_mod
        import bibr.input.pdf_outline as outline_mod
        import bibr.ocr.native_text as native_mod

        layout = layout_results
        if fill_native_text:
            layout = native_mod.fill_native_text_and_fonts(
                pdf_bytes,
                layout,
                min_chars=min_chars,
                min_printable_ratio=min_printable_ratio,
            )
        metadata = metadata_mod.harvest_pdf_metadata(pdf_bytes, first_page_text)
        outline = outline_mod.extract_pdf_outline(pdf_bytes) if include_outline else []
        return PdfInspection((), layout, metadata, outline, [])

    monkeypatch.setattr(stage_mod, "inspect_pdf", adapter)


def _ctx(file_states):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(),
    )


def _fs():
    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    return fs


@pytest.mark.asyncio
async def test_outline_not_harvested_when_flag_off(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.pipeline, "outline_headings", False)

    # Neutralize the native-fill + metadata helpers so the stage runs cleanly.
    import bibr.ocr.native_text as native_mod

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", lambda pdf, layout, **_k: layout)
    monkeypatch.setattr("bibr.input.pdf_metadata.harvest_pdf_metadata", lambda *_a, **_k: {})

    import bibr.input.pdf_outline as outline_mod

    def _boom(*_a, **_k):
        raise AssertionError("extract_pdf_outline must not run when the flag is off")

    monkeypatch.setattr(outline_mod, "extract_pdf_outline", _boom)

    fs = _fs()
    await NativeTextStage().run(_ctx([fs]))

    assert fs.pdf_outline is None
    assert fs.error is None


@pytest.mark.asyncio
async def test_outline_harvested_when_flag_on(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.pipeline, "outline_headings", True)

    import bibr.ocr.native_text as native_mod

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", lambda pdf, layout, **_k: layout)
    monkeypatch.setattr("bibr.input.pdf_metadata.harvest_pdf_metadata", lambda *_a, **_k: {})

    items = [OutlineItem(title="Introduction", level=0, page_no=1)]
    import bibr.input.pdf_outline as outline_mod

    monkeypatch.setattr(outline_mod, "extract_pdf_outline", lambda _b: items)

    fs = _fs()
    await NativeTextStage().run(_ctx([fs]))

    assert fs.pdf_outline == items
    assert fs.error is None


@pytest.mark.asyncio
async def test_empty_outline_stored_as_none(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.pipeline, "outline_headings", True)

    import bibr.ocr.native_text as native_mod

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", lambda pdf, layout, **_k: layout)
    monkeypatch.setattr("bibr.input.pdf_metadata.harvest_pdf_metadata", lambda *_a, **_k: {})
    monkeypatch.setattr("bibr.input.pdf_outline.extract_pdf_outline", lambda _b: [])

    fs = _fs()
    await NativeTextStage().run(_ctx([fs]))

    # An outline-less PDF yields [] -> normalized to None (feature stays inert).
    assert fs.pdf_outline is None
