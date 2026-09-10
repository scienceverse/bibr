"""NativeTextStage — pre-fill layout regions from native PDF text.

These tests mock ``fill_regions_from_native_text`` so they don't depend on
pypdfium2 or any real PDF bytes.  They verify wiring:

* feature-flag gating
* skip files without layout_results / pdf_bytes
* helper is invoked and marks regions with ``_native_text_used``
* on helper failure, partial fills are cleared and no error is set on the
  FileState (the stage is a best-effort bypass, not a hard pipeline stage)
"""

import asyncio
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bibr.config import Settings
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.stages.native_text import NativeTextStage
from bibr.pipeline.state import FileState


def _ctx(file_states, config=None):
    return PipelineContext(
        file_states=file_states,
        progress=NullProgress(),
        resources=MagicMock(),
        config=config or RunConfig(),
    )


@pytest.fixture(autouse=True)
def _adapt_legacy_helper_mocks(monkeypatch):
    """Route legacy helper spies through the fused inspection boundary."""
    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.ocr.pdf_inspection import PdfInspection
    from bibr.ocr.ref_geometry import record_to_dict, recover_reference_lines

    monkeypatch.setattr(
        stage_mod, "recover_reference_lines", recover_reference_lines, raising=False
    )
    monkeypatch.setattr(stage_mod, "record_to_dict", record_to_dict, raising=False)

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

        layout = deepcopy(layout_results)
        if fill_native_text:
            layout = native_mod.fill_native_text_and_fonts(
                pdf_bytes,
                layout,
                min_chars=min_chars,
                min_printable_ratio=min_printable_ratio,
            )
        verification = "\n".join(str(r.get("content") or "") for r in (layout[0] if layout else []))
        metadata = metadata_mod.harvest_pdf_metadata(pdf_bytes, verification or first_page_text)
        outline = outline_mod.extract_pdf_outline(pdf_bytes) if include_outline else []
        reference_lines = []
        if include_ref_geometry:
            reference_lines = [
                stage_mod.record_to_dict(record)
                for record in stage_mod.recover_reference_lines(pdf_bytes)
            ]
        return PdfInspection((), layout, metadata, outline, reference_lines)

    monkeypatch.setattr(stage_mod, "inspect_pdf", adapter)


def test_stage_name():
    assert NativeTextStage().name == "native_text"


@pytest.mark.asyncio
async def test_stage_uses_one_inspection_for_all_pdf_components(monkeypatch):
    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.input.pdf_outline import OutlineItem
    from bibr.ocr.pdf_inspection import PdfInspection, PdfPageInspection

    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    monkeypatch.setattr(Settings.pipeline, "outline_headings", True)
    calls = []
    inspection = PdfInspection(
        pages=(PdfPageInspection(0, 600.0, 800.0, (0.0, 0.0, 600.0, 800.0), 10),),
        layout_results=[[{"label": "text", "content": "native", "_native_text_used": True}]],
        metadata={"title": "Embedded"},
        outline=[OutlineItem("Methods", 0, 1, 0.2)],
        reference_lines=[{"text": "Ref", "page": 0}],
    )

    def fake_inspect(pdf_bytes, layout_results, **kwargs):
        calls.append((pdf_bytes, layout_results, kwargs))
        return inspection

    monkeypatch.setattr(stage_mod, "inspect_pdf", fake_inspect)
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    fs.layout_results = [[{"label": "text", "content": ""}]]

    await NativeTextStage().run(_ctx([fs], config=RunConfig(ref_seg_strategy="geom")))

    assert len(calls) == 1
    assert calls[0][2]["fill_native_text"] is True
    assert calls[0][2]["include_outline"] is True
    assert calls[0][2]["include_ref_geometry"] is True
    assert fs.pdf_inspection is inspection
    assert fs.layout_results == inspection.layout_results
    assert fs.native_metadata == inspection.metadata
    assert fs.pdf_outline == inspection.outline
    assert fs.ref_line_geometry == inspection.reference_lines


@pytest.mark.asyncio
async def test_stage_passes_physical_page_indices_to_inspection(monkeypatch):
    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.ocr.pdf_inspection import PdfInspection

    captured = {}

    def fake_inspect(pdf_bytes, layout_results, **kwargs):
        captured.update(kwargs)
        return PdfInspection((), deepcopy(layout_results), {}, [], [])

    monkeypatch.setattr(stage_mod, "inspect_pdf", fake_inspect)
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    fs.page_indices = [1]
    fs.layout_results = [[{"label": "text", "content": ""}]]

    await NativeTextStage().run(_ctx([fs]))

    assert captured["page_indices"] == fs.page_indices


@pytest.mark.asyncio
async def test_fill_skipped_when_flag_disabled(monkeypatch):
    """With the fill flag off the native-text FILL is skipped (helper never
    called). The independent metadata/geom harvests still run — see
    ``test_metadata_and_geom_harvest_run_when_fill_disabled`` — so they are
    neutralized here to isolate the fill gate."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", False)

    import bibr.input.pdf_metadata as pdf_meta_mod
    import bibr.ocr.native_text as native_mod
    import bibr.pipeline.stages.native_text as stage_mod

    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", lambda *a, **k: {})
    monkeypatch.setattr(stage_mod, "recover_reference_lines", lambda _b: [])

    def _boom(*_a, **_k):
        raise AssertionError("fill helper must not be called when flag is off")

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _boom)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]

    await NativeTextStage().run(_ctx([fs]))

    # Fill was skipped: region untouched, no error.
    assert fs.error is None
    assert "_native_text_used" not in fs.layout_results[0][0]


@pytest.mark.asyncio
async def test_metadata_and_geom_harvest_run_when_fill_disabled(monkeypatch):
    """A1 regression: disabling the native-text FILL must NOT strand the
    independent doc-info metadata and geom reference-geometry harvests — they
    need only pdf_bytes, and geom feeds the DEFAULT ref-seg strategy. Before the
    fix the stage's top-level early return skipped all three."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", False)
    monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "geom", raising=False)

    import bibr.input.pdf_metadata as pdf_meta_mod
    import bibr.ocr.native_text as native_mod
    import bibr.pipeline.stages.native_text as stage_mod

    def _boom_fill(*_a, **_k):
        raise AssertionError("fill helper must not run when native_text_enabled is False")

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _boom_fill)

    harvested: dict = {}

    def _harvest(pdf_bytes, first_page_text):
        harvested["called"] = True
        return {"title": "T"}

    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", _harvest)

    captured: list[bytes] = []

    def _spy_recover(pdf_bytes):
        captured.append(pdf_bytes)
        return []

    monkeypatch.setattr(stage_mod, "recover_reference_lines", _spy_recover)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs], config=RunConfig()))

    assert harvested.get("called") is True  # metadata harvest ran
    assert captured == [b"%PDF-1.4"]  # geom geometry capture ran
    assert fs.native_metadata == {"title": "T"}
    assert fs.error is None


@pytest.mark.asyncio
async def test_skips_file_without_layout_results(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = None  # e.g. DOCX-native path
    ctx = _ctx([fs])

    import bibr.ocr.native_text as native_mod

    calls: list[tuple] = []

    def _spy(pdf_bytes, layout, *, min_chars, min_printable_ratio=0.85):
        calls.append((pdf_bytes, layout, min_chars))
        return layout

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _spy)

    await NativeTextStage().run(ctx)

    assert calls == []  # helper not called
    assert fs.error is None


@pytest.mark.asyncio
async def test_skips_file_without_pdf_bytes(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = None
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    ctx = _ctx([fs])

    import bibr.ocr.native_text as native_mod

    calls: list[tuple] = []

    def _spy(pdf_bytes, layout, *, min_chars, min_printable_ratio=0.85):
        calls.append((pdf_bytes, layout, min_chars))
        return layout

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _spy)

    await NativeTextStage().run(ctx)

    assert calls == []
    assert fs.error is None


@pytest.mark.asyncio
async def test_helper_marks_regions_filled(monkeypatch):
    """Stage wraps the helper and leaves its ``_native_text_used`` marks in place."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [
        [
            {"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""},
            {"label": "figure", "task_type": "figure", "content": ""},
        ]
    ]
    ctx = _ctx([fs])

    import bibr.ocr.native_text as native_mod

    def _fake_fill(pdf_bytes, layout, *, min_chars, min_printable_ratio=0.85):
        # Simulate helper: mark the text region as filled from native text.
        layout[0][0]["content"] = "hello world"
        layout[0][0]["_native_text_used"] = True
        return layout

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _fake_fill)

    await NativeTextStage().run(ctx)

    assert fs.layout_results[0][0]["_native_text_used"] is True
    assert fs.layout_results[0][0]["content"] == "hello world"
    # Non-text region untouched.
    assert "_native_text_used" not in fs.layout_results[0][1]
    assert fs.error is None


@pytest.mark.asyncio
async def test_helper_failure_clears_partial_fills(monkeypatch):
    """If the helper raises after a partial fill, the stage must wipe the marks
    so OcrStage runs over all regions (fall-back-to-full-OCR guarantee).
    It must NOT set fs.error — this stage is a best-effort bypass.
    """
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    # Simulate a partially-filled layout at the moment the helper raised.
    fs.layout_results = [
        [
            {
                "label": "text",
                "task_type": "text",
                "content": "partial",
                "_native_text_used": True,
            },
            {"label": "text", "task_type": "text", "content": ""},
        ]
    ]
    ctx = _ctx([fs])

    import bibr.ocr.native_text as native_mod

    def _boom(pdf_bytes, layout, *, min_chars, min_printable_ratio=0.85):
        raise RuntimeError("native text blew up mid-fill")

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _boom)

    await NativeTextStage().run(ctx)

    # Stage is best-effort: no fs.error.
    assert fs.error is None
    # Partial fill cleared so OCR will re-process the region.
    region = fs.layout_results[0][0]
    assert "_native_text_used" not in region
    assert region["content"] == ""


@pytest.mark.asyncio
async def test_errored_files_are_skipped(monkeypatch):
    """Files with ``error`` set prior to the stage must not be touched."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    fs_err = FileState(path=Path("bad.pdf"))
    fs_err.pdf_bytes = b"%PDF-1.4"
    fs_err.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    fs_err.set_error("earlier failure", code="layout_failed", stage="layout")

    fs_ok = FileState(path=Path("ok.pdf"))
    fs_ok.pdf_bytes = b"%PDF-1.4"
    fs_ok.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]

    ctx = _ctx([fs_err, fs_ok])

    import bibr.ocr.native_text as native_mod

    seen_paths: list[str] = []

    def _spy(pdf_bytes, layout, *, min_chars, min_printable_ratio=0.85):
        # We don't have the filename in the helper, but we can assert count.
        seen_paths.append("called")
        layout[0][0]["_native_text_used"] = True
        layout[0][0]["content"] = "ok"
        return layout

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _spy)

    await NativeTextStage().run(ctx)

    # Helper called exactly once — only for the alive file.
    assert len(seen_paths) == 1
    assert fs_ok.layout_results[0][0]["_native_text_used"] is True
    # Errored file is left alone.
    assert "_native_text_used" not in fs_err.layout_results[0][0]


@pytest.mark.asyncio
async def test_sync_helpers_run_off_the_event_loop(monkeypatch):
    """A slow pdfium fill must not stall the event loop — in serve mode one
    worker's loop carries every in-flight request, so blocking here stalls
    all of them."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    import bibr.ocr.native_text as native_mod

    def _slow_fill(pdf_bytes, layout, *, min_chars, min_printable_ratio=0.85):
        time.sleep(0.25)
        return layout

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _slow_fill)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    ctx = _ctx([fs])

    ticks = 0

    async def _heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task start
    await NativeTextStage().run(ctx)
    hb.cancel()

    # Blocking implementation: the loop never runs the heartbeat during the
    # 0.25s fill → 0 ticks. Off-loop implementation: ~20 ticks.
    assert ticks >= 5
    assert fs.error is None


@pytest.mark.asyncio
async def test_docinfo_metadata_harvested(monkeypatch):
    """The stage harvests guarded doc-info metadata onto fs.native_metadata,
    passing the filled first-page native text as verification context."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    import bibr.input.pdf_metadata as pdf_meta_mod
    import bibr.ocr.native_text as native_mod

    def _fill(pdf_bytes, layout_results, min_chars, min_printable_ratio=0.85):
        for r in layout_results[0]:
            r["content"] = "Attention Is All You Need"
            r["_native_text_used"] = True
        return layout_results

    seen = {}

    def _harvest(pdf_bytes, first_page_text):
        seen["pdf_bytes"] = pdf_bytes
        seen["first_page_text"] = first_page_text
        return {"title": "Attention Is All You Need"}

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _fill)
    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", _harvest)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs]))

    assert fs.native_metadata == {"title": "Attention Is All You Need"}
    assert seen["pdf_bytes"] == b"%PDF-1.4"
    assert "Attention Is All You Need" in seen["first_page_text"]


@pytest.mark.asyncio
async def test_single_pdf_open_for_text_and_font_passes(monkeypatch):
    """The combined native-text fill + font-metadata path must open the PDF
    exactly once (one document, one textpage per page serving both passes),
    not once per pass. Runs the real helper against a native fixture PDF and
    counts ``pypdfium2.PdfDocument`` constructions during the stage run.

    The doc-info harvest and geom ref-geometry passes legitimately open the PDF
    on their own (out of scope for this win), so they are isolated out here.
    """
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    # Isolate the fill+font path: neutralize the other pdfium-opening passes.
    monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "ner", raising=False)
    import bibr.input.pdf_metadata as pdf_meta_mod

    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", lambda *a, **k: {})

    import pypdfium2

    real_ctor = pypdfium2.PdfDocument
    opens = 0

    def _counting_ctor(*a, **k):
        nonlocal opens
        opens += 1
        return real_ctor(*a, **k)

    monkeypatch.setattr(pypdfium2, "PdfDocument", _counting_ctor)

    fixture = Path(__file__).parent.parent / "fixtures" / "native_text_sample.pdf"
    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = fixture.read_bytes()
    fs.layout_results = [
        [{"label": "text", "task_type": "text", "bbox_2d": [0, 0, 1000, 1000], "content": ""}]
    ]

    await NativeTextStage().run(_ctx([fs]))

    assert opens == 1, f"expected a single PDF open for text+font passes, got {opens}"
    # Sanity: both passes ran off that single open — the text fill marked the
    # region and the font pass attached page geometry (attached to every region
    # regardless of measurable glyphs).
    region = fs.layout_results[0][0]
    assert region["_native_text_used"] is True
    assert "_page_w" in region and "_page_h" in region


def _geom_capture_stub(monkeypatch):
    """Neutralize the fill/harvest passes and spy on the ref-geometry capture.

    Returns the list that records each ``recover_reference_lines`` call so a
    test can assert whether geometry capture ran for the effective strategy.
    """
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)
    import bibr.input.pdf_metadata as pdf_meta_mod
    import bibr.ocr.native_text as native_mod
    import bibr.pipeline.stages.native_text as stage_mod

    monkeypatch.setattr(
        native_mod,
        "fill_native_text_and_fonts",
        lambda b, lr, min_chars, min_printable_ratio=0.85: lr,
    )
    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", lambda *a, **k: {})

    captured: list[bytes] = []

    def _spy_recover(pdf_bytes):
        captured.append(pdf_bytes)
        return []

    monkeypatch.setattr(stage_mod, "recover_reference_lines", _spy_recover)
    return captured


@pytest.mark.asyncio
async def test_geometry_capture_honors_per_run_geom_override(monkeypatch):
    """A per-run ``--ref-seg geom`` override must enable geometry capture even
    when the global Settings strategy isn't geom — otherwise the override's
    geom tier always declines ('no ref-line geometry')."""
    captured = _geom_capture_stub(monkeypatch)
    monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "region", raising=False)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs], config=RunConfig(ref_seg_strategy="geom")))

    assert captured == [b"%PDF-1.4"]


@pytest.mark.asyncio
async def test_geometry_capture_skipped_when_override_not_geom(monkeypatch):
    """A per-run override away from geom must skip the geometry capture even
    when the global strategy is geom — no wasted pdfium pass."""
    captured = _geom_capture_stub(monkeypatch)
    monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "geom", raising=False)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs], config=RunConfig(ref_seg_strategy="region")))

    assert captured == []


@pytest.mark.asyncio
async def test_geometry_capture_default_global_geom_no_override(monkeypatch):
    """With no per-run override and the global default (geom), capture runs."""
    captured = _geom_capture_stub(monkeypatch)
    monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "geom", raising=False)

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs], config=RunConfig()))

    assert captured == [b"%PDF-1.4"]


@pytest.mark.asyncio
async def test_geometry_capture_runs_when_native_text_fill_disabled(monkeypatch):
    """Disabling the native-text FILL must NOT strip the geom segmenter's
    reference-line geometry. geom is the DEFAULT ref-seg strategy, and its
    geometry harvest needs only pdf_bytes — so it runs even when
    ``native_text_enabled`` is False."""
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", False)
    monkeypatch.setattr(Settings, "REF_SEG_STRATEGY", "geom", raising=False)

    import bibr.input.pdf_metadata as pdf_meta_mod
    import bibr.ocr.native_text as native_mod
    import bibr.pipeline.stages.native_text as stage_mod

    # The fill helper must NOT be called when the flag is off.
    def _boom(*_a, **_k):
        raise AssertionError("native-text fill must not run when the flag is off")

    monkeypatch.setattr(native_mod, "fill_native_text_and_fonts", _boom)
    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", lambda *a, **k: {})

    known_record = object()
    monkeypatch.setattr(stage_mod, "recover_reference_lines", lambda pdf_bytes: [known_record])
    monkeypatch.setattr(stage_mod, "record_to_dict", lambda rec: {"y0": 0.5, "text": "[1] ref"})

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs], config=RunConfig()))

    # geom geometry harvested despite the fill being disabled.
    assert fs.ref_line_geometry == [{"y0": 0.5, "text": "[1] ref"}]
    # The fill was genuinely skipped (region left untouched).
    assert "_native_text_used" not in fs.layout_results[0][0]
    assert fs.error is None


@pytest.mark.asyncio
async def test_docinfo_harvest_empty_result_stays_none(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "native_text_enabled", True)

    import bibr.input.pdf_metadata as pdf_meta_mod
    import bibr.ocr.native_text as native_mod

    monkeypatch.setattr(
        native_mod,
        "fill_native_text_and_fonts",
        lambda b, lr, min_chars, min_printable_ratio=0.85: lr,
    )
    monkeypatch.setattr(pdf_meta_mod, "harvest_pdf_metadata", lambda *a: {})

    fs = FileState(path=Path("a.pdf"))
    fs.pdf_bytes = b"%PDF-1.4"
    fs.layout_results = [[{"label": "text", "task_type": "text", "content": ""}]]
    await NativeTextStage().run(_ctx([fs]))

    assert fs.native_metadata is None
