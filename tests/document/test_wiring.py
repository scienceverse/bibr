"""The document layer's wiring: off by default, kept out of the exports, and
never able to fail a paper."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import logging
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibr.config import GlobalSettings, Settings
from bibr.document import rebuild as rebuild_mod
from bibr.document import serialize
from bibr.document.rebuild import (
    attach_blocks,
    ensure_document_layer,
    layout_page_range,
    render_budget,
)
from bibr.ocr.pdf_inspection import PdfInspection, inspect_pdf
from bibr.ocr.types import OcrRegionResult
from bibr.paper_contents import PaperContents
from bibr.pipeline.context import PipelineContext, RunConfig
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.state import FileState
from tests.document import _pdfs

_REPO = Path(__file__).resolve().parents[2]


# --- bibr.export never sees the layer ---------------------------------------------


def _imported_modules(path: Path, tree: ast.AST) -> list[tuple[int, str]]:
    """``(line, absolute module)`` of every import, relative ones resolved."""
    # The package a relative import starts from (for __init__.py, the package itself).
    package = list(path.relative_to(_REPO).with_suffix("").parts)[:-1]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.append((node.lineno, module))
            found.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names)
    return found


def test_export_code_never_imports_the_document_layer():
    offenders = []
    for path in sorted((_REPO / "bibr" / "export").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, module in _imported_modules(path, tree):
            if module == "bibr.document" or module.startswith("bibr.document."):
                offenders.append(f"{path.relative_to(_REPO)}:{line}")
    assert offenders == []


def test_importing_every_export_module_leaves_the_document_layer_unloaded():
    code = (
        "import importlib, pkgutil, sys\n"
        "import bibr.export as export\n"
        "names = [info.name for info in pkgutil.walk_packages(export.__path__, 'bibr.export.')]\n"
        "for name in names:\n"
        "    importlib.import_module(name)\n"
        "print(len(names))\n"
        "print(sorted(name for name in sys.modules if name.startswith('bibr.document')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=_REPO
    )
    imported, loaded = result.stdout.strip().splitlines()
    assert int(imported) >= 5
    assert loaded == "[]"


# --- containers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "name"),
    [(FileState, "doc_layer"), (PaperContents, "document"), (PdfInspection, "document")],
)
def test_layer_fields_stay_out_of_equality_and_repr(cls, name):
    found = {item.name: item for item in dataclasses.fields(cls)}[name]

    assert found.default is None
    assert found.compare is False
    assert found.repr is False


def test_the_cached_page_inspection_keeps_its_fields():
    # OCR bundles store PdfPageInspection, and the bundle reader deletes any
    # entry it cannot decode: the layer rides on PdfInspection, never here.
    from bibr.ocr.pdf_inspection import PdfPageInspection

    assert [item.name for item in dataclasses.fields(PdfPageInspection)] == [
        "index",
        "width",
        "height",
        "crop_box",
        "char_count",
        "invisible_text_layer",
        "watermarks",
    ]


def test_file_state_ignores_its_layer_and_frees_it():
    first, second = FileState(path=Path("a.pdf")), FileState(path=Path("a.pdf"))
    second.doc_layer = object()

    assert first == second
    assert repr(first) == repr(second)
    second.free_all()
    assert second.doc_layer is None


def _layer_with_blocks():
    pdf_bytes = _pdfs.synthetic_paper()
    inspection = _inspect(pdf_bytes, _pdfs.band_layout(len(_pdfs.SYNTHETIC_TEXT_SOURCES)))
    attach_blocks(inspection.document, _regions(inspection.layout_results))
    return inspection.document


@pytest.mark.parametrize("mode", ["aggressive", "balanced", "keep_all"])
def test_every_memory_mode_frees_only_the_columns_after_post_parse(mode):
    layer = _layer_with_blocks()
    expected = serialize.to_dict(layer)
    for page in expected["pages"]:
        page["cols"] = None
    expected["columns_freed"] = True
    fs = FileState(path=Path("a.pdf"))
    fs.doc_layer = layer
    fs.contents = MagicMock()
    fs.contents.document = layer
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(memory_mode=mode),
    )

    # ParseSegment attaches the blocks and PostParse may read the columns.
    ctx.free_after_stage("parse")
    assert not layer.columns_freed
    assert any(page.cols is not None for page in layer.pages)

    ctx.free_after_stage("extract")
    assert fs.doc_layer is layer
    assert fs.contents.document is layer
    assert layer.nbytes == 0
    # Blocks and their lines, furniture, roles, fonts and presence stay.
    assert serialize.to_dict(layer) == expected

    fs.free_all()
    assert fs.doc_layer is None
    assert fs.contents is None


@pytest.mark.parametrize("mode", ["local", "serve"])
def test_the_columns_are_freed_after_the_last_stage_that_requires_the_layer(mode):
    from bibr.pipeline.context import LAYER_COLUMNS_FREED_AFTER
    from bibr.pipeline.plans import build_stage_plan

    stages = build_stage_plan(mode=mode, stream_backhalf=False, enrichers=[])
    requiring = [stage.name for stage in stages if "doc_layer" in getattr(stage, "requires", ())]

    assert requiring[-1:] == [LAYER_COLUMNS_FREED_AFTER]


@pytest.mark.parametrize("stand_in", [object(), "parsed text"])
def test_freeing_the_columns_leaves_stand_ins_alone(stand_in):
    # The streaming back-half tests stand plain objects in for PaperContents,
    # and the free runs for every file whether or not it has a layer.
    fs = FileState(path=Path("a.pdf"))
    fs.contents = stand_in
    fs.doc_layer = stand_in
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=MagicMock(),
        config=RunConfig(memory_mode="balanced"),
    )

    ctx.free_after_stage("extract")

    assert fs.contents is stand_in
    assert fs.doc_layer is stand_in


def test_paper_contents_takes_the_layer_by_keyword_only():
    fields = dataclasses.fields(PaperContents)
    positional = [item.name for item in fields if not item.kw_only]

    assert {item.name: item for item in fields}["document"].kw_only
    assert "document" not in positional
    assert positional[-1] == "caption_assignment_receipt"


# --- NativeTextStage and ParseSegmentStage ------------------------------------------


@pytest.mark.parametrize("on", [False, True])
async def test_native_text_stage_asks_for_the_layer_only_when_on(monkeypatch, on):
    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.pipeline.stages.native_text import NativeTextStage

    monkeypatch.setattr(Settings.pipeline, "document_layer", on)
    calls = []
    layer = object()

    def fake_inspect(pdf_bytes, layout_results, **kwargs):
        calls.append(kwargs)
        return PdfInspection((), deepcopy(layout_results), {}, [], [], document=layer)

    monkeypatch.setattr(stage_mod, "inspect_pdf", fake_inspect)
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    fs.layout_results = [[{"label": "text", "content": ""}]]
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=MagicMock(), config=RunConfig()
    )

    await NativeTextStage().run(ctx)

    assert len(calls) == 1
    assert fs.doc_layer_attempted is on
    if on:
        assert calls[0]["include_doc_layer"] is True
        assert calls[0]["render_budget"] == render_budget(ctx.settings)
        assert fs.doc_layer is layer
    else:
        assert "include_doc_layer" not in calls[0]
        assert "render_budget" not in calls[0]
        assert fs.doc_layer is None


@pytest.mark.parametrize("raises", [False, True])
async def test_a_failed_inline_build_is_not_rebuilt(monkeypatch, raises):
    import bibr.pipeline.stages.native_text as stage_mod
    from bibr.pipeline.stages.native_text import NativeTextStage

    monkeypatch.setattr(Settings.pipeline, "document_layer", True)

    def failed_inspect(pdf_bytes, layout_results, **kwargs):
        if raises:
            raise RuntimeError("the PDF does not open")
        # The layer failed to finish: the inspection carries no document.
        return PdfInspection((), deepcopy(layout_results), {}, [], [])

    def no_rebuild(*_args, **_kwargs):
        raise AssertionError("a failed inline build must not be rebuilt")

    monkeypatch.setattr(stage_mod, "inspect_pdf", failed_inspect)
    monkeypatch.setattr(rebuild_mod, "rebuild_document_layer", no_rebuild)
    fs = FileState(path=Path("paper.pdf"), pdf_bytes=b"%PDF")
    fs.layout_results = [[{"label": "text", "content": ""}]]
    ctx = PipelineContext(
        file_states=[fs], progress=NullProgress(), resources=MagicMock(), config=RunConfig()
    )

    await NativeTextStage().run(ctx)

    assert fs.doc_layer is None
    assert fs.doc_layer_attempted is True
    assert ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None) is None


def _parse_context(fs: FileState, contents):
    parser = MagicMock()
    parser.parse.return_value = contents
    parser._deferred_texts = []
    resources = MagicMock()
    resources.segmenter = MagicMock(segment_batch=MagicMock(return_value=[]))
    ctx = PipelineContext(
        file_states=[fs],
        progress=NullProgress(),
        resources=resources,
        config=RunConfig(start_page=2, end_page=5),
    )
    return ctx, parser


@pytest.mark.parametrize("on", [False, True])
async def test_parse_stage_keeps_the_layer_only_when_on(monkeypatch, on):
    from bibr.pipeline.stages.parse_segment import ParseSegmentStage

    monkeypatch.setattr(Settings.pipeline, "document_layer", on)
    calls = []
    layer = object()

    def fake_ensure(fs, settings, *, start_page, end_page):
        calls.append((fs, start_page, end_page))
        return layer

    monkeypatch.setattr(rebuild_mod, "ensure_document_layer", fake_ensure)
    fs = FileState(path=Path("x.pdf"))
    fs.ocr_regions = [[{"content": "hello world.", "task_type": "text"}]]
    contents = MagicMock()
    contents.document = None
    ctx, parser = _parse_context(fs, contents)

    with patch("bibr.structure.pdf_parser.PDFParser", return_value=parser):
        await ParseSegmentStage().run(ctx)

    assert fs.error is None
    if on:
        assert calls == [(fs, 2, 5)]
        assert contents.document is layer
    else:
        assert calls == []
        assert contents.document is None


async def test_a_layer_failure_never_fails_the_parse(monkeypatch, caplog):
    from bibr.pipeline.stages.parse_segment import ParseSegmentStage

    monkeypatch.setattr(Settings.pipeline, "document_layer", True)

    def broken(*_args, **_kwargs):
        raise RuntimeError("layer failed")

    monkeypatch.setattr(rebuild_mod, "ensure_document_layer", broken)
    fs = FileState(path=Path("x.pdf"))
    fs.ocr_regions = [[{"content": "hello world.", "task_type": "text"}]]
    contents = MagicMock()
    contents.document = None
    ctx, parser = _parse_context(fs, contents)

    with (
        caplog.at_level(logging.WARNING),
        patch("bibr.structure.pdf_parser.PDFParser", return_value=parser),
    ):
        await ParseSegmentStage().run(ctx)

    assert fs.error is None
    assert fs.contents is contents
    assert contents.document is None
    assert "Document layer failed" in caplog.text


# --- ensure_document_layer ------------------------------------------------------------


def _inspect(pdf_bytes: bytes, layout, **kwargs):
    return inspect_pdf(
        pdf_bytes,
        deepcopy(layout),
        fill_native_text=True,
        include_outline=False,
        include_ref_geometry=True,
        min_chars=3,
        min_printable_ratio=0.85,
        include_doc_layer=True,
        render_budget=render_budget(GlobalSettings()),
        **kwargs,
    )


def _regions(layout_results: list[list[dict]]) -> list[list[OcrRegionResult]]:
    return [
        [
            OcrRegionResult.from_layout_region(region, slot_idx=slot, content=region["content"])
            for slot, region in enumerate(page)
        ]
        for page in layout_results
    ]


def _bundle_hit(directory: Path, pdf_bytes: bytes) -> FileState:
    """A file whose OCR came from a bundle: no inline layer, PDF bytes freed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "paper.pdf"
    path.write_bytes(pdf_bytes)
    fs = FileState(path=path)
    fs.content_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    return fs


def test_a_bundle_hit_rebuilds_the_layer_the_native_stage_builds(tmp_path):
    pdf_bytes = _pdfs.synthetic_paper()
    inspection = _inspect(pdf_bytes, _pdfs.band_layout(len(_pdfs.SYNTHETIC_TEXT_SOURCES)))
    regions = _regions(inspection.layout_results)
    attach_blocks(inspection.document, regions)
    fs = _bundle_hit(tmp_path, pdf_bytes)
    fs.ocr_regions = regions

    layer = ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None)

    assert layer is fs.doc_layer
    assert serialize.canonical_bytes(layer) == serialize.canonical_bytes(inspection.document)


def test_a_bundle_hit_with_a_page_range_rebuilds_those_pages(tmp_path):
    pdf_bytes = _pdfs.synthetic_paper()
    pages = [2, 3, 4]
    inspection = _inspect(pdf_bytes, _pdfs.band_layout(len(pages)), page_indices=pages)
    # OcrStage pads the pages before start_page with empty lists.
    regions = [[], [], *_regions(inspection.layout_results)]
    attach_blocks(inspection.document, regions)
    fs = _bundle_hit(tmp_path, pdf_bytes)
    fs.ocr_regions = regions

    layer = ensure_document_layer(fs, GlobalSettings(), start_page=2, end_page=4)

    assert [page.index for page in layer.pages] == pages
    assert serialize.digest(layer) == serialize.digest(inspection.document)


def test_an_inline_layer_is_kept_and_given_blocks(tmp_path, monkeypatch):
    pdf_bytes = _pdfs.synthetic_paper()
    inspection = _inspect(pdf_bytes, _pdfs.band_layout(len(_pdfs.SYNTHETIC_TEXT_SOURCES)))
    fs = FileState(path=tmp_path / "missing.pdf")
    fs.doc_layer = inspection.document
    fs.ocr_regions = _regions(inspection.layout_results)

    def no_rebuild(*_args, **_kwargs):
        raise AssertionError("the inline layer must be reused")

    monkeypatch.setattr(rebuild_mod, "rebuild_document_layer", no_rebuild)
    layer = ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None)

    assert layer is inspection.document
    assert layer.page(0).blocks
    assert layer.page(0).blocks[0].block_id == "p0.r0"


def test_a_layer_whose_columns_were_freed_comes_back_as_it_is(tmp_path, monkeypatch):
    layer = _layer_with_blocks()
    lines = [block.lines for page in layer.pages for block in page.blocks]
    assert any(lines)
    layer.free_columns()
    fs = FileState(path=tmp_path / "missing.pdf")
    fs.doc_layer = layer
    fs.doc_layer_attempted = True
    fs.ocr_regions = [[] for _page in layer.pages]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a freed layer is neither rebuilt nor given new blocks")

    monkeypatch.setattr(rebuild_mod, "rebuild_document_layer", unexpected)
    monkeypatch.setattr(rebuild_mod, "attach_blocks", unexpected)

    assert ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None) is layer
    assert layer.columns_freed
    assert [block.lines for page in layer.pages for block in page.blocks] == lines


def test_no_layer_without_the_processed_pdf(tmp_path):
    not_pdf = _bundle_hit(tmp_path / "text", b"plain text input")
    changed = _bundle_hit(tmp_path / "changed", _pdfs.synthetic_paper())
    changed.path.write_bytes(_pdfs.synthetic_paper() + b"\n% edited")
    never_hashed = FileState(path=changed.path)

    for fs in (not_pdf, changed, never_hashed):
        assert ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None) is None
        assert fs.doc_layer is None


def test_a_rebuild_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    fs = _bundle_hit(tmp_path, _pdfs.synthetic_paper())
    calls = []

    def broken(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(rebuild_mod, "rebuild_document_layer", broken)
    with caplog.at_level(logging.WARNING):
        result = ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None)
        again = ensure_document_layer(fs, GlobalSettings(), start_page=None, end_page=None)

    assert result is None
    assert again is None
    assert calls == [1]
    assert fs.doc_layer is None
    assert "Could not build the document layer" in caplog.text


@pytest.mark.parametrize(
    ("n_pages", "start", "end", "max_pages", "expected"),
    [
        (10, None, None, 0, range(0, 10)),
        (10, None, None, 4, range(0, 4)),
        (10, 3, None, 4, range(3, 7)),
        (10, 3, 5, 4, range(3, 6)),
        (10, 3, 20, 0, range(3, 10)),
        (5, 2, None, 10, range(2, 5)),
    ],
)
def test_layout_page_range_follows_the_layout_stage(n_pages, start, end, max_pages, expected):
    assert (
        layout_page_range(n_pages, start_page=start, end_page=end, max_pages=max_pages) == expected
    )


def test_layout_page_range_rejects_what_the_renderer_rejects():
    with pytest.raises(ValueError):
        layout_page_range(3, start_page=5, end_page=None, max_pages=0)
