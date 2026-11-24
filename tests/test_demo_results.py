"""What the web demo shows for a finished run, and for every kind of failure.

The click handler is driven directly with a stand-in pipeline, so no model
loads: a run that raises must end in an explanation on the page (not only a
toast, not a traceback), and a run that succeeds must fill the paper card,
the notes and the downloads.
"""

import asyncio
import json
import threading
import zipfile
from html import unescape
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

import bibr.config
import bibr.demo.local_app as local_app
from bibr.exceptions import ProcessingError, UpstreamServiceError

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_full_export.json"


@pytest.fixture
def export() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def _demo_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    # Gradio would otherwise report each new Blocks to its maintainers.
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("DEMO_MAX_FILE_SIZE_MB", raising=False)
    section = bibr.config.Settings.cache
    original = section.ocr
    was_explicit = "ocr" in section.model_fields_set
    yield
    section.ocr = original
    if not was_explicit:
        section.model_fields_set.discard("ocr")


def _demo_with(monkeypatch, process_file, *, ocr_backend="text-layer", on_close=None):
    """A demo whose pipeline runs *process_file*; returns (demo, click handler)."""

    class _Pipeline:
        def __init__(self, **_kwargs):
            self.ocr_backend = ocr_backend
            self.mode = "fast"
            self.llm_backend = None

        async def process_file(self, path, progress=None):
            return await process_file(path, progress)

        async def aclose(self):
            if on_close is not None:
                on_close()

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)
    demo = local_app.create_local_demo(mode="fast")
    handler = next(
        f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "_process_file"
    )
    return demo, handler


def _by_label(demo, updates: dict) -> dict:
    """The handler's updates keyed by component type and label, for readable asserts."""
    keyed = {}
    for component, value in updates.items():
        label = getattr(component, "label", None) or getattr(component, "value", None)
        keyed[(type(component).__name__, str(label)[:40])] = value
    return keyed


def _card(demo, updates: dict) -> str:
    """The result card's HTML, with entities decoded so asserts read as the page does."""
    card = next(c for c in updates if "bibr-result" in (c.elem_classes or []))
    return unescape(updates[card])


def _paper(tmp_path, name="paper.pdf") -> str:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 stand-in")
    return str(path)


def _no_progress(*_args, **_kwargs):
    return None


async def _run(handler, path):
    return await handler(path, progress=_no_progress)


async def test_a_scanned_pdf_is_explained_on_the_page(monkeypatch, tmp_path, _demo_env):
    async def scanned(_path, _progress):
        raise ProcessingError(
            "no usable text layer ... needs an OCR route: --ocr anthropic ...",
            error_code="scan_needs_ocr",
            failed_stage="ocr",
        )

    demo, handler = _demo_with(monkeypatch, scanned)
    updates = await _run(handler, _paper(tmp_path, "scan.pdf"))
    card = _card(demo, updates)

    assert 'class="bibr-error"' in card
    assert "This PDF is a scan" in card
    assert "scan.pdf" in card
    # The restart option names the flag; bibr's own message (with its flags) does not show.
    assert "bibr demo --ocr anthropic" in card
    assert "--ocr paddle" not in card
    tabs = next(c for c in updates if isinstance(c, gr.Tabs))
    assert updates[tabs].visible is False  # no stale results beside the error


async def test_a_corrupt_file_is_explained_on_the_page(monkeypatch, tmp_path, _demo_env):
    async def corrupt(_path, _progress):
        raise ProcessingError(
            "File appears to be corrupted or unreadable",
            error_code="corrupted_file",
            failed_stage="validate",
        )

    demo, handler = _demo_with(monkeypatch, corrupt)
    card = _card(demo, await _run(handler, _paper(tmp_path, "notes.pdf")))

    assert "couldn't open this file" in card
    assert "renamed to .pdf" in card


async def test_an_unexpected_error_is_explained_and_logged(
    monkeypatch, tmp_path, _demo_env, caplog
):
    async def boom(_path, _progress):
        raise RuntimeError("internal detail /home/someone/models")

    demo, handler = _demo_with(monkeypatch, boom)
    card = _card(demo, await _run(handler, _paper(tmp_path)))

    assert "Something went wrong" in card
    assert "/home/someone" not in card  # the detail stays in the log
    assert any("internal detail" in (r.exc_text or "") for r in caplog.records)


async def test_an_outage_is_not_blamed_on_the_paper(monkeypatch, tmp_path, _demo_env):
    async def down(_path, _progress):
        raise UpstreamServiceError("OCR", "connection refused")

    demo, handler = _demo_with(monkeypatch, down)
    card = _card(demo, await _run(handler, _paper(tmp_path)))

    assert "didn't respond" in card
    assert "not a problem with your paper" in card


async def test_no_file_asks_for_one(monkeypatch, _demo_env):
    async def never(_path, _progress):
        raise AssertionError("must not run")

    demo, handler = _demo_with(monkeypatch, never)
    card = _card(demo, await _run(handler, None))

    assert "Choose a paper first" in card
    assert "Try a sample paper" in card


async def test_a_file_over_the_limit_is_explained(monkeypatch, tmp_path, _demo_env):
    monkeypatch.setenv("DEMO_MAX_FILE_SIZE_MB", "1")

    async def never(_path, _progress):
        raise AssertionError("must not run")

    demo, handler = _demo_with(monkeypatch, never)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"0" * (1024 * 1024 + 1))
    card = _card(demo, await _run(handler, str(big)))

    assert "too large" in card
    assert "up to 1 MB" in card


async def test_a_finished_run_fills_the_card_notes_and_downloads(
    monkeypatch, tmp_path, _demo_env, export
):
    export["table"][0]["contents"] = [["Group", "N"], ["A", "12"]]

    async def done(_path, _progress):
        return export

    demo, handler = _demo_with(monkeypatch, done)
    updates = await _run(handler, _paper(tmp_path))
    card = _card(demo, updates)

    assert "Deep Learning for Something Great" in card
    assert 'href="https://doi.org/10.1234/example.5678"' in card
    assert "2 references" in card

    buttons = {b.label: updates[b] for b in updates if isinstance(b, gr.DownloadButton)}
    shown = {
        label: Path(u.value["path"] if isinstance(u.value, dict) else u.value)
        for label, u in buttons.items()
        if u.visible
    }
    assert set(shown) >= {
        "Download JSON",
        "BibTeX",
        "RIS",
        "Data tables (Parquet, .zip)",
        "Download these tables as CSV (.zip)",
    }
    for path in shown.values():
        assert path.is_relative_to(tmp_path / "gradio")  # served in place, deleted by Gradio
    assert json.loads(shown["Download JSON"].read_text(encoding="utf-8")) == export
    assert shown["BibTeX"].read_text(encoding="utf-8").count("@") == 2
    assert shown["RIS"].read_text(encoding="utf-8").count("TY  - ") == 2
    with zipfile.ZipFile(shown["Data tables (Parquet, .zip)"]) as zf:
        assert "bib.parquet" in zf.namelist()
    with zipfile.ZipFile(shown["Download these tables as CSV (.zip)"]) as zf:
        assert zf.namelist() == ["table_1.csv"]
        assert zf.read("table_1.csv").decode().splitlines() == ["Group,N", "A,12"]

    tabs = next(c for c in updates if isinstance(c, gr.Tabs))
    assert updates[tabs].visible is True


async def test_a_new_upload_clears_the_last_results(monkeypatch, _demo_env):
    async def never(_path, _progress):
        raise AssertionError("must not run")

    demo, _handler = _demo_with(monkeypatch, never)
    file_input = next(b for b in demo.blocks.values() if isinstance(b, gr.File))
    clear = next(f.fn for f in demo.fns.values() if (file_input._id, "upload") in f.targets)
    updates = clear()

    assert "Ready" in _card(demo, updates)
    tabs = next(c for c in updates if isinstance(c, gr.Tabs))
    assert updates[tabs].visible is False
    assert all(updates[b].visible is False for b in updates if isinstance(b, gr.DownloadButton))


def test_the_sample_paper_is_copied_where_gradio_serves_it(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))

    path = Path(local_app._copy_sample_paper())

    assert path.name == "sample_paper.pdf"
    assert path.is_relative_to(tmp_path / "gradio")
    assert path.read_bytes().startswith(b"%PDF")


def test_progress_says_the_text_layer_is_read_and_models_load_on_the_first_run():
    seen = []

    def progress(frac, desc=""):
        seen.append(desc)

    first = local_app._GradioProgress(progress, text_layer_only=True, first_run=True)
    first.stage_start("ocr")
    first.stage_start("core_checkpoint")  # an internal step keeps the last label
    later = local_app._GradioProgress(progress, text_layer_only=False)
    later.stage_start("ocr")

    assert seen[0].startswith("Reading the PDF text")
    assert "first paper: loading models" in seen[0]
    assert len(seen) == 2
    assert seen[1] == "Reading the text with OCR…"


def test_notes_put_warnings_in_plain_words(export):
    export["extraction"]["warnings"] = [
        {
            "code": "LLM_SKIPPED",
            "message": "Fast mode made no LLM calls. Not produced: author roles; "
            "statistics in 4 sentences left to the regex pass.",
        },
        {"code": "TEXT_LAYER_TABLE_REBUILT", "message": "2 table(s) were rebuilt"},
        {"code": "FRONT_TAGGER_FLAGGED", "message": "low_confidence_authors"},
        {"code": "REF_SEG_MERGE_SPLIT", "message": "split merged reference strings"},
        {"code": "REF_SEG_REGION_RECOVERY", "message": "recovered 3 segment(s)"},
    ]
    export["extraction"]["validation"] = {
        "issues": [{"code": "VAL_ABSTRACT_SUSPECT", "severity": "warning", "message": "x"}]
    }

    notes = local_app._build_notes_md(export)

    assert "fast mode" in notes
    assert "author roles" in notes
    assert "regex" not in notes
    assert "merged or multi-line cells" in notes
    assert "the authors, the affiliations and the abstract" in notes
    assert "2 more technical notes" in notes
    assert "REF_SEG" not in notes


def test_status_summary_is_plain_words():
    fast = local_app._build_status_summary("text-layer", "fast", None)
    assert "Fast mode" in fast
    assert "scanned PDFs can't be read" in fast
    full = local_app._build_status_summary("glm-mlx", "full", "cloud")
    assert "Full mode" in full
    assert "sent to it" in full
    assert "Scanned pages are read with OCR" in full
    assert "this machine" in local_app._build_status_summary("glm-mlx", "full", "llama-cpp")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"doi": "10.1/x"}, "10.1_x"),
        (
            {"title": "Quality of relationship-building in low-threshold addiction and mental"},
            "Quality_of_relationship-building_in_low-threshold_addiction",
        ),
        ({}, "bibr"),
    ],
)
def test_download_names_cut_at_a_word(raw, expected):
    assert local_app._slug({"metadata": raw}) == expected


def test_the_pipeline_closes_on_the_loop_its_runs_used(monkeypatch, tmp_path, _demo_env, export):
    """Its batchers belong to that loop; closing on a new one ended Ctrl-C in a traceback."""
    closed_on = []

    async def done(_path, _progress):
        return export

    demo, handler = _demo_with(
        monkeypatch, done, on_close=lambda: closed_on.append(asyncio.get_running_loop())
    )
    # Gradio runs the handlers on its server's loop, in another thread.
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        asyncio.run_coroutine_threadsafe(_run(handler, _paper(tmp_path)), loop).result(30)
        demo.close_pipeline(timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()

    assert closed_on == [loop]


def test_before_any_run_the_pipeline_closes_on_a_new_loop(monkeypatch, _demo_env):
    closed = []

    async def never(_path, _progress):
        raise AssertionError("must not run")

    demo, _handler = _demo_with(monkeypatch, never, on_close=lambda: closed.append(True))
    demo.close_pipeline()

    assert closed == [True]


def test_presets_with_none_saved_say_how_to_save_one(monkeypatch, _demo_env):
    """``--presets`` with nothing saved used to show nothing at all."""
    monkeypatch.setattr("bibr.presets.PresetManager.list_presets", lambda self: [])

    class _Pipeline:
        def __init__(self, **_kwargs):
            self.ocr_backend = "text-layer"

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)
    demo = local_app.create_local_demo(mode="fast", presets_enabled=True)

    texts = [b.value or "" for b in demo.blocks.values() if isinstance(b, gr.Markdown)]
    assert any("bibr preset save NAME" in t for t in texts)
    assert not any(isinstance(b, gr.Dropdown) for b in demo.blocks.values())


def test_first_run_notice_mentions_downloads_when_models_are_uncached(monkeypatch):
    monkeypatch.setattr("bibr.utils.hf_cache.hub_offline", lambda: False)
    pending = [
        ("layout", "scienceverse/bibr-layout-onnx", "will download (~0.13 GB)"),
        ("sentence", "segment-any-text/sat-6l-sm", "will download (~0.45 GB)"),
    ]
    cold = local_app._build_first_run_notice(pending, warm=False)
    assert "First paper takes longer" in cold
    assert "2 models" in cold
    assert "580 MB" in cold

    cached = local_app._build_first_run_notice([], warm=False)
    assert "First paper takes longer" in cached
    assert "models load into memory" in cached

    assert local_app._build_first_run_notice(pending, warm=True) == ""

    monkeypatch.setattr("bibr.utils.hf_cache.hub_offline", lambda: True)
    offline = local_app._build_first_run_notice(pending, warm=False)
    assert "HF_HUB_OFFLINE" in offline


def test_progress_says_downloading_and_loading_on_cold_first_run():
    seen = []

    def progress(_frac, desc=""):
        seen.append(desc)

    cold = local_app._GradioProgress(
        progress, text_layer_only=True, first_run=True, downloading=True
    )
    cold.stage_start("ocr")
    assert "downloading and loading models" in seen[0]


async def test_first_run_notice_hides_after_first_successful_paper(
    monkeypatch, tmp_path, _demo_env, export
):
    monkeypatch.setattr(
        local_app,
        "_pending_demo_downloads",
        lambda *_a, **_k: [("layout", "scienceverse/bibr-layout-onnx", "will download (~0.13 GB)")],
    )

    async def done(_path, _progress):
        return export

    demo, handler = _demo_with(monkeypatch, done)
    notice = next(
        b
        for b in demo.blocks.values()
        if isinstance(b, gr.Markdown) and "bibr-notice" in (b.elem_classes or [])
    )
    assert notice.visible is True
    assert "downloads 1 model" in (notice.value or "")

    updates = await _run(handler, _paper(tmp_path))
    assert updates[notice].visible is False
    assert updates[notice].value == ""
