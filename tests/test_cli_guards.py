"""Regression tests for three CLI guards against silent data loss in ``bibr chew``.

1. Single-file ``-o deep/new/dir/out.json`` must create the parent directory
   instead of crashing with ``FileNotFoundError`` at write time.
2. Batch input whose files collide on ``.stem`` (e.g. ``a/x.pdf`` + ``b/x.pdf``)
   would silently overwrite one ``<stem>.json`` with the other — this must
   now be a hard error (exit 2) listing every collision group, raised before
   the pipeline is constructed.
3. ``--paper-id`` is only meaningful for single-file input; passing it with
   batch input (directory or multi-file glob) must now be a hard error
   (exit 2) instead of being silently discarded, also raised before the
   pipeline is constructed.

The stem-collision and --paper-id guards fire before ``LocalPipeline`` is
ever constructed, so those tests deliberately do NOT mock the pipeline layer
— if the guard didn't fire first, the test would blow up trying to build a
real pipeline (OCR/LLM/layout models), proving the ordering.
"""

from __future__ import annotations

import json

import pytest


class _FakePipeline:
    """Stand-in for ``LocalPipeline`` — construction is a no-op, and
    ``process_chunk`` marks every file as succeeded."""

    def __init__(self, **_kwargs):
        pass

    async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
        for fs in file_states:
            fs.result_json = {"info": {"title": fs.path.name}}

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


def _disable_ocr_runtime_preflight(monkeypatch):
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    # The OCR runtime preflight fails where no automatic runtime can start
    # (Windows CI: llama.cpp only, no llama-server); the pipeline is a fake.
    monkeypatch.setattr("bibr.local.cli.process._preflight_ocr_runtime", lambda config: None)


# --- Guard 1: parent dirs for single-file -o ---------------------------------


async def test_single_file_output_creates_parent_dirs(tmp_path, monkeypatch):
    """`-o deep/new/dir/out.json` must create the missing directories, not
    crash with FileNotFoundError at write time."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    good_file = tmp_path / "paper.pdf"
    good_file.write_bytes(b"%PDF-1.4\n")
    out_file = tmp_path / "deep" / "new" / "dir" / "out.json"
    assert not out_file.parent.exists()

    args = _build_parser().parse_args(["chew", str(good_file), "-o", str(out_file), "--no-llm"])
    await _run_process(args)

    assert out_file.exists()


async def test_explicit_sink_honors_compact_and_pretty_json_formatting(tmp_path, monkeypatch):
    from bibr.local.cli import _build_parser, _run_process

    class _DurableFakePipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            from bibr.pipeline.artifacts import RunState

            for fs in file_states:
                payload = {"info": {"title": fs.path.name}}
                fs.result_json = payload
                fs.core_sha256 = fs.artifact_sink.write_core(fs, payload)
                fs.artifact_sink.materialize(fs, payload)
                fs.artifact_sink.record(
                    fs,
                    RunState.CORE_WRITTEN,
                    detail="enrichment_not_requested",
                )

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _DurableFakePipeline)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    pretty = tmp_path / "pretty.json"
    compact = tmp_path / "compact.json"

    await _run_process(
        _build_parser().parse_args(["chew", str(source), "-o", str(pretty), "--no-llm"])
    )
    await _run_process(
        _build_parser().parse_args(
            ["chew", str(source), "-o", str(compact), "--no-llm", "--compact"]
        )
    )

    pretty_text = pretty.read_text()
    compact_text = compact.read_text()
    assert json.loads(pretty_text) == json.loads(compact_text)
    assert '\n  "info": {' in pretty_text
    assert compact_text == '{"info":{"title":"paper.pdf"}}'


# --- Guard 2: batch stem collisions ------------------------------------------


async def test_batch_stem_collision_exits_2_and_lists_paths(tmp_path, monkeypatch, capsys):
    """Two files from different directories sharing a stem must fail loudly
    instead of silently overwriting one <stem>.json with the other."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "x.pdf"
    file_b = dir_b / "x.pdf"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_batch_stem_collision_via_glob_exits_2(tmp_path, monkeypatch, capsys):
    """The collision guard must also fire when the collision comes from a
    glob expansion collecting files from sibling directories."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "x.pdf").write_bytes(b"%PDF-1.4\n")
    (dir_b / "x.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "*" / "x.pdf"), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2


async def test_batch_stem_collision_across_extensions_exits_2(tmp_path, monkeypatch, capsys):
    """x.pdf + x.docx both write x.json — same overwrite, different extension."""
    from bibr.local.cli import _build_parser, _run_process

    file_a = tmp_path / "x.pdf"
    file_b = tmp_path / "x.docx"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"PK\x03\x04")

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_batch_stem_collision_case_insensitive_exits_2(tmp_path, monkeypatch, capsys):
    """a/x.pdf + b/X.PDF collide on the default macOS/Windows filesystems,
    where x.json and X.json are the same directory entry."""
    from bibr.local.cli import _build_parser, _run_process

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "x.pdf"
    file_b = dir_b / "X.PDF"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(tmp_path / "out"), "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert str(file_a) in err
    assert str(file_b) in err


async def test_batch_no_stem_collision_still_works(tmp_path, monkeypatch):
    """Sanity check: distinct stems in batch mode must not trip the guard."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    file_a.write_bytes(b"%PDF-1.4\n")
    file_b.write_bytes(b"%PDF-1.4\n")
    out_dir = tmp_path / "out"

    args = _build_parser().parse_args(
        ["chew", str(file_a), str(file_b), "-o", str(out_dir), "--no-llm"]
    )
    await _run_process(args)

    assert (out_dir / "a.json").exists()
    assert (out_dir / "b.json").exists()


# --- Guard 3: --paper-id rejected for batch input ----------------------------


async def test_paper_id_with_directory_input_exits_2(tmp_path, monkeypatch, capsys):
    """--paper-id with a directory of 2+ files must hard-error instead of
    being silently discarded."""
    from bibr.local.cli import _build_parser, _run_process

    (tmp_path / "one.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "two.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(tmp_path), "--paper-id", "my-id", "--no-llm"])

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "--paper-id" in err
    assert "single-file" in err


async def test_paper_id_with_multi_file_glob_exits_2(tmp_path, monkeypatch):
    """--paper-id with a multi-file glob must also hard-error (not just
    directory input)."""
    from bibr.local.cli import _build_parser, _run_process

    (tmp_path / "one.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "two.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(
        ["chew", str(tmp_path / "*.pdf"), "--paper-id", "my-id", "--no-llm"]
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2


async def test_paper_id_with_single_file_still_works(tmp_path, monkeypatch):
    """--paper-id with a genuine single-file input must be unaffected."""
    from bibr.local.cli import _build_parser, _run_process

    _disable_ocr_runtime_preflight(monkeypatch)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    good_file = tmp_path / "paper.pdf"
    good_file.write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(good_file), "--paper-id", "my-id", "--no-llm"])
    await _run_process(args)


async def test_paper_id_with_single_file_directory_exits_2(tmp_path, monkeypatch):
    """A directory is batch input however many files it holds (it writes
    ``<dir>/<stem>.json`` with one paper as with two), so --paper-id is
    rejected for a directory that collects to exactly ONE file too."""
    from bibr.local.cli import _build_parser, _run_process

    (tmp_path / "only.pdf").write_bytes(b"%PDF-1.4\n")

    args = _build_parser().parse_args(["chew", str(tmp_path), "--paper-id", "my-id", "--no-llm"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2


# --- Guard 4: the first-run walls (no key, no OCR runtime, no vLLM) -----------------

_KEY_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "LANGEXTRACT_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BACKEND",
    "PIPELINE_MODE",
    "OCR_BACKEND",
)


class _RecordingPipeline(_FakePipeline):
    """A fake pipeline that keeps the kwargs ``bibr chew`` built it with."""

    built: list[dict] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _RecordingPipeline.built.append(kwargs)


@pytest.fixture
def first_run(monkeypatch):
    """A machine with no LLM configured, a fake pipeline and a credential check that fails.

    Returns the list of credential-check calls: a run that must not ask for a key leaves it empty.
    """
    from bibr.config import GlobalSettings

    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("bibr.config.Settings", GlobalSettings())
    _disable_ocr_runtime_preflight(monkeypatch)
    # What the real download announcement would say depends on this machine's cache.
    monkeypatch.setattr("bibr.local.cli.process.announce_downloads", lambda *a, **k: None)
    _RecordingPipeline.built = []
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _RecordingPipeline)

    asked: list[str] = []

    def no_key(settings=None):  # noqa: ARG001
        asked.append("asked")
        raise ValueError("Google API key required: set GOOGLE_API_KEY")

    monkeypatch.setattr("bibr.clients.llm.preflight_credentials", no_key)
    return asked


def _pdf(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return pdf


async def test_a_run_with_no_key_and_no_flag_runs_fast_without_asking_for_one(
    tmp_path, first_run, capsys
):
    """The credential check was the first wall of a fresh install; with nothing to call it is moot."""
    from bibr.local.cli import _build_parser, _run_process

    await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path))]))

    assert first_run == []
    (kwargs,) = _RecordingPipeline.built
    assert kwargs["mode"] == "fast"
    assert kwargs["llm_backend"] == "cloud"
    assert kwargs["no_llm"] is False
    assert kwargs["ocr_requested"] is False
    err = capsys.readouterr().err
    assert "mode fast" in err
    assert "no LLM calls" in err


async def test_fast_flag_runs_fast_even_with_a_key(tmp_path, first_run, monkeypatch):
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path)), "--fast"]))

    assert first_run == []
    assert _RecordingPipeline.built[0]["mode"] == "fast"


async def test_full_flag_with_no_key_stops_at_the_credential_check_and_suggests_fast(
    tmp_path, first_run, capsys
):
    from bibr.local.cli import _build_parser, _run_process

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path)), "--full"]))

    assert exc_info.value.code == 1
    assert first_run == ["asked"]
    assert _RecordingPipeline.built == []
    err = capsys.readouterr().err
    assert "Google API key required" in err
    assert "--fast" in err
    assert "--no-llm" in err


async def test_a_configured_llm_with_no_flag_still_asks_for_its_key(
    tmp_path, first_run, monkeypatch
):
    """Naming a provider is choosing an LLM: the run is full and a missing key is an error."""
    from bibr.config import GlobalSettings
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr("bibr.config.Settings", GlobalSettings())

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path))]))

    assert exc_info.value.code == 1
    assert first_run == ["asked"]


async def test_no_llm_does_not_ask_for_a_key_and_stays_structure_only(tmp_path, first_run):
    from bibr.local.cli import _build_parser, _run_process

    await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path)), "--no-llm"]))

    assert first_run == []
    (kwargs,) = _RecordingPipeline.built
    assert kwargs["no_llm"] is True
    assert kwargs["mode"] == "full"


async def test_the_ocr_check_comes_before_the_credential_check(
    tmp_path, first_run, monkeypatch, capsys
):
    """A machine that cannot OCR at all hears that before it is asked for a key."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr(
        "bibr.local.cli.process._preflight_ocr_runtime",
        lambda config: "No local OCR runtime can start on this machine for PDF input",
    )

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path)), "--full"]))

    assert exc_info.value.code == 1
    assert first_run == []
    err = capsys.readouterr().err
    assert "No local OCR runtime" in err
    assert "API key" not in err


async def test_fast_refuses_an_llm_only_option_before_doing_anything(tmp_path, first_run, capsys):
    from bibr.local.cli import _build_parser, _run_process

    with pytest.raises(SystemExit) as exc_info:
        await _run_process(
            _build_parser().parse_args(["chew", str(_pdf(tmp_path)), "--fast", "--refs", "llm"])
        )

    assert exc_info.value.code == 2
    assert "calls an LLM" in capsys.readouterr().err
    assert _RecordingPipeline.built == []
    assert first_run == []


async def test_a_fast_run_announces_the_downloads_before_the_pipeline_is_built(
    tmp_path, first_run, monkeypatch
):
    from bibr.local.cli import _build_parser, _run_process

    order: list[str] = []
    monkeypatch.setattr(
        "bibr.local.cli.process.announce_downloads", lambda *a, **k: order.append("announce")
    )

    class _Ordered(_RecordingPipeline):
        def __init__(self, **kwargs):
            order.append("pipeline")
            super().__init__(**kwargs)

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Ordered)
    await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path))]))

    assert order == ["announce", "pipeline"]


# --- a fast run reads the text layer unless the user chose an OCR route ---------------


@pytest.fixture
def gpu_without_vllm(monkeypatch):
    """A Linux x86 box with a roomy GPU, where vLLM is not installed."""
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: 24.0)
    monkeypatch.setattr(registry, "paddle_vllm_unavailable_reason", lambda: None)
    monkeypatch.setattr(registry, "vllm_compute_capability_blocker", lambda: None)


async def test_a_fast_run_with_no_chosen_ocr_route_reads_the_text_layer_on_a_gpu_box(
    tmp_path, first_run, gpu_without_vllm, capsys
):
    from bibr.local.cli import _build_parser, _run_process

    await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path))]))

    err = capsys.readouterr().err
    assert "OCR backend: text-layer" in err
    assert "mode fast" in err and "OCR text-layer" in err
    assert "paddle-vllm" not in err
    built = _RecordingPipeline.built[0]
    assert built["ocr_backend"] == "text-layer"
    assert built["ocr_requested"] is False


async def test_a_fast_run_that_chose_paddle_ocr_keeps_the_gpu_route(
    tmp_path, first_run, gpu_without_vllm, capsys
):
    from bibr.local.cli import _build_parser, _run_process

    await _run_process(_build_parser().parse_args(["chew", str(_pdf(tmp_path)), "--ocr", "paddle"]))

    err = capsys.readouterr().err
    assert "text-layer" not in err
    assert "paddle-vllm (automatic)" in err
    built = _RecordingPipeline.built[0]
    assert built["ocr_backend"] == "paddle"
    assert built["ocr_requested"] is True
