"""Regression tests for audit action S2 (CLI findings, 2026-09-24 audit).

Each test goes through the real argument parser and entry functions with
fakes for the pipeline (no real chew, no models, no network). Every
regression test fails on base (33e6c47) and passes with the S2 fixes;
guards (marked as such) pass on base too.
"""

from __future__ import annotations

import io
import json
import sys

import pytest


class _FakePipeline:
    """Stand-in for ``LocalPipeline`` — no-op construction, instant success."""

    def __init__(self, **_kwargs):
        pass

    async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
        for fs in file_states:
            fs.result_json = {"info": {"title": fs.path.name}}

    async def aclose(self):
        pass

    def llm_usage_snapshot(self):
        return {}


class _ForbiddenPipeline:
    """Stand-in that blows up if ever constructed (ordering proofs)."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("pipeline must not be constructed on this path")


def _astral_payload() -> dict:
    return {"title": "Bounds on \U0001d465 in \u03b1-stable \u4e2d caf\u00e9"}


# --- local-cli-2: stdout JSON under a legacy encoding --------------------------


async def test_stdout_json_under_cp1252_parses_back_identically(tmp_path, monkeypatch):
    """JSON on stdout must survive a cp1252 console: UTF-8 bytes, not
    backslashreplace escapes (``\\U0001d465`` is not legal JSON)."""
    from bibr.local.cli import _build_parser, _run_process, ui

    class _AstralPipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            for fs in file_states:
                fs.result_json = _astral_payload()

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _AstralPipeline)

    src = tmp_path / "paper.xml"
    src.write_text("<article><title>t</title></article>")
    args = _build_parser().parse_args(["chew", str(src), "--no-llm"])

    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="backslashreplace")
    monkeypatch.setattr(sys, "stdout", wrapper)
    ui.configure_output_streams()  # the legacy-console condition
    await _run_process(args)
    wrapper.flush()

    raw = buf.getvalue()
    assert b"\\U0001d465" not in raw  # the base failure mode
    assert json.loads(raw.decode("utf-8")) == _astral_payload()


def test_write_stdout_json_roundtrip_with_astral_chars(monkeypatch):
    """Direct unit proof: the stdout sink emits UTF-8 that parses back
    identically, even when stdout is a cp1252 backslashreplace wrapper."""
    from bibr.local.cli import ui
    from bibr.local.cli.process import _write_stdout_json

    payload = _astral_payload()
    text = json.dumps(payload, ensure_ascii=False)

    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="backslashreplace")
    monkeypatch.setattr(sys, "stdout", wrapper)
    ui.configure_output_streams()
    _write_stdout_json(text)
    wrapper.flush()

    raw = buf.getvalue()
    assert json.loads(raw.decode("utf-8")) == payload


def test_write_stdout_json_falls_back_without_buffer(monkeypatch, capsys):
    """Guard: when stdout has no buffer (StringIO capture), plain print still
    emits the payload."""

    from bibr.local.cli.process import _write_stdout_json

    payload = {"title": "plain ascii"}
    text = json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    _write_stdout_json(text)
    assert json.loads(sys.stdout.getvalue()) == payload
    capsys.readouterr()  # drain pytest capture, keeps the suite hermetic


# --- local-cli-5: -o directory intent with one resolved file -------------------


async def test_single_file_dir_with_trailing_slash_writes_inside(tmp_path, monkeypatch):
    """``chew papers/ -o results/`` (one paper) must create ``results/`` and
    write ``only.json`` inside — not a FILE named ``results``."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "only.xml").write_text("<article/>")
    out_raw = str(tmp_path / "results") + "/"

    args = _build_parser().parse_args(["chew", str(papers), "-o", out_raw, "--no-llm"])
    await _run_process(args)

    out_dir = tmp_path / "results"
    assert out_dir.is_dir()
    assert (out_dir / "only.json").is_file()
    assert json.loads((out_dir / "only.json").read_text())["info"]["title"] == "only.xml"

    # A second run with the same -o must write into it again, not FileExistsError.
    args = _build_parser().parse_args(["chew", str(papers), "-o", out_raw, "--no-llm"])
    await _run_process(args)
    assert (out_dir / "only.json").is_file()


async def test_single_file_dir_with_existing_directory_writes_inside(tmp_path, monkeypatch):
    """``-o`` pointing at an existing directory (no trailing slash) is also
    directory intent."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "only.xml").write_text("<article/>")
    existing = tmp_path / "existing"
    existing.mkdir()

    args = _build_parser().parse_args(["chew", str(papers), "-o", str(existing), "--no-llm"])
    await _run_process(args)

    assert (existing / "only.json").is_file()


async def test_paper_id_with_new_directory_output(tmp_path, monkeypatch):
    """``chew paper.xml --paper-id X -o newdir/`` keeps the id and writes
    ``newdir/paper.json`` — directory intent must not turn the run into a
    batch that rejects ``--paper-id``."""
    from bibr.local.cli import _build_parser, _run_process

    seen = {}

    class _CapturePipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            for fs in file_states:
                seen[fs.path.name] = fs.paper_id
            await super().process_chunk(file_states, progress=progress)

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _CapturePipeline)

    src = tmp_path / "paper.xml"
    src.write_text("<article/>")
    out_raw = str(tmp_path / "results") + "/"

    args = _build_parser().parse_args(
        ["chew", str(src), "--paper-id", "my-id", "-o", out_raw, "--no-llm"]
    )
    await _run_process(args)  # must not raise SystemExit(2)

    assert (tmp_path / "results" / "paper.json").is_file()
    assert seen == {"paper.xml": "my-id"}


async def test_paper_id_with_existing_directory_output(tmp_path, monkeypatch):
    """``chew paper.xml --paper-id X -o <existing dir>`` keeps the id and
    writes ``<dir>/paper.json``."""
    from bibr.local.cli import _build_parser, _run_process

    seen = {}

    class _CapturePipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            for fs in file_states:
                seen[fs.path.name] = fs.paper_id
            await super().process_chunk(file_states, progress=progress)

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _CapturePipeline)

    src = tmp_path / "paper.xml"
    src.write_text("<article/>")
    existing = tmp_path / "existing"
    existing.mkdir()

    args = _build_parser().parse_args(
        ["chew", str(src), "--paper-id", "my-id", "-o", str(existing), "--no-llm"]
    )
    await _run_process(args)  # must not raise SystemExit(2)

    assert (existing / "paper.json").is_file()
    assert seen == {"paper.xml": "my-id"}


async def test_dry_run_previews_directory_output_without_creating_it(tmp_path, monkeypatch, capsys):
    """Guard: `chew paper.xml -o newdir/ --dry-run` previews
    `paper.xml -> newdir/paper.json` (what the real run writes) and
    creates nothing."""
    from bibr.local.cli import _build_parser, _run_process

    src = tmp_path / "paper.xml"
    src.write_text("<article/>")
    out_raw = str(tmp_path / "results") + "/"
    args = _build_parser().parse_args(["chew", str(src), "-o", out_raw, "--dry-run", "--no-llm"])
    await _run_process(args)  # must not raise

    assert f"paper.xml -> {tmp_path / 'results' / 'paper.json'}" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


async def test_single_file_output_unchanged(tmp_path, monkeypatch):
    """Guard: ``chew paper.xml -o result.json`` still writes the file itself."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FakePipeline)

    src = tmp_path / "paper.xml"
    src.write_text("<article/>")
    out_file = tmp_path / "result.json"

    args = _build_parser().parse_args(["chew", str(src), "-o", str(out_file), "--no-llm"])
    await _run_process(args)

    assert out_file.is_file()
    assert json.loads(out_file.read_text())["info"]["title"] == "paper.xml"


async def test_output_blocked_by_existing_file_exits_2_before_pipeline(
    tmp_path, monkeypatch, capsys
):
    """``-o blocked/`` where ``blocked`` is a FILE must be a clean exit 2
    before any model loads — not an uncaught FileExistsError."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _ForbiddenPipeline)

    src = tmp_path / "paper.xml"
    src.write_text("<article/>")
    (tmp_path / "blocked").write_text("not a dir")

    args = _build_parser().parse_args(
        ["chew", str(src), "-o", str(tmp_path / "blocked") + "/", "--no-llm"]
    )
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2
    assert "Cannot write output" in capsys.readouterr().err


# --- local-cli-14: option errors name the option --------------------------------


async def test_ocr_model_without_profile_names_ocr_model_not_pages(tmp_path, monkeypatch, capsys):
    """``--ocr-model`` without ``--ocr-profile`` must not blame ``--pages``."""
    from bibr.local.cli import _build_parser, _run_process

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    args = _build_parser().parse_args(["chew", str(src), "--ocr-model", "my-ocr", "--dry-run"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Invalid option:" in err
    assert "--ocr-profile" in err
    assert "Invalid --pages" not in err


async def test_bad_pages_still_reports_cleanly(tmp_path, monkeypatch, capsys):
    """Guard: a genuinely bad ``--pages`` still fails fast with exit 2."""
    from bibr.local.cli import _build_parser, _run_process

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    args = _build_parser().parse_args(["chew", str(src), "--pages", "0", "--dry-run"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 2
    assert "Invalid option:" in capsys.readouterr().err


# --- local-cli-16: hints keyed by error_code ------------------------------------


def _state(path, error, code):
    from pathlib import Path as _Path

    from bibr.pipeline.state import FileState

    fs = FileState(path=_Path(path))
    fs.error = error
    fs.error_code = code
    return fs


@pytest.mark.parametrize(
    ("error", "code", "hint_fragment"),
    [
        ("Unsupported file format", "unsupported_format", "bibr accepts"),
        ("File is password-protected", "encrypted_file", "Remove the password"),
        (
            "OCR backend init failed: No local OCR runtime could start",
            "ocr_failed",
            "Check your OCR backend",
        ),
        (
            "OCR backend init failed: paddle-rapid-mlx: launcher exited",
            "ocr_failed",
            "Check your OCR backend",
        ),
        (
            "LLM server start failed: rapid-mlx executable not found",
            "llm_server_failed",
            "Check your LLM backend",
        ),
        # Code-keyed, not message-keyed: the message matches no fallback
        # substring, so only the error_code can produce the hint.
        ("x", "encrypted_file", "Remove the password"),
        # A misleading message must not win over the code: the 'api key'
        # fallback would blame API keys for a local OCR failure.
        ("api key expired for rapid-mlx", "ocr_failed", "Check your OCR backend"),
    ],
)
def test_hints_follow_error_code(error, code, hint_fragment):
    """Every structured code maps to its hint — including the real stage
    messages the old substrings never matched."""
    from bibr.local.cli.process import _hint_for_file_error

    hint = _hint_for_file_error(_state("a.pdf", error, code))
    assert hint_fragment in hint
    assert "API keys" not in hint


def test_rapid_mlx_failures_never_suggest_api_keys():
    """Guard: 'rapid-mlx' contains 'api' — the old substring misfire must not
    return, with or without a structured code."""
    from bibr.local.cli.process import _hint_for_file_error

    for code in ("ocr_failed", "llm_server_failed", None):
        hint = _hint_for_file_error(_state("a.pdf", "rapid-mlx launcher exited", code))
        assert "API keys" not in hint


async def test_chunk_results_report_hints_through_entry_function(tmp_path, monkeypatch, capsys):
    """End-to-end through ``_run_process`` with a failing fake pipeline:
    an encrypted PDF gets the password hint, an OCR failure the OCR hint."""

    from bibr.local.cli import _build_parser, _run_process

    class _FailingPipeline(_FakePipeline):
        async def process_chunk(self, file_states, progress=None):  # noqa: ARG002
            for fs in file_states:
                if fs.path.stem == "locked":
                    fs.error = "File is password-protected"
                    fs.error_code = "encrypted_file"
                else:
                    fs.error = "OCR backend init failed: boom"
                    fs.error_code = "ocr_failed"

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _FailingPipeline)

    (tmp_path / "locked.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        [
            "chew",
            str(tmp_path / "locked.pdf"),
            str(tmp_path / "scan.pdf"),
            "-o",
            str(out),
            "--no-llm",
        ]
    )
    with pytest.raises(SystemExit):
        await _run_process(args)
    err = capsys.readouterr().err
    assert "Remove the password" in err
    assert "Check your OCR backend" in err


# --- local-cli-9: dry-run OCR weight repo for the automatic chain --------------


async def test_dry_run_automatic_paddle_checks_weight_repo_not_served_alias(
    tmp_path, capsys, monkeypatch
):
    """On the Linux/CUDA automatic chain the first candidate is paddle-vllm,
    whose display model is the served alias ``paddle-ocr-vl-1.6`` — but the
    weights line must cache-check the real repo
    ``PaddlePaddle/PaddleOCR-VL-1.6`` (cached here), not the alias."""
    import pytest as _pytest

    hf_hub = _pytest.importorskip("huggingface_hub")
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: 24.0)

    class _FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    class _FakeCacheInfo:
        repos = [_FakeRepo("PaddlePaddle/PaddleOCR-VL-1.6")]

    monkeypatch.setattr(hf_hub, "scan_cache_dir", lambda *a, **k: _FakeCacheInfo())
    # Neutralize the dry-run OCR-runtime blocker (see above): this test
    # asserts cache rendering, not readiness.
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--no-llm"])
    await _run_process(args)

    out = capsys.readouterr().out
    assert "OCR weights: PaddlePaddle/PaddleOCR-VL-1.6 — cached" in out


def test_dry_run_ocr_model_separates_served_identity_from_weights(monkeypatch):
    """Unit pin: automatic ``paddle`` returns the served alias for display
    and the weight repo for the cache check."""
    from bibr.config import Settings
    from bibr.local.cli.dry_run import _dry_run_ocr_model
    from bibr.local.cli.parser import _build_parser
    from bibr.local.cli.run_config import resolve_run_config

    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: 24.0)
    args = _build_parser().parse_args(["chew", "paper.pdf", "--no-llm"])
    config = resolve_run_config(args)
    assert config.ocr_backend == "paddle"
    label, repo = _dry_run_ocr_model(config)
    assert repo == Settings.ocr.paddle_model
    assert repo != label or config.ocr_model is not None


# --- local-cli-19: parser without installed metadata ----------------------------


def test_build_parser_without_package_metadata(monkeypatch):
    """A source-tree checkout (no installed ``bibr`` metadata) must still
    build the parser — every command, including doctor, starts."""
    from importlib.metadata import PackageNotFoundError

    from bibr.local.cli.parser import _build_parser

    def _missing(_name):
        raise PackageNotFoundError("bibr")

    monkeypatch.setattr("importlib.metadata.version", _missing)
    parser = _build_parser()  # must not raise (base: PackageNotFoundError)
    args = parser.parse_args(["chew", "a.pdf"])
    assert args.command == "chew"

    from bibr.local.cli.parser import _safe_version

    assert _safe_version() == "?"


def test_safe_version_matches_installed_metadata():
    """Guard: with metadata present, the helper returns the real version."""
    from bibr.local.cli.parser import _get_version, _safe_version

    assert _safe_version() == _get_version() != "?"


# --- local-cli-10: dry-run blockers ----------------------------------------------


def _clear_llm_keys(monkeypatch, provider="google", key_attr="GOOGLE_API_KEY"):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.llm, "provider", provider)
    monkeypatch.setattr(Settings.llm, "api_key", None)
    monkeypatch.setattr(Settings.llm, "base_url", None)
    # The openai adapter reads only llm.api_key / llm.base_url, so it has
    # no dedicated Settings key to clear (key_attr=None).
    if key_attr is not None:
        monkeypatch.setattr(Settings, key_attr, None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)


async def test_dry_run_reports_missing_inputs_and_credentials(tmp_path, monkeypatch, capsys):
    """``chew a.xml nope.xml --dry-run`` with no LLM key prints a Blockers
    section and exits 1 — the base behaviour (clean plan, exit 0) hid a run
    that fails immediately."""
    from bibr.local.cli import _build_parser, _run_process

    _clear_llm_keys(monkeypatch)
    good = tmp_path / "a.xml"
    good.write_text("<article/>")
    args = _build_parser().parse_args(["chew", str(good), str(tmp_path / "nope.xml"), "--dry-run"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Blockers" in out
    assert "not found" in out
    assert "API key" in out


async def test_dry_run_ocr_blocker_for_unstartable_backend(tmp_path, monkeypatch, capsys):
    """A PDF dry-run whose OCR runtime cannot start reports the blocker and
    exits 1 instead of previewing a doomed plan."""
    from bibr.local.cli import _build_parser, _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        "bibr.local.cli.dry_run._preflight_ocr_runtime",
        lambda config: "No local OCR runtime can start on this machine for PDF input",
    )
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--no-llm"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 1
    assert "Blockers" in capsys.readouterr().out


async def test_dry_run_local_backend_blocker(tmp_path, monkeypatch, capsys):
    """A managed local LLM backend that cannot start is a Blocker with
    exit 1 — disabling the check must not pass silently."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr(
        "bibr.local.cli.dry_run._preflight_local_backend",
        lambda backend: f"{backend} cannot start here: no NVIDIA GPU",
    )
    good = tmp_path / "a.xml"
    good.write_text("<article/>")
    args = _build_parser().parse_args(["chew", str(good), "--dry-run", "--llm", "vllm"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Blockers" in out
    assert "no NVIDIA GPU" in out


async def test_dry_run_opencv_blocker(tmp_path, monkeypatch, capsys):
    """A PDF dry-run on a torch install without cv2 reports the opencv
    blocker and exits 1 — disabling the check must not pass silently."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr(
        "bibr.local.cli.dry_run._preflight_opencv",
        lambda: (
            "Layout/OCR image runtime unavailable: opencv (cv2) not installed",
            "uv sync --extra torch",
        ),
    )
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    args = _build_parser().parse_args(["chew", str(pdf), "--dry-run", "--no-llm"])
    with pytest.raises(SystemExit) as exc_info:
        await _run_process(args)
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Blockers" in out
    assert "opencv" in out


async def test_clean_dry_run_still_exits_0_without_blockers(tmp_path, monkeypatch, capsys):
    """Guard: a dry-run with nothing failing keeps exit 0 and no section."""
    from bibr.local.cli import _build_parser, _run_process

    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)
    good = tmp_path / "a.xml"
    good.write_text("<article/>")
    args = _build_parser().parse_args(["chew", str(good), "--dry-run", "--no-llm"])
    await _run_process(args)  # must not raise
    assert "Blockers" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("provider", "key_attr", "message_fragment"),
    [
        ("google", "GOOGLE_API_KEY", "GOOGLE_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        ("groq", "GROQ_API_KEY", "GROQ_API_KEY"),
        ("openai", None, "LLM_API_KEY"),
    ],
)
def test_dry_run_credential_message_matches_provider_preflight(
    monkeypatch, provider, key_attr, message_fragment
):
    """Guard: the non-constructing dry-run message cannot drift from the
    provider adapter's own error (both read the same Settings fields) —
    for every bundled provider whose lookup the dry run mirrors."""
    from bibr.clients.llm import preflight_credentials
    from bibr.local.cli.dry_run import _dry_run_cloud_credential_blocker

    _clear_llm_keys(monkeypatch, provider=provider, key_attr=key_attr)
    expected = _dry_run_cloud_credential_blocker()
    assert expected is not None and message_fragment in expected
    with pytest.raises(ValueError) as exc_info:
        preflight_credentials()
    assert str(exc_info.value) == expected


def test_dry_run_credential_openai_base_url_exemption(monkeypatch):
    """Guard: the openai base_url exemption (self-hosted server needs no
    key) matches the adapter — both stay silent together."""
    from bibr.clients.llm import preflight_credentials
    from bibr.config import Settings
    from bibr.local.cli.dry_run import _dry_run_cloud_credential_blocker

    _clear_llm_keys(monkeypatch, provider="openai", key_attr=None)
    monkeypatch.setattr(Settings.llm, "base_url", "http://localhost:8080/v1")
    assert _dry_run_cloud_credential_blocker() is None
    preflight_credentials()  # must not raise
