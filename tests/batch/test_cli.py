"""``bibr batch`` argument parsing and the thin CLI layer."""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

import pytest

from bibr.local.cli import _build_parser
from bibr.local.cli.batch import _form_fields, _run_batch, chew_options_from_config
from bibr.local.cli.run_config import ResolvedRunConfig


def _pdf(tmp_path, name="paper"):
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    return path


def _pin_install(monkeypatch, *, torch, cv2):
    """Pin whether torch is installed and what ``import cv2`` finds (``None``: not installed)."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "torch":
            return ModuleSpec("torch", None) if torch else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setitem(sys.modules, "cv2", cv2)


def _stub_chew_many(monkeypatch):
    """Stand in for the warm local pipeline; returns the paths it was asked to chew."""
    chewed = []

    class _Result:
        ok = True

        def __init__(self, path):
            self.data = {"info": {"title": path.stem}, "bib": [], "text": []}

    def chew_many(paths, batch_size):
        chewed.extend(paths)
        return [_Result(p) for p in paths]

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", lambda local: chew_many)
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "sha")
    return chewed


def _seed_ledger(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "paper_id": "a",
            "status": "ok",
            "duration_s": 12.0,
            "n_refs": 4,
            "n_matched": 2,
            "started_at": "2026-09-02T10:00:00+00:00",
            "finished_at": "2026-09-02T10:30:00+00:00",
        },
        {"paper_id": "b", "status": "failed", "error_code": "poll_timeout"},
    ]
    (out / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_dry_run_prints_the_plan_without_running(tmp_path, capsys, monkeypatch):
    def boom(local):
        raise AssertionError("dry-run must not open a pipeline")

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", boom)
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        ["batch", str(pdf), "--out", str(out), "--dry-run", "--no-llm", "--batch-size", "3"]
    )

    assert _run_batch(args) == 0

    printed = capsys.readouterr().out
    assert "bibr batch · dry run" in printed
    assert "Input (1 files)" in printed
    assert str(pdf) in printed
    assert "local" in printed and "batch size 3" in printed
    assert "llm disabled (--no-llm)" in printed
    assert "Dry run — nothing was processed." in printed
    assert not out.exists()


def test_dry_run_reports_a_failing_preflight_like_the_real_run(tmp_path, monkeypatch, capsys):
    """``batch --dry-run`` is the plan check: when the run's own preflight
    fails, the preview must fail the same way (exit 1) instead of a clean
    exit 0."""
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        "bibr.local.cli.run_config._preflight_ocr_runtime",
        lambda config: "No local OCR runtime can start on this machine for PDF input",
    )
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"

    dry_args = _build_parser().parse_args(
        ["batch", str(pdf), "--out", str(out), "--dry-run", "--no-llm"]
    )
    assert _run_batch(dry_args) == 1
    assert "No local OCR runtime can start" in capsys.readouterr().err

    real_args = _build_parser().parse_args(["batch", str(pdf), "--out", str(out), "--no-llm"])
    assert _run_batch(real_args) == 1
    assert "No local OCR runtime can start" in capsys.readouterr().err
    assert not out.exists()


def test_dry_run_sources_line_counts_unreadable(tmp_path, capsys, monkeypatch):
    """The dry-run plan names the unreadable bucket, not just missing/unsupported."""

    def boom(local):
        raise AssertionError("dry-run must not open a pipeline")

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", boom)
    pdf = _pdf(tmp_path)
    binary = tmp_path / "list.txt"
    binary.write_bytes(bytes([0xD0, 0xCF, 0x11, 0xE0, 0x80, 0x41]) * 30)
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        ["batch", str(pdf), str(binary), "--out", str(out), "--dry-run", "--no-llm"]
    )

    assert _run_batch(args) == 0

    printed = capsys.readouterr().out
    assert "1 unreadable" in printed


def test_remote_dry_run_shows_form_and_warns_about_local_only_flags(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    pdf = _pdf(tmp_path)
    args = _build_parser().parse_args(
        [
            "batch",
            str(pdf),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
            "--serve-url",
            "http://serve:8000/",
            "--token",
            "t0k",
            "--refs",
            "llm",
            "--pages",
            "2-4",
            "--figure-images",
            "--include-regions",
            "--form",
            "extra=1",
            "--ocr",
            "glm",
            "--concurrency",
            "3",
        ]
    )

    assert _run_batch(args) == 0

    captured = capsys.readouterr()
    assert "remote       http://serve:8000" in captured.out
    assert "in-flight    3 (min 1, max 4)" in captured.out
    assert (
        "refs=llm start_page=1 end_page=3 include_figures=true include_regions=true extra=1"
        in captured.out
    )
    assert "token        set" in captured.out
    assert "ignored by the remote executor" in captured.err
    assert "--ocr" in captured.err


def test_report_subcommand_text_and_json(tmp_path, capsys):
    out = tmp_path / "run"
    _seed_ledger(out)

    assert _run_batch(_build_parser().parse_args(["batch", "report", str(out)])) == 0
    text = capsys.readouterr().out
    assert text.startswith("bibr batch report")
    assert "1 ok · 1 failed · 2 total" in text
    assert "poll_timeout ×1" in text

    assert (
        _run_batch(_build_parser().parse_args(["batch", "--report", "--out", str(out), "--json"]))
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["papers"] == {"total": 2, "ok": 1, "failed": 1, "attempts": 2}
    assert report["references"]["match_rate"] == 0.5


def test_report_without_ledger_is_an_error(tmp_path, capsys):
    assert _run_batch(_build_parser().parse_args(["batch", "report", str(tmp_path)])) == 2
    assert "No ledger found" in capsys.readouterr().err
    assert _run_batch(_build_parser().parse_args(["batch", "report"])) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["batch", "--out", "x"],  # no inputs
        ["batch", "paper.pdf"],  # no --out
        ["batch", "paper.pdf", "--out", "x", "--deadline", "soon"],
    ],
)
def test_usage_errors_exit_2(argv, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _pdf(tmp_path)
    assert _run_batch(_build_parser().parse_args(argv)) == 2
    assert capsys.readouterr().err.strip()


@pytest.fixture
def _restore_logging():
    """``main()`` configures the root logger and pins bibr.* levels; undo it."""
    import logging

    names = [
        "bibr.local",
        "bibr.pipeline",
        "bibr.structure",
        "bibr.extract",
        "httpcore",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "filelock",
        "asyncio",
        "hf_xet",
    ]
    root = logging.getLogger()
    saved_root = (root.level, list(root.handlers))
    saved = {name: logging.getLogger(name).level for name in names}
    yield
    root.setLevel(saved_root[0])
    for handler in list(root.handlers):
        if handler not in saved_root[1]:
            root.removeHandler(handler)
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


def test_main_dispatches_batch_report(tmp_path, monkeypatch, capsys, _restore_logging):
    from bibr.local.cli import main

    out = tmp_path / "run"
    _seed_ledger(out)
    monkeypatch.setattr(sys, "argv", ["bibr", "batch", "report", str(out)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "bibr batch report" in capsys.readouterr().out


def test_cli_local_run_end_to_end_with_stubbed_pipeline(tmp_path, monkeypatch, capsys):
    from bibr.api import ChewFailure

    class _Result:
        ok = True

        def __init__(self, path):
            self.data = {"info": {"title": path.stem}, "bib": [], "text": []}

    class _Fake:
        def __init__(self):
            self.local = None

        def __call__(self, paths, batch_size):
            return [
                ChewFailure(p, "corrupt", error_code="CORRUPT") if p.stem == "bad" else _Result(p)
                for p in paths
            ]

    fake = _Fake()

    def factory(local):
        fake.local = local
        return fake

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", factory)
    monkeypatch.setattr("bibr.batch.runner.local_build_sha", lambda: "sha")
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.run_config._preflight_ocr_runtime", lambda config: None)
    good = _pdf(tmp_path, "good")
    _pdf(tmp_path, "bad")
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        [
            "batch",
            str(good),
            str(tmp_path / "bad.pdf"),
            "--out",
            str(out),
            "--no-llm",
            "--refs",
            "off",
        ]
    )

    assert _run_batch(args) == 1

    assert fake.local.chew_options["no_llm"] is True
    assert fake.local.chew_options["refs"] == "off"
    assert fake.local.stages[0] == "validate"
    assert (out / "good.json").is_file()
    rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text().splitlines()]
    assert {r["paper_id"]: r["status"] for r in rows} == {"good": "ok", "bad": "failed"}
    info = json.loads((out / "run_info.json").read_text())
    assert info["options"]["no_llm"] is True
    assert "token" not in info["options"]
    assert "1 ok · 1 failed" in capsys.readouterr().out


def test_local_pdfs_need_no_opencv_on_a_core_install(tmp_path, monkeypatch, capsys):
    """A core install runs layout through ONNX Runtime, so ``bibr chew`` takes
    its PDFs without cv2 — and so must the batch preflight."""
    _pin_install(monkeypatch, torch=False, cv2=None)
    chewed = _stub_chew_many(monkeypatch)
    papers = tmp_path / "papers"
    papers.mkdir()
    _pdf(papers, "a")
    _pdf(papers, "b")
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        ["batch", str(papers), "--out", str(out), "--ocr-url", "http://ocr.invalid:8080"]
    )

    assert _run_batch(args) == 0

    assert sorted(p.name for p in chewed) == ["a.pdf", "b.pdf"]
    assert (out / "a.json").is_file() and (out / "b.json").is_file()
    assert "Layout/OCR image runtime unavailable" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("cv2", "reason", "repair"),
    [
        (None, "opencv (cv2) not installed", "uv sync --extra torch"),
        (
            SimpleNamespace(),
            "cv2 module is incomplete",
            "uv pip install --reinstall opencv-python-headless",
        ),
    ],
    ids=["missing", "broken"],
)
def test_local_pdfs_need_a_working_opencv_when_torch_is_installed(
    tmp_path, monkeypatch, capsys, cv2, reason, repair
):
    """The torch layout path imports cv2: refuse before any model loads, with
    the same repair ``bibr chew`` gives."""
    _pin_install(monkeypatch, torch=True, cv2=cv2)
    chewed = _stub_chew_many(monkeypatch)
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        ["batch", str(_pdf(tmp_path)), "--out", str(out), "--ocr-url", "http://ocr.invalid:8080"]
    )

    assert _run_batch(args) == 1

    err = capsys.readouterr().err
    assert f"Layout/OCR image runtime unavailable: {reason}" in err
    assert f"(repair with: {repair})" in err
    assert chewed == []
    assert not out.exists()


def test_local_pdfs_still_get_the_ocr_runtime_check_on_a_core_install(
    tmp_path, monkeypatch, capsys
):
    """Skipping the opencv check on a core install must not skip the local
    OCR runtime check that follows it."""
    _pin_install(monkeypatch, torch=False, cv2=None)
    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: None)
    chewed = _stub_chew_many(monkeypatch)
    out = tmp_path / "out"
    args = _build_parser().parse_args(
        ["batch", str(_pdf(tmp_path)), "--out", str(out), "--ocr", "paddle-vllm"]
    )

    assert _run_batch(args) == 1

    err = capsys.readouterr().err
    assert "OCR backend cannot start here — paddle-vllm: no NVIDIA GPU" in err
    assert "Layout/OCR image runtime unavailable" not in err
    assert chewed == []


def test_form_fields_map_flags_to_the_job_api(tmp_path):
    from rich.console import Console

    console = Console(stderr=True)
    parse = _build_parser().parse_args
    base = ["batch", "x", "--out", "y", "--serve-url", "http://s"]
    assert _form_fields(parse(base), console) == {}
    args = parse(
        [
            *base,
            "--refs",
            "ner",
            "--ref-seg",
            "geom",
            "--consolidate",
            "--pages",
            "3",
            "--figure-images",
            "--regions",
            "--form",
            "k=v=w",
        ]
    )
    assert _form_fields(args, console) == {
        "refs": "ner",
        "ref_seg": "geom",
        "consolidate": "fill",
        "start_page": "2",
        "end_page": "2",
        "include_figures": "true",
        "include_regions": "true",
        "k": "v=w",
    }
    assert _form_fields(parse([*base, "--form", "novalue"]), console) is None
    assert _form_fields(parse([*base, "--pages", "0-3"]), console) is None


def test_chew_options_from_config_mirrors_the_pipeline_construction():
    config = ResolvedRunConfig(
        ocr_backend="glm-http",
        memory_mode="balanced",
        llm_backend="cloud",
        ocr_url="http://ocr",
        device="cpu",
        crossref=False,
        no_llm=False,
        figure_images=True,
        start_page=0,
        end_page=4,
        consolidate="replace",
        ref_seg="geom",
        refs="ner",
    )
    options = chew_options_from_config(config)
    assert options["ocr_backend"] == "glm-http"
    assert options["llm_backend"] == "cloud"
    assert options["memory_mode"] == "balanced"
    assert options["ocr_url"] == "http://ocr"
    assert options["device"] == "cpu"
    assert options["crossref"] is False
    assert options["figure_images"] is True
    assert (options["start_page"], options["end_page"]) == (0, 4)
    assert options["consolidate"] == "replace"
    assert options["ref_seg"] == "geom"
    assert options["refs"] == "ner"
    bare = chew_options_from_config(ResolvedRunConfig("paddle", "balanced", "cloud"))
    assert "consolidate" not in bare and "ref_seg" not in bare and "refs" not in bare


def test_chew_and_batch_share_pipeline_options():
    parser = _build_parser()
    chew = parser.parse_args(["chew", "p.pdf", "--include-regions", "--refs", "off"])
    batch = parser.parse_args(
        ["batch", "p.pdf", "--out", "o", "--include-regions", "--refs", "off"]
    )
    assert chew.regions is True and batch.regions is True
    assert chew.refs == batch.refs == "off"
    for name in ("ocr", "llm", "memory", "pages", "device", "batch_size", "preset", "consolidate"):
        assert hasattr(batch, name), name


@pytest.mark.parametrize(
    ("serve_url", "extra", "code"),
    [
        ("http://bibr.example.org:8000", [], 2),
        ("http://bibr.example.org:8000", ["--allow-insecure-http"], 0),
        ("https://bibr.example.org", [], 0),
        ("http://gpu-box:8000", [], 0),
        ("http://127.0.0.1:8000", [], 0),
    ],
)
def test_remote_refuses_to_send_the_token_to_a_public_http_host(
    serve_url, extra, code, tmp_path, capsys, monkeypatch
):
    """x-security-9: the bearer token rides every submit and poll."""
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    args = _build_parser().parse_args(
        [
            "batch",
            str(_pdf(tmp_path)),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
            "--serve-url",
            serve_url,
            "--token",
            "serve-token-placeholder",
            *extra,
        ]
    )

    assert _run_batch(args) == code
    captured = capsys.readouterr()
    refused = "Refusing to send the serve bearer token over plain HTTP" in (
        captured.out + captured.err
    )
    assert refused is (code == 2)


def _remote_batch_output(tmp_path, capsys, monkeypatch, serve_url, *token_args):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    monkeypatch.delenv("BIBR_SERVE_TOKEN", raising=False)
    monkeypatch.setattr("bibr.batch.remote.resolve_token", lambda explicit: explicit)
    args = _build_parser().parse_args(
        [
            "batch",
            str(_pdf(tmp_path)),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
            "--serve-url",
            serve_url,
            *token_args,
        ]
    )
    code = _run_batch(args)
    captured = capsys.readouterr()
    return code, " ".join((captured.out + captured.err).split())


@pytest.mark.parametrize(
    ("serve_url", "warned"),
    [
        ("http://gpu-box:8000", True),  # LAN: allowed, but readable on the segment
        ("http://192.168.1.20:8000", True),
        ("http://100.113.200.117:8000", False),  # tailnet: WireGuard-encrypted
        ("http://gpu.tail1234.ts.net:8000", False),
        ("http://127.0.0.1:8000", False),
        ("https://bibr.example.org", False),
    ],
)
def test_remote_warns_when_the_token_crosses_a_network_in_clear_text(
    serve_url, warned, tmp_path, capsys, monkeypatch
):
    code, output = _remote_batch_output(
        tmp_path, capsys, monkeypatch, serve_url, "--token", "serve-token-placeholder"
    )
    assert code == 0
    assert ("the bearer token goes to" in output) is warned


def test_remote_without_a_token_may_use_plain_http_to_any_host(tmp_path, capsys, monkeypatch):
    """Nothing secret rides the requests, so there is nothing to refuse."""
    code, output = _remote_batch_output(
        tmp_path, capsys, monkeypatch, "http://bibr.example.org:8000"
    )
    assert code == 0
    assert "Refusing" not in output
    assert "the bearer token goes to" not in output
