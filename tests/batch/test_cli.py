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
    # The plan check runs the OCR runtime preflight, which fails on hosts with
    # no startable automatic runtime (Windows CI); this test is about the plan.
    monkeypatch.setattr("bibr.local.cli.run_config._preflight_ocr_runtime", lambda config: None)
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
    monkeypatch.setattr("bibr.local.cli.run_config._preflight_ocr_runtime", lambda config: None)
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
    assert report["papers"] == {
        "total": 2,
        "ok": 1,
        "failed": 1,
        "stopped_on_model_override": 0,
        "fell_back_on_model_override": 0,
        "attempts": 2,
    }
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
            "--region-meta",
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
        "include_region_meta": "true",
        "k": "v=w",
    }
    assert _form_fields(parse([*base, "--form", "novalue"]), console) is None
    assert _form_fields(parse([*base, "--pages", "0-3"]), console) is None


@pytest.mark.parametrize(
    ("flags", "expected"),
    [(["--fast"], {"mode": "fast"}), (["--full"], {"mode": "full"}), ([], {})],
)
def test_form_fields_map_the_run_mode_flags_to_mode(flags, expected):
    from rich.console import Console

    args = _build_parser().parse_args(
        ["batch", "x", "--out", "y", "--serve-url", "http://s", *flags]
    )
    assert _form_fields(args, Console(stderr=True)) == expected


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


# --- run mode: --fast / --full on bibr batch ----------------------------------------------

_BATCH_KEY_VARS = (
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


@pytest.fixture
def keyless(monkeypatch):
    """No LLM configured, the OCR and opencv preflights out of the way, downloads silent."""
    from bibr.config import GlobalSettings

    for var in _BATCH_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("bibr.config.Settings", GlobalSettings())
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.run_config._preflight_ocr_runtime", lambda config: None)
    announced = []
    monkeypatch.setattr(
        "bibr.local.cli.dry_run.announce_downloads",
        lambda config, files, console: announced.append(list(files)),
    )
    return announced


def _capture_local(monkeypatch):
    """Record the ``LocalOptions`` the batch run opens its pipeline with."""
    seen = {}
    chewed = _stub_chew_many(monkeypatch)
    stub_factory = importlib.import_module("bibr.batch.runner").open_chew_many

    def factory(local):
        seen["local"] = local
        return stub_factory(local)

    monkeypatch.setattr("bibr.batch.runner.open_chew_many", factory)
    seen["chewed"] = chewed
    return seen


def test_batch_parser_takes_fast_and_full_and_refuses_both():
    parse = _build_parser().parse_args
    assert parse(["batch", "p.pdf", "--out", "o", "--fast"]).fast is True
    assert parse(["batch", "p.pdf", "--out", "o", "--full"]).full is True
    assert parse(["batch", "p.pdf", "--out", "o"]).fast is False
    for flags in (["--fast", "--full"], ["--fast", "--no-llm"], ["--full", "--no-llm"]):
        with pytest.raises(SystemExit):
            parse(["batch", "p.pdf", "--out", "o", *flags])


def test_chew_options_carry_the_mode():
    full = chew_options_from_config(ResolvedRunConfig("glm-http", "balanced", "vllm", mode="full"))
    assert full["mode"] == "full"
    assert full["llm_backend"] == "vllm"

    structure_only = chew_options_from_config(
        ResolvedRunConfig("glm-http", "balanced", "cloud", no_llm=True)
    )
    assert structure_only["mode"] == "full"
    assert structure_only["no_llm"] is True


def test_fast_chew_options_name_no_llm_backend():
    """A fast run starts no LLM, so naming a backend beside it would be refused by the API."""
    from bibr.api import _pipeline_kwargs

    options = chew_options_from_config(
        ResolvedRunConfig("glm-http", "balanced", "cloud", mode="fast")
    )

    assert options["mode"] == "fast"
    assert "llm_backend" not in options
    assert _pipeline_kwargs(options)["mode"] == "fast"


def test_an_ocr_route_nobody_chose_stays_unset_in_the_chew_options():
    """So a fast run can tell the automatic chain from a request to bootstrap vLLM."""
    automatic = chew_options_from_config(
        ResolvedRunConfig("paddle", "balanced", "cloud", mode="fast", ocr_requested=False)
    )
    chosen = chew_options_from_config(
        ResolvedRunConfig("paddle", "balanced", "cloud", mode="fast", ocr_requested=True)
    )

    assert automatic["ocr_backend"] is None
    assert chosen["ocr_backend"] == "paddle"


def test_batch_without_an_llm_runs_fast_and_asks_for_no_key(tmp_path, keyless, monkeypatch):
    seen = _capture_local(monkeypatch)
    monkeypatch.setattr(
        "bibr.clients.llm.preflight_credentials",
        lambda settings=None: pytest.fail("a fast run must not ask for a key"),
    )
    args = _build_parser().parse_args(["batch", str(_pdf(tmp_path)), "--out", str(tmp_path / "o")])

    _run_batch(args)

    options = seen["local"].chew_options
    assert options["mode"] == "fast"
    assert "llm_backend" not in options
    assert options["ocr_backend"] is None


def test_batch_full_flag_is_carried_to_the_pipeline(tmp_path, keyless, monkeypatch):
    seen = _capture_local(monkeypatch)
    args = _build_parser().parse_args(
        ["batch", str(_pdf(tmp_path)), "--out", str(tmp_path / "o"), "--full"]
    )

    _run_batch(args)

    options = seen["local"].chew_options
    assert options["mode"] == "full"
    assert options["llm_backend"] == "cloud"


def test_batch_fast_with_an_llm_only_option_is_a_usage_error(tmp_path, keyless, capsys):
    args = _build_parser().parse_args(
        ["batch", str(_pdf(tmp_path)), "--out", str(tmp_path / "o"), "--fast", "--refs", "llm"]
    )

    assert _run_batch(args) == 2
    assert "calls an LLM" in capsys.readouterr().err


def test_batch_dry_run_states_the_mode(tmp_path, keyless, monkeypatch, capsys):
    monkeypatch.setattr(
        "bibr.batch.runner.open_chew_many",
        lambda local: pytest.fail("dry-run must not open a pipeline"),
    )
    args = _build_parser().parse_args(
        ["batch", str(_pdf(tmp_path)), "--out", str(tmp_path / "o"), "--dry-run"]
    )

    assert _run_batch(args) == 0

    printed = capsys.readouterr().out
    assert "mode fast: no LLM calls" in printed
    assert "llm disabled (fast mode)" in printed


def test_batch_announces_the_downloads_once_for_the_whole_run(tmp_path, keyless, monkeypatch):
    _stub_chew_many(monkeypatch)
    first, second = _pdf(tmp_path, "one"), _pdf(tmp_path, "two")
    args = _build_parser().parse_args(
        ["batch", str(first), str(second), "--out", str(tmp_path / "o"), "--batch-size", "1"]
    )

    _run_batch(args)

    assert len(keyless) == 1


def test_batch_announces_no_download_for_a_run_that_cannot_start(tmp_path, keyless, monkeypatch):
    """A refused run has fetched nothing, so there is nothing to announce."""
    monkeypatch.setattr(
        "bibr.local.cli.run_config._preflight_ocr_runtime",
        lambda config: "No local OCR runtime can start on this machine for PDF input",
    )
    _stub_chew_many(monkeypatch)
    args = _build_parser().parse_args(["batch", str(_pdf(tmp_path)), "--out", str(tmp_path / "o")])

    assert _run_batch(args) == 1
    assert keyless == []


def _remote_dry_run(tmp_path, *flags):
    return _build_parser().parse_args(
        [
            "batch",
            str(_pdf(tmp_path)),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
            "--serve-url",
            "http://serve:8000/",
            "--token",
            "t0k",
            *flags,
        ]
    )


@pytest.mark.parametrize("flag", ["--fast", "--full"])
def test_remote_batch_sends_the_run_mode_as_a_form_field(tmp_path, capsys, monkeypatch, flag):
    """The serve takes a mode per request, so the flag is sent and not reported as ignored."""
    monkeypatch.delenv("AUTH_API_KEY", raising=False)

    assert _run_batch(_remote_dry_run(tmp_path, flag)) == 0

    captured = capsys.readouterr()
    assert f"mode={flag[2:]}" in captured.out
    assert flag not in captured.err


def test_remote_batch_without_a_mode_flag_leaves_the_mode_to_the_serve(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)

    assert _run_batch(_remote_dry_run(tmp_path)) == 0

    assert "mode=" not in capsys.readouterr().out


def test_remote_batch_still_warns_that_no_llm_is_ignored(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)

    assert _run_batch(_remote_dry_run(tmp_path, "--no-llm")) == 0

    assert "--no-llm" in capsys.readouterr().err


def test_batch_refuses_a_model_override_that_does_not_exist(tmp_path, monkeypatch, capsys):
    from bibr.config import Settings

    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.run_config._preflight_ocr_runtime", lambda config: None)
    # The front-role classifier is the one model a --no-llm run on a PDF asks.
    monkeypatch.setattr(Settings.ml, "front_role_model_id", str(tmp_path / "no-front-role"))
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"
    args = _build_parser().parse_args(["batch", str(pdf), "--out", str(out), "--no-llm"])
    assert _run_batch(args) == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "Model override problem: ML_FRONT_ROLE_MODEL_ID=" in err
    assert not out.exists()
    # A --no-llm run extracts no references: a stale parser setting does not block it,
    # and one warning says so.
    monkeypatch.setattr(Settings.ml, "front_role_model_id", None)
    monkeypatch.setattr(Settings, "NER_PARSER_CKPT", str(tmp_path / "no-parser"))
    monkeypatch.setattr("bibr.batch.runner.open_chew_many", lambda local: _no_pipeline())
    args = _build_parser().parse_args(
        ["batch", str(pdf), "--out", str(out), "--no-llm", "--dry-run"]
    )
    assert _run_batch(args) == 0
    err = " ".join(capsys.readouterr().err.split())
    assert "Model override problem" not in err
    assert err.count("Model override: NER_PARSER_CKPT=") == 1
    assert "--no-llm extracts no references" in err


def _no_pipeline():
    raise AssertionError("dry-run must not open a pipeline")
