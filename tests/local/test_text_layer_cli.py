"""``--ocr text-layer`` and ``OCR_FALLBACK_TO_TEXT_LAYER`` in the CLI and the OCR registry.

The text-layer backend runs no OCR model, so it has no runtime to vet; with
``OCR_FALLBACK_TO_TEXT_LAYER=auto`` an automatic backend that cannot start here
becomes the text layer with one warning instead of refusing the run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bibr.config import GlobalSettings, Settings
from bibr.local.cli import _build_parser
from bibr.local.cli.run_config import ResolvedRunConfig, apply_text_layer_fallback

# The automatic chain's refusal, as run_config._ocr_runtime_blocker words it.
_BLOCKER = (
    "No local OCR runtime can start on this machine for PDF input:\n"
    "  - glm-llama: llama.cpp not installed\n"
    "Alternatives: point --ocr-url at an external OCR server, use a cloud vision backend "
    "(--ocr gemini|openai|anthropic), or read born-digital PDFs from their text layer "
    "without OCR (--ocr text-layer, or OCR_FALLBACK_TO_TEXT_LAYER=auto)."
)


def _config(backend: str = "paddle", url: str | None = None) -> ResolvedRunConfig:
    return ResolvedRunConfig(
        ocr_backend=backend,
        memory_mode="balanced",
        llm_backend="vllm",
        ocr_url=url,
        ocr_model="some-ocr-model",
        ocr_profile="glm",
        ocr_summary_model="some-ocr-model",
    )


def _blocked(config) -> str | None:
    return None if config.ocr_backend == "text-layer" else _BLOCKER


@pytest.fixture
def auto(monkeypatch):
    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", "auto")


# --- Registry and identity -----------------------------------------------------


def test_text_layer_is_a_single_model_free_candidate():
    from bibr.ocr.registry import resolve_backend_candidates

    [candidate] = resolve_backend_candidates("text-layer", GlobalSettings())

    assert (candidate.backend, candidate.model, candidate.profile) == (
        "text-layer",
        "pdf-text-layer",
        None,
    )


def test_an_ocr_url_keeps_an_explicit_text_layer_request_only():
    from bibr.ocr.registry import resolve_url_backend, url_request

    url = "http://ocr.invalid:8080"
    assert resolve_url_backend(url_request("text-layer", None), url) == "text-layer"
    # A configured OCR_BACKEND=text-layer gives way to the server the caller named.
    assert url_request(None, "text-layer") is None


def test_runtime_identity_and_served_model():
    from bibr.ocr.profiles import resolve_ocr_runtime_identity, resolve_served_model
    from bibr.pipeline.context import RunConfig

    identity = resolve_ocr_runtime_identity(RunConfig(ocr_backend="text-layer"), GlobalSettings())

    assert (identity.backend, identity.model, identity.profile) == (
        "text-layer",
        "pdf-text-layer",
        None,
    )
    assert identity.normalizer_version == "text-layer-v1"
    served = resolve_served_model(
        requested_backend="text-layer",
        concrete_backend="text-layer",
        settings=GlobalSettings(),
        explicit_profile=None,
    )
    assert served == "pdf-text-layer"


def test_settings_accept_the_backend_and_the_fallback(monkeypatch):
    from bibr.config import OcrOptions

    monkeypatch.setenv("OCR_BACKEND", "text-layer")
    monkeypatch.setenv("OCR_FALLBACK_TO_TEXT_LAYER", "auto")

    options = OcrOptions()

    assert (options.backend, options.fallback_to_text_layer) == ("text-layer", "auto")
    assert OcrOptions.model_fields["fallback_to_text_layer"].default == "off"


def test_registry_creates_a_client_that_never_starts():
    from bibr.ocr import registry, text_layer  # noqa: F401 — registers text-layer

    client = registry.create("text-layer")

    assert client.loaded is True
    with pytest.raises(RuntimeError, match="no OCR model"):
        import asyncio

        asyncio.run(client.recognize(None, "prompt"))


# --- Parser and preflight -------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["chew", "paper.pdf", "--ocr", "text-layer"],
        ["batch", "papers", "--out", "out", "--ocr", "text-layer"],
        ["mcp", "--ocr", "text-layer"],
    ],
    ids=["chew", "batch", "mcp"],
)
def test_parser_accepts_text_layer(argv):
    assert _build_parser().parse_args(argv).ocr == "text-layer"


def test_text_layer_needs_no_ocr_runtime(monkeypatch):
    from bibr.local.cli import run_config

    monkeypatch.setattr(
        run_config, "_ocr_runtime_blocker", lambda backend: pytest.fail("not vetted")
    )

    assert run_config._preflight_ocr_runtime(_config("text-layer")) is None


def test_the_refusal_names_the_text_layer_backend(monkeypatch):
    from bibr.local.cli import run_config
    from bibr.ocr import registry

    monkeypatch.setattr(registry.sys, "platform", "linux")
    monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: None)
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)

    message = run_config._preflight_ocr_runtime(SimpleNamespace(ocr_backend="paddle", ocr_url=None))

    assert message is not None
    assert "--ocr text-layer" in message and "OCR_FALLBACK_TO_TEXT_LAYER=auto" in message


def test_fallback_swaps_a_blocked_automatic_backend(auto):
    config, warning = apply_text_layer_fallback(_config(), _blocked)

    assert config.ocr_backend == "text-layer"
    assert (config.ocr_model, config.ocr_profile) == (None, None)
    assert config.ocr_summary_model == "pdf-text-layer"
    assert config.ocr_text_layer_fallback == "glm-llama: llama.cpp not installed"
    assert warning is not None
    assert "text layer" in warning and "scanned PDFs fail" in warning
    assert "  - glm-llama: llama.cpp not installed" in warning
    # The refusal's alternatives suggest the very setting that is on: dropped.
    assert "Alternatives" not in warning
    assert warning.count("OCR_FALLBACK_TO_TEXT_LAYER=auto") == 1


def test_the_pipeline_gets_the_automatic_backend_back_after_the_fallback(auto):
    """So the library finds the same blocker and marks each export it reads."""
    swapped, _ = apply_text_layer_fallback(_config(), _blocked)

    assert swapped.pipeline_ocr_backend == "paddle"
    assert _config("text-layer").pipeline_ocr_backend == "text-layer"
    assert _config().pipeline_ocr_backend == "paddle"


def test_a_refusal_without_a_list_is_kept_up_to_its_alternatives(auto):
    def single(config):
        return None if config.ocr_backend == "text-layer" else "cannot start. Alternatives: x"

    config, warning = apply_text_layer_fallback(_config(), single)

    assert config.ocr_text_layer_fallback == "cannot start."
    assert warning is not None and "Alternatives" not in warning


@pytest.mark.parametrize(
    ("config", "preflight"),
    [
        (_config(), lambda config: None),
        (_config(url="http://ocr.invalid:8080"), _blocked),
        (_config("glm-llama"), _blocked),
    ],
    ids=["runtime-ok", "ocr-url", "explicit-backend"],
)
def test_fallback_leaves_other_runs_alone(auto, config, preflight):
    assert apply_text_layer_fallback(config, preflight) == (config, None)


def test_fallback_is_off_by_default():
    config = _config()

    assert apply_text_layer_fallback(config, _blocked) == (config, None)


# --- chew and batch ---------------------------------------------------------------


class _Recorded(Exception):
    pass


def _recording_pipeline(seen: dict):
    class _Pipeline:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _Recorded

    return _Pipeline


async def test_chew_falls_back_with_one_warning(tmp_path, monkeypatch, capsys, auto):
    from bibr.local.cli import _run_process

    seen: dict = {}
    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.process._preflight_ocr_runtime", _blocked)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _recording_pipeline(seen))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(_Recorded):
        await _run_process(_build_parser().parse_args(["chew", str(pdf), "--no-llm"]))

    # The pipeline gets the automatic backend: it falls back itself and marks
    # each export with OCR_BACKEND_UNAVAILABLE, as a library run does.
    assert seen["ocr_backend"] == "paddle"
    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert printed.count("OCR_FALLBACK_TO_TEXT_LAYER=auto") == 1
    assert "llama.cpp not installed" in printed
    assert "Alternatives" not in printed
    # The run summary names the backend the run will use, not paddle.
    assert "text-layer (no OCR backend can start here)" in printed
    assert "paddle (automatic" not in printed


async def test_chew_summary_comes_before_the_fallback_warning(tmp_path, monkeypatch, capsys, auto):
    from bibr.local.cli import _run_process

    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.process._preflight_ocr_runtime", _blocked)
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _recording_pipeline({}))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(_Recorded):
        await _run_process(_build_parser().parse_args(["chew", str(pdf), "--no-llm"]))

    printed = capsys.readouterr().err
    assert printed.index("text-layer (no OCR backend can start here)") < printed.index(
        "No OCR backend can start on this machine"
    )


async def test_chew_without_fallback_refuses_and_names_the_option(tmp_path, monkeypatch, capsys):
    from bibr.local.cli import _run_process
    from bibr.local.cli import run_config as run_config_module

    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        run_config_module,
        "_ocr_runtime_blocker",
        lambda backend: f"OCR backend cannot start here. (--ocr text-layer) [{backend}]",
    )
    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _recording_pipeline({}))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(SystemExit) as exc:
        await _run_process(_build_parser().parse_args(["chew", str(pdf), "--no-llm"]))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--ocr text-layer" in err
    assert "OCR_FALLBACK_TO_TEXT_LAYER=auto: reading" not in err


async def test_chew_dry_run_shows_the_text_layer_plan(tmp_path, monkeypatch, capsys, auto):
    from bibr.local.cli import _run_process

    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.process._preflight_ocr_runtime", _blocked)
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", _blocked)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    await _run_process(_build_parser().parse_args(["chew", str(pdf), "--no-llm", "--dry-run"]))

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert "OCR backend: text-layer" in printed
    assert "OCR model: pdf-text-layer" in printed
    assert "none (no OCR model)" in printed
    assert printed.count("OCR_FALLBACK_TO_TEXT_LAYER=auto") == 1


def test_batch_falls_back_and_warns_once(monkeypatch, capsys, auto, tmp_path):
    from rich.console import Console

    from bibr.local.cli.batch import _local_options

    monkeypatch.setattr("bibr.local.cli._opencv_unavailable_reason", lambda: None)
    monkeypatch.setattr("bibr.local.cli.run_config._preflight_ocr_runtime", _blocked)
    args = _build_parser().parse_args(["batch", str(tmp_path), "--out", "out", "--no-llm"])
    console = Console(stderr=True)

    options = _local_options(args, console)

    assert options.chew_options["ocr_backend"] == "paddle"
    assert any(line.startswith("ocr text-layer") for line in options.summary)
    pdfs = [tmp_path / "a.pdf"]
    assert options.preflight([tmp_path / "a.docx"]) is None
    assert options.preflight(pdfs) is None
    assert options.preflight(pdfs) is None
    assert capsys.readouterr().err.count("OCR_FALLBACK_TO_TEXT_LAYER=auto") == 1


# --- doctor -------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def ok(self, msg: str) -> None:
        self.calls.append(("ok", msg, ""))

    def warn(self, msg: str, hint: str = "") -> None:
        self.calls.append(("warn", msg, hint))

    def fail(self, msg: str, hint: str = "") -> None:
        self.calls.append(("fail", msg, hint))


def test_doctor_passes_the_text_layer_backend(monkeypatch):
    from bibr.local.cli import _check_ocr_backend

    monkeypatch.setattr(Settings.ocr, "backend", "text-layer")
    rec = _Recorder()

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    [(kind, message, _)] = rec.calls
    assert kind == "ok"
    assert "text-layer" in message and "scanned PDFs fail" in message


@pytest.mark.parametrize("fallback", ["off", "auto"])
def test_doctor_warns_instead_of_failing_with_the_fallback(monkeypatch, fallback):
    from bibr.local.cli import _check_ocr_backend
    from bibr.ocr.registry import OcrBackendCandidate

    monkeypatch.setattr(Settings.ocr, "backend", "paddle")
    monkeypatch.setattr(Settings.ocr, "fallback_to_text_layer", fallback)
    monkeypatch.setattr(
        "bibr.ocr.registry.resolve_backend_candidates",
        lambda backend, settings: (OcrBackendCandidate("glm-llama", "glm-ocr", "glm"),),
    )
    monkeypatch.setattr("bibr.local.llama_cpp.find_llama_server", lambda: None)
    monkeypatch.setattr("bibr.local.llama_cpp.install_hint", lambda: "Install llama.cpp")
    rec = _Recorder()

    _check_ocr_backend(rec.ok, rec.warn, rec.fail)

    [(kind, message, hint)] = rec.calls
    if fallback == "off":
        assert (kind, hint) == ("fail", "Install llama.cpp")
    else:
        assert kind == "warn"
        assert "falling back to the PDF text layer" in message
        assert "OCR_FALLBACK_TO_TEXT_LAYER=auto" in hint and "Install llama.cpp" in hint
