"""bibr serve logging: one scrubbed sink per process, metering in both processes."""

from __future__ import annotations

import io
import logging
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.serve.logsetup import (
    configure_metering_logging,
    configure_serve_logging,
    configure_worker_logging,
    install_litserve_logging_hook,
    metering_logger,
)
from bibr.utils.redact import SecretScrubbingFilter

_LEVELS = ("bibr", "bibr.serve.metering", "httpx", "httpcore", "urllib3", "google_genai")


@pytest.fixture
def clean_logging():
    root = logging.getLogger()
    before = list(root.handlers)
    root_level = root.level
    levels = {name: logging.getLogger(name).level for name in _LEVELS}
    metering_before = list(metering_logger.handlers)
    yield
    for handler in root.handlers[:]:
        if handler not in before:
            root.removeHandler(handler)
    root.setLevel(root_level)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    for handler in metering_logger.handlers[:]:
        if handler not in metering_before:
            metering_logger.removeHandler(handler)
            handler.close()


def _settings(level: str = "info", log_path: str | None = None):
    return SimpleNamespace(
        SERVE_LOG_LEVEL=level,
        metering=SimpleNamespace(log_path=log_path, log_max_bytes=10_000, log_backup_count=1),
    )


def _sinks():
    return [h for h in logging.getLogger().handlers if getattr(h, "_bibr_serve_sink", False)]


def test_serve_sink_is_installed_once_and_emits_bibr_info(clean_logging):
    stream = io.StringIO()
    first = configure_serve_logging(_settings(), stream=stream)
    second = configure_serve_logging(_settings(), stream=stream)
    assert first is second
    assert _sinks() == [first]

    logging.getLogger("bibr.serve.test").info("worker ready on %s", "cpu")
    logging.getLogger("httpx").info("HTTP Request: GET https://llm.example/v1?key=secret")
    logging.getLogger("some.library").info("chatter")
    out = stream.getvalue()
    assert "INFO bibr.serve.test: worker ready on cpu" in out
    assert "HTTP Request" not in out
    assert "chatter" not in out


def test_serve_log_level_applies_to_bibr_loggers(clean_logging):
    stream = io.StringIO()
    configure_serve_logging(_settings("warning"), stream=stream)
    logging.getLogger("bibr.pipeline").info("not shown")
    logging.getLogger("bibr.pipeline").warning("shown")
    assert logging.getLogger("bibr").level == logging.WARNING
    assert "not shown" not in stream.getvalue()
    assert "WARNING bibr.pipeline: shown" in stream.getvalue()


def test_serve_sink_scrubs_secrets_from_messages_and_tracebacks(clean_logging):
    stream = io.StringIO()
    configure_serve_logging(_settings(), stream=stream)
    key = "AIza" + "S" * 35
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        raise RuntimeError(f"POST {url} failed")
    except RuntimeError:
        logging.getLogger("bibr.clients.llm").warning(
            "LLM call failed: %s", f"see {url}", exc_info=True
        )
    out = stream.getvalue()
    assert key not in out
    assert "key=***" in out
    assert "Traceback" in out


def test_metering_records_reach_stderr_without_a_file_sink(clean_logging):
    stream = io.StringIO()
    settings = _settings("warning")
    configure_serve_logging(settings, stream=stream)
    configure_metering_logging(settings)
    metering_logger.info('{"event": "extract"}')
    # Metering is its own switch: emitted even at SERVE_LOG_LEVEL=warning.
    assert 'bibr.serve.metering: {"event": "extract"}' in stream.getvalue()


def test_metering_file_sink_takes_records_off_stderr(clean_logging, tmp_path):
    path = tmp_path / "meter.jsonl"
    stream = io.StringIO()
    settings = _settings(log_path=str(path))
    configure_serve_logging(settings, stream=stream)
    configure_metering_logging(settings)
    metering_logger.info('{"event": "request", "auth": "Bearer sk_abcdefghijklmnop"}')
    assert path.read_text().strip() == '{"event": "request", "auth": "Bearer ***"}'
    assert "event" not in stream.getvalue()


def test_worker_logging_installs_both_sinks(clean_logging, tmp_path):
    path = tmp_path / "meter.jsonl"
    configure_worker_logging(_settings(log_path=str(path)))
    assert len(_sinks()) == 1
    # The worker rotates its own suffixed sibling: sharing one rotating file
    # between the API and worker processes loses records at rollover.
    assert [getattr(h, "_bibr_metering_sink", None) for h in metering_logger.handlers][-1] == str(
        tmp_path / "meter.worker.jsonl"
    )
    assert metering_logger.level == logging.INFO


def test_worker_metering_path_suffixing(clean_logging, tmp_path):
    configure_metering_logging(_settings(log_path=str(tmp_path / "meter.jsonl")), role="worker")
    configure_metering_logging(_settings(log_path=str(tmp_path / "plain")), role="worker")
    markers = [getattr(h, "_bibr_metering_sink", None) for h in metering_logger.handlers]
    assert str(tmp_path / "meter.worker.jsonl") in markers
    assert str(tmp_path / "plain.worker") in markers


def test_api_and_worker_handlers_keep_every_record_across_rotation(clean_logging, tmp_path):
    """Two role handlers (one process standing in for API + worker) lose nothing.

    Each role writes its own file, so each rotation is single-writer and every
    tagged record survives exactly once in its own file. On the pre-fix code
    both handlers rotate the same path and records are lost to clobbered
    backups.
    """
    import glob
    import json

    path = tmp_path / "meter.jsonl"
    settings = _settings(log_path=str(path))
    settings.metering.log_max_bytes = 1000
    settings.metering.log_backup_count = 10
    configure_metering_logging(settings)  # API role: METER_LOG_PATH itself
    configure_metering_logging(settings, role="worker")  # worker: .worker sibling
    handlers = {
        str(path): next(
            h
            for h in metering_logger.handlers
            if getattr(h, "_bibr_metering_sink", None) == str(path)
        ),
        str(tmp_path / "meter.worker.jsonl"): next(
            h
            for h in metering_logger.handlers
            if getattr(h, "_bibr_metering_sink", None) == str(tmp_path / "meter.worker.jsonl")
        ),
    }

    per_role = 100

    def _write(handler, tag):
        for seq in range(per_role):
            handler.handle(
                logging.LogRecord(
                    metering_logger.name,
                    logging.INFO,
                    __file__,
                    0,
                    json.dumps({"tag": tag, "seq": seq}),
                    (),
                    None,
                )
            )

    # One handler per simulated process: each rotates only its own file.
    _write(handlers[str(path)], "api")
    _write(handlers[str(tmp_path / "meter.worker.jsonl")], "worker")

    kept: dict[str, set] = {}
    for role_path, tag in ((str(path), "api"), (str(tmp_path / "meter.worker.jsonl"), "worker")):
        seqs = set()
        for rotated in glob.glob(role_path + "*"):
            with open(rotated) as f:
                for line in f.read().splitlines():
                    record = json.loads(line)
                    assert record["tag"] == tag
                    seqs.add(record["seq"])
        kept[tag] = seqs
    assert kept == {"api": set(range(per_role)), "worker": set(range(per_role))}


def test_litserve_hook_scrubs_the_handler_that_run_rebuilds(clean_logging):
    pytest.importorskip("litserve")
    import litserve.server as ls_server

    install_litserve_logging_hook()
    install_litserve_logging_hook()  # idempotent
    ls_server.configure_logging("info")
    library = logging.getLogger("litserve")
    assert library.handlers
    assert all(
        any(isinstance(f, SecretScrubbingFilter) for f in h.filters) for h in library.handlers
    )


def test_worker_setup_configures_logging(tmp_path, monkeypatch):
    pytest.importorskip("litserve")
    from bibr.config import Settings
    from bibr.serve.deployments import pipeline as pipeline_mod
    from bibr.serve.deployments.pipeline import BibrPipelineAPI

    calls = []
    monkeypatch.setattr(pipeline_mod, "configure_worker_logging", calls.append)
    api = BibrPipelineAPI(upload_root=tmp_path, settings=Settings)
    with (
        mock.patch("bibr.serve.deployments.layout.LayoutDetector"),
        mock.patch("bibr.serve.deployments.segmenter.SentenceSegmenter"),
        mock.patch("bibr.serve.pipeline.ServePipeline"),
    ):
        api.setup(device="cpu")
    assert calls == [api._settings]


def test_serve_main_installs_the_sink_and_hands_uvicorn_no_log_config(clean_logging, monkeypatch):
    pytest.importorskip("litserve")
    from bibr.config import Settings
    from bibr.serve import app as app_mod

    captured = {}

    class _Server:
        def run(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(app_mod, "build_server", lambda: _Server())
    monkeypatch.setattr(sys, "argv", ["bibr serve", "--port", "8123"])
    app_mod.main()

    assert len(_sinks()) == 1
    assert captured["log_config"] is None
    assert captured["log_level"] == Settings.SERVE_LOG_LEVEL
    assert captured["num_api_servers"] == 1
