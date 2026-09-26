"""Logging for ``bibr serve``: one configured, scrubbed sink per process.

``bibr serve`` returns from the CLI before the CLI's own ``basicConfig`` runs, and
LitServe spawns the inference worker as a fresh interpreter, so without this module
neither process has a root handler: ``bibr.*`` INFO records (metering included) are
dropped, and warnings fall through ``logging.lastResort`` — unformatted and
unscrubbed, SDK URLs carrying ``?key=`` included. :func:`configure_serve_logging`
runs in the API process (``main``) and :func:`configure_worker_logging` in the
inference worker (``BibrPipelineAPI.setup``); both are idempotent.

Library loggers that keep their own handlers are covered too: LitServe's handler
(rebuilt by ``LitServer.run``, hence :func:`install_litserve_logging_hook`) gets the
scrubber, and uvicorn is started without its own ``log_config`` so its records
propagate to the serve sink.
"""

from __future__ import annotations

import functools
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bibr.utils.redact import install_secret_scrubbing

_SINK_MARKER = "_bibr_serve_sink"
_METERING_SINK_MARKER = "_bibr_metering_sink"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# HTTP client libraries log full request URLs at INFO; the CLI pins them too.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "google_genai")
_LIBRARY_LOGGERS = ("litserve", "uvicorn", "uvicorn.error", "uvicorn.access")

# Dedicated logger for per-request / per-extraction usage metering (D2). One
# JSON line per record; formatting is done at the call site with json.dumps.
metering_logger = logging.getLogger("bibr.serve.metering")


def _metering_has_sink() -> bool:
    return any(getattr(h, _METERING_SINK_MARKER, None) for h in metering_logger.handlers)


class _MeteringSinkFilter(logging.Filter):
    """Keep metering records off stderr once METER_LOG_PATH routes them to a file."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return record.name != metering_logger.name or not _metering_has_sink()


def serve_log_level(settings) -> int:
    return getattr(logging, str(settings.SERVE_LOG_LEVEL).upper(), logging.INFO)


def _serve_sink(root: logging.Logger) -> logging.Handler | None:
    return next((h for h in root.handlers if getattr(h, _SINK_MARKER, False)), None)


def configure_serve_logging(settings, *, stream=None) -> logging.Handler:
    """Install the process-wide serve log sink (idempotent) and return it.

    A single stderr handler on the root logger, formatted and secret-scrubbed;
    ``bibr.*`` at ``SERVE_LOG_LEVEL``, the root (third-party) level no lower than
    WARNING, HTTP client libraries held at WARNING. ``stream`` is for tests.
    """
    root = logging.getLogger()
    sink = _serve_sink(root)
    if sink is None:
        sink = logging.StreamHandler(stream or sys.stderr)
        sink.setFormatter(logging.Formatter(_FORMAT))
        sink.addFilter(_MeteringSinkFilter())
        setattr(sink, _SINK_MARKER, True)
        root.addHandler(sink)
    elif stream is not None:
        sink.setStream(stream)

    level = serve_log_level(settings)
    root.setLevel(max(logging.WARNING, level))
    logging.getLogger("bibr").setLevel(level)
    # Metering has its own switch (METER_ENABLED); its records exist at any level.
    metering_logger.setLevel(logging.INFO)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, level))
    scrub_library_handlers()
    return sink


def configure_metering_logging(settings, *, role: str = "api") -> None:
    """Ensure metering records are emitted, and (idempotently) attach a file sink.

    Sets the metering logger to INFO so records aren't dropped at source, and —
    when ``METER_LOG_PATH`` is set — attaches a single size-capped JSONL
    ``RotatingFileHandler`` (the serve sink then stops echoing them to stderr).
    Rotation is what bounds disk use: the metering middleware runs outside the
    auth gate (it deliberately logs 401s), so unauthenticated request spam would
    otherwise grow the log without bound and exhaust disk (audit M5). Idempotent
    across repeated ``build_server`` calls (tests) so handlers don't accumulate.

    ``role`` selects the file: the API process writes ``METER_LOG_PATH``
    itself, while the spawned inference worker (``role="worker"``) writes a
    sibling ``<stem>.worker<suffix>`` file. ``RotatingFileHandler`` is not
    multi-process safe — two processes rotating one path rename each other's
    live file and drop records — so exactly one process may rotate each file.
    """
    metering_logger.setLevel(logging.INFO)
    log_path = settings.metering.log_path
    if not log_path:
        return
    if role == "worker":
        stemmed = Path(log_path)
        suffix = stemmed.suffix
        log_path = (
            str(stemmed.with_name(f"{stemmed.stem}.worker{suffix}"))
            if suffix
            else f"{log_path}.worker"
        )
    for handler in metering_logger.handlers:
        if getattr(handler, _METERING_SINK_MARKER, None) == log_path:
            return
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.metering.log_max_bytes,
        backupCount=settings.metering.log_backup_count,
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(file_handler, _METERING_SINK_MARKER, log_path)
    install_secret_scrubbing(file_handler)
    metering_logger.addHandler(file_handler)


def scrub_library_handlers() -> None:
    """Attach the secret scrubber to every handler that writes serve logs."""
    targets: list[logging.Handler] = list(logging.getLogger().handlers)
    for name in _LIBRARY_LOGGERS:
        targets.extend(logging.getLogger(name).handlers)
    targets.extend(metering_logger.handlers)
    if targets:
        install_secret_scrubbing(*targets)


def install_litserve_logging_hook() -> None:
    """Re-scrub after ``LitServer.run`` rebuilds the ``litserve`` handler.

    ``run()`` calls ``configure_logging``, which discards every handler on the
    ``litserve`` logger — and with it the filter :func:`scrub_library_handlers`
    attached at build time. Wrap it so the replacement is scrubbed as well.
    """
    try:
        import litserve.server as ls_server
    except ImportError:  # pragma: no cover - serve requires litserve
        return
    original = ls_server.configure_logging
    if getattr(original, "_bibr_scrubbing_hook", False):
        return

    @functools.wraps(original)
    def configure_logging(*args, **kwargs):
        result = original(*args, **kwargs)
        scrub_library_handlers()
        return result

    configure_logging._bibr_scrubbing_hook = True  # type: ignore[attr-defined]
    ls_server.configure_logging = configure_logging


def configure_worker_logging(settings) -> None:
    """Logging for the spawned inference worker: serve sink plus metering sink.

    The worker is a fresh interpreter (LitServe uses the ``spawn`` context), so
    nothing configured in the API process reaches it; without this every
    per-extraction metering record — the ones carrying LLM token usage — was
    dropped at source, and worker warnings went out unformatted and unscrubbed.
    The worker's metering file is a ``.worker``-suffixed sibling of
    ``METER_LOG_PATH`` (see ``configure_metering_logging``): sharing one
    rotating file between processes loses records at rollover.
    """
    configure_serve_logging(settings)
    configure_metering_logging(settings, role="worker")
