"""CLI entry point for local single-machine pipeline.

Usage:
    bibr chew paper.pdf                     # single file → stdout JSON
    bibr chew paper.pdf -o result.json      # single file → file
    bibr chew papers/ -o results/           # batch → directory
    bibr chew paper.pdf --memory aggressive # 8GB machines
    bibr chew paper.pdf --ocr-url http://localhost:8080  # external OCR server

Implementation lives in this package's submodules (``parser``, ``run_config``,
``inputs``, ``dry_run``, ``doctor``, ``presets``, ``process``); this
``__init__`` wires them together into ``main()`` and re-exports every name
consumed elsewhere in ``bibr`` (and by the test suite) under the historical
``bibr.local.cli`` path.
"""

import argparse
import asyncio
import atexit
import contextlib
import logging
import os
import signal
import sys
import threading
import warnings
from typing import NoReturn

from bibr.exceptions import BibrError, ConfigurationError
from bibr.local.cli import ui
from bibr.local.cli.doctor import (
    _check_device,
    _check_llm_local_backend,
    _check_ocr_backend,
    _check_onnx_runtime,
    _check_ref_strategies,
    _opencv_unavailable_reason,
    _probe_ocr_url,
    _run_doctor,
)
from bibr.local.cli.dry_run import (
    _dry_run_blockers,
    _dry_run_cloud_credential_blocker,
    _print_dry_run_plan,
)
from bibr.local.cli.inputs import (
    _collect_files,
    _find_stem_collisions,
    _prepare_output_path,
    _resolve_single_output_path,
    _suffix_for_stdin_payload,
    _zip_payload_suffix,
)
from bibr.local.cli.parser import (
    _EXAMPLES,
    _build_parser,
    _get_version,
    _safe_version,
    normalize_ocr_backend,
)
from bibr.local.cli.presets import _run_preset
from bibr.local.cli.process import (
    ChunkProcessor,
    _as_count,
    _format_validation_line,
    _hint_for_file_error,
    _print_validation_line,
    _run_process,
    _validation_counts,
    _write_chunk_results,
    _write_stdout_json,
)
from bibr.local.cli.run_config import (
    ResolvedRunConfig,
    _apply_runtime_settings,
    _managed_llm_model,
    _managed_llm_weight_repo,
    _preflight_local_backend,
    _resolve_llm_backend,
    resolve_run_config,
)
from bibr.utils.pages import parse_pages as _parse_pages

__all__ = [
    "BibrError",
    "ChunkProcessor",
    "ResolvedRunConfig",
    "_EXAMPLES",
    "_apply_runtime_settings",
    "_as_count",
    "_dry_run_blockers",
    "_dry_run_cloud_credential_blocker",
    "_build_parser",
    "_check_device",
    "_check_llm_local_backend",
    "_check_ocr_backend",
    "_check_onnx_runtime",
    "_check_ref_strategies",
    "_collect_files",
    "_find_stem_collisions",
    "_format_validation_line",
    "_get_version",
    "_hint_for_file_error",
    "_safe_version",
    "_managed_llm_model",
    "_managed_llm_weight_repo",
    "_opencv_unavailable_reason",
    "_parse_pages",
    "_prepare_output_path",
    "_preflight_local_backend",
    "_print_dry_run_plan",
    "_print_validation_line",
    "_probe_ocr_url",
    "_resolve_llm_backend",
    "_resolve_single_output_path",
    "_run_doctor",
    "_run_preset",
    "_run_process",
    "_suffix_for_stdin_payload",
    "_validation_counts",
    "_write_chunk_results",
    "_write_stdout_json",
    "_zip_payload_suffix",
    "entrypoint",
    "main",
    "normalize_ocr_backend",
    "resolve_run_config",
]


class _ShowOnceFilter(logging.Filter):
    """Let the first record whose message contains a marker through and drop the repeats.

    Hugging Face Hub warns that requests are unauthenticated once per process in
    each library that talks to it (and per worker), so a first run printed the
    same advice several times between the progress bars.
    """

    def __init__(self, *markers: str) -> None:
        super().__init__()
        self._markers = markers
        self._shown: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for marker in self._markers:
            if marker in message:
                if marker in self._shown:
                    return False
                self._shown.add(marker)
        return True


#: Messages that are advice repeated by every client of a library, shown once.
_REPEATED_ADVICE = ("unauthenticated requests",)


class _LiveStderrHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stderr`` at emit time.

    The OCR progress bar (``RichProgress``) is a Rich ``Live`` display; while
    it is active Rich replaces ``sys.stderr`` with a proxy that reprints
    whole lines above the bar. A plain ``StreamHandler`` binds the stream
    object once at ``basicConfig`` time, so ``bibr.*`` INFO lines emitted
    while a bar is live — routine in streaming mode, where a window's back
    half overlaps a later window's OCR — would bypass the proxy and garble
    the bar. Resolving the stream per record routes them through the proxy
    exactly when a Live display is active; otherwise behavior is identical.
    """

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        # Always dynamic; ignore the base __init__'s assignment.
        pass


def _print_error(message: str) -> None:
    """Print a fatal CLI error as a styled one-liner on stderr."""
    from rich.console import Console

    ui.error(Console(stderr=True), message)


def _suppress_progress_bars_if_not_tty() -> None:
    """Silence tqdm and HF Hub progress bars when output is being piped.

    Transformers / vllm-mlx / safetensors emit per-weight tqdm bars during
    model loading. In a TTY they redraw in place; redirected to a file or
    pipe they balloon to thousands of lines that drown out everything else.
    Set the env vars before any heavy import so per-instance ``tqdm``
    reads them on construction.
    """
    if sys.stderr.isatty():
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


# Signals the active ``_interrupt_on_termination`` has turned into an interrupt.
_received_signals: list[int] = []
# Set once the running command has nothing left to shut down.
_exit_at_once = threading.Event()


def _terminating_signal() -> int | None:
    """The SIGTERM or SIGHUP stopping the running command, if one arrived.

    ``bibr mcp`` exits with it as soon as its cleanup has run, since its stdio
    transport cannot finish unwinding while the client keeps stdin open.
    """
    return _received_signals[0] if _received_signals else None


def _exit_at_once_on_termination() -> None:
    """Make a later SIGTERM or SIGHUP end the running command at once.

    For ``bibr mcp`` once its cleanup has run: the session may have ended
    without one (the host stopped reading stdout) while the stdio transport
    still waits on a stdin the host keeps open, where an interrupt no longer
    reaches anything.
    """
    _exit_at_once.set()


def _exit_now(code: int) -> NoReturn:
    """Flush stdout and stderr, then exit without unwinding the interpreter.

    Output lost in the flush turns a success into status 120, as at a normal exit.
    """
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, ValueError):
                pass  # no stream, or one that is closed
            except Exception:  # noqa: BLE001 - the exit goes ahead whatever failed
                if stream is sys.stdout and code == 0:
                    code = 120
    finally:
        os._exit(code)


@contextlib.contextmanager
def _interrupt_on_termination():
    """Make SIGTERM and SIGHUP take Ctrl-C's path while a command runs.

    Managed inference servers run in their own session, so a closed terminal
    or SSH session never sends them SIGHUP, and Python's default SIGTERM exits
    without running ``finally`` blocks: ``kill``, ``docker stop``, ``timeout``
    or an MCP host stopping ``bibr mcp`` left vLLM or llama.cpp holding VRAM
    and the port. Raising SIGINT reaches whatever handles Ctrl-C right then
    (KeyboardInterrupt, ``asyncio.run`` cancelling its main task, the batch
    runner's graceful stop), so the same shutdown paths run. An interrupt that
    began as one of these signals then exits 128 + its number; ``bibr batch``
    exits 130 for any interrupt, and ``bibr mcp`` exits from its lifespan once
    its cleanup has run.
    """
    received = _received_signals
    received.clear()
    _exit_at_once.clear()

    def handler(signum, _frame) -> None:
        received.append(signum)
        if _exit_at_once.is_set():
            _exit_now(128 + received[0])
        if callable(signal.getsignal(signal.SIGINT)):
            signal.raise_signal(signal.SIGINT)
        else:
            # SIGINT is ignored (a background job): interrupt directly.
            raise KeyboardInterrupt

    previous = {}
    if os.name == "posix" and threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGHUP):
            # Keep an inherited SIG_IGN (nohup) or an embedder's own handler.
            if signal.getsignal(signum) is signal.SIG_DFL:
                previous[signum] = signal.signal(signum, handler)
    try:
        yield
    except KeyboardInterrupt:
        if not received:
            raise
        sys.exit(128 + received[0])
    finally:
        received.clear()
        _exit_at_once.clear()
        for signum, old in previous.items():
            signal.signal(signum, old)


def main():
    """Entry point for ``bibr`` CLI command."""
    # serve runs under LitServe and uvicorn, which install, restore and
    # re-raise their own SIGINT/SIGTERM handlers.
    serving = len(sys.argv) >= 2 and sys.argv[1] == "serve"
    with contextlib.nullcontext() if serving else _interrupt_on_termination():
        _main()


def _main():
    ui.configure_output_streams()
    _suppress_progress_bars_if_not_tty()
    parser = _build_parser()

    # For delegated subcommands (setup, serve, demo), we need to intercept
    # before argparse consumes their flags. Check sys.argv manually.
    if len(sys.argv) >= 2 and sys.argv[1] in ("setup", "serve", "demo"):
        subcmd = sys.argv[1]
        # Remove our subcommand from argv so the delegate sees its own args
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        try:
            if subcmd == "setup":
                from bibr.setup_wizard import main as setup_main

                setup_main()
            elif subcmd == "serve":
                from bibr.serve.app import main as serve_main

                serve_main()
            elif subcmd == "demo":
                from bibr.demo.server import main as demo_main

                demo_main()
        except ConfigurationError as e:
            _print_error(str(e))
            sys.exit(1)
        return

    args = parser.parse_args()

    if args.command is None:
        # Bare ``bibr``: the main parser's format_help() renders the designed
        # landing screen (brand header, command list, quick start).
        parser.print_help()
        sys.exit(0)

    # Configure logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[_LiveStderrHandler()],
    )
    # Scrub credential-shaped substrings (?key=…, Bearer …, AIza…/sk-…) from every
    # emitted record, including SDK tracebacks logged with exc_info (audit L12).
    from bibr.utils.redact import install_secret_scrubbing

    install_secret_scrubbing(*logging.getLogger().handlers)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_ShowOnceFilter(*_REPEATED_ADVICE))
    # The same advice raised as a Python warning rather than a log record.
    warnings.filterwarnings("once", message=".*unauthenticated requests.*")
    # Always show INFO for bibr.* even without --verbose so the user sees stage
    # transitions (native-text bypass, OCR readiness) without enabling -v.
    if log_level > logging.INFO:
        for name in ("bibr.local", "bibr.pipeline", "bibr.structure", "bibr.extract"):
            logging.getLogger(name).setLevel(logging.INFO)
    # Silence noisy 3rd-party HTTP/transport loggers even with -v — they bury
    # our own messages under thousands of frame-level DEBUG lines.
    for name in (
        "httpcore",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "filelock",
        "asyncio",
        "hf_xet",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # A bad .env surfaces as ConfigurationError the moment any subcommand
    # touches Settings; convert it to a clean stderr message + exit 1 for every
    # path (chew catches it inside its own BibrError handler; doctor degrades it
    # into a failed check itself and never re-raises).
    try:
        if args.command == "chew":
            try:
                asyncio.run(_run_process(args))
            except BibrError as e:
                # Convert known bibr exceptions to a clean stderr message + exit 1.
                # Without this, users see a Python traceback for things like
                # "vllm-mlx not installed" or "OCR server unreachable" that we
                # already provide a clear message for.
                _print_error(str(e))
                sys.exit(1)
        elif args.command == "batch":
            from bibr.local.cli.batch import _run_batch

            try:
                sys.exit(_run_batch(args))
            except KeyboardInterrupt:
                # The executors already map an interrupt mid-run to 130; one
                # during discovery, planning, teardown or the closing table
                # rebuild exits 130 too, not 128 + the signal or a traceback.
                from bibr.batch.runner import EXIT_INTERRUPTED

                sys.exit(EXIT_INTERRUPTED)
            except BibrError as e:
                _print_error(str(e))
                sys.exit(1)
            except ValueError as e:
                # ``Chewer`` preflight: missing LLM credentials surface as ValueError.
                _print_error(str(e))
                sys.exit(1)
        elif args.command == "doctor":
            _run_doctor()
        elif args.command == "ref-model":
            from pathlib import Path

            from bibr.local.ref_model import download

            try:
                print(str(download(Path(args.output))))
            except BibrError as e:
                _print_error(str(e))
                sys.exit(1)
        elif args.command == "inspect":
            from bibr.local.inspect import run_inspect

            sys.exit(run_inspect(args.json_file))
        elif args.command == "tables":
            from bibr.local.cli.tables import run_tables

            sys.exit(run_tables(args))
        elif args.command == "references":
            from bibr.local.cli.references import run_references

            sys.exit(run_references(args))
        elif args.command == "mcp":
            try:
                from bibr.mcp_server import run_mcp
            except ModuleNotFoundError as e:
                if e.name and e.name.split(".")[0] == "mcp":
                    _print_error(
                        "bibr mcp requires the optional 'mcp' dependency — install it "
                        "with: uv tool install 'bibr[mcp]' "
                        "(in a source checkout: uv sync --extra mcp)."
                    )
                    sys.exit(1)
                raise
            try:
                sys.exit(run_mcp(args))
            except BibrError as e:
                _print_error(str(e))
                sys.exit(1)
            # No ``except ValueError`` here: run_mcp turns the build-time
            # ValueError (a bad option or --allow-dir) into ConfigurationError, so a
            # ValueError escaping the server session keeps its traceback.
        elif args.command == "preset":
            # Pull the preset subparser out of argparse's tree so ``bibr preset``
            # (no subcommand) can render its help via the standard argparse path
            # instead of a hand-rolled usage line.
            preset_parser = None
            for action in parser._actions:  # noqa: SLF001
                if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                    preset_parser = action.choices.get("preset")
                    break
            _run_preset(args, parser=preset_parser)
        elif args.command == "config":
            # Pull the config subparser out of argparse's tree so ``bibr config``
            # (no subcommand) can render its help via the standard argparse path
            # instead of a hand-rolled usage line (mirrors ``bibr preset`` above).
            config_parser = None
            for action in parser._actions:  # noqa: SLF001
                if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                    config_parser = action.choices.get("config")
                    break
            from bibr.config_cli import run_config_command

            sys.exit(run_config_command(args, parser=config_parser))
        else:
            parser.print_help()
            sys.exit(1)
    except ConfigurationError as e:
        _print_error(str(e))
        sys.exit(1)


# Commands whose process ends right after a run that loaded several ONNX models
# and a few GB of weights. See ``entrypoint``.
_HARD_EXIT_COMMANDS = frozenset({"batch", "chew"})


def _hard_exit_wanted() -> bool:
    """Whether this ``bibr`` invocation should skip interpreter teardown.

    ``BIBR_FAST_EXIT=0`` (or ``false``/``no``/``off``) opts out.
    """
    if os.environ.get("BIBR_FAST_EXIT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return len(sys.argv) >= 2 and sys.argv[1] in _HARD_EXIT_COMMANDS


def _exit_status(code: object) -> int:
    """The process status a ``SystemExit`` carries (``None`` is 0, a message is 1)."""
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def _hard_exit_hook(status: list[int | None]) -> None:
    """Last ``atexit`` handler: end the process once everything else has run.

    A finished ``bibr batch`` or ``bibr chew`` has written and fsynced its
    outputs; what is left of the interpreter's teardown is freeing the loaded
    models object by object. On a machine under memory pressure that pages the
    swapped-out heap back in first, which measured 17 to 20 s after the last log
    line against 1.3 s without it. Registered before anything else, so the other
    handlers (temporary-file cleanup, ``logging``) have already run.
    """
    if status[0] is None:
        # No normal finish (an unhandled exception): leave Python's own path.
        return
    with contextlib.suppress(Exception):
        logging.shutdown()
    _exit_now(status[0])


def entrypoint() -> None:
    """Console-script entry point: ``main`` plus a hard exit for ``batch``/``chew``.

    ``main`` itself stays free of ``os._exit``, so tests and embedders that call
    it keep the normal ``SystemExit`` and teardown.
    """
    if not _hard_exit_wanted():
        main()
        return
    status: list[int | None] = [None]
    atexit.register(_hard_exit_hook, status)
    try:
        main()
    except SystemExit as exc:
        status[0] = _exit_status(exc.code)
        raise
    status[0] = 0


if __name__ == "__main__":
    entrypoint()
