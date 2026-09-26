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
import logging
import os
import sys

from bibr.exceptions import BibrError, ConfigurationError
from bibr.local.cli import ui
from bibr.local.cli.doctor import (
    _check_device,
    _check_llm_local_backend,
    _check_ocr_backend,
    _check_ref_strategies,
    _opencv_unavailable_reason,
    _probe_ocr_url,
    _run_doctor,
)
from bibr.local.cli.dry_run import _print_dry_run_plan
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
    normalize_ocr_backend,
)
from bibr.local.cli.presets import _run_preset
from bibr.local.cli.process import (
    ChunkProcessor,
    _as_count,
    _format_validation_line,
    _print_validation_line,
    _run_process,
    _validation_counts,
    _write_chunk_results,
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
    "_build_parser",
    "_check_device",
    "_check_llm_local_backend",
    "_check_ocr_backend",
    "_check_ref_strategies",
    "_collect_files",
    "_find_stem_collisions",
    "_format_validation_line",
    "_get_version",
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
    "_zip_payload_suffix",
    "main",
    "normalize_ocr_backend",
    "resolve_run_config",
]


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


def main():
    """Entry point for ``bibr`` CLI command."""
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
            except BibrError as e:
                _print_error(str(e))
                sys.exit(1)
            except ValueError as e:
                # ``Chewer`` preflight: missing LLM credentials surface as ValueError.
                _print_error(str(e))
                sys.exit(1)
        elif args.command == "doctor":
            _run_doctor()
        elif args.command == "inspect":
            from bibr.local.inspect import run_inspect

            sys.exit(run_inspect(args.json_file))
        elif args.command == "tables":
            from bibr.local.cli.tables import run_tables

            sys.exit(run_tables(args))
        elif args.command == "mcp":
            try:
                from bibr.mcp_server import run_mcp
            except ModuleNotFoundError as e:
                if e.name and e.name.split(".")[0] == "mcp":
                    _print_error(
                        "bibr mcp requires the optional 'mcp' dependency — install it "
                        "with 'uv sync --extra mcp' (source checkout) or "
                        "pip install 'bibr[mcp]'."
                    )
                    sys.exit(1)
                raise
            try:
                sys.exit(run_mcp(args))
            except BibrError as e:
                _print_error(str(e))
                sys.exit(1)
            # No ``except ValueError`` here: run_mcp turns the build-time
            # credential ValueError into ConfigurationError, so a ValueError
            # escaping the server session keeps its traceback.
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


if __name__ == "__main__":
    main()
