"""CLI entry point for the bibr web demo.

Usage:
    bibr demo [OPTIONS]
"""

import argparse
import asyncio
import atexit
import logging
import os
import shutil
import sys
import tempfile
import time

logger = logging.getLogger(__name__)

# With logging's own aliases, so every value that worked before still does.
_LOG_LEVELS = ("debug", "info", "warning", "warn", "error", "critical", "fatal", "notset")


def _private_gradio_temp_dir() -> None:
    """Give Gradio a private temp folder unless the operator chose one.

    Gradio's default is the shared, predictable ``<tmp>/gradio``, which another
    local user can pre-create and own, then read uploads from or plant files
    in. Gradio reads ``GRADIO_TEMP_DIR`` when it is imported, so this must run
    first. An operator-provided value is left alone.
    """
    if os.environ.get("GRADIO_TEMP_DIR"):
        return
    # mkdtemp creates the folder with mode 0700.
    path = tempfile.mkdtemp(prefix="bibr-demo-")
    os.environ["GRADIO_TEMP_DIR"] = path
    atexit.register(shutil.rmtree, path, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``bibr demo`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="bibr demo",
        description="bibr demo: read a paper in your browser and see what bibr extracts",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help=(
            "Allow --share or a non-loopback --host without GRADIO_PASSWORD. Anyone "
            "who reaches the demo can then process papers on your LLM quota."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.lower,
        choices=_LOG_LEVELS,
        default="warning",
        help="Log level for the terminal (default: warning)",
    )
    parser.add_argument(
        "--ocr",
        default=None,
        help="OCR backend (default: from .env or auto-detected)",
    )
    parser.add_argument(
        "--memory",
        choices=["aggressive", "balanced", "keep_all"],
        default=None,
        help="Memory mode (default: PIPELINE_MEMORY_MODE from .env, else auto-detected)",
    )
    parser.add_argument(
        "--llm",
        choices=["cloud", "local", "vllm", "vllm-mlx", "rapid-mlx", "llama-cpp", "llmster"],
        default=None,
        help=(
            "LLM backend: cloud, or a managed local server. 'local' auto-picks "
            "vllm-mlx/Rapid-MLX, llama.cpp, or vLLM for this machine."
        ),
    )
    parser.add_argument(
        "--refs",
        choices=["llm", "ner"],
        default=None,
        help=(
            "Reference extraction strategy: 'ner' (default) parses each reference "
            "with the local encoder-CRF model (no per-reference LLM cost); 'llm' "
            "parses the bibliography with the configured LLM. "
            "Overrides REF_PARSE_STRATEGY for the demo."
        ),
    )
    parser.add_argument(
        "--presets",
        action="store_true",
        help="Enable preset switching dropdown in the demo UI",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Fast mode: no LLM call and no API key, local models only (as for 'bibr chew'). "
            "What the demo does with no LLM configured, without the flag. Overrides PIPELINE_MODE."
        ),
    )
    run_mode.add_argument(
        "--full",
        action="store_true",
        help=(
            "Full mode: the configured LLM reads what the local models are unsure of "
            "(as for 'bibr chew'). Overrides PIPELINE_MODE."
        ),
    )
    return parser


def main():
    """Run the bibr Gradio demo."""
    parser = build_parser()
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if level > logging.DEBUG:
        # Gradio logs each request it makes to itself through httpx: noise here.
        logging.getLogger("httpx").setLevel(max(level, logging.WARNING))

    # Gradio sends usage analytics to its maintainers unless this is off. bibr
    # keeps it off for the demo; a value already in the environment wins.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    _private_gradio_temp_dir()

    try:
        import gradio

        gradio.Blocks  # noqa: B018
    except (ImportError, AttributeError):
        from rich.console import Console

        from bibr.local.cli import ui

        ui.error(
            Console(stderr=True),
            "The web demo needs Gradio, which is not installed.",
            hint="Add the demo extra: [cyan]uv tool install 'bibr\\[demo]'[/cyan] if bibr is "
            "installed as a tool, or [cyan]uv add 'bibr\\[demo]'[/cyan] in a uv project.",
        )
        sys.exit(1)

    import gradio as gr

    from bibr.demo.local_app import (
        DEMO_CSS,
        DEMO_JS,
        _cache_lifetime,
        _max_file_size_mb,
        create_local_demo,
    )

    for read_setting, hint in (
        (_cache_lifetime, "Leave it unset to delete files after an hour."),
        (_max_file_size_mb, "Leave it unset for the 10 MB default."),
    ):
        try:
            read_setting()
        except ValueError as e:
            from rich.console import Console
            from rich.markup import escape

            from bibr.local.cli import ui

            ui.error(Console(stderr=True), escape(str(e)), hint=hint)
            sys.exit(1)

    password = os.environ.get("GRADIO_PASSWORD", "")
    username = os.environ.get("GRADIO_USERNAME", "demo")
    if not password and not args.allow_unauthenticated:
        from bibr.utils.hosts import is_loopback_host

        if args.share or not is_loopback_host(args.host):
            from rich.console import Console
            from rich.markup import escape

            from bibr.local.cli import ui

            # A share link or a network-visible bind lets anyone run papers on
            # the operator's LLM quota, and with --presets switch the
            # server-wide configuration.
            exposure = "--share" if args.share else f"--host {args.host}"
            ui.error(
                Console(stderr=True),
                escape(f"Refusing to start an unauthenticated demo with {exposure}."),
                hint="Set GRADIO_PASSWORD (and GRADIO_USERNAME), or pass "
                "--allow-unauthenticated to run it open to anyone who reaches it.",
            )
            sys.exit(2)

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": True,
        "theme": gr.themes.Soft(),
        "css": DEMO_CSS,
        "js": DEMO_JS,
        # bibr prints its own start-up lines (below), and waits for Ctrl-C itself.
        "quiet": True,
        "prevent_thread_lock": True,
        # The demo has no API routes, so no "Use via API" link to an empty page.
        "footer_links": ["gradio", "settings"],
        # Refuse an oversized upload while it arrives (HTTP 413) instead of
        # storing all of it before the size check in the click handler.
        "max_file_size": f"{_max_file_size_mb()}mb",
    }

    if password:
        launch_kwargs["auth"] = (username, password)

    try:
        demo = create_local_demo(
            ocr_backend=args.ocr,
            memory_mode=args.memory,
            llm_backend=args.llm,
            presets_enabled=args.presets,
            refs=args.refs,
            mode="fast" if args.fast else "full" if args.full else None,
        )
    except ValueError as e:
        # --fast together with --llm or --refs llm.
        from rich.console import Console
        from rich.markup import escape

        from bibr.local.cli import ui

        ui.error(Console(stderr=True), escape(str(e)))
        sys.exit(2)

    try:
        try:
            demo.launch(**launch_kwargs)
        except OSError as e:
            # Gradio's words for any port it could not bind.
            if "Cannot find empty port" not in str(e):
                raise
            from rich.console import Console

            from bibr.local.cli import ui

            ui.error(
                Console(stderr=True),
                f"Port {args.port} is already in use (is another bibr demo running?).",
                hint=f"Stop the other one, or pick another port: bibr demo --port {args.port + 1}",
            )
            sys.exit(2)
        _print_banner(demo, args, username if password else None)
        _wait_until_stopped(demo)
    except KeyboardInterrupt:
        # Ctrl-C, or a SIGTERM bibr turned into one: the normal way to stop the demo.
        print("\nStopping the demo…", file=sys.stderr)
    finally:
        _shut_down(demo)


def _print_banner(demo, args, username: str | None) -> None:
    """Where the demo is, how it reads papers, and how to stop it."""
    from rich.console import Console
    from rich.markup import escape

    console = Console()
    url = getattr(demo, "local_url", None) or f"http://{args.host}:{args.port}/"
    console.print()
    console.print(f"[bold]bibr demo[/bold] is running at [cyan]{escape(url)}[/cyan]")
    summary = getattr(demo, "status_summary", "")
    if summary:
        console.print(f"  {escape(summary.replace('**', ''))}")
    first_run_notice = getattr(demo, "first_run_notice", "")
    if first_run_notice and "downloads" in first_run_notice:
        console.print(f"  [dim]{escape(first_run_notice.replace('**', ''))}[/dim]")
    share_url = getattr(demo, "share_url", None)
    if share_url:
        console.print(f"  Public link: [cyan]{escape(share_url)}[/cyan]")
    if username:
        console.print(f"  Sign in as {escape(username)} with the GRADIO_PASSWORD you set.")
    console.print("  Press Ctrl+C to stop.")


def _wait_until_stopped(demo) -> None:
    """Block until Ctrl-C (a KeyboardInterrupt) or the server stops by itself."""
    while getattr(demo, "is_running", False):
        time.sleep(0.2)


def _shut_down(demo) -> None:
    """Close the pipeline (and the managed servers it started), then the web server.

    The pipeline goes first, on the event loop its runs used, while that loop
    still runs. A second Ctrl-C, or a pipeline that does not close in time,
    ends the wait without a traceback.
    """
    try:
        close_pipeline = getattr(demo, "close_pipeline", None)
        if close_pipeline is not None:
            close_pipeline()
        else:
            aclose_pipeline = getattr(demo, "aclose_pipeline", None)
            if aclose_pipeline is not None:
                asyncio.run(aclose_pipeline())
    except KeyboardInterrupt:
        pass
    except (Exception, asyncio.CancelledError) as e:  # noqa: BLE001 — shutting down anyway
        logger.warning("The demo's models did not close cleanly: %r", e)
        logger.debug("Pipeline close failed", exc_info=True)
    close = getattr(demo, "close", None)
    if close is not None and getattr(demo, "is_running", False):
        try:
            close(verbose=False)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
