"""CLI Entry Point for bibr Demo

Usage:
    bibr demo [OPTIONS]
"""

import argparse
import logging
import os
import sys

_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``bibr demo`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="bibr demo",
        description="bibr demo — interactive scientific paper metadata extraction",
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
        "--log-level", type=str.lower, choices=_LOG_LEVELS, default="info", help="Log level"
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
            "with the local ModernBERT-CRF model (no per-reference LLM cost); 'llm' "
            "parses the bibliography with the configured LLM. "
            "Overrides REF_PARSE_STRATEGY for the demo."
        ),
    )
    parser.add_argument(
        "--presets",
        action="store_true",
        help="Enable preset switching dropdown in the demo UI",
    )
    return parser


def main():
    """Run the bibr Gradio demo."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import gradio

        gradio.Blocks  # noqa: B018
    except (ImportError, AttributeError):
        from rich.console import Console

        from bibr.local.cli import ui

        ui.error(
            Console(stderr=True),
            "Gradio is not installed.",
            hint="Install with: [cyan]uv sync --extra=demo[/cyan]",
        )
        sys.exit(1)

    import gradio as gr

    from bibr.demo.local_app import (
        _TABLE_SCROLL_CSS,
        _TABLE_SCROLL_JS,
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
        "css": _TABLE_SCROLL_CSS,
        "js": _TABLE_SCROLL_JS,
        # Refuse an oversized upload while it arrives (HTTP 413) instead of
        # storing all of it before the size check in the click handler.
        "max_file_size": f"{_max_file_size_mb()}mb",
    }

    if password:
        launch_kwargs["auth"] = (username, password)

    demo = create_local_demo(
        ocr_backend=args.ocr,
        memory_mode=args.memory,
        llm_backend=args.llm,
        presets_enabled=args.presets,
        refs=args.refs,
    )

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
