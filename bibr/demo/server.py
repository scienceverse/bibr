"""CLI Entry Point for bibr Demo

Usage:
    bibr demo [OPTIONS]
"""

import argparse
import logging
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the ``bibr demo`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="bibr demo",
        description="bibr demo — interactive scientific paper metadata extraction",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--log-level", default="info", help="Log level")
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

    from bibr.demo.local_app import _TABLE_SCROLL_CSS, _TABLE_SCROLL_JS, create_local_demo

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": True,
        "theme": gr.themes.Soft(),
        "css": _TABLE_SCROLL_CSS,
        "js": _TABLE_SCROLL_JS,
    }

    password = os.environ.get("GRADIO_PASSWORD", "")
    username = os.environ.get("GRADIO_USERNAME", "demo")
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
