"""
bibr Demo Module

Optional Gradio demo for interactive paper processing.
Install with: uv sync --extra=demo

Usage:
    # CLI
    bibr demo

    # Programmatic
    from bibr.demo import create_local_demo
    demo = create_local_demo()
    demo.launch(max_file_size="10mb")

Pass ``max_file_size`` so Gradio refuses an oversized upload while it arrives
(HTTP 413). ``bibr demo`` sets it from ``DEMO_MAX_FILE_SIZE_MB``; without it the
upload is stored in full before the demo's own size check rejects it.
"""


def create_local_demo(*args, **kwargs):
    """Create the local Gradio demo that runs the pipeline in-process."""
    try:
        from bibr.demo.local_app import create_local_demo as _create
    except ImportError as e:
        raise ImportError("Gradio not installed. Install with: uv sync --extra=demo") from e
    return _create(*args, **kwargs)


__all__ = ["create_local_demo"]
