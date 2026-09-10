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
    demo.launch()
"""


def create_local_demo(*args, **kwargs):
    """Create the local Gradio demo that runs the pipeline in-process."""
    try:
        from bibr.demo.local_app import create_local_demo as _create
    except ImportError as e:
        raise ImportError("Gradio not installed. Install with: uv sync --extra=demo") from e
    return _create(*args, **kwargs)


__all__ = ["create_local_demo"]
