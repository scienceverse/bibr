"""
bibr web demo

Optional Gradio app for reading one paper at a time in the browser.
Install the extra: uv tool install 'bibr[demo]' (bibr installed as a tool),
or uv add 'bibr[demo]' (in a uv project).

Usage:
    # CLI
    bibr demo

    # Programmatic
    from bibr.demo import create_local_demo
    from bibr.demo.local_app import DEMO_CSS, DEMO_JS

    demo = create_local_demo()
    demo.launch(max_file_size="10mb", css=DEMO_CSS, js=DEMO_JS)

Pass ``max_file_size`` so Gradio refuses an oversized upload while it arrives
(HTTP 413). ``bibr demo`` sets it from ``DEMO_MAX_FILE_SIZE_MB``; without it the
upload is stored in full before the demo's own size check rejects it.
``DEMO_CSS`` and ``DEMO_JS`` style the page as ``bibr demo`` does.
"""


def create_local_demo(*args, **kwargs):
    """Create the local Gradio demo that runs the pipeline in-process."""
    try:
        from bibr.demo.local_app import create_local_demo as _create
    except ImportError as e:
        raise ImportError(
            "The web demo needs Gradio, which is not installed. Add the demo extra: "
            "uv tool install 'bibr[demo]' if bibr is installed as a tool, or "
            "uv add 'bibr[demo]' in a uv project."
        ) from e
    return _create(*args, **kwargs)


__all__ = ["create_local_demo"]
