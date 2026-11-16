"""Tests for the ``bibr demo`` CLI parser.

The parser is pure ``argparse`` (gradio is imported lazily inside ``main``),
so these tests do not need the demo extra installed.
"""

import os

import pytest

from bibr.demo.server import build_parser


def test_demo_parser_accepts_refs_strategy():
    args = build_parser().parse_args(["--refs", "ner"])
    assert args.refs == "ner"


def test_demo_parser_defaults_refs_to_none():
    assert build_parser().parse_args([]).refs is None


def test_demo_parser_rejects_unknown_refs_strategy():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--refs", "bogus"])


def test_demo_parser_accepts_llama_cpp_llm_backend():
    args = build_parser().parse_args(["--llm", "llama-cpp"])
    assert args.llm == "llama-cpp"


def test_demo_parser_defaults_llm_to_none():
    assert build_parser().parse_args([]).llm is None


def test_demo_parser_rejects_unknown_llm_backend():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--llm", "sglang"])


def test_demo_parser_defaults_memory_to_none():
    assert build_parser().parse_args([]).memory is None


def test_demo_parser_accepts_explicit_memory_mode():
    args = build_parser().parse_args(["--memory", "aggressive"])

    assert args.memory == "aggressive"


def test_demo_parser_run_mode_flags_default_off():
    args = build_parser().parse_args([])
    assert args.fast is False
    assert args.full is False


@pytest.mark.parametrize("flag", ["fast", "full"])
def test_demo_parser_accepts_a_run_mode_flag(flag):
    args = build_parser().parse_args([f"--{flag}"])
    assert getattr(args, flag) is True
    assert getattr(args, "full" if flag == "fast" else "fast") is False


def test_demo_parser_run_mode_flags_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--fast", "--full"])
    assert exc_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_missing_gradio_names_the_tool_install(monkeypatch, capsys):
    """``bibr demo`` without Gradio tells an installed-tool user how to add the extra."""
    import sys

    from bibr.demo.server import main

    monkeypatch.setitem(sys.modules, "gradio", None)
    monkeypatch.setattr(sys, "argv", ["bibr-demo"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "uv tool install 'bibr[demo]'" in err
    assert "uv add 'bibr[demo]'" in err


def test_create_local_demo_without_gradio_names_the_tool_install(monkeypatch):
    import sys

    from bibr.demo import create_local_demo

    monkeypatch.setitem(sys.modules, "bibr.demo.local_app", None)
    with pytest.raises(ImportError, match=r"uv tool install 'bibr\[demo\]'.*uv add 'bibr\[demo\]'"):
        create_local_demo()


def test_private_gradio_temp_dir_is_created_and_removed_at_exit(monkeypatch):
    import os
    import stat

    from bibr.demo import server

    exit_hooks = []
    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)
    monkeypatch.setattr(
        server.atexit, "register", lambda fn, *a, **k: exit_hooks.append((fn, a, k))
    )
    server._private_gradio_temp_dir()
    path = os.environ["GRADIO_TEMP_DIR"]
    try:
        assert os.path.basename(path).startswith("bibr-demo-")
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o700
        ((fn, a, k),) = exit_hooks
        fn(*a, **k)
        assert not os.path.exists(path)
    finally:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


def test_operator_gradio_temp_dir_is_left_alone(monkeypatch, tmp_path):
    from bibr.demo import server

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    server._private_gradio_temp_dir()
    assert os.environ["GRADIO_TEMP_DIR"] == str(tmp_path)


def test_demo_logs_warnings_only_by_default():
    assert build_parser().parse_args([]).log_level == "warning"


class _Demo:
    """Stands in for the Gradio app: records how it was launched."""

    status_summary = "**Fast mode:** local models only."

    def __init__(self, launch_error: BaseException | None = None):
        self.launched: dict = {}
        self.closed: list[str] = []
        self._launch_error = launch_error

    def launch(self, **kwargs):
        self.launched.update(kwargs)
        if self._launch_error is not None:
            raise self._launch_error
        self.local_url = f"http://127.0.0.1:{kwargs['server_port']}/"

    def close_pipeline(self):
        self.closed.append("pipeline")


def _run_main(monkeypatch, demo, argv=()):
    pytest.importorskip("gradio")
    import sys

    import bibr.demo.local_app as local_app
    from bibr.demo.server import main

    monkeypatch.setattr(local_app, "create_local_demo", lambda **_kw: demo)
    monkeypatch.setattr(sys, "argv", ["bibr demo", *argv])
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("DEMO_MAX_FILE_SIZE_MB", raising=False)
    main()


def test_demo_prints_its_own_banner(monkeypatch, capsys):
    demo = _Demo()
    _run_main(monkeypatch, demo, ["--port", "7999"])

    out = " ".join(capsys.readouterr().out.split())
    assert "bibr demo is running at http://127.0.0.1:7999/" in out
    assert "Fast mode: local models only." in out  # the page's summary, without Markdown
    assert "Press Ctrl+C to stop." in out
    # Gradio's own lines (and its share=True advice) are off; bibr waits for Ctrl-C itself.
    assert demo.launched["quiet"] is True
    assert demo.launched["prevent_thread_lock"] is True
    # No "Use via API" link: the demo has no API routes, so it led to an empty page.
    assert "api" not in demo.launched["footer_links"]
    assert demo.closed == ["pipeline"]


def test_a_port_in_use_is_one_line_with_a_hint(monkeypatch, capsys):
    demo = _Demo(OSError("Cannot find empty port in range: 7999-7999. You can specify ..."))
    with pytest.raises(SystemExit) as exc_info:
        _run_main(monkeypatch, demo, ["--port", "7999"])

    assert exc_info.value.code == 2
    err = " ".join(capsys.readouterr().err.split())
    assert "Port 7999 is already in use" in err
    assert "bibr demo --port 8000" in err
    assert "GRADIO_SERVER_PORT" not in err  # Gradio's advice for its own developers
    assert demo.closed == ["pipeline"]  # the pipeline built before launch is still closed


def test_another_launch_error_is_not_reported_as_a_busy_port(monkeypatch):
    demo = _Demo(OSError("something else"))
    with pytest.raises(OSError, match="something else"):
        _run_main(monkeypatch, demo)
    assert demo.closed == ["pipeline"]


def test_httpx_request_logs_are_quiet_unless_debugging(monkeypatch):
    import logging

    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.NOTSET)
    _run_main(monkeypatch, _Demo(), ["--log-level", "info"])
    assert logging.getLogger("httpx").level == logging.WARNING
