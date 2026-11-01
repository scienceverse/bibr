"""Tests for the ``bibr demo`` CLI parser.

The parser is pure ``argparse`` (gradio is imported lazily inside ``main``),
so these tests do not need the demo extra installed.
"""

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


def test_create_local_demo_without_gradio_names_the_tool_install(monkeypatch):
    import sys

    from bibr.demo import create_local_demo

    monkeypatch.setitem(sys.modules, "bibr.demo.local_app", None)
    with pytest.raises(ImportError, match=r"uv tool install 'bibr\[demo\]'"):
        create_local_demo()
