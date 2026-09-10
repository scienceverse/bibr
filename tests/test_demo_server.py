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
