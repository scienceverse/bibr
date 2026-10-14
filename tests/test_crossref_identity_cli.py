"""``--crossref-identity`` on ``bibr chew``/``batch``/``mcp`` and ``chew(crossref_identity=)``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from bibr.local.cli import ResolvedRunConfig, _build_parser, resolve_run_config


def _chew_args(*argv: str):
    return _build_parser().parse_args(["chew", *argv])


def _config(**overrides) -> ResolvedRunConfig:
    base = {"ocr_backend": "glm-llama", "memory_mode": "balanced", "llm_backend": "cloud"}
    base.update(overrides)
    return ResolvedRunConfig(**base)


def test_chew_parser_identity_flags_are_tri_state():
    assert resolve_run_config(_chew_args("p.pdf", "--ocr", "glm-llama")).crossref_identity is None
    on = resolve_run_config(_chew_args("p.pdf", "--ocr", "glm-llama", "--crossref-identity"))
    assert on.crossref_identity is True
    off = resolve_run_config(_chew_args("p.pdf", "--ocr", "glm-llama", "--no-crossref-identity"))
    assert off.crossref_identity is False
    with pytest.raises(SystemExit):
        _chew_args("p.pdf", "--crossref-identity", "--no-crossref-identity")


def test_batch_and_mcp_parsers_accept_identity_flags():
    parser = _build_parser()
    assert parser.parse_args(["batch", "x", "--out", "y", "--crossref-identity"]).crossref_identity
    assert parser.parse_args(["mcp", "--no-crossref-identity"]).no_crossref_identity


@pytest.mark.parametrize(
    ("overrides", "setting", "expected"),
    [
        ({}, False, False),
        ({}, True, True),
        ({"crossref_identity": True}, False, True),
        ({"crossref_identity": False}, True, False),
        # Reference enrichment already makes the lookup.
        ({"crossref_identity": True, "crossref": True}, False, False),
        # refs=off does not stop it; --no-llm does.
        ({"crossref_identity": True, "crossref": True, "refs": "off"}, False, True),
        ({"crossref_identity": True, "no_llm": True}, False, False),
    ],
)
def test_identity_lookup_enabled(monkeypatch, overrides, setting, expected):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.crossref, "enrich", False)
    monkeypatch.setattr(Settings.crossref, "identity", setting)
    assert _config(**overrides).identity_lookup_enabled() is expected


def test_active_stages_keep_enrich_for_the_identity_lookup(monkeypatch):
    from bibr.config import Settings

    monkeypatch.setattr(Settings.crossref, "enrich", False)
    monkeypatch.setattr(Settings.crossref, "identity", False)
    assert "enrich" not in _config().active_stages([Path("paper.pdf")])
    assert "enrich" in _config(crossref_identity=True).active_stages([Path("paper.pdf")])
    assert "enrich" not in _config(crossref_identity=True, no_llm=True).active_stages(
        [Path("paper.pdf")]
    )


def test_batch_forwards_the_identity_switch():
    from rich.console import Console

    from bibr.local.cli.batch import _form_fields, chew_options_from_config

    parse = _build_parser().parse_args
    base = ["batch", "x", "--out", "y", "--serve-url", "http://s"]
    console = Console(stderr=True)
    assert _form_fields(parse([*base, "--crossref-identity"]), console) == {
        "crossref_identity": "true"
    }
    assert _form_fields(parse([*base, "--no-crossref-identity"]), console) == {
        "crossref_identity": "false"
    }
    assert "crossref_identity" not in _form_fields(parse(base), console)
    assert chew_options_from_config(_config(crossref_identity=True))["crossref_identity"] is True


def test_run_mcp_maps_identity_flags(monkeypatch):
    from bibr.mcp_server import run_mcp

    captured = {}

    class FakeServer:
        def run(self, transport):
            captured["transport"] = transport

    def fake_build_server(*, refs=None, settings=None, allowed_dirs=None, **options):
        captured["options"] = options
        return FakeServer()

    monkeypatch.setattr("bibr.mcp_server.build_server", fake_build_server)
    args = argparse.Namespace(crossref_identity=True, no_crossref_identity=False, verbose=False)
    assert run_mcp(args) == 0
    assert captured["options"] == {"crossref_identity": True}
    args = argparse.Namespace(crossref_identity=False, no_crossref_identity=True, verbose=False)
    assert run_mcp(args) == 0
    assert captured["options"] == {"crossref_identity": False}


def test_chew_option_reaches_the_pipeline():
    from bibr.api import _pipeline_kwargs

    assert _pipeline_kwargs({"crossref_identity": True}) == {"crossref_identity": True}


@pytest.fixture
def _no_ocr_runtime_blocker(monkeypatch):
    monkeypatch.setattr("bibr.local.cli.dry_run._preflight_ocr_runtime", lambda config: None)


@pytest.mark.usefixtures("_no_ocr_runtime_blocker")
async def test_dry_run_reports_the_identity_lookup(tmp_path, capsys, monkeypatch):
    from bibr.config import Settings
    from bibr.local.cli import _run_process

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%dry-run fixture\n")
    monkeypatch.setattr(Settings.crossref, "enrich", False)
    monkeypatch.setattr(Settings.crossref, "identity", False)

    await _run_process(_chew_args(str(pdf), "--dry-run", "--crossref-identity"))

    out = capsys.readouterr().out
    assert re.search(
        r"^  crossref\s+paper DOI lookup only \(--crossref-identity\); references off",
        out,
        re.MULTILINE,
    )
