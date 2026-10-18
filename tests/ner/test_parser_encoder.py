"""Which encoder the torch parser loads, at which revision, and whether its code may run.

v5 checkpoints ship their encoder (config, modeling code, tokenizer) in the parser
repo and load it from there at the parser's pinned revision; older checkpoints fall
back to ModernBERT-base. Custom encoder code runs only from a commit SHA or a local
directory.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from bibr.ner import parser as parser_mod
from bibr.ner.parser import DEFAULT_ENCODER, resolve_encoder, trusts_remote_code

SHA = "f508ada9013b74443d7d872ab2de181f5b497ad2"


def _hub_files(monkeypatch, present: set[str]) -> list:
    """Stub the Hub: ``present`` holds the ``repo/filename`` pairs that exist."""
    import bibr.utils.hf_cache as hf_cache

    calls = []

    def fake_download(repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        if f"{repo_id}/{filename}" not in present:
            raise FileNotFoundError(filename)
        return f"/cache/{repo_id}/{filename}"

    monkeypatch.setattr(hf_cache, "hf_download_or_cached", fake_download)
    return calls


def test_parser_repo_with_encoder_config_is_its_own_encoder_at_the_pin(monkeypatch):
    calls = _hub_files(monkeypatch, {"org/parser-v5/config.json"})
    assert resolve_encoder("org/parser-v5", SHA) == ("org/parser-v5", SHA)
    assert resolve_encoder("org/parser-v5:best.pt", SHA) == ("org/parser-v5", SHA)
    assert calls[0] == ("org/parser-v5", "config.json", SHA)


def test_parser_repo_without_encoder_config_falls_back_to_modernbert(monkeypatch):
    _hub_files(monkeypatch, set())
    assert resolve_encoder("org/parser-v4-5", "ff50a83e") == (DEFAULT_ENCODER, None)


def test_local_checkpoint_uses_a_sibling_encoder_config(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")
    assert resolve_encoder(ckpt) == (DEFAULT_ENCODER, None)
    (tmp_path / "config.json").write_text("{}")
    assert resolve_encoder(ckpt) == (str(tmp_path.resolve()), None)
    assert resolve_encoder(str(ckpt)) == (str(tmp_path.resolve()), None)


@pytest.mark.parametrize(
    ("encoder", "revision", "trusted"),
    [
        ("org/parser-v5", SHA, True),
        ("org/parser-v5", "main", False),
        ("org/parser-v5", SHA[:8], False),
        ("org/parser-v5", None, False),
        (DEFAULT_ENCODER, None, False),
    ],
)
def test_remote_code_is_trusted_only_at_a_commit_sha(encoder, revision, trusted):
    assert trusts_remote_code(encoder, revision) is trusted


def test_local_encoder_directory_is_trusted(tmp_path):
    assert trusts_remote_code(str(tmp_path), None) is True


class _StubModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return self


def _capture_loads(monkeypatch) -> dict:
    seen: dict = {}

    def fake_tokenizer(name, **kwargs):
        seen["tokenizer"] = (name, kwargs)
        return object()

    def fake_model(**kwargs):
        seen["model"] = kwargs
        return _StubModel()

    monkeypatch.setattr(parser_mod, "resolve_checkpoint", lambda ckpt, revision=None: "/c/best.pt")
    monkeypatch.setattr(parser_mod.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(parser_mod, "FeatureGatedEncoderCRF", fake_model)
    monkeypatch.setattr(parser_mod.torch, "load", lambda *a, **k: {})
    return seen


def test_refparser_loads_a_v5_encoder_from_the_pinned_parser_repo(monkeypatch):
    _hub_files(monkeypatch, {"org/parser-v5/config.json"})
    seen = _capture_loads(monkeypatch)

    parser = parser_mod.RefParser("org/parser-v5", device="cpu", revision=SHA)

    assert parser.encoder_name == "org/parser-v5"
    assert seen["tokenizer"] == ("org/parser-v5", {"revision": SHA, "trust_remote_code": True})
    assert seen["model"]["encoder_name"] == "org/parser-v5"
    assert seen["model"]["revision"] == SHA
    assert seen["model"]["trust_remote_code"] is True


def test_refparser_keeps_modernbert_without_remote_code_for_v4_5(monkeypatch):
    _hub_files(monkeypatch, set())
    seen = _capture_loads(monkeypatch)

    parser_mod.RefParser("org/parser-v4-5", device="cpu", revision="ff50a83e" * 5)

    assert seen["tokenizer"] == (DEFAULT_ENCODER, {"revision": None, "trust_remote_code": False})
    assert seen["model"]["revision"] is None
    assert seen["model"]["trust_remote_code"] is False


def test_explicit_encoder_setting_reaches_refparser(monkeypatch):
    from bibr.ner import runtime

    seen = {}
    monkeypatch.setattr(
        "bibr.utils.ml_runtime.resolve_runtime", lambda *a, **k: ("torch", None), raising=True
    )
    monkeypatch.setattr(parser_mod, "RefParser", lambda ckpt, **k: seen.update(k) or "parser")
    settings = SimpleNamespace(NER_PARSER_ENCODER="org/enc", NER_PARSER_ENCODER_REVISION=SHA)

    assert runtime.load_ref_parser("org/parser", revision="r", settings=settings) == "parser"
    assert seen["encoder_name"] == "org/enc"
    assert seen["encoder_revision"] == SHA


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)


@pytest.mark.parametrize("trust", [True, False])
def test_model_fetches_custom_encoder_code_at_the_config_revision(monkeypatch, trust):
    import bibr.ner.model as model_mod

    seen = {}

    def fake_config(name, **kwargs):
        seen["config"] = (name, kwargs)
        return "cfg"

    def fake_from_config(config, **kwargs):
        seen["from_config"] = (config, kwargs)
        return _TinyEncoder()

    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(model_mod.AutoModel, "from_config", fake_from_config)

    model_mod.FeatureGatedEncoderCRF(
        "org/parser-v5",
        num_tags=3,
        bio_tags=["O", "B-X", "I-X"],
        revision=SHA,
        trust_remote_code=trust,
    )

    assert seen["config"] == ("org/parser-v5", {"revision": SHA, "trust_remote_code": trust})
    # Without code_revision, from_config would fetch the modeling code from `main`.
    expected = {"trust_remote_code": True, "code_revision": SHA} if trust else {}
    assert seen["from_config"] == ("cfg", expected)
