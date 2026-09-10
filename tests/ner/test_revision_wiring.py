"""The pinned ``revision`` must reach ``resolve_checkpoint``.

Two seams: the NER constructors forward their ``revision`` arg to the resolver,
and the extractor singletons pass ``Settings.NER_*_REVISION`` into them.
"""

from __future__ import annotations

import pytest

from bibr.config import Settings


class _StubModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return self


def test_config_exposes_pinned_ner_revisions():
    # Defaults must be real commit SHAs (40 hex chars), not a moving branch —
    # that is what makes loads reproducible and lets hf_hub_download skip the
    # network when the snapshot is cached.
    for rev in (Settings.NER_SEG_REVISION, Settings.NER_PARSER_REVISION):
        assert len(rev) == 40
        assert all(c in "0123456789abcdef" for c in rev)


def test_refparser_forwards_revision_to_resolver(monkeypatch):
    pytest.importorskip("torch")
    from bibr.ner import parser as parser_mod

    seen = {}

    def fake_resolve(ckpt, revision=None):
        seen.update(ckpt=ckpt, revision=revision)
        return "/cache/best.pt"

    monkeypatch.setattr(parser_mod, "resolve_checkpoint", fake_resolve)
    monkeypatch.setattr(parser_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(parser_mod, "FeatureGatedEncoderCRF", lambda **k: _StubModel())
    monkeypatch.setattr(parser_mod.torch, "load", lambda *a, **k: {})

    parser_mod.RefParser("org/repo", device="cpu", revision="parser-sha")

    assert seen == {"ckpt": "org/repo", "revision": "parser-sha"}


def test_refsegmenter_forwards_revision_to_resolver(monkeypatch):
    pytest.importorskip("torch")
    from bibr.ner import segmenter as seg_mod

    seen = {}

    def fake_resolve(ckpt, revision=None):
        seen.update(ckpt=ckpt, revision=revision)
        return "/cache/seg.pt"

    monkeypatch.setattr(seg_mod, "resolve_checkpoint", fake_resolve)
    monkeypatch.setattr(seg_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(seg_mod, "EncoderCRFModel", lambda **k: _StubModel())
    monkeypatch.setattr(seg_mod.torch, "load", lambda *a, **k: {})

    seg_mod.RefSegmenter("org/repo", device="cpu", revision="seg-sha")

    assert seen == {"ckpt": "org/repo", "revision": "seg-sha"}


def test_extractor_passes_configured_revisions(monkeypatch):
    pytest.importorskip("torch")
    import bibr.extract.ref_extractor as ex
    import bibr.ner.parser as parser_mod
    import bibr.ner.segmenter as seg_mod

    calls = {}

    def rec_parser(ckpt, device=None, revision=None):
        calls["parser"] = revision
        return object()

    def rec_seg(ckpt, device=None, revision=None):
        calls["seg"] = revision
        return object()

    monkeypatch.setattr(parser_mod, "RefParser", rec_parser)
    monkeypatch.setattr(seg_mod, "RefSegmenter", rec_seg)
    monkeypatch.setattr(ex, "_NER_PARSER", None)
    monkeypatch.setattr(ex, "_NER_SEGMENTER", None)

    ex._get_ner_parser()
    ex._get_ner_segmenter()

    assert calls["parser"] == Settings.NER_PARSER_REVISION
    assert calls["seg"] == Settings.NER_SEG_REVISION
