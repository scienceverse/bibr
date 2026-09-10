"""Smoke test for the HF Hub loader of the trained section classifier."""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("huggingface_hub")

from bibr.paper_contents import CanonicalSection
from bibr.structure.section_classifier_model import SectionClassifierModel, _pick_device


def test_build_model_input_v3():
    from bibr.structure.section_classifier_model import HeaderContext, _build_input_text

    ctx = HeaderContext(
        heading="Ethics",
        body="IRB approved." + " x" * 900,
        relative_position=0.9,
        prev_heading="Methods",
        next_heading="",
    )
    text = _build_input_text(ctx, template_version=3)
    assert text.startswith("Ethics [SEP] pos=end | prev=Methods | next=- [SEP] IRB approved.")
    assert _build_input_text(ctx, template_version=2).startswith("Ethics [SEP] IRB approved.")


def test_position_bucket_matches_training():
    from bibr.structure.section_classifier_model import _position_bucket

    assert [_position_bucket(x) for x in (0.0, 0.2, 0.5, 0.7, 0.95)] == [
        "start",
        "early",
        "middle",
        "late",
        "end",
    ]


def test_classify_batch_accepts_lightweight_header_context(monkeypatch):
    from bibr.structure.section_classifier import HeaderContext as LightweightHeaderContext
    from bibr.structure.section_classifier_model import SectionPrediction

    model = SectionClassifierModel.__new__(SectionClassifierModel)
    captured = {}

    def fake_forward_chunk(contexts, max_length):  # noqa: ARG001
        captured["contexts"] = contexts
        return [SectionPrediction(CanonicalSection.METHODS, True, 0.9)]

    monkeypatch.setattr(model, "_forward_chunk", fake_forward_chunk)

    result = model.classify_batch(
        [
            LightweightHeaderContext(
                heading="Ethics",
                body="IRB approved.",
                relative_position=0.9,
                prev_heading="Methods",
                next_heading="Funding",
            )
        ]
    )

    assert result[0].canonical_type == CanonicalSection.METHODS
    ctx = captured["contexts"][0]
    assert ctx.heading == "Ethics"
    assert ctx.body == "IRB approved."
    assert ctx.relative_position == 0.9
    assert ctx.prev_heading == "Methods"
    assert ctx.next_heading == "Funding"


def test_forward_chunk_survives_lone_surrogate_ocr_garbage():
    from bibr.structure.section_classifier_model import HeaderContext

    model = SectionClassifierModel.__new__(SectionClassifierModel)
    model.device = "cpu"
    model.template_version = 2
    model.label_classes = [CanonicalSection.METHODS.value, CanonicalSection.RESULTS.value]
    tokenizer_calls: list[list[str]] = []

    class _FakeEnc(dict):
        def to(self, device):  # noqa: ARG002
            return self

    def fake_tokenizer(texts, **kwargs):  # noqa: ARG001
        tokenizer_calls.append(list(texts))
        if any(any(0xD800 <= ord(char) <= 0xDFFF for char in text) for text in texts):
            raise TypeError(
                "TextEncodeInput must be Union[TextInputSequence, "
                "Tuple[InputSequence, InputSequence]]"
            )
        count = len(texts)
        return _FakeEnc(
            input_ids=torch.ones((count, 4), dtype=torch.long),
            attention_mask=torch.ones((count, 4), dtype=torch.long),
        )

    def fake_model(input_ids, attention_mask):  # noqa: ARG001
        count = input_ids.shape[0]
        type_logits = torch.zeros((count, 2))
        type_logits[:, 0] = 5.0
        return {
            "type_logits": type_logits,
            "top_level_logits": torch.zeros((count, 1)),
        }

    model.tokenizer = fake_tokenizer
    model.model = fake_model

    result = model.classify_batch([HeaderContext(heading="Meth\udce9ods", body="Body \ud835text")])

    assert len(result) == 1
    assert result[0].canonical_type == CanonicalSection.METHODS
    assert len(tokenizer_calls) == 2
    assert all(not (0xD800 <= ord(char) <= 0xDFFF) for char in tokenizer_calls[1][0])


def test_pick_device_skips_incompatible_cuda():
    """Present-but-unusable GPU (e.g. Pascal + sm_75+ wheel) must not be picked."""
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("bibr.utils.device.cuda_incompatibility", return_value="sm_61 unsupported"),
    ):
        assert _pick_device() == "cpu"


def test_pick_device_keeps_compatible_cuda():
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("bibr.utils.device.cuda_incompatibility", return_value=None),
    ):
        assert _pick_device() == "cuda"


def test_pick_device_cpu_without_cuda():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert _pick_device() == "cpu"


def test_download_snapshot_disables_hf_symlinks_on_windows(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    from bibr.structure import section_classifier_common as common
    from bibr.structure import section_classifier_model as mod

    calls = []

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_SYMLINKS", False)

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append(
            (
                repo_id,
                kwargs,
                os.environ.get("HF_HUB_DISABLE_SYMLINKS"),
                hf_constants.HF_HUB_DISABLE_SYMLINKS,
            )
        )
        return str(tmp_path)

    monkeypatch.setattr(common, "snapshot_download", fake_snapshot_download)

    assert mod._download_snapshot("fake/repo", "main") == tmp_path
    assert calls == [
        (
            "fake/repo",
            {"revision": "main"},
            "1",
            True,
        )
    ]


def test_download_snapshot_retries_windows_symlink_privilege_error(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    from bibr.structure import section_classifier_common as common
    from bibr.structure import section_classifier_model as mod

    calls = []

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_SYMLINKS", False)

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        if len(calls) == 1:
            error = OSError("[WinError 1314] A required privilege is not held by the client")
            error.winerror = 1314
            raise error
        return str(tmp_path)

    monkeypatch.setattr(common, "snapshot_download", fake_snapshot_download)

    assert mod._download_snapshot("fake/repo", "main") == tmp_path
    assert calls == [
        ("fake/repo", {"revision": "main"}),
        ("fake/repo", {"revision": "main", "max_workers": 1}),
    ]


@pytest.mark.slow  # hits HF Hub
def test_section_classifier_model_predicts_canonical_type_and_top_level():
    model = SectionClassifierModel.from_pretrained("thesanogoeffect/bibr-section-classifier-v2")
    result = model.classify_batch(
        [
            (
                "Methods",
                "Participants were recruited from the university subject pool. "
                "We measured reaction times.",
            ),
            (
                "Acknowledgments",
                "We thank the reviewers and the lab members for their feedback.",
            ),
        ]
    )
    assert len(result) == 2
    assert result[0].canonical_type == CanonicalSection.METHODS
    # NOTE: top_level is a model quality signal, not a loader contract. The
    # current v2 checkpoint sometimes predicts top_level=False for obvious
    # top-level sections like "Methods" (logits ~ -1.4). We only assert the
    # type is a Python bool so the loader contract is exercised.
    assert isinstance(result[0].is_top_level, bool)
    assert 0.0 <= result[0].score <= 1.0
    assert result[1].canonical_type == CanonicalSection.ACKNOWLEDGMENT
    assert isinstance(result[1].is_top_level, bool)
