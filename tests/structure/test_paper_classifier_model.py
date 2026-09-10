"""Loader + inference tests for the trained multitask paper classifier.

No real HF model is downloaded — inference is exercised with an injected fake
tokenizer + fake torch module, and the input-template builder is checked
byte-for-byte against bibr-training's ``build_input_text``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bibr.structure.paper_classifier_model import (
    PaperClassificationPrediction,
    PaperClassifierModel,
    _build_input_text,
)


class TestInputTemplate:
    """``_build_input_text`` must match bibr-training's title_abstract_v1 exactly.

    Source: bibr-training/src/bibr_training/paper_classifier/dataset.py
      TEXT_SEPARATOR = " [SEP] "  (line 18)
      build_input_text -> " [SEP] ".join(non-empty cleaned parts)  (lines 68-71)
      _clean_str -> re.sub(r"\\s+", " ", value).strip()  (lines 38-44)
    """

    def test_both_present(self):
        assert _build_input_text("My Title", "My abstract") == "My Title [SEP] My abstract"

    def test_whitespace_collapsed(self):
        assert (
            _build_input_text("  My\n\tTitle ", "Line one\nline two")
            == "My Title [SEP] Line one line two"
        )

    def test_empty_abstract_drops_separator(self):
        assert _build_input_text("Only Title", "") == "Only Title"
        assert _build_input_text("Only Title", None) == "Only Title"

    def test_empty_title_drops_separator(self):
        assert _build_input_text("", "Only abstract") == "Only abstract"

    def test_both_empty(self):
        assert _build_input_text("", "") == ""


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.captured_texts: list[str] = []

    def __call__(self, texts, **_kwargs):
        self.captured_texts = list(texts)
        n = len(texts)
        return _FakeEncoding(
            input_ids=torch.zeros((n, 4), dtype=torch.long),
            attention_mask=torch.ones((n, 4), dtype=torch.long),
        )


class _FakeModel:
    """Returns fixed per-head logits; argmax picks a known class each time."""

    def __init__(self, l1_logits, l2_logits, pt_logits):
        self._l1 = l1_logits
        self._l2 = l2_logits
        self._pt = pt_logits

    def __call__(self, input_ids, attention_mask):  # noqa: ARG002
        n = input_ids.shape[0]
        return {
            "l1_logits": self._l1.repeat(n, 1),
            "l2_logits": self._l2.repeat(n, 1),
            "paper_type_logits": self._pt.repeat(n, 1),
        }


def _make_model(l1_logits, l2_logits, pt_logits):
    model = PaperClassifierModel.__new__(PaperClassifierModel)
    model.device = "cpu"
    model.max_length = 256
    model.paper_type_temperature = 1.0
    model.tokenizer = _FakeTokenizer()
    model.l1_classes = ["Natural Sciences", "Social Sciences"]
    model.l2_classes = ["Physical Sciences", "Psychology and Cognitive Sciences"]
    model.paper_type_classes = ["empirical", "review", "commentary"]
    model.model = _FakeModel(
        torch.tensor([l1_logits]),
        torch.tensor([l2_logits]),
        torch.tensor([pt_logits]),
    )
    return model


class TestClassifyBatch:
    def test_builds_exact_training_input_string(self):
        model = _make_model([0.1, 5.0], [0.1, 5.0], [0.1, 0.2, 5.0])
        model.classify_batch([("My Title", "My abstract")])
        assert model.tokenizer.captured_texts == ["My Title [SEP] My abstract"]

    def test_returns_argmax_labels_and_softmax_scores(self):
        # L1 argmax -> index 1 "Social Sciences"; L2 -> "Psychology..."; pt -> "commentary"
        model = _make_model([0.0, 10.0], [0.0, 10.0], [0.0, 0.0, 10.0])
        preds = model.classify_batch([("T", "A")])
        assert len(preds) == 1
        p = preds[0]
        assert isinstance(p, PaperClassificationPrediction)
        assert p.oecd_l1 == "Social Sciences"
        assert p.oecd_l2 == "Psychology and Cognitive Sciences"
        assert p.paper_type == "commentary"
        # softmax over a decisive logit gap is high; each score in [0, 1].
        assert 0.9 < p.oecd_l1_score <= 1.0
        assert 0.9 < p.oecd_l2_score <= 1.0
        assert 0.9 < p.paper_type_score <= 1.0

    def test_empty_input_returns_empty(self):
        model = _make_model([1.0, 0.0], [1.0, 0.0], [1.0, 0.0, 0.0])
        assert model.classify_batch([]) == []

    def test_survives_lone_surrogate_ocr_garbage(self):
        model = _make_model([0.0, 10.0], [0.0, 10.0], [0.0, 0.0, 10.0])
        base = model.tokenizer
        calls: list[list[str]] = []

        class _SurrogateRejectingTokenizer:
            def __call__(self, texts, **kwargs):
                batch = [texts] if isinstance(texts, str) else list(texts)
                calls.append(batch)
                if any(any(0xD800 <= ord(char) <= 0xDFFF for char in text) for text in batch):
                    raise TypeError(
                        "TextEncodeInput must be Union[TextInputSequence, "
                        "Tuple[InputSequence, InputSequence]]"
                    )
                return base(batch, **kwargs)

        model.tokenizer = _SurrogateRejectingTokenizer()

        predictions = model.classify_batch([("Ti\udce9tle", "Abs\ud835tract")])

        assert len(predictions) == 1
        assert predictions[0].oecd_l1 == "Social Sciences"
        assert len(calls) == 2
        assert all(not (0xD800 <= ord(char) <= 0xDFFF) for char in calls[1][0])

    def test_paper_type_temperature_softens_confidence(self):
        # T > 1 lowers the paper_type softmax score without changing the argmax label,
        # and leaves the OECD heads untouched.
        base = _make_model([0.0, 10.0], [0.0, 10.0], [0.0, 0.0, 10.0])
        hot = _make_model([0.0, 10.0], [0.0, 10.0], [0.0, 0.0, 10.0])
        hot.paper_type_temperature = 3.0
        p_base = base.classify_batch([("T", "A")])[0]
        p_hot = hot.classify_batch([("T", "A")])[0]
        assert p_hot.paper_type == p_base.paper_type == "commentary"
        assert p_hot.paper_type_score < p_base.paper_type_score
        assert p_hot.oecd_l1_score == p_base.oecd_l1_score
