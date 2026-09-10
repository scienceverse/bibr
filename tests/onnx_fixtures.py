"""Tiny randomly initialised models + ONNX bundles for the runtime parity tests.

Everything here is built in-process with a fixed seed and exported with the
same recipe the ``scripts/export_onnx_*.py`` exporters use (TorchScript
exporter, opset 17, dynamic batch/sequence axes). No network access: the
encoders are tiny ``BertModel``s built from an explicit config and the
tokenizer is a hand-rolled WordPiece vocabulary.

Callers must ``pytest.importorskip`` ``torch``, ``transformers`` and ``onnx``
before using these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORDS = [
    "smith",
    "j",
    "doe",
    "a",
    "2020",
    "2021",
    "memory",
    "and",
    "consciousness",
    "journal",
    "of",
    "science",
    "methods",
    "results",
    "discussion",
    "introduction",
    "participants",
    "were",
    "recruited",
    "the",
    "sep",
    "pos",
    "prev",
    "next",
    "start",
    "early",
    "middle",
    "late",
    "end",
    "-",
    "|",
    "=",
    ".",
    ",",
    "(",
    ")",
    ":",
    "1",
    "12",
    "26",
    "title",
    "abstract",
    "sleep",
    "ethics",
    "irb",
    "approved",
]
PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"


def tiny_tokenizer():
    """A WordPiece tokenizer over ``WORDS`` with BERT special tokens (ids 0-3)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    vocab = {PAD: 0, UNK: 1, CLS: 2, SEP: 3}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordPiece(vocab, unk_token=UNK))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single=f"{CLS} $A {SEP}",
        pair=f"{CLS} $A {SEP} $B:1 {SEP}:1",
        special_tokens=[(CLS, 2), (SEP, 3)],
    )
    # Register them as *added* tokens, not just vocabulary entries, exactly as a
    # published tokenizer.json does. Both classifier templates put a literal
    # "[SEP]" inside the input string, and only an added token survives the
    # lowercasing normalizer to come back as one id.
    tok.add_special_tokens([PAD, UNK, CLS, SEP])
    return tok


def hf_tokenizer(tok):
    """Wrap the ``tokenizers`` object the way ``AutoTokenizer`` would."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token=PAD, unk_token=UNK, cls_token=CLS, sep_token=SEP
    )


def tiny_bert(seed: int = 0, vocab_size: int = 64):
    import torch
    from transformers import BertConfig, BertModel

    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    return BertModel(config).eval()


def _export(module, example, path: Path, *, input_names, output_names, dynamic_axes):
    import torch

    torch.onnx.export(
        module,
        example,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _write_manifest(bundle: Path, manifest: dict) -> None:
    manifest = {"schema_version": 1, "model_file": "model.onnx", **manifest}
    (bundle / "bibr_onnx.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _text_axes(*outputs: str) -> dict:
    axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
    }
    for name in outputs:
        axes[name] = {0: "batch"}
    return axes


# --------------------------------------------------------------------------
# section classifier
# --------------------------------------------------------------------------

SECTION_LABELS = ["introduction", "method", "results", "discussion", "unknown"]


def tiny_section_model(monkeypatch, seed: int = 1):
    """A ``SectionMiniLMModel`` around a tiny BERT (from_config stubbed)."""
    import torch

    import bibr.structure._section_minilm_arch as arch

    encoder = tiny_bert(seed)
    monkeypatch.setattr(arch.AutoConfig, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(arch.AutoModel, "from_config", lambda *a, **k: encoder)
    torch.manual_seed(seed + 100)
    model = arch.SectionMiniLMModel(num_types=len(SECTION_LABELS)).eval()
    return model


def export_section_bundle(model, tok, out: Path, *, template_version: int = 3) -> Path:
    import torch

    bundle = out / "onnx"
    bundle.mkdir(parents=True, exist_ok=True)
    tok.save(str(bundle / "tokenizer.json"))

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            return out["type_logits"], out["top_level_logits"].reshape(-1)

    ids = torch.tensor([[2, 5, 6, 3, 0], [2, 7, 8, 9, 3]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    _export(
        Wrapped(model).eval(),
        (ids, mask),
        bundle / "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["type_logits", "top_level_logits"],
        dynamic_axes=_text_axes("type_logits", "top_level_logits"),
    )
    _write_manifest(
        bundle,
        {
            "model": "section_classifier",
            "tokenizer_file": "tokenizer.json",
            "pad_token_id": 0,
            "pad_token": PAD,
            "max_length": 32,
            "label_classes": SECTION_LABELS,
            "template_version": template_version,
        },
    )
    return bundle


def torch_section_classifier(model, tok, *, template_version: int = 3):
    from bibr.structure.section_classifier_model import SectionClassifierModel

    clf = SectionClassifierModel.__new__(SectionClassifierModel)
    clf.device = "cpu"
    clf.tokenizer = hf_tokenizer(tok)
    clf.label_classes = list(SECTION_LABELS)
    clf.template_version = template_version
    clf.model = model
    return clf


# --------------------------------------------------------------------------
# paper classifier
# --------------------------------------------------------------------------

L1 = ["Natural Sciences", "Social Sciences"]
L2 = ["Physical Sciences", "Psychology and Cognitive Sciences", "Economics and Business"]
PAPER_TYPES = ["empirical", "review", "commentary"]


def tiny_paper_model(seed: int = 2):
    import torch

    from bibr.structure._paper_classifier_arch import PaperClassifierMultitaskModel

    encoder = tiny_bert(seed)
    torch.manual_seed(seed + 100)
    return PaperClassifierMultitaskModel(
        num_l1=len(L1), num_l2=len(L2), num_paper_type=len(PAPER_TYPES), encoder=encoder
    ).eval()


def export_paper_bundle(model, tok, out: Path, *, temperature: float = 1.5) -> Path:
    import torch

    bundle = out / "onnx"
    bundle.mkdir(parents=True, exist_ok=True)
    tok.save(str(bundle / "tokenizer.json"))

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            return out["l1_logits"], out["l2_logits"], out["paper_type_logits"]

    ids = torch.tensor([[2, 5, 6, 3, 0], [2, 7, 8, 9, 3]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    _export(
        Wrapped(model).eval(),
        (ids, mask),
        bundle / "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["l1_logits", "l2_logits", "paper_type_logits"],
        dynamic_axes=_text_axes("l1_logits", "l2_logits", "paper_type_logits"),
    )
    _write_manifest(
        bundle,
        {
            "model": "paper_classifier",
            "tokenizer_file": "tokenizer.json",
            "pad_token_id": 0,
            "pad_token": PAD,
            "max_length": 32,
            "l1_classes": L1,
            "l2_classes": L2,
            "paper_type_classes": PAPER_TYPES,
            "paper_type_temperature": temperature,
        },
    )
    return bundle


def torch_paper_classifier(model, tok, *, temperature: float = 1.5):
    from bibr.structure.paper_classifier_model import PaperClassifierModel

    clf = PaperClassifierModel.__new__(PaperClassifierModel)
    clf.device = "cpu"
    clf.max_length = 32
    clf.paper_type_temperature = temperature
    clf.tokenizer = hf_tokenizer(tok)
    clf.l1_classes = list(L1)
    clf.l2_classes = list(L2)
    clf.paper_type_classes = list(PAPER_TYPES)
    clf.model = model
    return clf


# --------------------------------------------------------------------------
# NER reference parser
# --------------------------------------------------------------------------


def tiny_ner_model(monkeypatch, seed: int = 3):
    """A ``FeatureGatedEncoderCRF`` around a tiny BERT with random CRF params."""
    import torch

    import bibr.ner.model as model_mod
    from bibr.ner.tags import BIO_TAGS

    encoder = tiny_bert(seed)
    monkeypatch.setattr(model_mod.AutoConfig, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(model_mod.AutoModel, "from_config", lambda *a, **k: encoder)
    torch.manual_seed(seed + 100)
    model = model_mod.FeatureGatedEncoderCRF(
        "stub", num_tags=len(BIO_TAGS), dropout=0.0, bio_tags=BIO_TAGS
    )
    with torch.no_grad():
        # Non-trivial, non-zero residual and transitions so parity is meaningful.
        model.feature_proj.bias.normal_(0, 0.5)
        model.feature_gate.normal_(0, 1.0)
        model.crf.transitions.add_(torch.randn_like(model.crf.transitions) * 0.5)
        model.crf.start_transitions.add_(torch.randn_like(model.crf.start_transitions) * 0.5)
        model.crf.end_transitions.add_(torch.randn_like(model.crf.end_transitions) * 0.5)
    return model.eval()


def export_ner_bundle(model, tok, out: Path, *, max_seq_len: int = 16) -> Path:
    import torch

    from bibr.ner.tags import BIO_TAGS

    bundle = out / "onnx"
    bundle.mkdir(parents=True, exist_ok=True)
    tok.save(str(bundle / "tokenizer.json"))

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            hidden = self.inner.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            gate = torch.sigmoid(self.inner.feature_gate)
            zeros = torch.zeros(
                (hidden.shape[0], hidden.shape[1], self.inner.feature_dim), dtype=hidden.dtype
            )
            hidden = hidden + self.inner.feature_proj(zeros * gate)
            return self.inner.linear(hidden)

    # Apply the BIO masks as predict() does before reading the CRF tables.
    model.crf.transitions.data[model._transition_mask] = -10000.0
    model.crf.start_transitions.data[model._start_mask] = -10000.0
    ids = torch.tensor([[5, 6, 7, 0], [8, 9, 10, 11]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    _export(
        Wrapped(model).eval(),
        (ids, mask),
        bundle / "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["emissions"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "emissions": {0: "batch", 1: "sequence"},
        },
    )
    _write_manifest(
        bundle,
        {
            "model": "ner_parser",
            "tokenizer_file": "tokenizer.json",
            "add_special_tokens": False,
            "pad_token_id": 0,
            "pad_token": PAD,
            "max_length": max_seq_len,
            "bio_tags": list(BIO_TAGS),
            "crf": {
                "start_transitions": model.crf.start_transitions.detach().tolist(),
                "end_transitions": model.crf.end_transitions.detach().tolist(),
                "transitions": model.crf.transitions.detach().tolist(),
            },
        },
    )
    return bundle


def torch_ref_parser(model, tok, *, max_seq_len: int = 16):
    from bibr.ner.parser import RefParser

    parser = RefParser.__new__(RefParser)
    parser.device = "cpu"
    parser.max_seq_len = max_seq_len
    parser.tokenizer = hf_tokenizer(tok)
    parser.model = model
    return parser


# --------------------------------------------------------------------------
# layout detector stand-in
# --------------------------------------------------------------------------

LAYOUT_SIZE = 64
LAYOUT_QUERIES = 8
LAYOUT_CLASSES = 5


def tiny_layout_module(seed: int = 4):
    """Conv stand-in with PP-DocLayoutV3's output signature.

    Returns ``(logits (B,Q,C), pred_boxes (B,Q,4) in cxcywh ∈ (0,1), order_logits (B,Q,Q))``.
    """
    import torch
    from torch import nn

    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            # kernel == stride keeps the feature map an exact multiple of the
            # pool's output (64 -> 16 -> 4); torch.onnx has no symbolic for
            # adaptive_avg_pool2d when the sizes are not a clean factor.
            self.conv = nn.Conv2d(3, 4, kernel_size=4, stride=4)
            self.pool = nn.AdaptiveAvgPool2d(4)
            q, c = LAYOUT_QUERIES, LAYOUT_CLASSES
            self.fc = nn.Linear(4 * 16, q * (c + 4 + q))

        def forward(self, pixel_values):
            q, c = LAYOUT_QUERIES, LAYOUT_CLASSES
            x = torch.relu(self.conv(pixel_values))
            x = self.pool(x).flatten(1)
            out = self.fc(x) * 3.0
            b = out.shape[0]
            logits = out[:, : q * c].reshape(b, q, c)
            boxes = torch.sigmoid(out[:, q * c : q * c + q * 4]).reshape(b, q, 4)
            order = out[:, q * c + q * 4 :].reshape(b, q, q)
            return logits, boxes, order

    torch.manual_seed(seed)
    return TinyDetector().eval()


def export_layout_bundle(module, out: Path) -> Path:
    import torch

    bundle = out / "onnx"
    bundle.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, LAYOUT_SIZE, LAYOUT_SIZE))
    _export(
        module,
        (example,),
        bundle / "model.onnx",
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes", "order_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "order_logits": {0: "batch"},
        },
    )
    _write_manifest(
        bundle,
        {
            "model": "layout",
            "preprocessing": {
                "size": {"height": LAYOUT_SIZE, "width": LAYOUT_SIZE},
                "rescale_factor": 1 / 255,
                "image_mean": [0, 0, 0],
                "image_std": [1, 1, 1],
            },
            "num_queries": LAYOUT_QUERIES,
            "num_labels": LAYOUT_CLASSES,
        },
    )
    return bundle


def sample_pages(seed: int = 5, n: int = 2):
    """Page-like RGB images of different sizes with sharp bars (edge cases for bicubic)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pages = []
    for i in range(n):
        h, w = 96 + 40 * i, 80 + 24 * i
        page = np.full((h, w, 3), 255, dtype=np.uint8)
        for _ in range(12):
            y = int(rng.integers(0, h - 4))
            x = int(rng.integers(0, w // 2))
            page[y : y + 3, x : x + int(rng.integers(10, w // 2)), :] = rng.integers(0, 60)
        page[::7, :, 1] = 128  # thin lines → sharp transitions
        pages.append(Image.fromarray(page))
    return pages


def fake_hf_outputs(logits, boxes, order):
    """Mimic ``PPDocLayoutV3ForObjectDetectionOutput`` for HF post-processing."""
    import torch

    b, q, _ = logits.shape
    return SimpleNamespace(
        logits=torch.as_tensor(logits),
        pred_boxes=torch.as_tensor(boxes),
        order_logits=torch.as_tensor(order),
        out_masks=torch.zeros((b, q, 4, 4)),
    )
