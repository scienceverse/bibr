"""Export the trained MiniLM section classifier to ONNX and check parity.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/export_onnx_section_classifier.py \
        --out /mnt/bulk/datasets/onnx_exports/section_classifier [--revision <sha>]

Writes ``<out>/onnx/{model.onnx,bibr_onnx.json,tokenizer.json}`` (plus copies
of the torch bundle's sidecar files so ``<out>`` is a complete local bundle)
and compares ``OnnxSectionClassifierModel`` with ``SectionClassifierModel`` on
a handful of header contexts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from _onnx_export import (  # noqa: E402
    DEFAULT_OPSET,
    Timer,
    bundle_dir,
    check_onnx,
    copy_bundle_files,
    copy_tokenizer,
    export_torch_module,
    file_size_mb,
    report,
    write_manifest,
)

SAMPLES = [
    (
        "Methods",
        "Participants were recruited from three universities. " * 20,
        0.3,
        "Introduction",
        "Results",
    ),
    (
        "2. Data and sample",
        "We use the panel of firms described below. " * 10,
        0.25,
        "1. Introduction",
        "3. Results",
    ),
    ("Discussion", "Our findings extend prior work on memory. " * 30, 0.8, "Results", "References"),
    ("Ethics statement", "IRB approved. Informed consent was obtained.", 0.9, "Discussion", ""),
    ("Table 3", "Descriptive statistics for the main variables.", 0.55, "Results", "Results"),
    ("References", "Smith J (2020). A paper. Journal 1:1-2.", 0.97, "Conclusion", ""),
]


class _TwoHead:
    def __init__(self, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask):
                out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
                return out["type_logits"], out["top_level_logits"].reshape(-1)

        self.module = Wrapped(model).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--dynamo", action="store_true")
    args = parser.parse_args()

    import torch

    from bibr.config import GlobalSettings
    from bibr.structure.section_classifier_common import HeaderContext, _download_snapshot
    from bibr.structure.section_classifier_model import SectionClassifierModel
    from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

    settings = GlobalSettings()
    model_id = settings.ml.section_classifier_model_id or "scienceverse/bibr-section-classifier"
    revision = args.revision or settings.ml.section_classifier_revision
    timer = Timer()
    snapshot = _download_snapshot(model_id, revision)
    torch_model = SectionClassifierModel(snapshot, device="cpu")
    print(f"loaded {model_id}@{revision} in {timer.lap():.1f}s")

    bundle = bundle_dir(args.out)
    copy_bundle_files(
        snapshot, args.out, ["label_classes.json", "inference_config.json", "tokenizer_config.json"]
    )
    copy_tokenizer(snapshot, bundle)
    model_path = bundle / "model.onnx"
    enc = torch_model.tokenizer(
        [
            "Methods [SEP] some body",
            "A much longer heading here [SEP] with more body text than the other",
        ],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    export_torch_module(
        _TwoHead(torch_model.model).module,
        (enc["input_ids"], enc["attention_mask"]),
        model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["type_logits", "top_level_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "type_logits": {0: "batch"},
            "top_level_logits": {0: "batch"},
        },
        opset=args.opset,
        dynamo=args.dynamo,
    )
    print(f"exported in {timer.lap():.1f}s -> {model_path} ({file_size_mb(model_path):.1f} MB)")
    check_onnx(model_path)

    pad_id = int(torch_model.tokenizer.pad_token_id)
    write_manifest(
        bundle,
        {
            "model": "section_classifier",
            "architecture": "SectionMiniLMModel",
            "opset": args.opset,
            "exporter": "dynamo" if args.dynamo else "torchscript",
            "inputs": [
                {"name": "input_ids", "shape": ["batch", "sequence"], "dtype": "int64"},
                {"name": "attention_mask", "shape": ["batch", "sequence"], "dtype": "int64"},
            ],
            "outputs": [
                {"name": "type_logits", "shape": ["batch", len(torch_model.label_classes)]},
                {"name": "top_level_logits", "shape": ["batch"]},
            ],
            "tokenizer_file": "tokenizer.json",
            "pad_token_id": pad_id,
            "pad_token": str(torch_model.tokenizer.pad_token),
            "max_length": 256,
            "label_classes": list(torch_model.label_classes),
            "template_version": int(torch_model.template_version),
            "source": {"repo_id": model_id, "revision": revision},
        },
    )

    onnx_model = OnnxSectionClassifierModel(bundle, device="cpu")
    contexts = [HeaderContext(h, b, p, prev, nxt) for h, b, p, prev, nxt in SAMPLES]
    torch_preds = torch_model.classify_batch(contexts)
    onnx_preds = onnx_model.classify_batch(contexts)
    score_diff = max(abs(a.score - b.score) for a, b in zip(torch_preds, onnx_preds, strict=True))
    label_match = sum(
        a.canonical_type == b.canonical_type and a.is_top_level == b.is_top_level
        for a, b in zip(torch_preds, onnx_preds, strict=True)
    )
    # Raw logits on the same token ids, to separate graph error from tokenizer drift.
    from bibr.structure.section_classifier_common import _build_input_text

    texts = [_build_input_text(c, torch_model.template_version) for c in contexts]
    enc = torch_model.tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        t_out = torch_model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    o_type, o_top = onnx_model.session.run(
        ["type_logits", "top_level_logits"],
        {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
    )
    logit_diff = max(
        float(np.abs(t_out["type_logits"].numpy() - o_type).max()),
        float(np.abs(t_out["top_level_logits"].numpy().reshape(-1) - o_top).max()),
    )
    report(
        "section classifier parity",
        [
            ("samples", str(len(SAMPLES))),
            ("graph max|Δ| logits", f"{logit_diff:.3e}"),
            ("end-to-end max|Δ| score", f"{score_diff:.3e}"),
            ("labels + top-level agree", f"{label_match}/{len(SAMPLES)}"),
            (
                "predictions",
                ", ".join(f"{p.canonical_type.value}({p.score:.2f})" for p in onnx_preds),
            ),
            ("model.onnx", f"{file_size_mb(model_path):.1f} MB"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
