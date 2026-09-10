"""Export the trained multitask paper classifier to ONNX and check parity.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/export_onnx_paper_classifier.py \
        --out /mnt/bulk/datasets/onnx_exports/paper_classifier [--revision <sha>]
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
        "Working memory capacity predicts reading comprehension in children",
        "We tested 240 children aged 8-11 on a battery of working memory tasks and a standardised "
        "reading comprehension measure. Capacity predicted comprehension beyond decoding skill.",
    ),
    (
        "A review of deep learning for protein structure prediction",
        "This review surveys neural approaches to protein folding since AlphaFold, summarising "
        "architectures, benchmarks and open problems.",
    ),
    ("Comment on 'Estimating the reproducibility of psychological science'", ""),
    (
        "Monetary policy transmission in emerging markets: evidence from panel data",
        "Using quarterly data for 32 economies we estimate the pass-through of policy rates to "
        "lending rates with a panel VAR.",
    ),
]


class _ThreeHead:
    def __init__(self, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask):
                out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
                return out["l1_logits"], out["l2_logits"], out["paper_type_logits"]

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
    from bibr.structure.paper_classifier_common import _build_input_text
    from bibr.structure.paper_classifier_model import PaperClassifierModel, _download_snapshot
    from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel

    settings = GlobalSettings()
    model_id = settings.ml.paper_classifier_model_id or "scienceverse/bibr-paper-classifier"
    revision = args.revision or settings.ml.paper_classifier_revision
    timer = Timer()
    snapshot = _download_snapshot(model_id, revision)
    torch_model = PaperClassifierModel(snapshot, device="cpu")
    print(f"loaded {model_id}@{revision} in {timer.lap():.1f}s")

    bundle = bundle_dir(args.out)
    copy_bundle_files(
        snapshot, args.out, ["label_maps.json", "inference_config.json", "tokenizer_config.json"]
    )
    copy_tokenizer(snapshot, bundle)
    model_path = bundle / "model.onnx"
    enc = torch_model.tokenizer(
        [
            "Title [SEP] abstract",
            "A longer title here [SEP] and a longer abstract than the other one",
        ],
        padding=True,
        truncation=True,
        max_length=torch_model.max_length,
        return_tensors="pt",
    )
    export_torch_module(
        _ThreeHead(torch_model.model).module,
        (enc["input_ids"], enc["attention_mask"]),
        model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["l1_logits", "l2_logits", "paper_type_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "l1_logits": {0: "batch"},
            "l2_logits": {0: "batch"},
            "paper_type_logits": {0: "batch"},
        },
        opset=args.opset,
        dynamo=args.dynamo,
    )
    print(f"exported in {timer.lap():.1f}s -> {model_path} ({file_size_mb(model_path):.1f} MB)")
    check_onnx(model_path)

    write_manifest(
        bundle,
        {
            "model": "paper_classifier",
            "architecture": "PaperClassifierMultitaskModel",
            "opset": args.opset,
            "exporter": "dynamo" if args.dynamo else "torchscript",
            "inputs": [
                {"name": "input_ids", "shape": ["batch", "sequence"], "dtype": "int64"},
                {"name": "attention_mask", "shape": ["batch", "sequence"], "dtype": "int64"},
            ],
            "outputs": [
                {"name": "l1_logits", "shape": ["batch", len(torch_model.l1_classes)]},
                {"name": "l2_logits", "shape": ["batch", max(1, len(torch_model.l2_classes))]},
                {
                    "name": "paper_type_logits",
                    "shape": ["batch", max(1, len(torch_model.paper_type_classes))],
                },
            ],
            "tokenizer_file": "tokenizer.json",
            "pad_token_id": int(torch_model.tokenizer.pad_token_id),
            "pad_token": str(torch_model.tokenizer.pad_token),
            "max_length": int(torch_model.max_length),
            "l1_classes": list(torch_model.l1_classes),
            "l2_classes": list(torch_model.l2_classes),
            "paper_type_classes": list(torch_model.paper_type_classes),
            "paper_type_temperature": float(torch_model.paper_type_temperature),
            "source": {"repo_id": model_id, "revision": revision},
        },
    )

    onnx_model = OnnxPaperClassifierModel(bundle, device="cpu")
    torch_preds = torch_model.classify_batch(SAMPLES)
    onnx_preds = onnx_model.classify_batch(SAMPLES)
    score_diff = max(
        max(
            abs(a.oecd_l1_score - b.oecd_l1_score),
            abs(a.oecd_l2_score - b.oecd_l2_score),
            abs(a.paper_type_score - b.paper_type_score),
        )
        for a, b in zip(torch_preds, onnx_preds, strict=True)
    )
    label_match = sum(
        (a.oecd_l1, a.oecd_l2, a.paper_type) == (b.oecd_l1, b.oecd_l2, b.paper_type)
        for a, b in zip(torch_preds, onnx_preds, strict=True)
    )
    texts = [_build_input_text(t, a) for t, a in SAMPLES]
    enc = torch_model.tokenizer(
        texts, padding=True, truncation=True, max_length=torch_model.max_length, return_tensors="pt"
    )
    with torch.no_grad():
        t_out = torch_model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    o_out = onnx_model.session.run(
        ["l1_logits", "l2_logits", "paper_type_logits"],
        {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
    )
    logit_diff = max(
        float(np.abs(t_out[k].numpy() - o).max())
        for k, o in zip(("l1_logits", "l2_logits", "paper_type_logits"), o_out, strict=True)
    )
    report(
        "paper classifier parity",
        [
            ("samples", str(len(SAMPLES))),
            ("graph max|Δ| logits", f"{logit_diff:.3e}"),
            ("end-to-end max|Δ| score", f"{score_diff:.3e}"),
            ("labels agree (L1, L2, paper_type)", f"{label_match}/{len(SAMPLES)}"),
            (
                "predictions",
                "; ".join(
                    f"{p.oecd_l1}/{p.oecd_l2}/{p.paper_type}({p.paper_type_score:.2f})"
                    for p in onnx_preds
                ),
            ),
            ("model.onnx", f"{file_size_mb(model_path):.1f} MB"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
