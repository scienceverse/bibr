"""Export the ModernBERT+CRF reference parser to ONNX and check parity.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/export_onnx_ner_parser.py \
        --out /mnt/bulk/datasets/onnx_exports/ner_parser [--revision <sha>] [--dynamo] [--eager]

The graph emits per-token CRF ``emissions`` (encoder → gated feature residual
with all-zero features, i.e. the learned ``feature_proj.bias`` → linear); the
CRF parameters (with the BIO constraint masks applied, as ``predict`` does) go
into ``bibr_onnx.json`` and ``bibr.ner.crf_numpy.viterbi_decode`` decodes them.
Parity is checked on references of several lengths, batched with padding and
individually, against ``RefParser``.
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
    export_torch_module,
    file_size_mb,
    report,
    write_manifest,
)

SAMPLES = [
    "Tulving, E. (1985). Memory and consciousness. Canadian Psychology, 26(1), 1-12.",
    "Wongvibulsin S, Habeos EE, et al. Digital health. J Med Internet Res 2021; 23: e18773. "
    "doi:10.2196/18773",
    "Kahneman, D. (2011). Thinking, fast and slow. Farrar, Straus and Giroux.",
    "Smith J, Doe A. 2019. A very long title that goes on and on about the effects of sleep "
    "deprivation on working memory in undergraduate students across three cohorts. Journal of "
    "Sleep Research 28(4):e12845. https://doi.org/10.1111/jsr.12845",
    "Nosek BA (2015) Estimating the reproducibility of psychological science. Science 349:aac4716",
    "",
]


class _Emissions:
    def __init__(self, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask):
                outputs = self.inner.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                gate = torch.sigmoid(self.inner.feature_gate)
                zeros = torch.zeros(
                    (hidden.shape[0], hidden.shape[1], self.inner.feature_dim),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                hidden = hidden + self.inner.feature_proj(zeros * gate)
                return self.inner.linear(hidden)

        self.module = Wrapped(model).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument(
        "--eager", action="store_true", help="load the encoder with attn_implementation=eager"
    )
    args = parser.parse_args()

    import torch

    from bibr.config import GlobalSettings
    from bibr.ner.parser import DEFAULT_ENCODER, RefParser
    from bibr.ner.parser_onnx import OnnxRefParser
    from bibr.ner.tags import BIO_TAGS

    settings = GlobalSettings()
    ckpt = settings.NER_PARSER_CKPT
    revision = args.revision or settings.NER_PARSER_REVISION
    timer = Timer()
    if args.eager:
        from transformers import AutoConfig, AutoModel

        import bibr.ner.model as model_mod

        _orig = AutoModel.from_config

        def _eager_from_config(config, *a, **k):
            return _orig(config, *a, attn_implementation="eager", **k)

        model_mod.AutoModel.from_config = staticmethod(_eager_from_config)  # type: ignore[assignment]
        AutoConfig  # noqa: B018 — keep the import explicit for readers
    torch_parser = RefParser(ckpt, device="cpu", revision=revision)
    print(f"loaded {ckpt}@{revision} in {timer.lap():.1f}s")
    model = torch_parser.model
    # Apply the BIO constraint masks exactly as predict() does before reading the CRF.
    model.crf.transitions.data[model._transition_mask] = -10000.0
    model.crf.start_transitions.data[model._start_mask] = -10000.0

    bundle = bundle_dir(args.out)
    tokenizer_json = (
        Path(torch_parser.tokenizer.vocab_file).parent
        if getattr(torch_parser.tokenizer, "vocab_file", None)
        else None
    )
    # Save the exact tokenizer the torch parser uses (ModernBERT-base's tokenizer.json).
    torch_parser.tokenizer.save_pretrained(str(bundle / "_tok"))
    (bundle / "_tok" / "tokenizer.json").replace(bundle / "tokenizer.json")
    for leftover in (bundle / "_tok").iterdir():
        leftover.unlink()
    (bundle / "_tok").rmdir()
    del tokenizer_json

    model_path = bundle / "model.onnx"
    example = torch_parser.tokenizer(
        [SAMPLES[0], SAMPLES[2]],
        padding=True,
        truncation=True,
        max_length=torch_parser.max_seq_len,
        return_tensors="pt",
        add_special_tokens=False,
    )
    export_torch_module(
        _Emissions(model).module,
        (example["input_ids"], example["attention_mask"]),
        model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["emissions"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "emissions": {0: "batch", 1: "sequence"},
        },
        opset=args.opset,
        dynamo=args.dynamo,
    )
    print(f"exported in {timer.lap():.1f}s -> {model_path} ({file_size_mb(model_path):.1f} MB)")
    check_onnx(model_path)

    write_manifest(
        bundle,
        {
            "model": "ner_parser",
            "architecture": "FeatureGatedEncoderCRF",
            "encoder": DEFAULT_ENCODER,
            "opset": args.opset,
            "exporter": "dynamo" if args.dynamo else "torchscript",
            "attn_implementation": "eager" if args.eager else "default",
            "inputs": [
                {"name": "input_ids", "shape": ["batch", "sequence"], "dtype": "int64"},
                {"name": "attention_mask", "shape": ["batch", "sequence"], "dtype": "int64"},
            ],
            "outputs": [{"name": "emissions", "shape": ["batch", "sequence", len(BIO_TAGS)]}],
            "tokenizer_file": "tokenizer.json",
            "add_special_tokens": False,
            "pad_token_id": int(torch_parser.tokenizer.pad_token_id),
            "pad_token": str(torch_parser.tokenizer.pad_token),
            "max_length": int(torch_parser.max_seq_len),
            "bio_tags": list(BIO_TAGS),
            "crf": {
                "start_transitions": model.crf.start_transitions.detach().cpu().tolist(),
                "end_transitions": model.crf.end_transitions.detach().cpu().tolist(),
                "transitions": model.crf.transitions.detach().cpu().tolist(),
            },
            "source": {"repo_id": ckpt, "revision": revision},
        },
    )

    onnx_parser = OnnxRefParser(bundle, device="cpu")
    torch_batch = torch_parser.parse_batch(SAMPLES, batch_size=3)
    onnx_batch = onnx_parser.parse_batch(SAMPLES, batch_size=3)
    torch_single = [torch_parser.parse(s) for s in SAMPLES]
    onnx_single = [onnx_parser.parse(s) for s in SAMPLES]
    agree_batch = sum(a == b for a, b in zip(torch_batch, onnx_batch, strict=True))
    agree_single = sum(a == b for a, b in zip(torch_single, onnx_single, strict=True))

    # Emission parity on identical ids (batched with padding, and one long ref alone).
    diffs = []
    for texts in ([s for s in SAMPLES if s], [SAMPLES[3]]):
        enc = torch_parser.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=torch_parser.max_seq_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        with torch.no_grad():
            hidden = model.encoder(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).last_hidden_state
            zeros = torch.zeros((hidden.shape[0], hidden.shape[1], model.feature_dim))
            hidden = hidden + model.feature_proj(zeros * torch.sigmoid(model.feature_gate))
            t_em = model.linear(hidden).numpy()
        o_em = onnx_parser._emissions(enc["input_ids"].numpy(), enc["attention_mask"].numpy())
        mask = enc["attention_mask"].numpy().astype(bool)
        diffs.append(float(np.abs(t_em - o_em)[mask].max()))
    for a, b in zip(torch_batch, onnx_batch, strict=True):
        if a != b:
            print("  MISMATCH torch:", a)
            print("           onnx :", b)
    report(
        "NER parser parity",
        [
            ("samples", str(len(SAMPLES))),
            ("graph max|Δ| emissions (padded batch)", f"{diffs[0]:.3e}"),
            ("graph max|Δ| emissions (single long ref)", f"{diffs[1]:.3e}"),
            ("parse_batch agree", f"{agree_batch}/{len(SAMPLES)}"),
            ("parse agree", f"{agree_single}/{len(SAMPLES)}"),
            ("model.onnx", f"{file_size_mb(model_path):.1f} MB"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
