"""Export PP-DocLayoutV3 (the layout detector) to ONNX and check parity.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/export_onnx_layout.py \
        --out /mnt/bulk/datasets/onnx_exports/layout [--revision <sha>] \
        [--images page1.png page2.png ...] [--pdf paper.pdf]

Loads the pinned torch checkpoint on CPU, exports the graph with the
``logits``, ``pred_boxes`` and ``order_logits`` outputs (no mask head — bibr
never reads the polygons), writes ``<out>/onnx/{model.onnx,bibr_onnx.json}``
and then compares bibr's ONNX layout backend with the transformers path on
the sample pages: preprocessing (pixel values), raw graph outputs, and the
final region lists after ``BaseLayoutDetector._postprocess``.
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

MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"


def _sample_images(args) -> list:
    from PIL import Image

    images = []
    for path in args.images or []:
        images.append(Image.open(path).convert("RGB"))
    for pdf in args.pdf or []:
        from bibr.ocr.utils import render_pdf_pages

        pages = render_pdf_pages(Path(pdf).read_bytes(), dpi=args.dpi, end_page=args.pages - 1)
        images.extend(img for _, img in pages)
    if not images:
        # Deterministic synthetic pages: a white page with black text-like bars.
        rng = np.random.default_rng(0)
        for h, w in ((2200, 1700), (1754, 1240), (1100, 1700)):
            page = np.full((h, w, 3), 255, dtype=np.uint8)
            for _ in range(40):
                y = int(rng.integers(50, h - 60))
                x = int(rng.integers(50, w // 2))
                page[y : y + 18, x : x + int(rng.integers(200, w // 2))] = 0
            images.append(Image.fromarray(page))
    return images


def _freeze_position_embedding() -> None:
    """Bake the 2-D sin/cos position embedding into the graph as a constant.

    The layout preprocessor resizes every page to a fixed ``size``, so the
    embedding grid is fixed too and the table is a compile-time constant. But
    transformers builds it with float64 tensor ops, which the tracer emits as
    ``Sin``/``Cos`` on doubles — and ONNX Runtime ships no kernel for those, so
    the exported graph fails to load with "Could not find an implementation for
    Cos(7)". Evaluating the same float64 arithmetic eagerly in numpy keeps the
    values bit-for-bit and leaves an initializer where the subgraph was.
    """
    import numpy as np
    import torch
    from transformers.models.pp_doclayout_v3 import modeling_pp_doclayout_v3 as modeling

    def build(
        height,
        width,
        embed_dim,
        temperature: float = 10000.0,
        cls_token: bool = False,
        device=None,
        dtype=None,
    ):
        height, width, embed_dim = int(height), int(width), int(embed_dim)
        if embed_dim % 4 != 0:
            raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")
        pos_dim = embed_dim // 4
        omega = 1.0 / np.float64(temperature) ** (np.arange(pos_dim, dtype=np.float64) / pos_dim)
        grid_h, grid_w = np.meshgrid(
            np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij"
        )
        emb_h = np.outer(grid_h.ravel(), omega)
        emb_w = np.outer(grid_w.ravel(), omega)
        table = np.concatenate([np.sin(emb_h), np.cos(emb_h), np.sin(emb_w), np.cos(emb_w)], axis=1)
        if cls_token:
            table = np.concatenate([np.zeros((1, embed_dim), dtype=np.float64), table], axis=0)
        out = torch.from_numpy(np.ascontiguousarray(table))
        return out.to(device=device, dtype=dtype or torch.float32)

    modeling.build_2d_sinusoidal_position_embedding = build
    cached = (
        modeling.PPDocLayoutV3SinePositionEmbedding._cached_build_2d_sinusoidal_position_embedding
    )
    for name in ("cache_clear", "clear_cache"):
        clear = getattr(cached, name, None)
        if callable(clear):
            clear()
            break


class _Wrapper:
    """Graph = HF model minus the mask head outputs."""

    def __init__(self, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values):
                out = self.inner(pixel_values=pixel_values)
                return out.logits, out.pred_boxes, out.order_logits

        self.module = Wrapped(model).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--revision", default=None, help="defaults to LAYOUT_MODEL_REVISION")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--dynamo", action="store_true", help="use the torch.export exporter")
    parser.add_argument("--images", nargs="*", help="sample page images for the parity check")
    parser.add_argument(
        "--pdf", nargs="*", help="sample PDFs whose first --pages pages are rendered"
    )
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    from bibr.config import GlobalSettings
    from bibr.layout_onnx import OnnxLayoutBackend, order_sequences, preprocess_images

    settings = GlobalSettings()
    revision = args.revision or settings.layout.model_revision
    timer = Timer()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, revision=revision).eval()
    print(f"loaded {MODEL_ID}@{revision} in {timer.lap():.1f}s")
    _freeze_position_embedding()

    size = (int(processor.size["height"]), int(processor.size["width"]))
    bundle = bundle_dir(args.out)
    model_path = bundle / "model.onnx"
    example = torch.zeros((1, 3, *size), dtype=torch.float32)
    export_torch_module(
        _Wrapper(model).module,
        (example,),
        model_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes", "order_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "order_logits": {0: "batch"},
        },
        opset=args.opset,
        dynamo=args.dynamo,
    )
    print(f"exported in {timer.lap():.1f}s -> {model_path} ({file_size_mb(model_path):.1f} MB)")
    check_onnx(model_path)

    manifest = {
        "model": "layout",
        "architecture": "PPDocLayoutV3ForObjectDetection",
        "opset": args.opset,
        "exporter": "dynamo" if args.dynamo else "torchscript",
        "inputs": [
            {"name": "pixel_values", "shape": ["batch", 3, size[0], size[1]], "dtype": "float32"}
        ],
        "outputs": [
            {
                "name": "logits",
                "shape": ["batch", model.config.num_queries, model.config.num_labels],
            },
            {
                "name": "pred_boxes",
                "shape": ["batch", model.config.num_queries, 4],
                "format": "cxcywh_normalized",
            },
            {
                "name": "order_logits",
                "shape": ["batch", model.config.num_queries, model.config.num_queries],
            },
        ],
        "preprocessing": {
            "size": {"height": size[0], "width": size[1]},
            "resample": "bicubic_no_antialias",
            "rescale_factor": float(processor.rescale_factor),
            "image_mean": [float(v) for v in processor.image_mean],
            "image_std": [float(v) for v in processor.image_std],
        },
        "num_queries": int(model.config.num_queries),
        "num_labels": int(model.config.num_labels),
        "source": {"repo_id": MODEL_ID, "revision": revision},
    }
    write_manifest(bundle, manifest)

    # ---- parity ---------------------------------------------------------
    images = _sample_images(args)
    threshold = settings.layout.detection_threshold
    backend = OnnxLayoutBackend(bundle, device="cpu", threshold=threshold)

    hf_inputs = processor(images=images, return_tensors="pt")
    ours = preprocess_images(
        images,
        size=size,
        rescale_factor=float(processor.rescale_factor),
        image_mean=[float(v) for v in processor.image_mean],
        image_std=[float(v) for v in processor.image_std],
    )
    pre_diff = float(np.abs(hf_inputs["pixel_values"].numpy() - ours).max())

    with torch.inference_mode():
        out = model(**hf_inputs)
    logits_t, boxes_t, order_t = (
        out.logits.numpy(),
        out.pred_boxes.numpy(),
        out.order_logits.numpy(),
    )
    # Same inputs through the graph: isolates the export from preprocessing.
    logits_o, boxes_o, order_o = backend.forward(hf_inputs["pixel_values"].numpy())
    # Queries above the detection threshold are what reach postprocessing; the
    # order head carries ±1e4 mask values whose fp32 noise is irrelevant.
    kept = (1.0 / (1.0 + np.exp(-logits_t))).max(axis=-1) >= threshold
    graph_diff = {
        "logits": float(np.abs(logits_t - logits_o).max()),
        "logits_kept": float(np.abs(logits_t[kept] - logits_o[kept]).max()),
        "pred_boxes_kept": float(np.abs(boxes_t[kept] - boxes_o[kept]).max()),
        "order_logits": float(np.abs(order_t - order_o).max()),
        "order_seq_kept_equal": all(
            np.array_equal(
                order_sequences(order_t[b : b + 1])[0][kept[b]],
                order_sequences(order_o[b : b + 1])[0][kept[b]],
            )
            for b in range(len(images))
        ),
        "n_kept": int(kept.sum()),
    }

    # End to end: bibr's numpy pre/post vs HF pre/post, then bibr's postprocess.
    from bibr.layout_base import BaseLayoutDetector

    det = object.__new__(BaseLayoutDetector)
    det.threshold = threshold
    det._settings = settings
    from bibr.layout_utils import _CORRECT_ID2LABEL

    det._id2label = _CORRECT_ID2LABEL
    orig_sizes = [(img.height, img.width) for img in images]
    hf_results = processor.post_process_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor(orig_sizes, dtype=torch.float32)
    )
    onnx_results = backend.run(images)
    region_mismatch = 0
    max_box_delta = 0
    for hf_r, ox_r, (h, w) in zip(hf_results, onnx_results, orig_sizes, strict=True):
        a = det._postprocess(hf_r, w, h)
        b = det._postprocess(ox_r, w, h)
        if len(a) != len(b):
            region_mismatch += 1
            continue
        for ra, rb in zip(a, b, strict=True):
            if ra["label"] != rb["label"]:
                region_mismatch += 1
                break
            max_box_delta = max(
                max_box_delta,
                max(abs(x - y) for x, y in zip(ra["bbox_2d"], rb["bbox_2d"], strict=True)),
            )
    onnx_time = timer.lap()
    report(
        "layout parity",
        [
            ("sample pages", str(len(images))),
            ("preprocess max|Δ| (pixel_values)", f"{pre_diff:.3e}"),
            ("queries >= threshold", str(graph_diff["n_kept"])),
            (
                "graph max|Δ| logits (all / kept queries)",
                f"{graph_diff['logits']:.3e} / {graph_diff['logits_kept']:.3e}",
            ),
            ("graph max|Δ| pred_boxes (kept queries)", f"{graph_diff['pred_boxes_kept']:.3e}"),
            (
                "graph max|Δ| order_logits (all, incl. ±1e4 masks)",
                f"{graph_diff['order_logits']:.3e}",
            ),
            ("order sequence identical on kept queries", str(graph_diff["order_seq_kept_equal"])),
            (
                "regions (HF pre/post vs bibr numpy pre/post)",
                f"{sum(len(det._postprocess(r, w, h)) for r, (h, w) in zip(hf_results, orig_sizes, strict=True))} regions, {region_mismatch} pages differ, max bbox Δ {max_box_delta} (0-1000 scale)",
            ),
            ("model.onnx", f"{file_size_mb(model_path):.1f} MB"),
            ("onnx forward+post time", f"{onnx_time:.1f}s"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
