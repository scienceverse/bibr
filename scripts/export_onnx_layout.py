"""Export the PP-DocLayout layout detector (V3 or V4) to ONNX and check parity.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/export_onnx_layout.py \
        --out /mnt/bulk/datasets/onnx_exports/layout [--model-id <repo or dir>] \
        [--revision <sha>] [--images page1.png page2.png ...] [--pdf paper.pdf]

Loads the checkpoint (default ``LAYOUT_MODEL_ID`` at ``LAYOUT_MODEL_REVISION``)
on CPU and exports the graph bibr's ONNX backend reads:

- PP-DocLayoutV3: ``logits``, ``pred_boxes`` and ``order_logits`` (no mask
  head — bibr never reads the polygons);
- PP-DocLayoutV4: ``logits``, ``pred_boxes`` (quads), ``relative_order_logits``
  and ``successor_order_logits``. Needs a transformers with PP-DocLayoutV4 and,
  for the parity check's reference post-processing, scipy.

It writes ``<out>/onnx/{model.onnx,bibr_onnx.json}`` (the manifest names the
architecture, which is what selects bibr's pre/post-processing) and then
compares bibr's ONNX layout backend with the transformers path on the sample
pages: preprocessing (pixel values), raw graph outputs, the reading order and
the final region lists after ``BaseLayoutDetector._postprocess``.
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

# model_type -> what differs between the two generations.
ARCHITECTURES = {
    "pp_doclayout_v3": {
        "architecture": "PPDocLayoutV3ForObjectDetection",
        "sine": "PPDocLayoutV3SinePositionEmbedding",
        "outputs": ("logits", "pred_boxes", "order_logits"),
        "box_dims": 4,
        "box_format": "cxcywh_normalized",
        "resample": "bicubic_no_antialias",
        "rescale_before_resize": False,
    },
    "pp_doclayout_v4": {
        "architecture": "PPDocLayoutV4ForObjectDetection",
        "sine": "PPDocLayoutV4SinePositionEmbedding",
        "outputs": ("logits", "pred_boxes", "relative_order_logits", "successor_order_logits"),
        "box_dims": 10,
        "box_format": "quad_center_offsets_normalized",
        "resample": "bicubic_no_antialias_float",
        "rescale_before_resize": True,
    },
}


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


def _freeze_position_embedding(model_type: str) -> None:
    """Bake the 2-D sin/cos position embedding into the graph as a constant.

    The layout preprocessor resizes every page to a fixed ``size``, so the
    embedding grid is fixed too and the table is a compile-time constant. But
    transformers builds it with float64 tensor ops, which the tracer emits as
    ``Sin``/``Cos`` on doubles — and ONNX Runtime ships no kernel for those, so
    the exported graph fails to load with "Could not find an implementation for
    Cos(7)". Evaluating the same float64 arithmetic eagerly in numpy keeps the
    values bit-for-bit and leaves an initializer where the subgraph was.
    """
    import importlib

    import numpy as np
    import torch

    modeling = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")

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
    sine = getattr(modeling, ARCHITECTURES[model_type]["sine"])
    cached = sine._cached_build_2d_sinusoidal_position_embedding
    for name in ("cache_clear", "clear_cache"):
        clear = getattr(cached, name, None)
        if callable(clear):
            clear()
            break


class _Wrapper:
    """Graph = HF model restricted to the outputs bibr reads (V3: no mask head)."""

    def __init__(self, model, output_names: tuple[str, ...]):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values):
                out = self.inner(pixel_values=pixel_values)
                return tuple(getattr(out, name) for name in output_names)

        self.module = Wrapped(model).eval()


def _resolve_source(args, settings) -> tuple[str, str | None]:
    """``(model_id, revision)`` to export; the revision is a commit SHA or ``None`` (local dir).

    The configured pin applies only to the configured checkpoint: exporting
    another repo (say V4 while bibr still pins V3) without ``--revision``
    resolves that repo's current head to a SHA, so the manifest never records
    a revision the weights did not come from.
    """
    model_id = args.model_id or settings.layout.model_id
    if Path(model_id).is_dir():
        return model_id, None
    if args.revision:
        return model_id, args.revision
    if model_id == settings.layout.model_id:
        return model_id, settings.layout.model_revision
    from huggingface_hub import HfApi

    return model_id, HfApi().model_info(model_id).sha


def _detector(settings, threshold):
    """A postprocess-only ``BaseLayoutDetector`` (no model load)."""
    from bibr.layout_base import BaseLayoutDetector
    from bibr.layout_utils import _CORRECT_ID2LABEL

    det = object.__new__(BaseLayoutDetector)
    det.threshold = threshold
    det._settings = settings
    det._id2label = _CORRECT_ID2LABEL
    return det


def _region_diff(det, reference, ours, orig_sizes) -> tuple[int, int, int]:
    """``(regions, pages that differ, max bbox Δ on the 0-1000 scale)``."""
    regions = mismatch = max_box_delta = 0
    for ref_r, our_r, (h, w) in zip(reference, ours, orig_sizes, strict=True):
        a = det._postprocess(ref_r, w, h)
        b = det._postprocess(our_r, w, h)
        regions += len(a)
        if len(a) != len(b) or any(ra["label"] != rb["label"] for ra, rb in zip(a, b, strict=True)):
            mismatch += 1
            continue
        for ra, rb in zip(a, b, strict=True):
            max_box_delta = max(
                max_box_delta,
                max(abs(x - y) for x, y in zip(ra["bbox_2d"], rb["bbox_2d"], strict=True)),
            )
    return regions, mismatch, max_box_delta


def _parity_v4(args, settings, processor, model, bundle, model_path, size, timer) -> int:
    """PP-DocLayoutV4: bibr's numpy pipeline vs the transformers processor + model."""
    import torch

    from bibr.layout_onnx import OnnxLayoutBackend, decode_detections_v4, preprocess_images

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
        rescale_before_resize=True,
    )
    pre_diff = float(np.abs(hf_inputs["pixel_values"].numpy() - ours).max())

    with torch.inference_mode():
        out = model(**hf_inputs)
    names = ARCHITECTURES["pp_doclayout_v4"]["outputs"]
    torch_raw = [getattr(out, name).float().numpy() for name in names]
    # Same inputs through the graph: isolates the export from preprocessing.
    onnx_raw = backend.forward(hf_inputs["pixel_values"].numpy())
    kept = (1.0 / (1.0 + np.exp(-torch_raw[0]))).max(axis=-1) >= threshold
    graph = {"n_kept": int(kept.sum())}
    for name, t, o in zip(names, torch_raw, onnx_raw, strict=True):
        if name.endswith("order_logits"):  # pairwise (queries x queries)
            deltas = []
            for b in range(len(images)):
                idx = np.flatnonzero(kept[b])
                sub_t, sub_o = t[b][np.ix_(idx, idx)], o[b][np.ix_(idx, idx)]
                off_diagonal = ~np.eye(len(idx), dtype=bool)
                if off_diagonal.any():
                    deltas.append(float(np.abs(sub_t - sub_o)[off_diagonal].max()))
            graph[name] = max(deltas, default=0.0)
        else:
            graph[name] = float(np.abs(t[kept] - o[kept]).max()) if kept.any() else 0.0

    orig_sizes = [(img.height, img.width) for img in images]
    hf_results = processor.post_process_object_detection(
        out, threshold=threshold, target_sizes=orig_sizes
    )
    ours_torch = decode_detections_v4(*torch_raw, orig_sizes, threshold)
    order_equal = all(
        np.array_equal(h["order_seq"].numpy(), o["order_seq"])
        and np.array_equal(h["labels"].numpy(), o["labels"])
        for h, o in zip(hf_results, ours_torch, strict=True)
    )
    det = _detector(settings, threshold)
    onnx_results = backend.run(images)
    regions, mismatch, max_box_delta = _region_diff(det, hf_results, onnx_results, orig_sizes)
    onnx_time = timer.lap()
    report(
        "layout parity (PP-DocLayoutV4)",
        [
            ("sample pages", str(len(images))),
            ("preprocess max|Δ| (pixel_values)", f"{pre_diff:.3e}"),
            ("queries >= threshold", str(graph["n_kept"])),
            ("graph max|Δ| logits (kept)", f"{graph['logits']:.3e}"),
            ("graph max|Δ| pred_boxes quads (kept)", f"{graph['pred_boxes']:.3e}"),
            (
                "graph max|Δ| relative / successor order (kept, off-diagonal)",
                f"{graph['relative_order_logits']:.3e} / {graph['successor_order_logits']:.3e}",
            ),
            ("bibr decode == HF post-process (order, labels)", str(order_equal)),
            (
                "regions (HF pre/post vs bibr numpy pre/post)",
                f"{regions} regions, {mismatch} pages differ, max bbox Δ {max_box_delta} "
                "(0-1000 scale)",
            ),
            ("model.onnx", f"{file_size_mb(model_path):.1f} MB"),
            ("onnx forward+post time", f"{onnx_time:.1f}s"),
        ],
    )
    return 0 if order_equal and mismatch == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--model-id", default=None, help="Hub repo or local checkpoint; defaults to LAYOUT_MODEL_ID"
    )
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
    model_id, revision = _resolve_source(args, settings)
    timer = Timer()
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForObjectDetection.from_pretrained(model_id, revision=revision).eval()
    model_type = model.config.model_type
    if model_type not in ARCHITECTURES:
        raise SystemExit(f"{model_id} is a {model_type!r} checkpoint, not PP-DocLayoutV3/V4")
    arch = ARCHITECTURES[model_type]
    where = f"{model_id}@{revision}" if revision else f"{model_id} (local)"
    print(f"loaded {where} ({arch['architecture']}) in {timer.lap():.1f}s")
    _freeze_position_embedding(model_type)
    output_names = arch["outputs"]

    size = (int(processor.size["height"]), int(processor.size["width"]))
    bundle = bundle_dir(args.out)
    model_path = bundle / "model.onnx"
    example = torch.zeros((1, 3, *size), dtype=torch.float32)
    export_torch_module(
        _Wrapper(model, output_names).module,
        (example,),
        model_path,
        input_names=["pixel_values"],
        output_names=list(output_names),
        dynamic_axes={name: {0: "batch"} for name in ("pixel_values", *output_names)},
        opset=args.opset,
        dynamo=args.dynamo,
    )
    print(f"exported in {timer.lap():.1f}s -> {model_path} ({file_size_mb(model_path):.1f} MB)")
    check_onnx(model_path)

    queries, labels = int(model.config.num_queries), int(model.config.num_labels)
    outputs = [
        {"name": "logits", "shape": ["batch", queries, labels]},
        {
            "name": "pred_boxes",
            "shape": ["batch", queries, arch["box_dims"]],
            "format": arch["box_format"],
        },
    ] + [{"name": name, "shape": ["batch", queries, queries]} for name in output_names[2:]]
    preprocessing = {
        "size": {"height": size[0], "width": size[1]},
        "resample": arch["resample"],
        "rescale_factor": float(processor.rescale_factor),
        "image_mean": [float(v) for v in processor.image_mean],
        "image_std": [float(v) for v in processor.image_std],
    }
    manifest = {
        "model": "layout",
        "architecture": arch["architecture"],
        "opset": args.opset,
        "exporter": "dynamo" if args.dynamo else "torchscript",
        "inputs": [
            {"name": "pixel_values", "shape": ["batch", 3, size[0], size[1]], "dtype": "float32"}
        ],
        "outputs": outputs,
        "preprocessing": preprocessing,
        "num_queries": queries,
        "num_labels": labels,
        "source": {"repo_id": model_id, "revision": revision},
    }
    if arch["rescale_before_resize"]:
        preprocessing["rescale_before_resize"] = True
        # V3's hub config collapses five labels, so only V4 records its list
        # (bibr refuses a bundle whose list differs from the one it maps).
        manifest["id2label"] = {str(k): v for k, v in model.config.id2label.items()}
    write_manifest(bundle, manifest)

    if model_type == "pp_doclayout_v4":
        return _parity_v4(args, settings, processor, model, bundle, model_path, size, timer)

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
    det = _detector(settings, threshold)
    orig_sizes = [(img.height, img.width) for img in images]
    hf_results = processor.post_process_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor(orig_sizes, dtype=torch.float32)
    )
    onnx_results = backend.run(images)
    regions, region_mismatch, max_box_delta = _region_diff(
        det, hf_results, onnx_results, orig_sizes
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
                f"{regions} regions, {region_mismatch} pages differ, max bbox Δ {max_box_delta} "
                "(0-1000 scale)",
            ),
            ("model.onnx", f"{file_size_mb(model_path):.1f} MB"),
            ("onnx forward+post time", f"{onnx_time:.1f}s"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
