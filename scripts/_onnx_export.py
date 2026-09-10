"""Shared helpers for the ``scripts/export_onnx_*.py`` exporters.

Every exporter writes a bundle directory laid out as bibr's ONNX artifact
contract expects (see ``bibr/utils/ml_runtime.py``)::

    <out>/onnx/model.onnx
    <out>/onnx/bibr_onnx.json
    <out>/onnx/tokenizer.json      (text models)

so ``<out>`` can be pointed at directly by the model setting, and ``<out>/onnx``
is what ``hf upload <repo> <out>/onnx onnx`` publishes.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# CPU only: the export must never touch the shared GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ONNX_DIRNAME = "onnx"
DEFAULT_OPSET = 17


def bundle_dir(out: Path) -> Path:
    d = Path(out) / ONNX_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_torch_module(
    module,
    example_inputs: tuple,
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int = DEFAULT_OPSET,
    dynamo: bool = False,
) -> None:
    """``torch.onnx.export`` with the TorchScript exporter by default.

    ``dynamo=True`` switches to the torch.export-based exporter; the
    ``dynamic_axes`` mapping is translated into ``dynamic_shapes`` for it.
    """
    import torch

    module.eval()
    kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamo:
        shapes = tuple(dynamic_axes.get(name, {}) for name in input_names)
        torch.onnx.export(
            module, example_inputs, str(path), dynamic_shapes=shapes, dynamo=True, **kwargs
        )
    else:
        torch.onnx.export(
            module, example_inputs, str(path), dynamic_axes=dynamic_axes, dynamo=False, **kwargs
        )


def check_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)


def environment_stamp() -> dict:
    import numpy
    import onnx
    import onnxruntime
    import torch
    import transformers

    return {
        "exported_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def write_manifest(bundle: Path, manifest: dict) -> Path:
    manifest = {"schema_version": 1, **manifest}
    manifest.setdefault("model_file", "model.onnx")
    files = ["model.onnx"]
    if (bundle / "tokenizer.json").is_file():
        files.append("tokenizer.json")
    manifest["files"] = files
    manifest["source"] = {**manifest.get("source", {}), **environment_stamp()}
    path = bundle / "bibr_onnx.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def copy_tokenizer(src_dir: Path, bundle: Path) -> Path:
    src = Path(src_dir) / "tokenizer.json"
    if not src.is_file():
        raise FileNotFoundError(f"no tokenizer.json in {src_dir}")
    dst = bundle / "tokenizer.json"
    shutil.copyfile(src, dst)
    return dst


def copy_bundle_files(src_dir: Path, out: Path, names: list[str]) -> None:
    """Copy the torch bundle's small sidecar files so ``out`` is a complete local bundle."""
    for name in names:
        src = Path(src_dir) / name
        if src.is_file():
            shutil.copyfile(src, Path(out) / name)


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def report(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n== {title}")
    width = max(len(k) for k, _ in rows) if rows else 0
    for key, value in rows:
        print(f"  {key.ljust(width)}  {value}")


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = now - self.t0
        self.t0 = now
        return elapsed


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)
