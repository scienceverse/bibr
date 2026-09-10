"""The HTTP runtime path must not import torch when the ONNX classes are selected.

A fresh interpreter (clean ``sys.modules``) imports every module ``bibr chew``
touches for a PDF with ``OCR_BACKEND=paddle-http`` and a cloud LLM, then
constructs the ONNX layout detector, section/paper classifiers and NER parser
on tiny bundles exported here and runs each once. Only then is ``sys.modules``
checked for ``torch`` (and ``transformers``/``cv2``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchcrf")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from tests import onnx_fixtures as fx  # noqa: E402

CHILD = r"""
import json, os, sys
from pathlib import Path

cfg = json.loads(sys.argv[1])
os.environ.update(cfg["env"])

# Everything the chew path imports for a PDF over paddle-http + a cloud LLM.
import bibr  # noqa: F401
from bibr import LocalPipeline, Pipeline, Settings  # noqa: F401
import bibr.local.cli  # noqa: F401
import bibr.local.pipeline  # noqa: F401
import bibr.pipeline.resources  # noqa: F401
import bibr.pipeline.classifier_resources  # noqa: F401
import bibr.pipeline.stages.layout  # noqa: F401
import bibr.local.ocr  # noqa: F401
import bibr.ocr.image_processing  # noqa: F401
import bibr.clients.llm  # noqa: F401
import bibr.extract.ref_extractor  # noqa: F401
import bibr.structure.section_classifier  # noqa: F401
import bibr.structure.paper_classifier  # noqa: F401
import bibr.segmenter_base  # noqa: F401
import bibr.utils.device  # noqa: F401
from bibr.layout_base import BaseLayoutDetector  # noqa: F401
from bibr.local.layout import LayoutDetector
from bibr.ner.runtime import load_ref_parser
from bibr.structure.paper_classifier_common import load_paper_classifier
from bibr.structure.section_classifier_common import HeaderContext, load_section_classifier
from bibr.config import GlobalSettings
from PIL import Image

settings = GlobalSettings()
assert settings.ml.runtime == "onnx", settings.ml.runtime

det = LayoutDetector(device="cpu", settings=settings)
assert det._runtime == "onnx"
import asyncio
regions = asyncio.run(det.detect_batch([Image.new("RGB", (90, 120), "white")]))
assert len(regions) == 1

sec = load_section_classifier(cfg["section"], revision="main", device="cpu", settings=settings)
assert sec.runtime == "onnx"
sec.classify_batch([HeaderContext("Methods", "Participants were recruited.")])

paper = load_paper_classifier(cfg["paper"], revision="main", device="cpu", settings=settings)
assert paper.runtime == "onnx"
paper.classify_batch([("Memory and consciousness", "The journal of science")])

ner = load_ref_parser(cfg["ner"], device="cpu", revision=None, settings=settings)
assert ner.runtime == "onnx"
ner.parse_batch(["Smith J (2020). Memory. Journal 1:1.", ""])

leaked = sorted(m for m in ("torch", "transformers", "cv2", "torchvision", "torchcrf") if m in sys.modules)
print(json.dumps({"leaked": leaked, "ok": True}))
"""


def test_http_path_with_onnx_bundles_never_imports_torch(monkeypatch, tmp_path):
    tok = fx.tiny_tokenizer()
    layout_root = tmp_path / "layout"
    fx.export_layout_bundle(fx.tiny_layout_module(), layout_root)
    section_root = tmp_path / "section"
    fx.export_section_bundle(fx.tiny_section_model(monkeypatch), tok, section_root)
    paper_root = tmp_path / "paper"
    fx.export_paper_bundle(fx.tiny_paper_model(), tok, paper_root)
    ner_root = tmp_path / "ner"
    fx.export_ner_bundle(fx.tiny_ner_model(monkeypatch), tok, ner_root)

    cfg = {
        "env": {
            "ML_RUNTIME": "onnx",
            "LAYOUT_ONNX_MODEL_ID": str(layout_root),
            "ML_SECTION_CLASSIFIER_MODEL_ID": str(section_root),
            "ML_PAPER_CLASSIFIER_MODEL_ID": str(paper_root),
            "NER_PARSER_CKPT": str(ner_root),
            "OCR_BACKEND": "paddle-http",
            "OCR_BASE_URL": "http://127.0.0.1:1",
            "LLM_PROVIDER": "google",
            "GOOGLE_API_KEY": "test-key",
            "REF_PARSE_STRATEGY": "ner",
            "BIBR_DISABLE_DOTENV": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        "section": str(section_root),
        "paper": str(paper_root),
        "ner": str(ner_root),
    }
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(cfg)],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**cfg["env"], "PATH": "", "PYTHONPATH": str(root), "HOME": str(tmp_path)},
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"]
    assert result["leaked"] == [], f"heavy modules imported on the HTTP path: {result['leaked']}"
