"""Torch-free ONNX runtime tests (x-tests-8).

The default core install serves every bibr model through ONNX Runtime — no
torch, no transformers. The parity modules (``test_*_onnx.py``) build their
fixtures in-test with ``torch.onnx.export``, so in a core environment they
skip entirely and the runtime most users run goes untested. The tiny
deterministic bundles under ``tests/fixtures/onnx/`` (see
``scripts/generate_onnx_test_bundles.py``) let these tests load the real
``Onnx*`` classes and run real inference with only core dependencies.

No ``pytest.importorskip("torch")`` may appear in this module: it must run
in the core-compat and core-install CI jobs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from bibr.layout_onnx import OnnxLayoutBackend
from bibr.ner.parser_onnx import OnnxRefParser
from bibr.structure.paper_classifier_onnx import OnnxPaperClassifierModel
from bibr.structure.section_classifier_common import HeaderContext
from bibr.structure.section_classifier_onnx import OnnxSectionClassifierModel

BUNDLES = Path(__file__).parent / "fixtures" / "onnx"


def _bundle(name: str) -> Path:
    """The loadable ``onnx/`` child (exporter layout); loaders take the parent."""
    return BUNDLES / name / "onnx"


SECTION_LABELS = ["introduction", "method", "results", "discussion", "unknown"]
L1 = ["Natural Sciences", "Social Sciences"]
L2 = ["Physical Sciences", "Psychology and Cognitive Sciences", "Economics and Business"]
PAPER_TYPES = ["empirical", "review", "commentary"]

CONTEXTS = [
    HeaderContext("Methods", "Participants were recruited. " * 5, 0.3, "Introduction", "Results"),
    HeaderContext("Results", "The results of the sleep study.", 0.6, "Methods", "Discussion"),
    HeaderContext("Ethics", "IRB approved.", 0.95, "Discussion", ""),
    ("Introduction", "memory and consciousness"),  # tuple form
]

ITEMS = [
    ("Memory and consciousness", "Participants were recruited. Results and discussion."),
    ("Sleep science", ""),
    ("", "The journal of science, 2020."),
]

REFS = [
    "Smith J (2020). Memory and consciousness. Journal of science, 26(1), 1-12.",
    "Doe A. 2021. Sleep. Journal 12: 1.",
]


def _manifest(name: str) -> dict:
    path = _bundle(name) / "bibr_onnx.json"
    assert path.is_file(), f"committed ONNX bundle {name} missing ({path})"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert (_bundle(name) / manifest["model_file"]).is_file()
    return manifest


def test_section_bundle_manifest_is_well_formed():
    manifest = _manifest("section")
    assert manifest["label_classes"] == SECTION_LABELS
    assert manifest["template_version"] == 3


def test_section_classifier_runs_torch_free():
    model = OnnxSectionClassifierModel(_bundle("section"), "cpu")
    first = model.classify_batch(CONTEXTS, max_length=32)
    assert len(first) == len(CONTEXTS)
    for pred in first:
        assert pred.canonical_type.value in SECTION_LABELS
        assert 0.0 <= pred.score <= 1.0
    # Deterministic smoke bundles: same input, same prediction.
    second = model.classify_batch(CONTEXTS, max_length=32)
    assert [(p.canonical_type, p.score) for p in first] == [
        (p.canonical_type, p.score) for p in second
    ]


def test_paper_bundle_manifest_is_well_formed():
    manifest = _manifest("paper")
    assert manifest["l1_classes"] == L1
    assert manifest["l2_classes"] == L2
    assert manifest["paper_type_classes"] == PAPER_TYPES
    assert manifest["paper_type_temperature"] == 1.5


def test_paper_classifier_runs_torch_free():
    model = OnnxPaperClassifierModel(_bundle("paper"), "cpu")
    assert model.paper_type_temperature == 1.5
    first = model.classify_batch(ITEMS)
    assert len(first) == len(ITEMS)
    for pred in first:
        assert pred.oecd_l1 in L1
        assert pred.oecd_l2 in L2
        assert pred.paper_type in PAPER_TYPES
        assert 0.0 <= pred.oecd_l1_score <= 1.0
    second = model.classify_batch(ITEMS)
    assert [(p.oecd_l1, p.oecd_l2, p.paper_type) for p in first] == [
        (p.oecd_l1, p.oecd_l2, p.paper_type) for p in second
    ]


def test_ner_bundle_manifest_is_well_formed():
    from bibr.ner.tags import BIO_TAGS

    manifest = _manifest("ner")
    assert manifest["bio_tags"] == list(BIO_TAGS)
    assert manifest["max_length"] == 16
    assert set(manifest["crf"]) == {"start_transitions", "end_transitions", "transitions"}


def test_ner_parser_runs_torch_free():
    from bibr.ner.tags import BIO_TAGS

    parser = OnnxRefParser(_bundle("ner"), device="cpu")
    assert parser.tags == list(BIO_TAGS)
    assert parser.parse("") == {}
    assert parser.parse("   ") == {}
    first = parser.parse_batch(REFS)
    assert len(first) == len(REFS)
    assert all(isinstance(entry, dict) for entry in first)
    assert parser.parse_batch(REFS) == first


def test_layout_bundle_manifest_is_well_formed():
    manifest = _manifest("layout")
    assert manifest["model"] == "layout"
    assert manifest["preprocessing"]["size"] == {"height": 64, "width": 64}
    assert manifest["num_queries"] == 8
    assert manifest["num_labels"] == 5


def _sample_pages() -> list[Image.Image]:
    pages = []
    for i in range(2):
        h, w = 96 + 40 * i, 80 + 24 * i
        page = np.full((h, w, 3), 255, dtype=np.uint8)
        page[::7, :, 1] = 128
        pages.append(Image.fromarray(page))
    return pages


def test_layout_backend_runs_torch_free():
    backend = OnnxLayoutBackend(_bundle("layout"), threshold=0.05)
    pages = _sample_pages()
    first = backend.run(pages)
    assert len(first) == len(pages)
    for det in first:
        assert set(det) == {"scores", "labels", "boxes", "order_seq"}
        assert len(det["scores"]) > 0
        assert det["boxes"].shape[1] == 4
        assert (det["boxes"][:, 0] <= det["boxes"][:, 2]).all()
    second = backend.run(pages)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a["scores"], b["scores"])
        assert np.array_equal(a["labels"], b["labels"])


def test_layout_forward_shapes_match_the_v3_contract():
    from bibr.layout_onnx import preprocess_images

    backend = OnnxLayoutBackend(_bundle("layout"), threshold=0.99)
    pages = _sample_pages()
    pixel_values = preprocess_images(
        pages,
        size=backend.size,
        rescale_factor=backend.rescale_factor,
        image_mean=backend.image_mean,
        image_std=backend.image_std,
        rescale_before_resize=backend.rescale_before_resize,
    )
    assert pixel_values.shape == (2, 3, 64, 64)
    logits, boxes, order = backend.forward(pixel_values)
    assert logits.shape == (2, 8, 5)
    assert boxes.shape == (2, 8, 4)
    assert order.shape == (2, 8, 8)
    assert bool(((boxes >= 0.0) & (boxes <= 1.0)).all())
    # A strict threshold keeps nothing, but the run stays well-formed.
    assert all(len(det["scores"]) == 0 for det in backend.run(pages))
