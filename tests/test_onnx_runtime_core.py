"""Torch-free ONNX runtime tests (x-tests-8).

The default core install serves every bibr model through ONNX Runtime — no
torch, no transformers. The parity modules (``test_*_onnx.py``) build their
fixtures in-test with ``torch.onnx.export``, so in a core environment they
skip entirely and the runtime most users run goes untested. The tiny
deterministic bundles under ``tests/fixtures/onnx/`` (see
``scripts/generate_onnx_test_bundles.py``) let these tests load the real
``Onnx*`` classes and run real inference with only core dependencies.

The bundles are stand-ins, not trained models — but every head is a genuine
function of its inputs (parabolas over the token-id mean and length, a
per-token-id NER emission table with a nonzero CRF, a layout head reading
the image mean). The tests below pin golden outputs for the fixed inputs,
so a numpy-only preprocessing bug (tokenization, attention mask,
normalization, resize) or a decode regression (argmax flip, dropped
temperature, reversed tag path) that the torch path does not share fails
these tests instead of passing silently.

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


SECTION_LABELS = ["intro", "method", "results", "discussion", "unknown"]
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

# Golden outputs for the fixed inputs above, from the committed bundles.
# Each model predicts differently per input (no constant functions): the
# section head covers four of its five classes, every paper head varies,
# the NER parses carry several fields each, and the two layout pages give
# different boxes, scores and orderings.
SECTION_GOLDEN = [
    ("method", True, 1.0),
    ("results", False, 0.8973),
    ("unknown", True, 1.0),
    ("intro", False, 0.8975),
]

PAPER_GOLDEN = [
    ("Natural Sciences", 1.0, "Economics and Business", 0.9302, "review", 0.8299),
    ("Social Sciences", 0.9932, "Physical Sciences", 0.7561, "empirical", 0.6655),
    (
        "Natural Sciences",
        0.9938,
        "Psychology and Cognitive Sciences",
        0.7228,
        "commentary",
        0.6134,
    ),
]

NER_GOLDEN = [
    {
        "doi": "( 26(",
        "edition": "Smith . Memory and consciousness",
        "editors": ".",
        "first_page": "science",
        "pmid": ",",
        "title": "2020 )",
        "url": "Journal of",
    },
    {
        "edition": ". . .",
        "editors": "A.",
        "last_page": "Doe",
        "publisher": "12:",
        "year": 1,
    },
]

LAYOUT_GOLDEN = [
    {
        "labels": [4, 3, 2, 4, 3, 2, 1, 0],
        "scores": [0.65, 0.644, 0.637, 0.683, 0.677, 0.67, 0.664, 0.657],
        "order": [4, 4, 4, 6, 6, 6, 6, 6],
        "boxes": [
            23.51,
            28.55,
            72.78,
            88.36,
            23.51,
            28.55,
            72.78,
            88.36,
            23.51,
            28.55,
            72.78,
            88.36,
            24.65,
            29.91,
            76.14,
            92.36,
            24.65,
            29.91,
            76.14,
            92.36,
            24.65,
            29.91,
            76.14,
            92.36,
            24.65,
            29.91,
            76.14,
            92.36,
            24.65,
            29.91,
            76.14,
            92.36,
        ],
    },
    {
        "labels": [4, 3, 2, 4, 3, 2, 1, 0],
        "scores": [0.593, 0.588, 0.582, 0.62, 0.615, 0.61, 0.604, 0.599],
        "order": [3, 3, 3, 5, 5, 5, 5, 5],
        "boxes": [
            28.43,
            37.56,
            87.61,
            115.71,
            28.43,
            37.56,
            87.61,
            115.71,
            28.43,
            37.56,
            87.61,
            115.71,
            29.6,
            39.08,
            91.09,
            120.25,
            29.6,
            39.08,
            91.09,
            120.25,
            29.6,
            39.08,
            91.09,
            120.25,
            29.6,
            39.08,
            91.09,
            120.25,
            29.6,
            39.08,
            91.09,
            120.25,
        ],
    },
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


def test_section_classifier_predicts_golden_labels_torch_free():
    model = OnnxSectionClassifierModel(_bundle("section"), "cpu")
    first = model.classify_batch(CONTEXTS, max_length=32)
    assert len(first) == len(CONTEXTS)
    assert [(p.canonical_type.value, p.is_top_level, round(p.score, 4)) for p in first] == (
        SECTION_GOLDEN
    )
    # Repeat runs are deterministic: the same inputs give identical outputs.
    # (Input-dependence itself is pinned by the golden assertion above.)
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


def test_paper_classifier_predicts_golden_labels_torch_free():
    model = OnnxPaperClassifierModel(_bundle("paper"), "cpu")
    assert model.paper_type_temperature == 1.5
    first = model.classify_batch(ITEMS)
    assert len(first) == len(ITEMS)
    # The paper_type scores are temperature-scaled (T=1.5): dropping the
    # division changes them in the second decimal, failing this pin.
    assert [
        (
            p.oecd_l1,
            round(p.oecd_l1_score, 4),
            p.oecd_l2,
            round(p.oecd_l2_score, 4),
            p.paper_type,
            round(p.paper_type_score, 4),
        )
        for p in first
    ] == PAPER_GOLDEN
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
    # The emissions alone must not decide the path: nonzero transitions mean
    # the numpy Viterbi decode is exercised, not just the argmax.
    assert any(v != 0.0 for row in manifest["crf"]["transitions"] for v in row)


def test_ner_parser_predicts_golden_fields_torch_free():
    parser = OnnxRefParser(_bundle("ner"), device="cpu")
    assert parser.parse("") == {}
    assert parser.parse("   ") == {}
    first = parser.parse_batch(REFS)
    assert len(first) == len(REFS)
    assert all(isinstance(entry, dict) for entry in first)
    assert first == NER_GOLDEN
    assert parser.parse_batch(REFS) == first


def test_layout_bundle_manifest_is_well_formed():
    manifest = _manifest("layout")
    assert manifest["model"] == "layout"
    assert manifest["preprocessing"]["size"] == {"height": 64, "width": 64}
    assert manifest["num_queries"] == 8
    assert manifest["num_labels"] == 5


def _sample_pages() -> list[Image.Image]:
    """Two pages with clearly different pixel means: near-white and dark."""
    bright = np.full((96, 80, 3), 255, dtype=np.uint8)
    bright[::7, :, 1] = 128
    dark = np.full((136, 104, 3), 60, dtype=np.uint8)
    dark[::5, :, 0] = 220
    return [Image.fromarray(bright), Image.fromarray(dark)]


def _rounded(det: dict) -> dict:
    return {
        "labels": [int(v) for v in det["labels"]],
        "scores": [round(float(v), 3) for v in det["scores"]],
        "order": [int(v) for v in det["order_seq"]],
        "boxes": [round(float(v), 2) for v in det["boxes"].ravel()],
    }


def test_layout_backend_predicts_golden_detections_torch_free():
    backend = OnnxLayoutBackend(_bundle("layout"), threshold=0.05)
    pages = _sample_pages()
    first = backend.run(pages)
    assert len(first) == len(pages)
    for det in first:
        assert set(det) == {"scores", "labels", "boxes", "order_seq"}
        assert len(det["scores"]) > 0
        assert det["boxes"].shape[1] == 4
        assert (det["boxes"][:, 0] <= det["boxes"][:, 2]).all()
    assert [_rounded(det) for det in first] == LAYOUT_GOLDEN
    # The head reads the pixels: the dark page detects differently.
    assert _rounded(first[0])["boxes"] != _rounded(first[1])["boxes"]
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
