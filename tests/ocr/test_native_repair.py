import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from bibr.config import GlobalSettings
from bibr.ocr.native_repair import plan_native_repairs
from bibr.ocr.native_source import NativeChar, account_characters, normalized_box
from bibr.ocr.pdf_inspection import inspect_pdf, inspection_from_dict, inspection_to_dict
from bibr.ocr.types import OcrRegionResult
from bibr.pipeline.stages.ocr import _deduplicate_formula_text_regions, ocr_page_regions


def test_raster_accounting_detects_omitted_scan_ink_without_native_text():
    from PIL import ImageDraw

    from bibr.ocr.native_source import visual_coverage

    with Image.new("RGB", (300, 300), "white") as image:
        ImageDraw.Draw(image).rectangle((30, 30, 100, 45), fill="black")
        source = {"characters": []}
        missed = visual_coverage(image, [], source)
        covered = visual_coverage(image, [{"bbox_2d": [0, 0, 500, 500]}], source)
    assert missed["outside_layout_pixels"] > 100
    assert missed["outside_layout_tiles"]
    assert covered["outside_layout_pixels"] == 0
    assert covered["outside_native_pixels"] == missed["ink_pixels"]


def test_duplicate_layer_candidates_retain_both_source_ids():
    chars = [NativeChar(i, i + 1, "A", (0, 0, 10, 10)) for i in range(2)]
    source = account_characters(chars, [], [], (0, 0, 100, 100), 0)
    assert source["duplicate_glyph_candidates"] == 1
    first, second = source["characters"]
    assert second["duplicate_of"] == first["source_id"]
    assert first["source_id"] != second["source_id"]
    assert source["counts"]["unassigned"] == 2


def test_source_coordinates_use_inherited_rendered_page_box():
    import pypdfium2 as pdfium

    from bibr.ocr.native_text import _page_crop_box

    with pdfium.PdfDocument("bibr/data/sample_paper.pdf") as doc:
        page = doc[0]
        try:
            box = _page_crop_box(page)
            width, height = page.get_size()
            assert box[2] - box[0] == pytest.approx(width)
            assert box[3] - box[1] == pytest.approx(height)
        finally:
            page.close()


@pytest.mark.asyncio
async def test_cancelled_line_repair_propagates_and_closes_crop():
    regions, _ = _plan(["Damaged native line containing \ue001 glyph."])
    crops = []

    async def cancel(crop, prompt):
        crops.append(crop)
        raise asyncio.CancelledError()

    with Image.new("RGB", (1000, 1000)) as image:
        with pytest.raises(asyncio.CancelledError):
            await ocr_page_regions(image, regions, 4, "sample.pdf", cancel, None)
        assert image.getpixel((0, 0)) == (0, 0, 0)
    with pytest.raises(ValueError, match="closed"):
        crops[0].getpixel((0, 0))


@pytest.mark.asyncio
async def test_truncated_repair_retains_raw_text_but_is_unresolved():
    from bibr.ocr.backend import OcrText

    regions, _ = _plan(["Damaged native line containing \ue001 glyph."])

    async def truncate(crop, prompt):
        return OcrText("An incomplete repair", finish_reason="length")

    with Image.new("RGB", (1000, 1000)) as image:
        output = await ocr_page_regions(image, regions, 4, "sample.pdf", truncate, None)
    span = output[0]["_native_spans"][0]
    assert span["status"] == "unresolved"
    assert span["raw_response"] == "An incomplete repair"
    assert span["repair_failure"] == "truncated"


def _plan(lines, *, label="text", formulas=(), captions=False):
    regions = [{"label": label, "bbox_2d": [0, 0, 1000, 1000], "content": ""}, *deepcopy(formulas)]
    chars = []
    for i, line in enumerate(lines):
        for col, char in enumerate(line):
            index = len(chars)
            chars.append(
                NativeChar(
                    index, index + 1, char, (20 + col * 8, 800 - i * 30, 27 + col * 8, 812 - i * 30)
                )
            )
        index = len(chars)
        chars.append(NativeChar(index, index + 1, "\n", None))
    source = account_characters(chars, [], regions, (0, 0, 1000, 1000), 4)
    plan_native_repairs(
        regions, source, min_chars=20, min_printable_ratio=0.8, native_captions=captions
    )
    return regions, source


@pytest.mark.asyncio
async def test_repairs_one_line_and_retains_nineteen_native_lines():
    lines = [f"Reliable line {i:02d} keeps its source wording." for i in range(20)]
    lines[9] = "Publication year contains private glyph \ue001."
    regions, _ = _plan(lines)
    calls = []

    async def recognize(crop, prompt):
        calls.append((crop.size, prompt))
        return "Publication year contains private glyph 7."

    with Image.new("RGB", (1000, 1000), "white") as image:
        output = await ocr_page_regions(image, regions, 4, "sample.pdf", recognize, None)
    assert len(calls) == 1
    assert calls[0][0][1] < 20
    expected = list(lines)
    expected[9] = "Publication year contains private glyph 7."
    assert output[0]["content"] == "\n".join(expected)
    spans = output[0]["_native_spans"]
    repaired = [s for s in spans if s["status"] == "recognized_unverified"]
    assert len(repaired) == 1
    assert repaired[0]["candidate"] == lines[9]
    assert repaired[0]["reason"] == "private_use"
    typed = OcrRegionResult.from_dict(json.loads(json.dumps(output[0])))
    assert typed.to_dict() == output[0]


@pytest.mark.asyncio
async def test_failed_line_is_explicit_without_erasing_clean_siblings():
    regions, _ = _plan(
        ["Clean native sentence survives unchanged.", "Bad \ue001 mapped line needs recognition."]
    )
    warnings = []

    async def fail(crop, prompt):
        raise RuntimeError("recognizer failed")

    with Image.new("RGB", (1000, 1000)) as image:
        output = await ocr_page_regions(
            image, regions, 4, "sample.pdf", fail, None, warning_sink=warnings.append
        )
    assert output[0]["content"] == "Clean native sentence survives unchanged.\n[unresolved text]"
    assert warnings
    assert any(s["status"] == "unresolved" for s in output[0]["_native_spans"])


@pytest.mark.asyncio
async def test_inline_formula_has_one_owner_and_preserves_native_prose():
    text = "The value x2 is measured in the experiment."
    start = text.index("x2")
    formula = {
        "label": "inline_formula",
        "task_type": "formula",
        "bbox_2d": [20 + start * 8, 188, 20 + (start + 2) * 8, 200],
        "content": "",
    }
    regions, _ = _plan([text], formulas=[formula])
    calls = []

    async def recognize(crop, prompt):
        calls.append(prompt)
        return "x^{2}"

    with Image.new("RGB", (1000, 1000)) as image:
        output = await ocr_page_regions(image, regions, 4, "sample.pdf", recognize, None)
    assert len(calls) == len(output) == 1
    assert output[0]["content"] == "The value $x^{2}$ is measured in the experiment."
    formula_span = next(s for s in output[0]["_native_spans"] if s["type"] == "inline_formula")
    assert formula_span["formula_owner"] == "p4:r1"
    assert len(formula_span["source_ids"]) == 2


def test_caption_acceptance_requires_a_single_owned_caption_line():
    regions, _ = _plan(
        ["Figure 1. Source caption for this experiment."], label="figure_title", captions=True
    )
    assert regions[0]["_native_text_used"] is True
    regions, _ = _plan(
        [
            "Figure 1. Source caption for this experiment.",
            "Adjacent paragraph with unrelated words.",
        ],
        label="figure_title",
        captions=True,
    )
    assert not regions[0].get("_native_text_used")


def test_independent_accounting_and_cache_roundtrip():
    result = inspect_pdf(
        Path("tests/fixtures/native_text_sample.pdf").read_bytes(),
        [[]],
        fill_native_text=False,
        include_outline=False,
        include_ref_geometry=False,
        min_chars=20,
        min_printable_ratio=0.8,
    )
    source = result.pages[0].source
    assert source["counts"]["unassigned"] > 0
    assert source["visual_coverage"] == "not_checked"
    assert source["fonts"]
    restored = inspection_from_dict(
        json.loads(json.dumps(inspection_to_dict(result))),
        metadata={},
        outline=[],
        reference_lines=[],
    )
    assert restored.pages == result.pages


def test_overlapping_layout_is_accounted_as_ambiguous():
    regions = [{"label": "text", "bbox_2d": [0, 0, 1000, 1000]} for _ in range(2)]
    chars = [NativeChar(2, 4, "𝑆", (20, 20, 30, 30))]
    source = account_characters(chars, [], regions, (0, 0, 1000, 1000), 3)
    assert source["counts"]["ambiguous"] == 1
    assert source["characters"][0]["source_id"] == "p3:c2-4"
    assert source["characters"][0]["owners"] == ["p3:r0", "p3:r1"]


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, [100, 200, 300, 400]),
        (90, [600, 100, 800, 300]),
        (180, [700, 600, 900, 800]),
        (270, [200, 700, 400, 900]),
    ],
)
def test_cropbox_rotation_mapping(rotation, expected):
    assert normalized_box((20, 140, 40, 180), (10, 20, 110, 220), rotation) == pytest.approx(
        expected
    )


def test_partial_and_display_formula_proposals_survive():
    settings = GlobalSettings(_env_file=None)
    text = {"native_label": "text", "label": "text", "bbox_2d": [0, 0, 60, 100]}
    formula = {"native_label": "inline_formula", "label": "formula", "bbox_2d": [0, 0, 100, 100]}
    assert len(_deduplicate_formula_text_regions([[text, formula]], settings)[0]) == 2
    text["bbox_2d"] = [0, 0, 100, 100]
    formula["native_label"] = "display_formula"
    assert len(_deduplicate_formula_text_regions([[text, formula]], settings)[0]) == 2
    formula["native_label"] = "inline_formula"
    output = _deduplicate_formula_text_regions([[text, formula]], settings)[0]
    assert len(output) == 1
    assert output[0]["_formula_proposals"][0] == formula
