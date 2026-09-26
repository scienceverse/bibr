"""``_postprocess_ocr_regions``: OCR cleanup reaches OCR output only.

Regions filled from the PDF text layer (``_native_text_used``) never carry
OCR decode artifacts, so the repairs for them only damage the text. Formula
regions hold LaTeX, where the prose repairs change bytes, and the ``$$``
pre-wrap must pair an outer delimiter with its own closing one.
"""

from __future__ import annotations

import json
from pathlib import Path

from bibr.ocr.normalization import normalize_ocr_output
from bibr.ocr.profiles import PADDLE_PROFILE
from bibr.pipeline.stages.ocr import _postprocess_ocr_regions

PADDLE_OUTPUTS = json.loads(
    (Path(__file__).parents[1] / "ocr" / "fixtures" / "paddle_vl_outputs.json").read_text()
)


def _region(label: str, content: str, top: int, *, native: bool = False) -> dict:
    region = {
        "native_label": label,
        "label": "formula" if label in {"display_formula", "inline_formula"} else label,
        "content": content,
        "bbox_2d": [100, top, 900, top + 40],
    }
    if native:
        region["_native_text_used"] = True
    return region


def _contents(regions: list[dict]) -> list[str]:
    return [region["content"] for region in _postprocess_ocr_regions([regions])[0]]


def test_text_layer_regions_are_only_trimmed():
    toc = "\r\n".join(f"{n} Section heading number {n} {'. ' * 55}{n}" for n in range(1, 19))
    regions = [
        _region("doc_title", "U.S. Adults and Social Media Use\n", 0, native=True),
        _region("text", toc, 100, native=True),
        _region("footnote", " 10.1038/s41586-020-2649-2", 900, native=True),
        _region("text", "e.g. the effect was small.", 950, native=True),
    ]

    assert _contents(regions) == [
        "U.S. Adults and Social Media Use",
        toc,
        "10.1038/s41586-020-2649-2",
        "e.g. the effect was small.",
    ]


def test_ocr_regions_still_get_the_ocr_cleanup():
    regions = [
        _region("text", r"\tThe results were robust.", 0),
        _region("text", "1.text of the first item", 100),
    ]

    assert _contents(regions) == ["The results were robust.", "1. text of the first item"]


def test_a_printed_asterisk_in_the_text_layer_is_not_a_markdown_bullet():
    regions = [
        _region("text", "* p < .05; ** p < .01.", 0, native=True),
        _region("text", "* An OCR bullet item.", 100),
    ]

    assert _contents(regions) == ["* p < .05; ** p < .01.", "- An OCR bullet item."]


def test_bare_latex_formula_keeps_its_leading_command():
    """The Paddle profile strips the ``\\[ \\]`` wrapper before post-processing,
    so a formula can start with ``\\theta``, whose ``\\t`` was stripped."""
    raw = r"\[\theta_{t+1}=\theta_{t}-\eta\nabla L\]"
    content = normalize_ocr_output(PADDLE_PROFILE, "formula", raw).content

    assert _contents([_region("display_formula", content, 0)]) == [
        "$$\n\\theta_{t+1}=\\theta_{t}-\\eta\\nabla L\n$$"
    ]
    assert _contents([_region("inline_formula", r"\tau", 0)]) == ["$$\n\\tau\n$$"]


def test_paddle_fixture_formula_is_not_rewritten_as_a_list_item():
    content = normalize_ocr_output(PADDLE_PROFILE, "formula", PADDLE_OUTPUTS["formula_simple"])

    assert _contents([_region("display_formula", content.content, 0)]) == [
        "$$\n(a)_{n}=(a;q)_{n}=\\prod_{k=0}^{n-1}(1-aq^{k}),\n$$"
    ]


def test_formula_prewrap_strips_only_a_matched_outer_pair():
    assert _contents([_region("display_formula", r"\(a\) + \(b\)", 0)]) == [
        "$$\n\\(a\\) + \\(b\\)\n$$"
    ]


def test_formula_prewrap_unwraps_single_dollar_math():
    assert _contents([_region("display_formula", "$E=mc^2$", 0)]) == ["$$\nE=mc^2\n$$"]
    assert _contents([_region("display_formula", r"\[x^2\]", 0)]) == ["$$\nx^2\n$$"]


def test_formula_prewrap_unwraps_a_formula_with_an_escaped_dollar():
    """GLM output keeps its wrapper, and a literal ``\\$`` inside used to
    unbalance the dollar count, so the wrapper stayed nested in ``$$``."""
    assert _contents([_region("display_formula", r"$$\text{cost} = \$5$$", 0)]) == [
        "$$\n\\text{cost} = \\$5\n$$"
    ]
    assert _contents([_region("display_formula", r"\[p = \$5\]", 0)]) == ["$$\np = \\$5\n$$"]


def test_dehyphenated_merge_with_ocr_text_is_not_text_layer_text():
    native_half = _region("text", "The results were ro-", 0, native=True)
    ocr_half = _region("text", "bust across samples.", 100)

    merged = _postprocess_ocr_regions([[native_half, ocr_half]])[0]

    assert [region["content"] for region in merged] == ["The results were robust across samples."]
    assert not merged[0].get("_native_text_used")


def test_dehyphenated_merge_of_text_layer_regions_stays_text_layer_text():
    merged = _postprocess_ocr_regions(
        [
            [
                _region("text", "The results were ro-", 0, native=True),
                _region("text", "bust across samples.", 100, native=True),
            ]
        ]
    )[0]

    assert merged[0]["_native_text_used"] is True
