"""Regression coverage for model-specific OCR output normalization."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from bibr.ocr.normalization import normalize_ocr_output
from bibr.ocr.profiles import GLM_PROFILE, PADDLE_PROFILE, OcrProfileName

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "paddle_vl_outputs.json"
PADDLE_OUTPUTS = json.loads(FIXTURE_PATH.read_text())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (PADDLE_OUTPUTS["formula_simple"], "(a)_{n}=(a;q)_{n}=\\prod_{k=0}^{n-1}(1-aq^{k}),"),
        (
            PADDLE_OUTPUTS["formula_align"],
            "\\begin{align*}&\\sum_{k=0}^{n-1}(-q^2)_{2k}\\\\&\\quad=\\frac{(-q)_{n-1}}{(q^2)_{n-1}}.\\end{align*}",
        ),
        (r"\(\mathbf{\alpha}+\beta\)", r"\mathbf{\alpha}+\beta"),
        (
            "$$\\begin{aligned}a &= b\\\\\nc &= d\\end{aligned}$$",
            "\\begin{aligned}a &= b\\\\\nc &= d\\end{aligned}",
        ),
        ("```latex\n\\[x^2\\]\n```", "x^2"),
        ("  ```tex\n\\(\\boldsymbol{x}\\)\n```  ", r"\boldsymbol{x}"),
        (r"\frac{a}{b}", r"\frac{a}{b}"),
        (r"\[x + y", r"\[x + y"),
        (r"x + y\]", r"x + y\]"),
        (r"\[\[x\]", r"\[\[x\]"),
        (r"\[$$x\]", r"\[$$x\]"),
        (r"\[x$$\]", r"\[x$$\]"),
    ],
)
def test_paddle_formula_normalization_removes_one_balanced_outer_wrapper(raw, expected):
    normalized = normalize_ocr_output(PADDLE_PROFILE, "formula", raw)

    assert normalized.content == expected
    assert normalized.raw_content == raw


def test_glm_output_remains_unchanged():
    raw = "  \\[x^2\\]  "

    normalized = normalize_ocr_output(GLM_PROFILE, "formula", raw)

    assert normalized.content == raw
    assert normalized.raw_content is None


def test_paddle_table_output_is_decoded_and_retains_raw_content():
    raw = PADDLE_OUTPUTS["table_merged"]

    normalized = normalize_ocr_output(PADDLE_PROFILE, "table", raw)

    assert normalized.content == (
        '<table><tr><td colspan="2">Access this article online</td></tr>'
        "<tr><td>Quick Response Code:</td><td>Website:www.ijem.in</td></tr>"
        "<tr><td></td><td>DOI:<br>10.4103/2230-8210.152787</td></tr></table>"
    )
    assert normalized.raw_content == raw


def test_glm_table_output_remains_unchanged():
    raw = "<table><tr><td>DOI:\\n10.1000/example</td></tr></table>"

    normalized = normalize_ocr_output(GLM_PROFILE, "table", raw)

    assert normalized.content == raw
    assert normalized.raw_content is None


def test_non_paddle_profile_table_output_is_not_decoded():
    custom_profile = replace(
        PADDLE_PROFILE,
        name=cast(OcrProfileName, "custom"),
    )
    raw = "<table><tr><td>Already canonical</td></tr></table>"

    normalized = normalize_ocr_output(custom_profile, "table", raw)

    assert normalized.content == raw


def test_paddle_formula_rejects_fence_with_interior_closing_fence():
    raw = "```latex\nx\n```\ntrailer\n```"

    normalized = normalize_ocr_output(PADDLE_PROFILE, "formula", raw)

    assert normalized.content == raw
    assert normalized.raw_content == raw


def test_paddle_formula_preserves_crlf_inside_fence():
    raw = "```latex\r\n\\[x\r\n+y\\]\r\n```"

    normalized = normalize_ocr_output(PADDLE_PROFILE, "formula", raw)

    assert normalized.content == "x\r\n+y"
    assert normalized.raw_content == raw
