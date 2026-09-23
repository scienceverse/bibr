"""Corrupted OCR output must be visible in processing_warnings.

Control-char-riddled region text (the vllm-mlx --mllm NUL failure mode) is
stripped for parsing, but the paper must carry an OCR_CONTROL_CHARS warning so
a corrupted extraction is visibly suspect instead of silently wrong.
"""

from __future__ import annotations

from bibr.processing_warnings import WarningCode


def _region(index, label, content):
    return {
        "index": index,
        "label": label,
        "content": content,
        "bbox_2d": [0, 0, 100, 100],
    }


def _parse(json_result):
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(json_result)
    return parser.parse()


def test_corrupt_regions_surface_processing_warning():
    contents = _parse(
        [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "paragraph_title", "\x00Meth\x00ods\x00"),
                _region(2, "text", "\x00\x00Body sent\x00ence."),
            ]
        ]
    )
    warnings = [w for w in contents.processing_warnings if w.code == WarningCode.OCR_CONTROL_CHARS]
    assert len(warnings) == 1
    assert warnings[0].message.startswith("2 region(s)")


def test_clean_parse_has_no_corruption_warning():
    contents = _parse(
        [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "paragraph_title", "Methods"),
                _region(2, "text", "Body sentence."),
            ]
        ]
    )
    assert not [w for w in contents.processing_warnings if w.code == WarningCode.OCR_CONTROL_CHARS]


def test_stx_soft_hyphen_marks_not_flagged():
    # STX is deliberate GLM-OCR soft-hyphen output, not corruption.
    contents = _parse(
        [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "text", "An off\x02line study of cross\x02national data."),
            ]
        ]
    )
    assert not [w for w in contents.processing_warnings if w.code == WarningCode.OCR_CONTROL_CHARS]


def test_single_stray_control_char_not_flagged():
    # Broken ToUnicode CMaps can leak a single C0 glyph in otherwise-healthy
    # native text; one stray char is not corruption.
    contents = _parse(
        [
            [
                _region(0, "doc_title", "Paper Title"),
                _region(1, "text", "A formula glyph\x0f leaked here."),
            ]
        ]
    )
    assert not [w for w in contents.processing_warnings if w.code == WarningCode.OCR_CONTROL_CHARS]


def test_private_use_body_fallback_has_full_summary_and_one_ordinary_warning():
    native = "Ernst Lau’s survey article (). " + "native detail " * 30
    canonical = "Ernst Lau’s survey article (1927). " + "canonical detail " * 30
    raw = "Ernst Lau’s survey article (1927). " + "raw detail " * 30
    contents = _parse(
        [
            [
                _region(0, "doc_title", "Paper Title"),
                {
                    **_region(1, "text", canonical),
                    "native_label": "text",
                    "_native_text_candidate": native,
                    "_native_text_rejection_reason": "private_use",
                    "_raw_ocr_content": raw,
                },
                {
                    **_region(2, "reference_content", "Lau, Ernst (1927)."),
                    "native_label": "reference_content",
                    "_native_text_candidate": "Lau, Ernst ().",
                    "_native_text_rejection_reason": "private_use",
                    "_raw_ocr_content": "Lau, Ernst (1927).",
                },
            ]
        ]
    )

    summary = contents.region_summaries[1]
    assert summary.content == canonical[:200]
    assert summary.canonical_ocr_content == canonical
    assert summary.raw_ocr_content == raw
    assert summary.native_text_candidate == native
    assert summary.native_text_rejection_reason == "private_use"
    warnings = [
        w
        for w in contents.processing_warnings
        if w.code == WarningCode.OCR_NATIVE_TEXT_PUA_FALLBACK
    ]
    assert len(warnings) == 1
    assert warnings[0].message.startswith("2 region(s)")
    assert not any(w.code == WarningCode.OCR_CONTROL_CHARS for w in contents.processing_warnings)
