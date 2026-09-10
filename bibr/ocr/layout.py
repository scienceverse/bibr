"""
Document layout label classification constants.

These constants classify the 23 region types detected by PP-DocLayoutV3 into
semantic categories used by bibr's OCR and header/footer detection logic.

The ``LayoutDetector`` class and overlap filtering have been replaced by
the glmocr SDK, which handles layout detection internally.
"""

from bibr.ocr.profiles import GLM_PROFILE

# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------

# Labels for regions that should be stripped from body text
STRUCTURAL_LABELS: frozenset[str] = frozenset(
    {
        "header",
        "footer",
        "number",
    }
)

# Labels that provide hints for section classification
SECTION_HINT_LABELS: frozenset[str] = frozenset(
    {
        "abstract",
        "reference",
        "reference_content",
        "footnote",
        "vision_footnote",
    }
)

# Labels that should use specialised GLM-OCR prompts
FORMULA_LABELS: frozenset[str] = frozenset({"formula", "formula_number"})
TABLE_LABELS: frozenset[str] = frozenset({"table"})

# Deprecated compatibility alias for the vendored glmocr SDK. New bibr runtime
# code must use the resolved OcrProfile instead. Keep this as the exact GLM
# profile mapping so legacy consumers cannot drift from the compatibility profile.
# See: https://huggingface.co/zai-org/GLM-OCR
TASK_PROMPTS: dict[str, str] = GLM_PROFILE.prompts

# Labels that map to markdown heading levels
HEADING_LABELS: dict[str, int] = {
    "doc_title": 1,
    "paragraph_title": 2,
}
