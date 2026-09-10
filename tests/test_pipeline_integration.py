# Tests for _process_file / _run_ocr internals that existed prior to the T9
# rewrite have been removed.  The behaviors they covered are now tested at the
# stage level:
#
#   • input-validation rejection  → tests/pipeline/test_validate_stage.py
#   • max_pages clamping          → tests/pipeline/test_layout_stage.py (TestMaxPagesClamping)
#   • native DOCX skips OCR       → tests/pipeline/test_docx_stage.py

