"""Tests for space-serialized DOI line-wrap bridging (qual-grade30 M3).

glmocr sometimes serializes a printed DOI line-wrap as a literal SPACE rather
than a newline: the reference DOI ``doi:10.1037/0033-295X.113.4.842`` (Van Der
Maas et al. 2006, Psychological Science) reaches the text layer as
``doi:10.1037/0033-295X .113.4.842``. The existing wrap bridges
(``_bridge_doi_midword_wraps``, ``_bridge_doi_url_crlf_wraps``) only match
``\\r?\\n`` wraps, so the space survives and the LLM truncates the DOI at the
space -> ``10.1037/0033-295X`` (suffix ``.113.4.842`` dropped).

The fix re-joins ``<doi-fragment><space><.digit…>`` in ``fix_ocr_artifacts``.
The asymmetry that makes this safe: ordinary prose after a DOI reads
``…001. Smith`` (period BEFORE the space), whereas a wrapped DOI suffix reads
``…295X .113`` (space BEFORE the period). Only the latter is joined.
"""

from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.utils.text import normalize_doi


class TestBridgesSpaceWrappedDoi:
    def test_bridges_van_der_maas_real_case(self):
        # The verbatim text_id 159 fragment from 0956797619841265.
        out = fix_ocr_artifacts("doi:10.1037/0033-295X .113.4.842")
        assert "10.1037/0033-295X.113.4.842" in out
        assert "0033-295X .113" not in out

    def test_recovered_doi_normalizes_clean(self):
        out = fix_ocr_artifacts("doi:10.1037/0033-295X .113.4.842")
        # Extract the DOI token and confirm it normalizes to the full form.
        assert normalize_doi("10.1037/0033-295X .113.4.842".replace(" ", "")) == normalize_doi(
            out.split("doi:")[1].strip()
        )

    def test_bridges_when_fragment_ends_in_digit(self):
        out = fix_ocr_artifacts("https://doi.org/10.1016/j.cognition.2018.06 .015")
        assert "10.1016/j.cognition.2018.06.015" in out

    def test_bridges_multiple_spaces_fixpoint(self):
        # A DOI wrapped across several lines, each serialized as a space.
        out = fix_ocr_artifacts("10.1037/0033-295X .113 .4 .842")
        assert "10.1037/0033-295X.113.4.842" in out

    def test_idempotent(self):
        once = fix_ocr_artifacts("doi:10.1037/0033-295X .113.4.842")
        twice = fix_ocr_artifacts(once)
        assert once == twice


class TestDoesNotCorruptText:
    def test_does_not_join_sentence_period_then_capital(self):
        # Prose: period BEFORE the space, continuation is a Capitalized name.
        out = fix_ocr_artifacts("See 10.1037/abc123. Smith and Jones (2019) replicated this.")
        assert "abc123. Smith" in out
        assert "abc123.Smith" not in out

    def test_does_not_join_sentence_period_then_lowercase_word(self):
        out = fix_ocr_artifacts("10.1037/xge0000729. The result held.")
        assert "0000729. The" in out

    def test_does_not_join_period_then_space_then_bare_number(self):
        # "…2020. 4 studies" — period before space; the continuation digit is
        # NOT preceded by a dot, so the lookahead must not fire.
        out = fix_ocr_artifacts("10.5018/foo2020. 4 studies were run.")
        assert "foo2020. 4" in out
        assert "foo2020.4" not in out

    def test_does_not_join_doi_then_page_range(self):
        # Complete DOI, space, then a page range that does not start with a dot.
        out = fix_ocr_artifacts("10.1037/0033-295X 113-185.")
        assert "0033-295X 113-185" in out
        assert "0033-295X113" not in out

    def test_leaves_text_without_doi_untouched(self):
        s = "The value was 10 .5 percent higher in the treatment group."
        assert fix_ocr_artifacts(s) == s
