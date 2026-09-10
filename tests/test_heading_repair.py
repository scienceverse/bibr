"""Unit tests for repair_heading_artifacts (bibr/structure/text_repair.py)."""

from bibr.structure.text_repair import repair_heading_artifacts


class TestExactAliasHeadersUntouched:
    """A header that already matches a canonical alias is not an artifact.

    Regression: the compact-alias map collides "keywords" / "key words"
    (both compact to "keywords"), which rewrote every OCR-correct
    "Keywords" header to "Key Words".
    """

    def test_keywords_not_rewritten(self):
        assert repair_heading_artifacts("Keywords") == "Keywords"

    def test_lowercase_keywords_not_rewritten(self):
        assert repair_heading_artifacts("keywords") == "keywords"

    def test_uppercase_keywords_not_rewritten(self):
        assert repair_heading_artifacts("KEYWORDS") == "KEYWORDS"

    def test_spaced_key_words_still_untouched(self):
        assert repair_heading_artifacts("Key Words") == "Key Words"


class TestCompactedArtifactsStillRepaired:
    def test_camel_fused_keywords_repaired(self):
        # NUL-fusion artifact of a printed "Key Words" header.
        assert repair_heading_artifacts("KeyWords") == "Key Words"

    def test_camel_fused_author_contributions_repaired(self):
        assert repair_heading_artifacts("AuthorContributions") == "Author Contributions"

    def test_study_marker_spacing_repaired(self):
        assert repair_heading_artifacts("Study1") == "Study 1"
