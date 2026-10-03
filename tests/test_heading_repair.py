"""Unit tests for repair_heading_artifacts (bibr/structure/text_repair.py)."""

from bibr.structure.text_repair import repair_heading_artifacts, strip_heading_watermark_text


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


class TestWatermarkAndGutterText:
    """Heading boxes that caught a watermark glyph, a line number or an overprint (#123)."""

    def test_trailing_watermark_letter_is_dropped(self):
        # A diagonal "RETRACTED" stamp put one letter inside each heading box.
        assert strip_heading_watermark_text("IV PROPOSED METHOD \r\nR") == "IV PROPOSED METHOD"
        assert strip_heading_watermark_text("V MODULE DESCRIPTION \r\nT") == "V MODULE DESCRIPTION"

    def test_leading_lowercase_watermark_letter_is_dropped(self):
        # An accepted-manuscript stamp in the margin, one letter per heading.
        assert (
            strip_heading_watermark_text("m\r\nPotential confounding variables")
            == "Potential confounding variables"
        )
        assert strip_heading_watermark_text("c\r\nAuthor contributions") == "Author contributions"

    def test_letter_dropped_from_a_one_word_section_name(self):
        assert strip_heading_watermark_text("Results\r\nR") == "Results"

    def test_letters_that_belong_to_the_heading_are_kept(self):
        for text in (
            "Appendix\r\nA",
            "Study\r\nB",
            "A\r\nMethods",
            "I\r\nINTRODUCTION",
            # A letter inside a wrapped heading is a word of it, or maths.
            "TURN THE\t\r\n SCIENTIFIC CYCLE INTO\t\r\n A\t\r\n TEST BED",
            "2.2 Weighted L\r\np\r\nspaces",
            "Spaces of type L\r\np",
        ):
            assert strip_heading_watermark_text(text) == text

    def test_trailing_letter_of_a_mixed_case_heading_is_a_label(self):
        for text in (
            "Results for Hypothesis\r\nA",
            "Treatment arm\r\nB",
            "Proof of Lemma\r\nB",
            "Outcomes in the control\r\nC",
        ):
            assert strip_heading_watermark_text(text) == text

    def test_gutter_line_number_is_dropped_before_a_section_name(self):
        assert strip_heading_watermark_text("668 References") == "References"
        # A plausible section number, or a heading that is not a section name, stays.
        assert strip_heading_watermark_text("12 Results") == "12 Results"
        assert strip_heading_watermark_text("2019 Novel Coronavirus") == "2019 Novel Coronavirus"

    def test_letter_spaced_heading_is_closed_up(self):
        assert strip_heading_watermark_text("A B S T R A C T") == "ABSTRACT"
        assert strip_heading_watermark_text("K e y w o r d s") == "Keywords"
        # Word gaps printed as wider spaces are kept ...
        assert (
            strip_heading_watermark_text("A B O U T  T H E  A U T H O R S") == "ABOUT THE AUTHORS"
        )
        # ... and restored from the label when the gaps were lost.
        assert strip_heading_watermark_text("A R T I C L E I N F O") == "ARTICLE INFO"
        # Letters that spell no known name stay as printed.
        for text in ("Q A", "C O M P U T E R V I S I O N S T U D Y"):
            assert strip_heading_watermark_text(text) == text

    def test_overprinted_heading_is_kept_once(self):
        assert (
            strip_heading_watermark_text("2.1. Loading Tests 2.1. Loading Tests 2.1. Loading Tests")
            == "2.1. Loading Tests"
        )
        assert strip_heading_watermark_text("Methods Methods") == "Methods"
        assert strip_heading_watermark_text("Bora Bora") == "Bora Bora"

    def test_ordinary_headings_are_returned_unchanged(self):
        for text in ("3 Methods", "A Priori Strategy", "Title of a paper\r\nwith two lines"):
            assert strip_heading_watermark_text(text) == text


def test_parser_opens_the_section_without_the_watermark_letter():
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(
        [
            [
                {
                    "label": "paragraph_title",
                    "content": "o\r\nStatistical Analyses",
                    "bbox_2d": [100, 80, 900, 110],
                },
                {
                    "label": "text",
                    "content": "We fitted mixed models.",
                    "bbox_2d": [100, 150, 900, 200],
                },
                {
                    "label": "paragraph_title",
                    "content": "668 References",
                    "bbox_2d": [30, 300, 900, 330],
                },
            ]
        ]
    )
    parser.parse()
    headers = [s.header for s in parser.sections if s.section_id != 0]
    assert headers == ["Statistical Analyses", "References"]
