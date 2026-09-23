"""Tests for data model utilities in bibr.models."""

from bibr.models import (
    BibAuthor,
    ErrorCode,
    ProcessingStatus,
    canonicalize_orcid,
    format_bib_authors,
    migrate_bib_type,
)


class TestMigrateBibType:
    def test_none_returns_other(self):
        assert migrate_bib_type(None) == "other"

    def test_empty_returns_other(self):
        assert migrate_bib_type("") == "other"

    def test_article_maps_to_journal_article(self):
        assert migrate_bib_type("article") == "journal_article"

    def test_case_insensitive(self):
        assert migrate_bib_type("ARTICLE") == "journal_article"

    def test_journal_dash_article(self):
        assert migrate_bib_type("journal-article") == "journal_article"

    def test_inproceedings(self):
        assert migrate_bib_type("inproceedings") == "conference_paper"

    def test_techreport(self):
        assert migrate_bib_type("techreport") == "report"

    def test_posted_content_is_preprint(self):
        assert migrate_bib_type("posted-content") == "preprint"

    def test_unknown_returns_other(self):
        assert migrate_bib_type("haruspicy") == "other"

    def test_thesis_variants(self):
        for variant in ("thesis", "phdthesis", "mastersthesis", "masterthesis", "dissertation"):
            assert migrate_bib_type(variant) == "thesis", f"Failed for {variant}"

    def test_whitespace_stripped(self):
        assert migrate_bib_type("  article  ") == "journal_article"

    def test_book_chapter_variants(self):
        for variant in ("inbook", "incollection", "book_chapter", "book-chapter", "book-section"):
            assert migrate_bib_type(variant) == "book_chapter", f"Failed for {variant}"


class TestFormatBibAuthors:
    def test_single_author(self):
        authors = [BibAuthor(given="John", family="Doe")]
        assert format_bib_authors(authors) == "Doe, John"

    def test_multiple_authors(self):
        authors = [
            BibAuthor(given="John", family="Doe"),
            BibAuthor(given="Jane", family="Smith"),
        ]
        assert format_bib_authors(authors) == "Doe, John; Smith, Jane"

    def test_empty_list(self):
        assert format_bib_authors([]) == ""

    def test_family_only(self):
        authors = [BibAuthor(given="", family="Doe")]
        assert format_bib_authors(authors) == "Doe"

    def test_empty_family_skipped(self):
        authors = [
            BibAuthor(given="John", family=""),
            BibAuthor(given="Jane", family="Smith"),
        ]
        assert format_bib_authors(authors) == "Smith, Jane"


class TestCanonicalizeOrcid:
    def test_http_prefix_converted(self):
        assert (
            canonicalize_orcid("http://orcid.org/0000-0002-1825-0097")
            == "https://orcid.org/0000-0002-1825-0097"
        )

    def test_https_prefix_preserved(self):
        assert (
            canonicalize_orcid("https://orcid.org/0000-0002-1825-0097")
            == "https://orcid.org/0000-0002-1825-0097"
        )

    def test_orcid_with_x_checksum(self):
        assert canonicalize_orcid("0000-0002-1825-009X") == "https://orcid.org/0000-0002-1825-009X"

    def test_invalid_format_dropped(self):
        assert canonicalize_orcid("not-an-orcid") is None

    def test_byline_superscript_dropped(self):
        # Author-index superscripts leak into the LLM's orcid slot; a bare
        # digit is never an ORCID and must not reach the export.
        for junk in ("1", "2", "3", "4", "a", "*", "†"):
            assert canonicalize_orcid(junk) is None

    def test_partial_orcid_dropped(self):
        assert canonicalize_orcid("0000-0002-1825") is None

    def test_bare_digits_without_dashes_are_canonicalized(self):
        # Some JATS deposits carry the 16-digit form with no separators.
        assert canonicalize_orcid("0000000218250097") == "https://orcid.org/0000-0002-1825-0097"

    def test_lowercase_x_checksum_is_uppercased(self):
        assert canonicalize_orcid("0000-0001-5109-381x") == "https://orcid.org/0000-0001-5109-381X"

    def test_url_without_scheme_is_canonicalized(self):
        assert canonicalize_orcid("orcid.org/0000-0002-1825-0097") == (
            "https://orcid.org/0000-0002-1825-0097"
        )

    def test_trailing_slash_is_tolerated(self):
        assert canonicalize_orcid("https://orcid.org/0000-0002-1825-0097/") == (
            "https://orcid.org/0000-0002-1825-0097"
        )

    def test_none_returns_none(self):
        assert canonicalize_orcid(None) is None

    def test_empty_string_returns_none(self):
        assert canonicalize_orcid("") is None


class TestErrorCode:
    def test_error_codes_are_strings(self):
        assert ErrorCode.OCR_FAILED == "ocr_failed"
        assert ErrorCode.UNSUPPORTED_FORMAT == "unsupported_format"

    def test_llm_invalid_output_error_code_is_stable(self):
        assert ErrorCode.LLM_INVALID_OUTPUT.value == "llm_invalid_output"

    def test_all_codes_unique(self):
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))


class TestProcessingStatus:
    def test_default_is_unparsed(self):
        status = ProcessingStatus()
        assert status.parsed is False
        assert status.error_code is None
        assert status.error_message is None
        assert status.failed_stage is None
        assert status.stage_times == {}

    def test_backward_compat_parsed_true(self):
        status = ProcessingStatus(parsed=True)
        assert status.parsed is True
        assert status.error_code is None

    def test_full_error_status(self):
        status = ProcessingStatus(
            parsed=False,
            error_code=ErrorCode.OCR_FAILED,
            error_message="OCR server unreachable",
            failed_stage="ocr",
        )
        assert status.error_code == "ocr_failed"
        assert status.failed_stage == "ocr"

    def test_stage_times_independent(self):
        s1 = ProcessingStatus()
        s2 = ProcessingStatus()
        s1.stage_times["layout"] = 1.5
        assert "layout" not in s2.stage_times
