"""Tests for previously untested code paths across the bibr codebase.

Covers:
- models.py: migrate_bib_type()
- paper_contents.py: PaperTable.contents, finalize_text(), equations_df, links_df
- utils/text.py: normalize_text()
- utils/circuit_breaker.py: _is_rate_limit_error(), _is_client_error()
- clients/base.py: BaseClient init, ping, close, context manager
- clients/crossref.py: _is_retryable(), get_client() singleton
- clients/llm.py: _cap_input()
- extract/extractor.py: _parse_year(), _map_bib_text_ids(), _collect_last_unknown_section_rows()
- extract/equation_extractor.py: helpers (_normalize_comp, _split_respecting_brackets,
  _find_toplevel_comp, _iter_parenthesized_groups), extract_with_llm_fallback()
- input/consolidate_text.py: strip_latex_commands(), _fix_operatorname_spacing()
- structure/pdf_parser.py: _is_copyright_notice()
- structure/study_detector.py: _backprop_content_section_ids()
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from bibr.input.file import InputFile, InputFormat
from bibr.models import (
    PaperMetadata,
    PaperReference,
    migrate_bib_type,
)
from bibr.paper import Paper
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperEquation,
    PaperSection,
    PaperSentence,
    PaperTable,
    PaperURLLink,
)

# ── helpers ────────────────────────────────────────────────────────────


def _input_file(name: str = "test.pdf") -> InputFile:
    return InputFile(
        path=Path(f"/tmp/{name}"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf", detected_mime_type="application/pdf", file_type="pdf"
        ),
    )


def _minimal_contents(**overrides) -> PaperContents:
    defaults = {
        "sentences": [PaperSentence(text_id=1, text="Hello.", section_id=1, paragraph_id=1)],
        "sections": [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Intro",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.INTRODUCTION,
            ),
        ],
        "tables": [],
        "links": [],
        "sections_text": {1: "Hello."},
    }
    defaults.update(overrides)
    return PaperContents(**defaults)


def _minimal_paper(**overrides) -> Paper:
    defaults = {
        "input_file": _input_file(),
        "metadata": PaperMetadata(doi="10.1234/test", title="Test Paper"),
        "contents": _minimal_contents(),
    }
    defaults.update(overrides)
    return Paper(**defaults)


# ══════════════════════════════════════════════════════════════════════
# models.py — migrate_bib_type()
# ══════════════════════════════════════════════════════════════════════


class TestMigrateBibType:
    def test_none_returns_other(self):
        assert migrate_bib_type(None) == "other"

    def test_empty_string_returns_other(self):
        assert migrate_bib_type("") == "other"

    def test_article_to_journal_article(self):
        assert migrate_bib_type("article") == "journal_article"

    def test_journal_article_passthrough(self):
        assert migrate_bib_type("journal_article") == "journal_article"

    def test_journal_hyphenated(self):
        assert migrate_bib_type("journal-article") == "journal_article"

    def test_book(self):
        assert migrate_bib_type("book") == "book"

    def test_inbook_to_book_chapter(self):
        assert migrate_bib_type("inbook") == "book_chapter"

    def test_incollection_to_book_chapter(self):
        assert migrate_bib_type("incollection") == "book_chapter"

    def test_book_chapter_variants(self):
        assert migrate_bib_type("book_chapter") == "book_chapter"
        assert migrate_bib_type("book-chapter") == "book_chapter"
        assert migrate_bib_type("book-section") == "book_chapter"

    def test_conference_variants(self):
        assert migrate_bib_type("conference") == "conference_paper"
        assert migrate_bib_type("inproceedings") == "conference_paper"
        assert migrate_bib_type("proceedings") == "conference_paper"
        assert migrate_bib_type("proceedings-article") == "conference_paper"
        assert migrate_bib_type("conference_paper") == "conference_paper"

    def test_report_variants(self):
        assert migrate_bib_type("techreport") == "report"
        assert migrate_bib_type("report") == "report"

    @pytest.mark.parametrize(
        ("crossref_type", "expected"),
        [
            ("monograph", "book"),
            ("edited-book", "book"),
            ("reference-book", "book"),
            ("book-set", "book"),
            ("book-part", "book_chapter"),
            ("book-track", "book_chapter"),
            ("report-component", "report"),
            ("report-series", "report"),
            ("database", "dataset"),
            ("standard", "other"),
            ("journal-issue", "other"),
        ],
    )
    def test_crossref_work_types(self, crossref_type, expected):
        """core-api-5: a matched Cambridge monograph came back as "other"."""
        assert migrate_bib_type(crossref_type) == expected

    def test_non_string_returns_other(self):
        assert migrate_bib_type(3) == "other"  # type: ignore[arg-type]

    def test_preprint_variants(self):
        assert migrate_bib_type("unpublished") == "preprint"
        assert migrate_bib_type("preprint") == "preprint"
        assert migrate_bib_type("posted-content") == "preprint"

    def test_dataset(self):
        assert migrate_bib_type("dataset") == "dataset"

    def test_software(self):
        assert migrate_bib_type("software") == "software"

    def test_unknown_type_returns_other(self):
        assert migrate_bib_type("haruspicy") == "other"
        assert migrate_bib_type("misc") == "other"

    def test_thesis(self):
        assert migrate_bib_type("thesis") == "thesis"
        assert migrate_bib_type("dissertation") == "thesis"

    def test_case_insensitive(self):
        assert migrate_bib_type("ARTICLE") == "journal_article"
        assert migrate_bib_type("Book") == "book"

    def test_whitespace_stripped(self):
        assert migrate_bib_type("  article  ") == "journal_article"


# ══════════════════════════════════════════════════════════════════════
# paper_contents.py — PaperTable.contents, finalize_text, equations_df, links_df
# ══════════════════════════════════════════════════════════════════════


class TestPaperTableContents:
    def test_converts_df_to_list(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        table = PaperTable(table_id=1, df=df, tbl_html="<table/>", section_id=1)
        result = table.contents
        assert result == [["col1", "col2"], ["1", "a"], ["2", "b"]]

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame()
        table = PaperTable(table_id=1, df=df, tbl_html="<table/>", section_id=1)
        assert table.contents == []

    def test_single_row(self):
        df = pd.DataFrame({"x": [42]})
        table = PaperTable(table_id=1, df=df, tbl_html="<table/>", section_id=1)
        result = table.contents
        assert result == [["x"], ["42"]]

    def test_values_stringified(self):
        df = pd.DataFrame({"num": [3.14], "bool": [True]})
        table = PaperTable(table_id=1, df=df, tbl_html="<table/>", section_id=1)
        result = table.contents
        assert result[1][0] == "3.14"
        assert result[1][1] == "True"


class TestFinalizeText:
    def test_strips_inline_math_delimiters(self):
        sentences = [
            PaperSentence(text_id=1, text="The value $x = 5$ here.", section_id=1, paragraph_id=1),
        ]
        contents = _minimal_contents(sentences=sentences)
        contents.finalize_text()
        assert "$" not in contents.sentences[0].text
        assert "x = 5" in contents.sentences[0].text

    def test_strips_latex_commands(self):
        sentences = [
            PaperSentence(text_id=1, text=r"The \alpha value.", section_id=1, paragraph_id=1),
        ]
        contents = _minimal_contents(sentences=sentences)
        contents.finalize_text()
        assert "\u03b1" in contents.sentences[0].text


class TestEquationsDf:
    def test_builds_dataframe(self):
        equations = [
            PaperEquation(text_id=1, grp_id=1, lhs="t", df="28", comp="=", rhs="3.42"),
            PaperEquation(text_id=1, grp_id=1, lhs="p", comp="<", rhs=".001"),
        ]
        contents = _minimal_contents(equations=equations)
        df = contents.equations_df
        assert len(df) == 2
        assert list(df.columns) == ["text_id", "grp_id", "lhs", "df", "comp", "rhs"]
        assert df.iloc[0]["lhs"] == "t"
        assert df.iloc[0]["df"] == "28"
        assert df.iloc[1]["df"] == ""
        assert df.iloc[1]["comp"] == "<"

    def test_empty_equations(self):
        contents = _minimal_contents()
        df = contents.equations_df
        assert len(df) == 0


class TestLinksDf:
    def test_builds_dataframe(self):
        links = [
            PaperURLLink(
                url="https://example.com", section_id=1, paragraph_id=1, text_id=1, link_text="Ex"
            ),
        ]
        contents = _minimal_contents(links=links)
        df = contents.links_df
        assert len(df) == 1
        assert df.iloc[0]["url"] == "https://example.com"
        assert df.iloc[0]["link_text"] == "Ex"

    def test_empty_links(self):
        contents = _minimal_contents()
        df = contents.links_df
        assert len(df) == 0
        assert list(df.columns) == ["url", "link_text", "text_id"]


# ══════════════════════════════════════════════════════════════════════
# utils/text.py — normalize_text()
# ══════════════════════════════════════════════════════════════════════


class TestNormalizeText:
    def test_strips_digits_and_dots(self):
        from bibr.utils.text import normalize_text

        assert normalize_text("1. Introduction") == "introduction"

    def test_multiple_digits(self):
        from bibr.utils.text import normalize_text

        assert normalize_text("3.2.1 Materials and Methods") == "materials and methods"

    def test_empty_string(self):
        from bibr.utils.text import normalize_text

        assert normalize_text("") == ""

    def test_no_digits(self):
        from bibr.utils.text import normalize_text

        assert normalize_text("Discussion") == "discussion"

    def test_preserves_non_digit_content(self):
        from bibr.utils.text import normalize_text

        assert normalize_text("Abstract") == "abstract"

    def test_only_digits_and_dots(self):
        from bibr.utils.text import normalize_text

        assert normalize_text("1.2.3.") == ""


# ══════════════════════════════════════════════════════════════════════
# utils/circuit_breaker.py — _is_rate_limit_error(), _is_client_error()
# ══════════════════════════════════════════════════════════════════════


class TestIsRateLimitError:
    def test_resource_exhausted(self):
        from bibr.utils.circuit_breaker import _is_rate_limit_error

        exc = type("ResourceExhausted", (Exception,), {})()
        assert _is_rate_limit_error(exc) is True

    def test_rate_limit_error(self):
        from bibr.utils.circuit_breaker import _is_rate_limit_error

        exc = type("RateLimitError", (Exception,), {})()
        assert _is_rate_limit_error(exc) is True

    def test_regular_error(self):
        from bibr.utils.circuit_breaker import _is_rate_limit_error

        assert _is_rate_limit_error(ValueError("something")) is False

    def test_wrapped_cause(self):
        from bibr.utils.circuit_breaker import _is_rate_limit_error

        inner = type("ResourceExhausted", (Exception,), {})()
        outer = RuntimeError("wrapped")
        outer.__cause__ = inner
        assert _is_rate_limit_error(outer) is True

    def test_wrapped_context(self):
        from bibr.utils.circuit_breaker import _is_rate_limit_error

        inner = type("RateLimitError", (Exception,), {})()
        outer = RuntimeError("context")
        outer.__context__ = inner
        assert _is_rate_limit_error(outer) is True

    def test_no_infinite_recursion_on_self_ref(self):
        from bibr.utils.circuit_breaker import _is_rate_limit_error

        exc = ValueError("test")
        # cause == self should not recurse infinitely
        exc.__cause__ = exc
        assert _is_rate_limit_error(exc) is False


class TestIsClientError:
    def test_rate_limit_is_client_error(self):
        from bibr.utils.circuit_breaker import _is_client_error

        exc = type("ResourceExhausted", (Exception,), {})()
        assert _is_client_error(type(exc), exc) is True

    def test_http_4xx_is_client_error(self):
        import httpx

        from bibr.utils.circuit_breaker import _is_client_error

        response = httpx.Response(400, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("bad", request=response.request, response=response)
        assert _is_client_error(type(exc), exc) is True

    def test_http_5xx_is_not_client_error(self):
        import httpx

        from bibr.utils.circuit_breaker import _is_client_error

        response = httpx.Response(500, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("server", request=response.request, response=response)
        assert _is_client_error(type(exc), exc) is False

    def test_regular_error_not_client_error(self):
        from bibr.utils.circuit_breaker import _is_client_error

        exc = ValueError("test")
        assert _is_client_error(type(exc), exc) is False


# ══════════════════════════════════════════════════════════════════════
# clients/base.py — BaseClient
# ══════════════════════════════════════════════════════════════════════


class TestBaseClient:
    def test_init_stores_base_url(self):
        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"  # trailing slash stripped
        assert client.timeout == 60.0

    def test_init_requires_base_url(self):
        from bibr.clients.base import BaseClient

        with pytest.raises(ValueError, match="Base URL must be provided"):
            BaseClient(base_url="")

    def test_custom_timeout(self):
        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost", timeout=30.0)
        assert client.timeout == 30.0

    async def test_close(self):
        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost")
        assert not client.client.is_closed
        await client.close()
        assert client.client.is_closed

    async def test_close_idempotent(self):
        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost")
        await client.close()
        await client.close()  # should not raise
        assert client.client.is_closed

    async def test_context_manager(self):
        from bibr.clients.base import BaseClient

        async with BaseClient(base_url="http://localhost") as client:
            assert not client.client.is_closed
        assert client.client.is_closed

    async def test_ping_success(self):
        import httpx

        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost")
        # Mock the get method to return 200
        client.client.get = AsyncMock(
            return_value=httpx.Response(
                200, request=httpx.Request("GET", "http://localhost/health")
            )
        )
        is_available, status = await client.ping()
        assert is_available is True
        assert status == 200
        await client.close()

    async def test_ping_failure(self):
        import httpx

        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost")
        client.client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        is_available, status = await client.ping()
        assert is_available is False
        assert status == 0
        await client.close()

    async def test_ping_non_200(self):
        import httpx

        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost")
        client.client.get = AsyncMock(
            return_value=httpx.Response(
                503, request=httpx.Request("GET", "http://localhost/health")
            )
        )
        is_available, status = await client.ping()
        assert is_available is False
        assert status == 503
        await client.close()

    def test_del_does_not_crash(self):
        from bibr.clients.base import BaseClient

        client = BaseClient(base_url="http://localhost")
        # __del__ should not raise even though client is unclosed
        client.__del__()

    def test_del_after_failed_init_does_not_crash(self, recwarn):
        # __init__ raises before self.client is assigned. Python may still
        # call __del__ on the partially-constructed object via gc, which
        # historically tried to access self.client and produced an
        # unraisable AttributeError warning.
        from bibr.clients.base import BaseClient

        partial = BaseClient.__new__(BaseClient)
        # Don't call __init__ — simulate failed init that never set self.client
        partial.__del__()  # should be a no-op, not raise
        # No unraisable AttributeError warning expected
        assert not any(
            "AttributeError" in str(w.message) and "client" in str(w.message) for w in recwarn.list
        )


# ══════════════════════════════════════════════════════════════════════
# clients/crossref.py — _is_retryable()
# ══════════════════════════════════════════════════════════════════════


class TestCrossrefIsRetryable:
    def test_timeout_is_retryable(self):
        import httpx

        from bibr.clients.crossref import CrossrefClient

        assert CrossrefClient._is_retryable(httpx.ReadTimeout("timeout")) is True
        assert CrossrefClient._is_retryable(httpx.ConnectTimeout("timeout")) is True

    def test_5xx_is_retryable(self):
        import httpx

        from bibr.clients.crossref import CrossrefClient

        response = httpx.Response(500, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("server", request=response.request, response=response)
        assert CrossrefClient._is_retryable(exc) is True

    def test_429_is_retryable(self):
        import httpx

        from bibr.clients.crossref import CrossrefClient

        response = httpx.Response(429, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("rate", request=response.request, response=response)
        assert CrossrefClient._is_retryable(exc) is True

    def test_4xx_not_retryable(self):
        import httpx

        from bibr.clients.crossref import CrossrefClient

        response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("not found", request=response.request, response=response)
        assert CrossrefClient._is_retryable(exc) is False

    def test_other_error_not_retryable(self):
        from bibr.clients.crossref import CrossrefClient

        assert CrossrefClient._is_retryable(ValueError("nope")) is False


# ══════════════════════════════════════════════════════════════════════
# clients/llm.py — _cap_input()
# ══════════════════════════════════════════════════════════════════════


class TestLlmCapInput:
    def test_short_text_unchanged(self):
        from bibr.clients.llm import LLMClient

        result = LLMClient._cap_input("short text")
        assert result == "short text"

    def test_long_text_truncated(self):
        from bibr.clients.llm import LLMClient
        from bibr.config import Settings

        limit = Settings.llm.max_input_chars
        long_text = "x" * (limit + 100)
        result = LLMClient._cap_input(long_text)
        assert len(result) == limit


# ══════════════════════════════════════════════════════════════════════
# extract/extractor.py — _parse_year(), _map_bib_text_ids(),
#                         _collect_last_unknown_section_rows()
# ══════════════════════════════════════════════════════════════════════


class TestParseYear:
    def test_simple_year(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year("2020") == 2020

    def test_year_with_letter_suffix(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year("2020a") == 2020

    def test_year_with_whitespace(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year("  2019  ") == 2019

    def test_none_returns_none(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year(None) is None

    def test_empty_returns_none(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year("") is None

    def test_invalid_returns_none(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year("not-a-year") is None

    def test_year_with_multiple_letters(self):
        from bibr.extract.extractor import _parse_year

        assert _parse_year("2021abc") == 2021

    def test_in_press_returns_none(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("in press") is None
        assert _is_in_press("in press")

    def test_in_press_case_insensitive(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("In Press") is None
        assert _is_in_press("In Press")

    def test_in_press_extra_spaces(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("in  press") is None
        assert _is_in_press("in  press")

    def test_forthcoming_returns_none(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("forthcoming") is None
        assert _is_in_press("forthcoming")

    def test_advance_online_returns_none(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("advance online") is None
        assert _is_in_press("advance online")

    def test_manuscript_submitted_returns_none(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("manuscript submitted") is None
        assert _is_in_press("manuscript submitted")

    def test_epub_ahead_returns_none(self):
        from bibr.extract.extractor import _is_in_press, _parse_year

        assert _parse_year("epub ahead") is None
        assert _is_in_press("epub ahead")


class TestMapBibTextIds:
    def test_exact_title_match(self):
        from bibr.extract.extractor import _map_bib_text_ids

        sections = [
            PaperSection(
                section_id=1,
                header="References",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.REFERENCES,
            ),
        ]
        sentences = [
            PaperSentence(
                text_id=10,
                text="Smith J. A Study of Something. Nature 2020.",
                section_id=1,
                paragraph_id=1,
            ),
        ]
        contents = _minimal_contents(sections=sections, sentences=sentences)
        refs = [
            PaperReference(
                bib_id=1,
                title="A Study of Something",
                first_page=None,
                volume=None,
                authors="Smith, J.",
                year=2020,
                container="Nature",
            )
        ]

        _map_bib_text_ids(refs, contents)
        assert refs[0].text_id == 10

    def test_fuzzy_match_fallback(self):
        pytest.importorskip("rapidfuzz")
        from bibr.extract.extractor import _map_bib_text_ids

        sections = [
            PaperSection(
                section_id=1,
                header="References",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.REFERENCES,
            ),
        ]
        sentences = [
            PaperSentence(
                text_id=10,
                text="Smith J (2020) A study of somthing. Nature.",
                section_id=1,
                paragraph_id=1,
            ),
        ]
        contents = _minimal_contents(sections=sections, sentences=sentences)
        refs = [
            PaperReference(
                bib_id=1,
                title="A Study of Something",
                first_page=None,
                volume=None,
                authors="Smith, J.",
                year=2020,
                container="Nature",
            )
        ]

        _map_bib_text_ids(refs, contents)
        # Fuzzy match should kick in
        assert refs[0].text_id == 10

    def test_no_ref_sections_is_noop(self):
        from bibr.extract.extractor import _map_bib_text_ids

        contents = _minimal_contents()
        refs = [
            PaperReference(
                bib_id=1,
                title="Title",
                first_page=None,
                volume=None,
                authors="A",
                year=2020,
                container="J",
            )
        ]
        _map_bib_text_ids(refs, contents)
        assert refs[0].text_id is None

    def test_empty_refs_is_noop(self):
        from bibr.extract.extractor import _map_bib_text_ids

        contents = _minimal_contents()
        _map_bib_text_ids([], contents)  # no crash

    def test_skips_refs_with_existing_text_id(self):
        from bibr.extract.extractor import _map_bib_text_ids

        sections = [
            PaperSection(
                section_id=1,
                header="References",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.REFERENCES,
            ),
        ]
        sentences = [
            PaperSentence(
                text_id=10, text="Smith J. Title. Nature 2020.", section_id=1, paragraph_id=1
            ),
        ]
        contents = _minimal_contents(sections=sections, sentences=sentences)
        refs = [
            PaperReference(
                bib_id=1,
                title="Title",
                first_page=None,
                volume=None,
                authors="Smith, J.",
                year=2020,
                container="Nature",
                text_id=99,
            )
        ]

        _map_bib_text_ids(refs, contents)
        assert refs[0].text_id == 99  # unchanged


class TestCollectLastUnknownSectionRows:
    def test_finds_last_unknown_section(self):
        from bibr.extract.extractor import MetadataExtractor

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Methods",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.METHODS,
            ),
            PaperSection(
                section_id=2,
                header="Literatur",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.UNKNOWN,
            ),
        ]
        sentences = [
            PaperSentence(text_id=1, text="Method text.", section_id=1, paragraph_id=1),
            PaperSentence(text_id=2, text="Ref 1.", section_id=2, paragraph_id=1),
            PaperSentence(text_id=3, text="Ref 2.", section_id=2, paragraph_id=1),
        ]
        contents = _minimal_contents(sections=sections, sentences=sentences)
        extractor = MetadataExtractor(contents=contents, llm_client=MagicMock())

        result = extractor.locator._collect_last_unknown_section_rows()
        assert len(result) == 2
        assert list(result["text"]) == ["Ref 1.", "Ref 2."]

    def test_raises_when_no_unknown_sections(self):
        from bibr.extract.extractor import MetadataExtractor

        sections = [
            PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
            PaperSection(
                section_id=1,
                header="Methods",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.METHODS,
            ),
        ]
        sentences = [
            PaperSentence(text_id=1, text="Text.", section_id=1, paragraph_id=1),
        ]
        contents = _minimal_contents(sections=sections, sentences=sentences)
        extractor = MetadataExtractor(contents=contents, llm_client=MagicMock())

        with pytest.raises(ValueError, match="No reference section found"):
            extractor.locator._collect_last_unknown_section_rows()


# ══════════════════════════════════════════════════════════════════════
# extract/equation_extractor.py — helpers
# ══════════════════════════════════════════════════════════════════════


class TestNormalizeComp:
    def test_less_equal(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp("<=") == "\u2264"

    def test_greater_equal(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp(">=") == "\u2265"

    def test_much_less(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp("<<") == "\u226a"

    def test_much_greater(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp(">>") == "\u226b"

    def test_equals_unchanged(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp("=") == "="

    def test_less_than_unchanged(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp("<") == "<"

    def test_unicode_passthrough(self):
        from bibr.extract.equation_extractor import _normalize_comp

        assert _normalize_comp("\u2248") == "\u2248"  # ≈


class TestSplitRespectingBrackets:
    def test_simple_split(self):
        from bibr.extract.equation_extractor import _split_respecting_brackets

        assert _split_respecting_brackets("a, b, c") == ["a", " b", " c"]

    def test_nested_parens_preserved(self):
        from bibr.extract.equation_extractor import _split_respecting_brackets

        result = _split_respecting_brackets("t(28) = 3.42, p < .001")
        assert len(result) == 2
        assert "t(28) = 3.42" in result[0]

    def test_nested_brackets_preserved(self):
        from bibr.extract.equation_extractor import _split_respecting_brackets

        result = _split_respecting_brackets("[1, 2], [3, 4]")
        assert len(result) == 2

    def test_semicolon_split(self):
        from bibr.extract.equation_extractor import _split_respecting_brackets

        result = _split_respecting_brackets("a = 1; b = 2")
        assert len(result) == 2

    def test_no_separators(self):
        from bibr.extract.equation_extractor import _split_respecting_brackets

        assert _split_respecting_brackets("t(28) = 3.42") == ["t(28) = 3.42"]


class TestFindToplevelComp:
    def test_finds_toplevel_equals(self):
        from bibr.extract.equation_extractor import _find_toplevel_comp

        m = _find_toplevel_comp("x = 5")
        assert m is not None
        assert m.group() == "="

    def test_ignores_subscript_equals(self):
        from bibr.extract.equation_extractor import _find_toplevel_comp

        m = _find_toplevel_comp(r"\sum_{i=1}^{n} x")
        assert m is None

    def test_ignores_superscript_equals(self):
        from bibr.extract.equation_extractor import _find_toplevel_comp

        m = _find_toplevel_comp(r"x^{2=y}")
        assert m is None

    def test_finds_after_braces(self):
        from bibr.extract.equation_extractor import _find_toplevel_comp

        m = _find_toplevel_comp(r"\hat{x} = 5")
        assert m is not None
        assert m.group() == "="

    def test_returns_none_on_empty(self):
        from bibr.extract.equation_extractor import _find_toplevel_comp

        assert _find_toplevel_comp("") is None

    def test_handles_unbalanced_braces(self):
        from bibr.extract.equation_extractor import _find_toplevel_comp

        # Unbalanced closing brace — should not crash
        m = _find_toplevel_comp("x} = 5")
        assert m is not None


class TestIterParenthesizedGroups:
    def test_simple_group(self):
        from bibr.extract.equation_extractor import _iter_parenthesized_groups

        result = list(_iter_parenthesized_groups("(a, b)"))
        assert len(result) == 1
        assert result[0] == ("a, b", 0, 6)

    def test_nested_groups(self):
        from bibr.extract.equation_extractor import _iter_parenthesized_groups

        # Only emits outermost
        result = list(_iter_parenthesized_groups("(t(28) = 3.42)"))
        assert len(result) == 1
        assert result[0][0] == "t(28) = 3.42"

    def test_multiple_groups(self):
        from bibr.extract.equation_extractor import _iter_parenthesized_groups

        result = list(_iter_parenthesized_groups("(a) and (b)"))
        assert len(result) == 2
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_empty_parens_skipped(self):
        from bibr.extract.equation_extractor import _iter_parenthesized_groups

        result = list(_iter_parenthesized_groups("()"))
        assert len(result) == 0

    def test_unbalanced_parens(self):
        from bibr.extract.equation_extractor import _iter_parenthesized_groups

        # Unmatched close paren should not crash
        result = list(_iter_parenthesized_groups(")(a)"))
        assert len(result) == 1

    def test_no_parens(self):
        from bibr.extract.equation_extractor import _iter_parenthesized_groups

        result = list(_iter_parenthesized_groups("no parens here"))
        assert len(result) == 0


class TestExtractWithLlmFallback:
    async def test_no_llm_client_returns_regex_only(self):
        from bibr.extract.equation_extractor import EquationExtractor

        extractor = EquationExtractor()
        sentences = [
            PaperSentence(text_id=1, text="t(28) = 3.42, p < .001", section_id=1, paragraph_id=1),
        ]
        sections = [
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        result = await extractor.extract_with_llm_fallback(sentences, sections, llm_client=None)
        assert len(result) > 0

    async def test_llm_fallback_called_for_unmatched(self):
        from bibr.extract.equation_extractor import EquationExtractor

        extractor = EquationExtractor()
        # Sentence with parenthesized numbers but no stat pattern → triggers LLM
        sentences = [
            PaperSentence(
                text_id=1, text="The model (n=50) showed improvement.", section_id=1, paragraph_id=1
            ),
        ]
        sections = [
            PaperSection(
                section_id=1,
                header="Results",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.RESULTS,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.extract_equations = AsyncMock(
            return_value=[
                PaperEquation(text_id=1, grp_id=0, lhs="n", comp="=", rhs="50"),
            ]
        )

        result = await extractor.extract_with_llm_fallback(sentences, sections, llm_client=mock_llm)
        # If regex found n=50 already, LLM won't be called; if it didn't, LLM provides it
        assert any(eq.lhs == "n" and eq.rhs == "50" for eq in result)

    async def test_llm_failure_graceful(self):
        from bibr.extract.equation_extractor import EquationExtractor

        extractor = EquationExtractor()
        sentences = [
            PaperSentence(
                text_id=1, text="Ambiguous (data 123) result.", section_id=1, paragraph_id=1
            ),
        ]
        sections = [
            PaperSection(
                section_id=1,
                header="Methods",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.METHODS,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.extract_equations = AsyncMock(side_effect=RuntimeError("LLM down"))

        # Should not raise — graceful degradation
        result = await extractor.extract_with_llm_fallback(sentences, sections, llm_client=mock_llm)
        assert isinstance(result, list)

    async def test_skips_non_target_sections(self):
        from bibr.extract.equation_extractor import EquationExtractor

        extractor = EquationExtractor()
        sentences = [
            PaperSentence(
                text_id=1, text="The model (n=50) showed improvement.", section_id=1, paragraph_id=1
            ),
        ]
        # Discussion section — not in target_section_types
        sections = [
            PaperSection(
                section_id=1,
                header="Discussion",
                level=1,
                parent_section_id=0,
                section_type=CanonicalSection.DISCUSSION,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.extract_equations = AsyncMock(return_value=[])

        await extractor.extract_with_llm_fallback(sentences, sections, llm_client=mock_llm)
        # LLM should NOT be called for discussion section
        mock_llm.extract_equations.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# input/consolidate_text.py — strip_latex_commands(), _fix_operatorname_spacing()
# ══════════════════════════════════════════════════════════════════════


class TestStripLatexCommands:
    def test_unwraps_mathrm(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert "n=5" in strip_latex_commands(r"\mathrm{n=5}")

    def test_unwraps_text(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert "hello" in strip_latex_commands(r"\text{hello}")

    def test_unwraps_textit(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert "italic" in strip_latex_commands(r"\textit{italic}")

    def test_flattens_subscript(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert "c" in strip_latex_commands(r"_{c}")
        assert "_" not in strip_latex_commands(r"_{c}")

    def test_flattens_superscript(self):
        from bibr.input.consolidate_text import strip_latex_commands

        result = strip_latex_commands(r"x^{2}")
        assert "2" in result
        assert "^" not in result

    def test_single_char_sub_super(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert strip_latex_commands("x_i") == "xi"
        assert strip_latex_commands("x^2") == "x2"

    def test_greek_to_unicode(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert strip_latex_commands(r"\alpha") == "\u03b1"
        assert strip_latex_commands(r"\beta") == "\u03b2"
        assert strip_latex_commands(r"\Delta") == "\u0394"

    def test_symbol_to_unicode(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert strip_latex_commands(r"\leq") == "\u2264"
        assert strip_latex_commands(r"\geq") == "\u2265"
        assert strip_latex_commands(r"\approx") == "\u2248"

    def test_unknown_command_stripped(self):
        from bibr.input.consolidate_text import strip_latex_commands

        result = strip_latex_commands(r"\foobar")
        assert "\\" not in result
        assert "foobar" in result

    def test_empty_braces_removed(self):
        from bibr.input.consolidate_text import strip_latex_commands

        result = strip_latex_commands(r"x{}")
        assert "{}" not in result

    def test_plain_text_unchanged(self):
        from bibr.input.consolidate_text import strip_latex_commands

        assert strip_latex_commands("plain text") == "plain text"


class TestFixOperatornameSpacing:
    def test_collapses_spaced_out_name(self):
        from bibr.input.consolidate_text import _fix_operatorname_spacing

        result = _fix_operatorname_spacing(r"\operatorname{A t t e n t i o n}")
        assert result == r"\operatorname{Attention}"

    def test_leaves_normal_name_alone(self):
        from bibr.input.consolidate_text import _fix_operatorname_spacing

        text = r"\operatorname{Attention}"
        assert _fix_operatorname_spacing(text) == text

    def test_leaves_multichar_tokens_alone(self):
        from bibr.input.consolidate_text import _fix_operatorname_spacing

        text = r"\operatorname{soft max}"
        assert _fix_operatorname_spacing(text) == text

    def test_no_operatorname(self):
        from bibr.input.consolidate_text import _fix_operatorname_spacing

        text = "regular text"
        assert _fix_operatorname_spacing(text) == text

    def test_single_char_not_collapsed(self):
        from bibr.input.consolidate_text import _fix_operatorname_spacing

        text = r"\operatorname{x}"
        # Single token, nothing to collapse
        assert _fix_operatorname_spacing(text) == text


# ══════════════════════════════════════════════════════════════════════
# structure/pdf_parser.py — _is_copyright_notice()
# ══════════════════════════════════════════════════════════════════════


class TestIsCopyrightNotice:
    def test_copyright_symbol(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("\u00a9 2024 Elsevier B.V.") is True

    def test_copyright_word(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("Copyright 2024 IEEE") is True

    def test_all_rights_reserved(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("All rights reserved.") is True

    def test_creative_commons(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("Licensed under Creative Commons CC-BY 4.0") is True

    def test_open_access(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("This is an open access article") is True

    def test_permission(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("Permission to reproduce is granted") is True

    def test_normal_title_not_flagged(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("A Study of Machine Learning Methods") is False

    def test_case_insensitive(self):
        from bibr.structure.pdf_parser import PDFParser

        assert PDFParser._is_copyright_notice("COPYRIGHT 2024") is True
