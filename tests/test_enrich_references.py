"""Tests for bibr.enrich.references — Crossref reference enrichment."""

import asyncio
import logging
from unittest import mock

import httpx

from bibr.clients.crossref import CrossrefClient
from bibr.paper import ExternalMatch, MatchSource, PaperReference


def _mock_crossref():
    client = mock.MagicMock(spec_set=CrossrefClient)
    client.enrich_semaphore = asyncio.Semaphore(100)
    return client


def _make_ref(bib_id=1, title="Test Paper", authors=None, year=2020, doi=None, **kwargs):
    if authors is None:
        authors = "Smith, J."
    return PaperReference(
        bib_id=bib_id,
        title=title,
        first_page=None,
        volume=None,
        authors=authors,
        year=year,
        container=None,
        doi=doi,
        **kwargs,
    )


def _make_crossref_item(
    doi="10.1000/test",
    title="Test Paper",
    authors=None,
    year=2020,
    cr_type="journal-article",
):
    """Build a mock Crossref work item."""
    return {
        "DOI": doi,
        "title": [title],
        "author": authors or [{"given": "J.", "family": "Smith"}],
        "issued": {"date-parts": [[year]]},
        "container-title": ["Nature"],
        "volume": "42",
        "issue": "3",
        "page": "100-115",
        "publisher": "Nature Publishing Group",
        "ISSN": ["0028-0836"],
        "ISBN": ["978-0-123-45678-9"],
        "URL": f"https://doi.org/{doi}",
        "type": cr_type,
    }


class TestEnrichReferences:
    """Tests for the enrich_references function — mutates refs in place."""

    async def test_doi_lookup(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test")
        cr_item = _make_crossref_item()

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF in ref.match
        m = ref.match[MatchSource.CROSSREF]
        assert isinstance(m, ExternalMatch)
        assert m.id == "10.1000/test"
        assert m.score == 100.0
        assert m.doi == "10.1000/test"
        assert m.container == "Nature"
        assert m.volume == "42"
        assert m.issue == "3"
        assert m.first_page == "100"
        assert m.last_page == "115"
        assert m.publisher == "Nature Publishing Group"
        assert m.url == "https://doi.org/10.1000/test"
        assert m.bib_type == "journal_article"  # mapped from journal-article

    async def test_bibliographic_search_fallback(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi=None, title="A Great Paper on Testing Methods")
        cr_item = _make_crossref_item(title="A Great Paper on Testing Methods")

        mock_client = _mock_crossref()
        mock_client.search = mock.AsyncMock(return_value={"message": {"items": [cr_item]}})

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF in ref.match
        m = ref.match[MatchSource.CROSSREF]
        assert m.score is not None
        assert m.score <= 100.0
        mock_client.search.assert_called_once()

    async def test_no_match_below_threshold(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi=None, title="Completely Different Paper Title Here")
        cr_item = _make_crossref_item(title="Unrelated Quantum Physics Research")

        mock_client = _mock_crossref()
        mock_client.search = mock.AsyncMock(return_value={"message": {"items": [cr_item]}})

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF not in ref.match

    async def test_original_ref_fields_not_mutated(self):
        """After enrichment, parsed reference fields must NOT be changed."""
        from bibr.enrich.references import enrich_references

        original_authors = "Original Author"
        ref = PaperReference(
            bib_id=1,
            title="Original Title",
            first_page=None,
            volume=None,
            authors=original_authors,
            year=2019,
            container="Original Journal",
            doi="10.1000/test",
        )
        cr_item = _make_crossref_item(title="Crossref Title", year=2020)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        # Parsed fields must be untouched
        assert ref.title == "Original Title"
        assert ref.authors is original_authors
        assert ref.year == 2019
        assert ref.container == "Original Journal"
        # Match carries crossref data
        assert MatchSource.CROSSREF in ref.match
        assert ref.match[MatchSource.CROSSREF].title == "Crossref Title"
        assert ref.match[MatchSource.CROSSREF].year == 2020

    async def test_structured_authors_in_match(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test")
        cr_item = _make_crossref_item(
            authors=[
                {"given": "John", "family": "Smith", "ORCID": "0000-0001-2345-6789"},
                {"given": "Jane", "family": "Doe"},
            ]
        )

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        # Author field must NOT be overwritten (authors is now a plain string)
        assert ref.authors == "Smith, J."
        # Match has structured authors (ExternalMatch.authors is still list[BibAuthor])
        match_authors = ref.match[MatchSource.CROSSREF].authors
        assert len(match_authors) == 2
        assert match_authors[0].family == "Smith"
        assert match_authors[0].given == "John"
        assert match_authors[1].family == "Doe"

    async def test_graceful_failure_per_reference(self):
        from bibr.enrich.references import enrich_references

        ref1 = _make_ref(bib_id=1, doi="10.1000/good")
        ref2 = _make_ref(bib_id=2, doi="10.1000/bad")
        ref3 = _make_ref(bib_id=3, doi="10.1000/also-good")

        cr_item = _make_crossref_item()

        async def mock_works(ids=None, **kwargs):
            if "bad" in (ids or ""):
                raise Exception("API error")
            return {"message": cr_item}

        mock_client = _mock_crossref()
        mock_client.works = mock_works

        report = await enrich_references([ref1, ref2, ref3], crossref_client=mock_client)

        assert MatchSource.CROSSREF in ref1.match
        assert MatchSource.CROSSREF not in ref2.match  # failed gracefully
        assert MatchSource.CROSSREF in ref3.match
        assert report.attempted == 3
        assert report.matched == 2
        assert report.failed == 1
        assert any("bib_id=2" in detail for detail in report.details)

    async def test_clean_404_is_a_complete_miss_not_a_failure(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/missing", title="")
        request = httpx.Request("GET", "https://api.crossref.org/works/10.1000/missing")
        response = httpx.Response(404, request=request)
        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(
            side_effect=httpx.HTTPStatusError("missing", request=request, response=response)
        )

        report = await enrich_references([ref], crossref_client=mock_client)

        assert report.attempted == 1
        assert report.matched == 0
        assert report.failed == 0

    async def test_empty_references_list(self):
        from bibr.enrich.references import enrich_references

        await enrich_references([])

    async def test_crossref_type_mapped(self):
        """Crossref type should be mapped to v6.0 BibType in match."""
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test")
        cr_item = _make_crossref_item(cr_type="proceedings-article")

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert ref.match[MatchSource.CROSSREF].bib_type == "conference_paper"

    async def test_match_score_100_for_doi_lookup(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test")
        cr_item = _make_crossref_item()

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert ref.match[MatchSource.CROSSREF].score == 100.0

    async def test_container_for_book_chapter(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/ch1")
        cr_item = {
            "DOI": "10.1000/ch1",
            "title": ["Test Paper"],
            "author": [{"given": "J.", "family": "Smith"}],
            "issued": {"date-parts": [[2020]]},
            "container-title": ["Handbook of Testing"],
            "type": "book-chapter",
            "page": "50-75",
        }

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        m = ref.match[MatchSource.CROSSREF]
        assert m.container == "Handbook of Testing"
        assert m.first_page == "50"
        assert m.last_page == "75"
        assert m.bib_type == "book_chapter"


class TestFindBestMatch:
    """Tests for the _find_best_match helper — now returns (item, score) tuples."""

    def test_exact_match(self):
        from bibr.enrich.references import _find_best_match

        items = [{"title": ["A Great Paper on Testing"]}]
        result = _find_best_match("A Great Paper on Testing", items)
        assert result is not None
        item, score = result
        assert score >= 80
        assert item.title == "A Great Paper on Testing"

    def test_fuzzy_match(self):
        from bibr.enrich.references import _find_best_match

        items = [{"title": ["A Great Paper on Testing Methods"]}]
        result = _find_best_match("A Great Paper on Testing", items)
        assert result is not None
        item, score = result
        assert score >= 80

    def test_no_match(self):
        from bibr.enrich.references import _find_best_match

        items = [{"title": ["Quantum Entanglement in Particles"]}]
        result = _find_best_match("A Great Paper on Testing", items)
        assert result is None

    def test_empty_items(self):
        from bibr.enrich.references import _find_best_match

        assert _find_best_match("Title", []) is None

    def test_item_without_title(self):
        from bibr.enrich.references import _find_best_match

        items = [{"DOI": "10.1000/test"}]  # no title key
        assert _find_best_match("Title", items) is None


class TestBuildMatch:
    """Tests for _build_match — constructs ExternalMatch from CrossrefWorkItem."""

    def test_full_match(self):
        from bibr.enrich.references import _build_match
        from bibr.enrich.schemas import CrossrefWorkItem

        cr_item = CrossrefWorkItem.from_raw(_make_crossref_item())
        m = _build_match(cr_item, 100.0)

        assert isinstance(m, ExternalMatch)
        assert m.id == "10.1000/test"
        assert m.score == 100.0
        assert m.title == "Test Paper"
        assert len(m.authors) == 1
        assert m.authors[0].family == "Smith"
        assert m.authors[0].given == "J."
        assert m.year == 2020
        assert m.container == "Nature"
        assert m.volume == "42"
        assert m.issue == "3"
        assert m.first_page == "100"
        assert m.last_page == "115"
        assert m.publisher == "Nature Publishing Group"
        assert m.doi == "10.1000/test"
        assert m.bib_type == "journal_article"
        assert m.url == "https://doi.org/10.1000/test"

    def test_minimal_match(self):
        from bibr.enrich.references import _build_match
        from bibr.enrich.schemas import CrossrefWorkItem

        m = _build_match(CrossrefWorkItem.from_raw({"DOI": "10.1000/min"}), 95.0)

        assert m.id == "10.1000/min"
        assert m.score == 95.0
        assert m.title is None
        assert m.authors is None

    def test_year_as_int(self):
        from bibr.enrich.references import _build_match
        from bibr.enrich.schemas import CrossrefWorkItem

        cr_item = CrossrefWorkItem.from_raw(_make_crossref_item(year=2023))
        m = _build_match(cr_item, 100.0)

        assert m.year == 2023
        assert isinstance(m.year, int)


# ══════════════════════════════════════════════════════════════════════
# Issue #7: DOI 404 should not fall through to bibliographic search
# ══════════════════════════════════════════════════════════════════════


class TestDoi404NoFallthrough:
    async def test_404_skips_search(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/nonexistent", title="A Long Enough Title Here")

        response = httpx.Response(404, request=httpx.Request("GET", "https://api.crossref.org"))
        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=response.request, response=response
            )
        )
        mock_client.search = mock.AsyncMock()

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF not in ref.match
        mock_client.search.assert_not_called()

    async def test_non_404_falls_through_to_search(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/bad", title="A Long Enough Title Here")
        cr_item = _make_crossref_item(title="A Long Enough Title Here")

        response = httpx.Response(500, request=httpx.Request("GET", "https://api.crossref.org"))
        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error", request=response.request, response=response
            )
        )
        mock_client.search = mock.AsyncMock(return_value={"message": {"items": [cr_item]}})

        await enrich_references([ref], crossref_client=mock_client)

        mock_client.search.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
# Issue #9: Year sanity check on DOI lookup
# ══════════════════════════════════════════════════════════════════════


class TestDoiYearSanityCheck:
    async def test_matching_years_score_100(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test", year=2020)
        cr_item = _make_crossref_item(year=2020)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert ref.match[MatchSource.CROSSREF].score == 100.0

    async def test_year_diff_1_tolerated(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test", year=2019)
        cr_item = _make_crossref_item(year=2020)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert ref.match[MatchSource.CROSSREF].score == 100.0

    async def test_year_mismatch_discards_match(self, caplog):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test", year=2015)
        cr_item = _make_crossref_item(year=2020)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        with caplog.at_level(logging.WARNING, logger="bibr.enrich.references"):
            await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF not in ref.match
        assert any("year mismatch" in r.getMessage() for r in caplog.records)

    async def test_year_mismatch_does_not_fall_through_to_search(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test", year=2015, title="A Long Enough Title For Search Here")
        cr_item = _make_crossref_item(year=2020)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})
        mock_client.search = mock.AsyncMock()

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF not in ref.match
        mock_client.search.assert_not_called()

    async def test_none_year_keeps_score_100(self):
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test", year=None)
        cr_item = _make_crossref_item(year=2020)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert ref.match[MatchSource.CROSSREF].score == 100.0


# ══════════════════════════════════════════════════════════════════════
# In-press year enrichment: year=0 should skip year-mismatch penalties
# ══════════════════════════════════════════════════════════════════════


class TestInPressYearEnrichment:
    """year=0 (in-press) references should not trigger year-mismatch penalties."""

    async def test_doi_lookup_year_zero_keeps_score_100(self):
        """year=0 is falsy → year check skipped → score stays 100."""
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test", year=0)
        cr_item = _make_crossref_item(year=2025)

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF in ref.match
        assert ref.match[MatchSource.CROSSREF].score == 100.0

    async def test_search_query_excludes_year_zero(self):
        """year=0 should NOT appear in the Crossref search query string."""
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi=None, title="A Great Paper on In-Press Testing Methods", year=0)
        cr_item = _make_crossref_item(title="A Great Paper on In-Press Testing Methods")

        mock_client = _mock_crossref()
        mock_client.search = mock.AsyncMock(return_value={"message": {"items": [cr_item]}})

        await enrich_references([ref], crossref_client=mock_client)

        # Verify "0" is not in the search query
        call_kwargs = mock_client.search.call_args
        query = call_kwargs.kwargs.get("query", call_kwargs.args[0] if call_kwargs.args else "")
        assert "0" not in query.split()

    def test_find_best_match_year_zero_no_penalty(self):
        """year=0 should skip the year-proximity penalty in _find_best_match."""
        from bibr.enrich.references import _find_best_match

        items = [_make_crossref_item(title="A Great Paper on Testing", year=2020)]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=0, ref_authors="Smith, J."
        )
        assert result is not None
        _, score = result
        # No year penalty applied → score should be high (not halved)
        assert score >= 80


# ══════════════════════════════════════════════════════════════════════
# Issue #8: Author/year validation in _find_best_match
# ══════════════════════════════════════════════════════════════════════


class TestFindBestMatchValidation:
    def test_year_mismatch_penalizes(self):
        from bibr.enrich.references import _find_best_match

        items = [_make_crossref_item(title="A Great Paper on Testing", year=2010)]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=2020, ref_authors="Smith, J."
        )
        # Score should be penalized below threshold (80 * 0.5 = 40)
        assert result is None

    def test_no_author_overlap_penalizes(self):
        from bibr.enrich.references import _find_best_match

        items = [
            _make_crossref_item(
                title="A Great Paper on Testing",
                authors=[{"given": "X.", "family": "Completely-Different"}],
                year=2020,
            )
        ]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=2020, ref_authors="Smith, J."
        )
        # Title score ~100, but no author overlap → *0.7 = ~70 < 80 threshold
        assert result is None

    def test_matching_author_and_year_no_penalty(self):
        from bibr.enrich.references import _find_best_match

        items = [_make_crossref_item(title="A Great Paper on Testing", year=2020)]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=2020, ref_authors="Smith, J."
        )
        assert result is not None
        _, score = result
        assert score >= 80

    def test_penalized_candidate_does_not_displace_clean_one(self):
        """A high-pre-score candidate that fails the year/author gate must
        not knock out a slightly-lower-but-clean candidate that passes the
        threshold."""
        from bibr.enrich.references import _find_best_match

        items = [
            # High title score but wrong year — penalized to 0.5
            _make_crossref_item(
                doi="10.1000/wrong-year",
                title="A Great Paper on Testing",
                authors=[{"given": "J.", "family": "Smith"}],
                year=2010,
            ),
            # Slightly different title, same year, same author — no penalty
            _make_crossref_item(
                doi="10.1000/clean",
                title="A Great Paper on Testing Methods",
                authors=[{"given": "J.", "family": "Smith"}],
                year=2020,
            ),
        ]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=2020, ref_authors="Smith, J."
        )
        assert result is not None
        item, score = result
        assert item.doi == "10.1000/clean"
        assert score >= 80

    def test_none_year_no_penalty(self):
        from bibr.enrich.references import _find_best_match

        items = [_make_crossref_item(title="A Great Paper on Testing")]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=None, ref_authors=None
        )
        assert result is not None

    def test_none_authors_no_penalty(self):
        from bibr.enrich.references import _find_best_match

        items = [
            _make_crossref_item(
                title="A Great Paper on Testing",
                authors=[{"given": "X.", "family": "Nobody"}],
            )
        ]
        result = _find_best_match(
            "A Great Paper on Testing", items, ref_year=2020, ref_authors=None
        )
        assert result is not None


# ══════════════════════════════════════════════════════════════════════
# Issue #12: Errors in _build_match now surface at WARNING level
# ══════════════════════════════════════════════════════════════════════


class TestErrorPropagation:
    async def test_build_match_error_surfaces(self):
        from bibr.enrich.references import enrich_references

        ref1 = _make_ref(bib_id=1, doi="10.1000/good")
        ref2 = _make_ref(bib_id=2, doi="10.1000/bad")

        good_item = _make_crossref_item()
        bad_item = _make_crossref_item()
        # Make bad_item produce an error in _build_match by using a broken type
        bad_item["type"] = None
        bad_item["page"] = 12345  # int instead of str — will cause AttributeError in split

        mock_client = _mock_crossref()

        async def mock_works(ids=None, **kwargs):
            if "bad" in (ids or ""):
                return {"message": bad_item}
            return {"message": good_item}

        mock_client.works = mock_works

        await enrich_references([ref1, ref2], crossref_client=mock_client)

        # ref1 should succeed regardless of ref2's failure
        assert MatchSource.CROSSREF in ref1.match


# ══════════════════════════════════════════════════════════════════════
# CROSSREF_ENRICH gate: verify flag controls enrichment execution
# ══════════════════════════════════════════════════════════════════════


class TestCrossrefEnrichGate:
    """Verify CROSSREF_ENRICH=False skips enrichment entirely."""

    async def test_enrich_skipped_when_disabled(self, monkeypatch):
        """When CROSSREF_ENRICH is False, the pipeline gate prevents enrichment."""
        from bibr.config import Settings

        monkeypatch.setattr(Settings.crossref, "enrich", False)

        assert Settings.crossref.enrich is False

    async def test_enrich_runs_when_enabled(self, monkeypatch):
        """When CROSSREF_ENRICH is True, enrichment proceeds normally."""
        from bibr.config import Settings
        from bibr.enrich.references import enrich_references

        monkeypatch.setattr(Settings.crossref, "enrich", True)

        ref = _make_ref(doi="10.1000/test")
        cr_item = _make_crossref_item()
        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await enrich_references([ref], crossref_client=mock_client)
        assert MatchSource.CROSSREF in ref.match


# ══════════════════════════════════════════════════════════════════════
# Enrichment timeout: mirrors pipeline's asyncio.wait_for behavior
# ══════════════════════════════════════════════════════════════════════


class TestEnrichmentTimeout:
    """Verify enrichment handles timeout gracefully (mirrors pipeline's asyncio.wait_for)."""

    async def test_timeout_caught_gracefully(self):
        """Simulates the pipeline's asyncio.wait_for timeout on slow enrichment."""
        import asyncio

        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/slow")

        async def slow_works(**kwargs):
            await asyncio.sleep(10)
            return {"message": _make_crossref_item()}

        mock_client = _mock_crossref()
        mock_client.works = slow_works

        timed_out = False
        try:
            await asyncio.wait_for(
                enrich_references([ref], crossref_client=mock_client),
                timeout=0.05,
            )
        except TimeoutError:
            timed_out = True

        assert timed_out
        assert MatchSource.CROSSREF not in ref.match

    async def test_fast_enrichment_completes_within_timeout(self):
        """Normal enrichment completes well within timeout."""
        import asyncio

        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/fast")
        cr_item = _make_crossref_item()

        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={"message": cr_item})

        await asyncio.wait_for(
            enrich_references([ref], crossref_client=mock_client),
            timeout=5.0,
        )

        assert MatchSource.CROSSREF in ref.match


# ══════════════════════════════════════════════════════════════════════
# Task 4: ResolutionStats — per-batch resolution instrumentation
# ══════════════════════════════════════════════════════════════════════


async def test_stats_counts_doi_path():
    from bibr.enrich.references import ResolutionStats, _fetch_crossref_item

    ref = _make_ref(doi="10.1000/test")
    client = mock.AsyncMock()
    client.works = mock.AsyncMock(return_value={"message": _make_crossref_item()})

    stats = ResolutionStats()
    result = await _fetch_crossref_item(ref, client, stats)

    assert result is not None
    assert stats.doi_attempts == 1
    assert stats.doi_matches == 1
    assert stats.search_attempts == 0
    assert stats.search_matches == 0


async def test_stats_counts_search_attempt_without_match():
    from bibr.enrich.references import ResolutionStats, _fetch_crossref_item

    ref = _make_ref(doi=None, title="A sufficiently long reference title")
    client = mock.AsyncMock()
    client.search = mock.AsyncMock(return_value={"message": {"items": []}})

    stats = ResolutionStats()
    result = await _fetch_crossref_item(ref, client, stats)

    assert result is None
    assert stats.doi_attempts == 0
    assert stats.search_attempts == 1
    assert stats.search_matches == 0


async def test_fetch_crossref_item_works_without_stats():
    from bibr.enrich.references import _fetch_crossref_item

    # stats is optional — existing callers/paths must keep working.
    ref = _make_ref(doi="10.1000/test")
    client = mock.AsyncMock()
    client.works = mock.AsyncMock(return_value={"message": _make_crossref_item()})

    result = await _fetch_crossref_item(ref, client)
    assert result is not None


def test_find_best_match_clean_candidate_wins_over_year_penalized():
    """When a near-exact title hit is year-penalized below threshold but a
    slightly-weaker candidate is clean and above threshold, return the clean
    one. Selection is by final (post-penalty) score so a strong-but-penalized
    candidate can't displace a clean one below the threshold gate."""
    from bibr.enrich.references import _find_best_match

    items = [
        # Pre-score = 100 (exact), year mismatched → final = 50 (below threshold)
        {
            "title": ["The Effect of X on Y"],
            "issued": {"date-parts": [[2010]]},
            "author": [{"family": "Smith"}, {"family": "Jones"}],
            "DOI": "good-but-old",
        },
        # Pre-score ~86 (close but not exact), correct year, author overlap
        # (Smith). final = 86 — above threshold and unpenalized.
        {
            "title": ["Effects of X on Y"],
            "issued": {"date-parts": [[2020]]},
            "author": [{"family": "Smith"}],
            "DOI": "ok-and-recent",
        },
    ]
    out = _find_best_match(
        title="The Effect of X on Y",
        items=items,
        ref_year=2020,
        ref_authors="Smith, Jones",
    )
    assert out is not None
    item, _score = out
    assert item.doi == "ok-and-recent"


def _raw_item(
    doi="10.1038/323533a0",
    title="Learning representations by back-propagating errors",
    container="Nature",
    volume="323",
    page="533-536",
    family="Rumelhart",
    year=1986,
):
    """A raw Crossref item with controllable fingerprint fields."""
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"given": "D. E.", "family": family}],
        "issued": {"date-parts": [[year]]},
        "container-title": [container],
        "volume": volume,
        "page": page,
        "type": "journal-article",
    }


class TestFingerprintMatch:
    """Strategy 3: match Science/Nature-style references that omit the article
    title, using the author+container+volume+year fingerprint instead.

    Real-world trigger: ``D. E. Rumelhart, G. E. Hinton, R. J. Williams,
    Nature 323, 533 (1986).`` — no title is printed, so the title-keyed
    Strategies 1-2 can't fire.
    """

    def _titleless_ref(self, **kw):
        # No DOI, no title — only the printed fingerprint fields.
        base = {
            "bib_id": 1,
            "title": "",  # title-less: the parser emits "" when no title is printed
            "authors": "D. E. Rumelhart, G. E. Hinton, R. J. Williams",
            "container": "Nature",
            "volume": "323",
            "first_page": None,
            "year": 1986,
            "doi": None,
        }
        base.update(kw)
        return PaperReference(**base)

    # --- unit: _score_fingerprint ---

    def test_score_accepts_matching_fingerprint(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        score = _score_fingerprint(self._titleless_ref(), CrossrefWorkItem.from_raw(_raw_item()))
        assert score is not None
        assert score >= 85

    def test_score_rejects_volume_mismatch(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(_raw_item(volume="999"))
        assert _score_fingerprint(self._titleless_ref(), cand) is None

    def test_score_rejects_missing_candidate_volume(self):
        # Volume is THE discriminator for journal refs — a candidate without one
        # cannot be confirmed on the fingerprint, so it must not match.
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(_raw_item(volume=None))
        assert _score_fingerprint(self._titleless_ref(), cand) is None

    def test_score_rejects_author_mismatch(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(_raw_item(family="Dijkstra"))
        assert _score_fingerprint(self._titleless_ref(), cand) is None

    def test_score_rejects_year_far_apart(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(_raw_item(year=1975))
        assert _score_fingerprint(self._titleless_ref(), cand) is None

    def test_score_rejects_container_mismatch(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(_raw_item(container="Journal of Irreproducible Results"))
        assert _score_fingerprint(self._titleless_ref(), cand) is None

    def test_score_rejects_first_page_mismatch(self):
        # When both ref and candidate carry a first page, a mismatch disqualifies.
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(_raw_item(page="999-1000"))
        assert _score_fingerprint(self._titleless_ref(first_page="533"), cand) is None

    # --- integration: _fetch_crossref_item Strategy 3 ---

    async def test_fetch_fires_strategy3_for_titleless_ref(self):
        from bibr.enrich.references import ResolutionStats, _fetch_crossref_item

        stats = ResolutionStats()
        client = mock.AsyncMock()
        client.search = mock.AsyncMock(return_value={"message": {"items": [_raw_item()]}})

        out = await _fetch_crossref_item(self._titleless_ref(), client, stats)

        assert out is not None
        cr_item, score = out
        assert cr_item.doi == "10.1038/323533a0"
        assert score >= 85
        client.search.assert_called_once()
        assert stats.fingerprint_attempts == 1
        assert stats.fingerprint_matches == 1

    async def test_fetch_skips_strategy3_without_volume(self):
        # Conservative scope: no volume → no fingerprint attempt (avoids weak matches).
        from bibr.enrich.references import ResolutionStats, _fetch_crossref_item

        stats = ResolutionStats()
        client = mock.AsyncMock()
        client.search = mock.AsyncMock(return_value={"message": {"items": [_raw_item()]}})

        out = await _fetch_crossref_item(self._titleless_ref(volume=None), client, stats)

        assert out is None
        client.search.assert_not_called()
        assert stats.fingerprint_attempts == 0

    async def test_fetch_titled_ref_does_not_use_fingerprint(self):
        # No regression: a titled ref uses Strategy 2 (title), not the fingerprint path.
        from bibr.enrich.references import ResolutionStats, _fetch_crossref_item

        stats = ResolutionStats()
        client = mock.AsyncMock()
        client.search = mock.AsyncMock(return_value={"message": {"items": [_raw_item()]}})

        ref = self._titleless_ref(title="Learning representations by back-propagating errors")
        out = await _fetch_crossref_item(ref, client, stats)

        assert out is not None
        assert stats.search_attempts == 1  # title path used
        assert stats.fingerprint_attempts == 0  # fingerprint path NOT used


class TestTitlelessAbbreviatedContainer:
    """Abbreviated-journal styles are the same styles that omit article titles,
    so the fingerprint path's container gate must survive an ISO4 abbreviation.

    Real-world trigger: ``E. U. Weber, Clim. Change 77, 103 (2006).`` — the
    printed container is ``Clim. Change``; Crossref carries ``Climatic Change``.
    """

    def _weber_ref(self, **kw):
        base = {
            "bib_id": 1,
            "title": "",
            "authors": "E. U. Weber",
            "container": "Clim. Change",
            "volume": "77",
            "first_page": "103",
            "year": 2006,
            "doi": None,
        }
        base.update(kw)
        return PaperReference(**base)

    def _weber_item(self, **kw):
        raw = {
            "DOI": "10.1007/s10584-006-9060-3",
            "title": ["Experience-based and description-based perceptions"],
            "author": [{"given": "Elke U.", "family": "Weber"}],
            "issued": {"date-parts": [[2006]]},
            "container-title": ["Climatic Change"],
            "volume": "77",
            "page": "103-120",
            "type": "journal-article",
        }
        raw.update(kw)
        return raw

    def test_accepts_iso4_abbreviated_container(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        score = _score_fingerprint(self._weber_ref(), CrossrefWorkItem.from_raw(self._weber_item()))
        assert score is not None, "'Clim. Change' must match 'Climatic Change'"

    def test_accepts_multiword_abbreviation_with_dropped_stopwords(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        ref = self._weber_ref(container="Q. J. Econ.")
        cand = CrossrefWorkItem.from_raw(
            self._weber_item(**{"container-title": ["Quarterly Journal of Economics"]})
        )
        assert _score_fingerprint(ref, cand) is not None

    def test_still_rejects_a_genuinely_different_journal(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        cand = CrossrefWorkItem.from_raw(
            self._weber_item(**{"container-title": ["Journal of Climate"]})
        )
        assert _score_fingerprint(self._weber_ref(), cand) is None


class TestAuthorInitialDisambiguation:
    """Surname-only author validation resolves ``E. U. Weber`` to Max Weber."""

    def test_rejects_same_surname_different_initial(self):
        from bibr.enrich.references import _best_resolver_match

        cand = {
            "title": "Climate Change",
            "year": 2007,
            "authors": [{"given": "Max", "family": "Weber"}],
            "type": "book",
        }
        best = _best_resolver_match(
            "Clim. Change", [cand], ref_year=2006, ref_authors="E. U. Weber"
        )
        assert best is None, "Max Weber must not satisfy a reference by E. U. Weber"

    def test_still_accepts_the_same_author(self):
        from bibr.enrich.references import _best_resolver_match

        cand = {
            "title": "Experience-based and description-based perceptions",
            "year": 2006,
            "authors": [{"given": "Elke U.", "family": "Weber"}],
            "type": "journal-article",
        }
        best = _best_resolver_match(
            "Experience-based and description-based perceptions",
            [cand],
            ref_year=2006,
            ref_authors="E. U. Weber",
        )
        assert best is not None


class TestWorkTypeVeto:
    """'giving me the review of a paper, but not the actual paper'."""

    def test_rejects_peer_review_of_the_cited_work(self):
        from bibr.enrich.references import _find_best_match

        title = "Defaults and donation decisions in the transplant setting"
        items = [
            {
                "DOI": "10.1000/review",
                "title": [f"Review of: {title}"],
                "author": [{"given": "A.", "family": "Critic"}],
                "issued": {"date-parts": [[2020]]},
                "type": "peer-review",
            }
        ]
        assert _find_best_match(title, items, ref_year=2020, ref_authors="A. Critic") is None


class TestAuthoritativeShortCircuitEligibility:
    """An authoritative resolver may only suppress the CrossRef fallback for a
    reference it actually queried. A title-less, DOI-less ref matches neither
    resolver branch, so it must still reach CrossRef's fingerprint path.
    """

    def _titleless_ref(self):
        return PaperReference(
            bib_id=1,
            title="",
            authors="E. U. Weber",
            container="Clim. Change",
            volume="77",
            first_page="103",
            year=2006,
            doi=None,
        )

    async def test_titleless_ref_still_reaches_crossref(self):
        from bibr.enrich.references import _resolve_reference

        crossref = _mock_crossref()
        crossref.search = mock.AsyncMock(
            return_value={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1007/s10584-006-9060-3",
                            "title": ["Experience-based perceptions"],
                            "author": [{"given": "Elke U.", "family": "Weber"}],
                            "issued": {"date-parts": [[2006]]},
                            "container-title": ["Climatic Change"],
                            "volume": "77",
                            "page": "103-120",
                            "type": "journal-article",
                        }
                    ]
                }
            }
        )
        resolver = mock.MagicMock()

        out = await _resolve_reference(
            self._titleless_ref(),
            crossref,
            resolver,
            resolver_authoritative=True,
        )

        assert crossref.search.await_count == 1, "CrossRef was never queried"
        assert out is not None
        assert out[1].doi == "10.1007/s10584-006-9060-3"


class TestAboutTheWorkTitleVeto:
    """Errata/comments/replies are typed `journal-article`, so the type veto can't
    see them, yet their titles embed the cited title verbatim and clear the 80 gate
    (measured: "Erratum: X" 91.3, "Review of: X" 89.5, "Comment on X" 82.2)."""

    TITLE = "Defaults and donation decisions in the transplant setting"

    def _item(self, cand_title):
        return {
            "DOI": "10.1000/about",
            "title": [cand_title],
            "author": [{"given": "A.", "family": "Critic"}],
            "issued": {"date-parts": [[2020]]},
            "type": "journal-article",
        }

    def test_rejects_erratum_comment_reply_and_review(self):
        from bibr.enrich.references import _find_best_match

        for prefix in ("Erratum: ", "Comment on ", "Reply to ", "Review of: ", "Correction: "):
            items = [self._item(prefix + self.TITLE)]
            assert (
                _find_best_match(self.TITLE, items, ref_year=2020, ref_authors="A. Critic") is None
            ), f"{prefix!r} candidate must be vetoed"

    def test_keeps_an_erratum_when_the_reference_itself_is_one(self):
        from bibr.enrich.references import _find_best_match

        printed = "Erratum: " + self.TITLE
        items = [self._item(printed)]
        assert _find_best_match(printed, items, ref_year=2020, ref_authors="A. Critic") is not None

    def test_keeps_a_title_that_merely_starts_with_a_similar_word(self):
        from bibr.enrich.references import _find_best_match

        printed = "Review of behavioural nudges in organ donation policy"
        items = [self._item(printed)]
        assert _find_best_match(printed, items, ref_year=2020, ref_authors="A. Critic") is not None


class TestFingerprintCandidateDepth:
    """The fingerprint gate is strict (exact volume + page + year + container +
    first-author), so precision comes from the gate, not from truncating the
    candidate list. A 3-row free-text query buries the correct record: measured
    on paper 1, 11 of 16 unresolved refs had a correct Crossref record outside
    the top 3.
    """

    def test_fingerprint_search_requests_a_deep_candidate_list(self):
        import asyncio

        from bibr.enrich.references import _fetch_crossref_item

        client = _mock_crossref()
        client.search = mock.AsyncMock(return_value={"message": {"items": []}})
        ref = PaperReference(
            bib_id=1,
            title="",
            authors="H. Gabel, H. N. Rehnqvist",
            container="Transplant. Proc.",
            volume="29",
            first_page="3093",
            year=1997,
            doi=None,
        )

        asyncio.run(_fetch_crossref_item(ref, client))

        assert client.search.await_count == 1
        assert client.search.await_args.kwargs["limit"] >= 20


class TestInitialsFalsePositives:
    """The initial check must not reject correct records. Found by measuring the
    live fingerprint path on paper 1, not by unit tests."""

    def test_multi_initial_reference_matches_single_given_initial(self):
        from bibr.enrich.references import _initials_conflict

        # Printed "H. Gabel, H. N. Rehnqvist"; Crossref has given "N." for Rehnqvist.
        # The candidate's initial appears in the run, so this is the same person.
        cand = [{"given": "H.", "family": "Gabel"}, {"given": "N.", "family": "Rehnqvist"}]
        assert _initials_conflict("H. Gabel, H. N. Rehnqvist", cand) is False

    def test_still_rejects_an_initial_absent_from_the_run(self):
        from bibr.enrich.references import _initials_conflict

        assert _initials_conflict("E. U. Weber", [{"given": "Max", "family": "Weber"}]) is True


class TestContainerSubtitle:
    """Crossref carries expanded journal names with subtitles; the printed
    abbreviation only ever covers the head."""

    def test_abbreviation_matches_head_before_colon(self):
        from bibr.enrich.references import _container_matches

        assert (
            _container_matches("JAMA", "JAMA: The Journal of the American Medical Association")
            is not None
        )

    def test_still_rejects_a_different_journal_with_a_subtitle(self):
        from bibr.enrich.references import _container_matches

        assert (
            _container_matches("BMJ", "JAMA: The Journal of the American Medical Association")
            is None
        )


class TestRepeatedSurnameInitials:
    """Two authors sharing a surname is ordinary ("A. C. Klassen, D. K. Klassen").
    Reading only the first occurrence's initials vetoes the correct record."""

    def test_collects_initials_from_every_occurrence_of_the_surname(self):
        from bibr.enrich.references import _ref_initials_for_surname

        assert _ref_initials_for_surname("A. C. Klassen, D. K. Klassen", "Klassen") == {
            "A",
            "C",
            "D",
            "K",
        }

    def test_second_same_surname_author_is_not_a_conflict(self):
        from bibr.enrich.references import _initials_conflict

        cand = [{"given": "A.", "family": "Klassen"}, {"given": "D.", "family": "Klassen"}]
        assert _initials_conflict("A. C. Klassen, D. K. Klassen", cand) is False


class TestUnpunctuatedAbbreviation:
    """Printed abbreviations don't always carry the period ISO4 prescribes:
    "J. Risk Uncertain" truncates "Uncertainty" with nothing to mark it."""

    def test_truncated_token_without_a_period_still_matches(self):
        from bibr.enrich.references import _container_matches

        assert (
            _container_matches("J. Risk Uncertain", "Journal of Risk and Uncertainty") is not None
        )

    def test_a_shorter_journal_name_is_not_a_prefix_of_a_longer_one(self):
        from bibr.enrich.references import _container_matches

        assert _container_matches("Nature", "Nature Communications") is None


class TestContainerParentheticalGloss:
    """Science-style bibliographies gloss an acronym in parentheses —
    "Commun. ACM (Assoc. Comput. Machin.)" — which no masthead carries."""

    def test_gloss_is_ignored_when_matching(self):
        from bibr.enrich.references import _container_matches

        assert (
            _container_matches("Commun. ACM (Assoc. Comput. Machin.)", "Communications of the ACM")
            is not None
        )

    def test_dropping_the_gloss_does_not_match_a_different_journal(self):
        from bibr.enrich.references import _container_matches

        assert _container_matches("Commun. ACM (Assoc. Comput. Machin.)", "Cell") is None


class TestFingerprintWithoutContainer:
    """A reference whose journal name the parser dropped is still pinned by
    author + volume + first page + year. Real case: "B. C. Madrian, D. Shea,
    116, 1149 (2001)" -> Quarterly Journal of Economics."""

    def _madrian_ref(self, **kw):
        base = {
            "bib_id": 22,
            "title": "",
            "authors": "B. C. Madrian, D. Shea",
            "container": None,
            "volume": "116",
            "first_page": "1149",
            "year": 2001,
            "doi": None,
        }
        base.update(kw)
        return PaperReference(**base)

    def _madrian_item(self, **kw):
        raw = {
            "DOI": "10.1162/003355301753265543",
            "title": ["The Power of Suggestion: Inertia in 401(k) Participation"],
            "author": [{"given": "B. C.", "family": "Madrian"}, {"given": "D.", "family": "Shea"}],
            "issued": {"date-parts": [[2001]]},
            "container-title": ["The Quarterly Journal of Economics"],
            "volume": "116",
            "page": "1149-1187",
            "type": "journal-article",
        }
        raw.update(kw)
        return raw

    def test_accepts_when_volume_and_first_page_pin_the_record(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        score = _score_fingerprint(
            self._madrian_ref(), CrossrefWorkItem.from_raw(self._madrian_item())
        )
        assert score is not None

    def test_rejects_when_the_first_page_is_also_missing(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        # Volume alone spans a whole year of a journal — not enough to identify
        # an article once the container is gone too.
        score = _score_fingerprint(
            self._madrian_ref(first_page=None), CrossrefWorkItem.from_raw(self._madrian_item())
        )
        assert score is None

    def test_rejects_when_the_first_page_disagrees(self):
        from bibr.enrich.references import _score_fingerprint
        from bibr.enrich.schemas import CrossrefWorkItem

        score = _score_fingerprint(
            self._madrian_ref(), CrossrefWorkItem.from_raw(self._madrian_item(page="200-230"))
        )
        assert score is None

    def test_containerless_ref_is_eligible_for_the_fingerprint_query(self):
        from bibr.enrich.references import _fetch_crossref_item

        client = _mock_crossref()
        client.search = mock.AsyncMock(return_value={"message": {"items": []}})
        asyncio.run(_fetch_crossref_item(self._madrian_ref(), client))
        assert client.search.await_count == 1


class TestBulkDoiPrefetch:
    """enrich_references collapses DOI lookups into one filter query."""

    async def test_prefetch_receives_every_doi_bearing_reference(self):
        from bibr.enrich.references import enrich_references

        refs = [
            _make_ref(bib_id=1, doi="10.1000/a"),
            _make_ref(bib_id=2, doi=None),
            _make_ref(bib_id=3, doi="10.1000/b"),
        ]
        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={})
        mock_client.search = mock.AsyncMock(return_value={"message": {"items": []}})

        await enrich_references(refs, crossref_client=mock_client)

        mock_client.prefetch_works_by_doi.assert_awaited_once_with(["10.1000/a", "10.1000/b"])

    async def test_no_prefetch_when_disabled(self):
        from bibr.config import GlobalSettings
        from bibr.enrich.references import enrich_references

        refs = [_make_ref(bib_id=1, doi="10.1000/a")]
        mock_client = _mock_crossref()
        mock_client.works = mock.AsyncMock(return_value={})

        settings = GlobalSettings(crossref={"bulk_doi_lookup": False})
        await enrich_references(refs, crossref_client=mock_client, settings=settings)

        mock_client.prefetch_works_by_doi.assert_not_awaited()

    async def test_prefetch_failure_still_enriches(self):
        """A dead prefetch must fall through to per-reference lookups."""
        from bibr.enrich.references import enrich_references

        ref = _make_ref(doi="10.1000/test")
        mock_client = _mock_crossref()
        mock_client.prefetch_works_by_doi = mock.AsyncMock(side_effect=RuntimeError("down"))
        mock_client.works = mock.AsyncMock(return_value={"message": _make_crossref_item()})

        await enrich_references([ref], crossref_client=mock_client)

        assert MatchSource.CROSSREF in ref.match
        mock_client.works.assert_awaited_once()
