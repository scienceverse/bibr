"""Tests for enrich_paper_identity — self-DOI enrichment of the paper's own identity."""

from unittest import mock

import httpx

from bibr.enrich.references import enrich_paper_identity
from bibr.paper import ExternalMatch, MatchSource, PaperMetadata


def _crossref_item(doi="10.1234/self", year=2020):
    return {
        "DOI": doi,
        "title": ["Test Paper"],
        "author": [{"given": "A", "family": "B"}],
        "issued": {"date-parts": [[year]]},
        "container-title": ["Psychological Science"],
        "volume": "31",
        "issue": "1",
        "page": "65-74",
        "publisher": "SAGE Publications",
        "URL": f"https://doi.org/{doi}",
        "type": "journal-article",
    }


class TestEnrichPaperIdentity:
    async def test_self_doi_lookup_populates_match(self):
        meta = PaperMetadata(doi="10.1234/self", title="Test Paper", published="2020")
        crossref = mock.AsyncMock()
        crossref.works = mock.AsyncMock(return_value={"message": _crossref_item()})

        await enrich_paper_identity(meta, crossref_client=crossref)

        assert MatchSource.CROSSREF in meta.match
        m = meta.match[MatchSource.CROSSREF]
        assert isinstance(m, ExternalMatch)
        assert m.doi == "10.1234/self"
        assert m.score == 100.0
        assert m.container == "Psychological Science"
        assert m.volume == "31"

    async def test_printed_fields_not_overwritten(self):
        # Enrichment lives only in match; the printed self-identity is untouched.
        meta = PaperMetadata(
            doi="10.1234/self",
            title="Test Paper",
            journal="As Printed",
            volume="99",
        )
        crossref = mock.AsyncMock()
        crossref.works = mock.AsyncMock(return_value={"message": _crossref_item()})

        await enrich_paper_identity(meta, crossref_client=crossref)

        assert meta.journal == "As Printed"
        assert meta.volume == "99"
        assert meta.match[MatchSource.CROSSREF].volume == "31"

    async def test_only_one_lookup_no_title_search(self):
        meta = PaperMetadata(doi="10.1234/self", title="A Long Real Title Here")
        crossref = mock.AsyncMock()
        crossref.works = mock.AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=mock.Mock(), response=mock.Mock(status_code=404)
            )
        )
        crossref.search = mock.AsyncMock()

        await enrich_paper_identity(meta, crossref_client=crossref)

        # DOI lookup 404 → no match; title search must NOT be attempted.
        assert meta.match == {}
        crossref.search.assert_not_called()

    async def test_lookup_failure_leaves_match_empty(self):
        meta = PaperMetadata(doi="10.1234/self", title="Test Paper")
        crossref = mock.AsyncMock()
        crossref.works = mock.AsyncMock(side_effect=RuntimeError("boom"))
        crossref.search = mock.AsyncMock(return_value=None)

        await enrich_paper_identity(meta, crossref_client=crossref)

        assert meta.match == {}

    async def test_no_doi_skips(self):
        meta = PaperMetadata(doi="", title="Test Paper")
        crossref = mock.AsyncMock()
        crossref.works = mock.AsyncMock()

        await enrich_paper_identity(meta, crossref_client=crossref)

        assert meta.match == {}
        crossref.works.assert_not_called()

    async def test_authoritative_resolver_miss_skips_crossref(self, monkeypatch):
        from bibr.config import Settings

        monkeypatch.setattr(Settings.resolver, "authoritative", True)

        meta = PaperMetadata(doi="10.1234/self", title="Test Paper", published="2020")
        crossref = mock.AsyncMock()
        resolver = mock.AsyncMock()
        resolver.lookup_doi = mock.AsyncMock(return_value=None)  # clean miss

        await enrich_paper_identity(meta, crossref_client=crossref, resolver_client=resolver)

        assert meta.match == {}
        crossref.works.assert_not_called()

    async def test_resolver_preferred_over_crossref(self):
        meta = PaperMetadata(doi="10.1234/self", title="Test Paper", published="2020")
        crossref = mock.AsyncMock()
        resolver = mock.AsyncMock()
        resolver.lookup_doi = mock.AsyncMock(
            return_value={
                "doi": "10.1234/self",
                "year": 2020,
                "title": "Test Paper",
                "container": "Psychological Science",
                "source": "openalex",
            }
        )

        await enrich_paper_identity(meta, crossref_client=crossref, resolver_client=resolver)

        assert MatchSource.OPENALEX in meta.match
        assert MatchSource.CROSSREF not in meta.match
        crossref.works.assert_not_called()
