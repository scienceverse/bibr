"""Title-search acceptance: one matcher for Crossref and the resolver.

The Crossref search, the resolver's primary search and the resolver fallback
used to run separate copies of the candidate loop. These pin the rules they now
share: the initials veto reads every citation style, surnames compare without
diacritics, organization authors are no evidence against a match, a printed DOI
admits only its own work, printed volume/page/container can rule a candidate
out, and deposited title markup neither costs the match nor leaks into it.
"""

import asyncio
from unittest import mock

import httpx
import pytest

from bibr.clients.crossref import CrossrefClient
from bibr.config import GlobalSettings, ResolverOptions
from bibr.paper import MatchSource, PaperReference
from bibr.processing_warnings import WarningCode

TITLE = "Mindless eating: The 200 daily food decisions we overlook"


def _ref(**kw):
    base = {
        "bib_id": 1,
        "title": TITLE,
        "first_page": None,
        "volume": None,
        "authors": None,
        "year": 2007,
        "container": None,
    }
    base.update(kw)
    return PaperReference(**base)


def _item(doi="10.1000/exact", title=TITLE, authors=None, year=2007, **extra):
    raw = {
        "DOI": doi,
        "title": [title],
        "author": authors
        if authors is not None
        else [{"given": "Brian", "family": "Wansink"}, {"given": "Jeffery", "family": "Sobal"}],
        "issued": {"date-parts": [[year]]},
        "type": "journal-article",
    }
    raw.update(extra)
    return raw


def _crossref(items, works_error=None):
    client = mock.MagicMock(spec_set=CrossrefClient)
    client.enrich_semaphore = asyncio.Semaphore(100)
    client.prefetch_works_by_doi = mock.AsyncMock()
    client.search = mock.AsyncMock(return_value={"message": {"items": items}})
    if works_error is not None:
        client.works = mock.AsyncMock(side_effect=works_error)
    return client


async def _crossref_match(ref, items, works_error=None):
    from bibr.enrich.references import enrich_references

    client = _crossref(items, works_error)
    await enrich_references([ref], crossref_client=client, settings=GlobalSettings(_env_file=None))
    match = ref.match.get(MatchSource.CROSSREF)
    return None if match is None else match.doi


class TestInitialsAcrossCitationStyles:
    """clients-external-enrich-1: the veto took "and" for a first name and cut
    "Jeffery M. Sobal" down to {M}, rejecting the exact record."""

    @pytest.mark.parametrize(
        "authors",
        [
            "Wansink, B. and Sobal, J.",  # Harvard
            "Wansink, B., and Sobal, J.",
            "Wansink, Brian, and Jeffery M. Sobal",  # Chicago / MLA
            "Wansink, B. et Sobal, J.",  # French
            "Wansink, B. und Sobal, J.",  # German
            "Wansink, B. Sobal, J.",  # separator lost in extraction
        ],
    )
    async def test_exact_record_is_accepted(self, authors):
        assert await _crossref_match(_ref(authors=authors), [_item()]) == "10.1000/exact"

    @pytest.mark.parametrize(
        ("authors", "family", "expected"),
        [
            ("Smith, J. and Weber, E. U.", "Weber", {"E", "U"}),
            ("Weber, Elke U., and Eric J. Johnson", "Johnson", {"E", "J"}),
            ("Abbott, Will, Thomas E. Brownlee, Liam D. Harper", "Brownlee", {"T", "E"}),
            ("Abbott, Will, Thomas E. Brownlee, Liam D. Harper", "Abbott", {"W"}),
            ("Lam, Hui Kwan Nicholas, John Sproule", "Lam", {"H", "K", "N"}),
            ("Dupont, J. et Martin, P.", "Martin", {"P"}),
            ("Beg, M.U., Saeed, T., Al-Muzaini, Beg, K.R.", "Al-Muzaini", set()),
            ("Choi KH, Karkhoff-Schweizer RR, Schweizer HP", "Schweizer", set()),
            ("Boer H. Emons P.A.A.", "Emons", {"B", "H", "P", "A"}),
        ],
    )
    def test_initials_read_for_the_surname(self, authors, family, expected):
        from bibr.enrich.references import _ref_initials_for_surname

        assert _ref_initials_for_surname(authors, family) == expected

    @pytest.mark.parametrize(
        ("authors", "cand"),
        [
            ("Soltis, D. E., et al", [("Douglas E.", "Soltis"), ("Pamela S.", "Soltis")]),
            ("Mohr, M. and Krolak-Salmon, P.", [("M.", "Mohr"), ("Eric", "Salmon")]),
            ("Liu, H. and Xue, R.", [("Prof Haiyue", "Liu"), ("Dr Rui", "Xue")]),
            ("Giraldo Peláez, Santiago", [("SANTIAGO GIRALDO", "PELÁEZ")]),
        ],
    )
    def test_same_person_is_not_a_conflict(self, authors, cand):
        from bibr.enrich.references import _initials_conflict

        people = [{"given": given, "family": family} for given, family in cand]
        assert _initials_conflict(authors, people) is False

    @pytest.mark.parametrize(
        "authors",
        [
            "E. U. Weber",
            "Smith, J. and Weber, E. U.",
            "Smith, J., Weber, E. U., & Jones, K.",
            "Weber, Elke U., and Eric J. Johnson",
        ],
    )
    def test_a_different_same_surname_author_is_still_vetoed(self, authors):
        from bibr.enrich.references import _initials_conflict

        assert _initials_conflict(authors, [{"given": "Max", "family": "Weber"}]) is True


class TestSurnameOverlap:
    """clients-external-enrich-2: an exact title fell to 70 against an
    organization author or a surname printed without its accents."""

    async def test_organization_author_is_no_evidence_against_the_match(self):
        who = _item(
            doi="10.1000/who",
            title="World report on violence and health",
            authors=[{"name": "World Health Organization"}],
            year=2002,
        )
        ref = _ref(
            title="World report on violence and health",
            authors="World Health Organization",
            year=2002,
        )
        assert await _crossref_match(ref, [who]) == "10.1000/who"

    def test_organization_author_passes_the_resolver_path_too(self):
        from bibr.enrich.references import _best_resolver_match

        cand = {
            "title": "World report on violence and health",
            "year": 2002,
            "authors": [{"given": "", "family": ""}],
        }
        best = _best_resolver_match(
            "World report on violence and health",
            [cand],
            ref_year=2002,
            ref_authors="World Health Organization",
        )
        assert best == (cand, 100.0)

    @pytest.mark.parametrize(
        ("printed", "deposited"),
        [
            ("González, M.", "Gonzalez"),
            ("Gonzalez, M.", "González"),
            ("Müller, K.", "Mueller"),
            ("Mueller, K.", "Müller"),
            ("Ørsted, H.", "Orsted"),
        ],
    )
    async def test_accents_and_transliterations_overlap(self, printed, deposited):
        given = printed.split(", ")[1]
        title = "A study of accents in reference matching"
        item = _item(title=title, authors=[{"given": given, "family": deposited}], year=2010)
        ref = _ref(title=title, authors=printed, year=2010)
        assert await _crossref_match(ref, [item]) == "10.1000/exact"


class TestPrintedDoiAgreement:
    """clients-external-enrich-4: after a failed DOI lookup the title search
    accepted a different work (a preprint) with its own DOI."""

    PRINTED = "10.1037/dev0000123"

    async def test_a_candidate_with_another_doi_is_rejected(self):
        preprint = _item(doi="10.9999/preprint.abc", **{"container-title": ["PsyArXiv"]})
        ref = _ref(authors="Wansink, B., & Sobal, J.", doi=self.PRINTED)
        assert await _crossref_match(ref, [preprint], httpx.ReadTimeout("slow")) is None

    async def test_the_candidate_carrying_the_printed_doi_is_accepted(self):
        items = [_item(doi="10.9999/preprint.abc"), _item(doi="10.1037/DEV0000123")]
        ref = _ref(authors="Wansink, B., & Sobal, J.", doi=self.PRINTED)
        assert await _crossref_match(ref, items, httpx.ReadTimeout("slow")) == "10.1037/DEV0000123"

    async def test_title_less_fingerprint_match_obeys_the_printed_doi(self):
        other = _item(
            doi="10.9999/other",
            title="Anything",
            authors=[{"given": "B.", "family": "Wansink"}],
            **{"container-title": ["Environment and Behavior"], "volume": "39", "page": "106-123"},
        )
        ref = _ref(
            title="",
            authors="Wansink, B.",
            container="Environment and Behavior",
            volume="39",
            first_page="106",
            doi=self.PRINTED,
        )
        assert await _crossref_match(ref, [other], httpx.ReadTimeout("slow")) is None


class TestPrintedFieldsRuleOutACandidate:
    """clients-external-enrich-5: "Introduction" by the same author in the same
    year matched any journal article of that title."""

    async def test_generic_title_in_another_container_is_rejected(self):
        article = _item(
            doi="10.1000/unrelated",
            title="Introduction",
            authors=[{"given": "John", "family": "Smith"}],
            year=2015,
            **{"container-title": ["Journal of Unrelated Studies"], "volume": "3"},
        )
        ref = _ref(
            title="Introduction",
            authors="Smith, J.",
            year=2015,
            container="Handbook of Emotion Regulation",
            first_page="1",
        )
        assert await _crossref_match(ref, [article]) is None

    async def test_generic_title_in_the_printed_container_is_accepted(self):
        chapter = _item(
            doi="10.1000/chapter",
            title="Introduction",
            authors=[{"given": "John", "family": "Smith"}],
            year=2015,
            **{"container-title": ["Handbook of Emotion Regulation"], "page": "1-5"},
        )
        ref = _ref(
            title="Introduction",
            authors="Smith, J.",
            year=2015,
            container="Handbook of Emotion Regulation",
            first_page="1",
        )
        assert await _crossref_match(ref, [chapter]) == "10.1000/chapter"

    async def test_volume_and_page_both_disagreeing_reject_any_title(self):
        reprint = _item(doi="10.1000/reprint", volume="80", page="1105-1120")
        ref = _ref(authors="Wansink, B., & Sobal, J.", volume="39", first_page="106")
        assert await _crossref_match(ref, [reprint]) is None

    async def test_one_misread_field_does_not_reject_a_distinctive_title(self):
        record = _item(volume="39", page="1-20")
        ref = _ref(authors="Wansink, B., & Sobal, J.", volume="39", first_page="106")
        assert await _crossref_match(ref, [record]) == "10.1000/exact"


class TestDepositedTitleMarkup:
    """clients-external-enrich-6: <sub>/<i> tags cost the fuzzy score and were
    exported into bib_match."""

    async def test_subscript_markup_does_not_cost_the_match(self):
        title = "Effects of elevated CO2 and O3 on N2O emissions from soil"
        deposited = (
            "Effects of elevated CO<sub>2</sub> and O<sub>3</sub> on N<sub>2</sub>O emissions"
            " from soil"
        )
        item = _item(title=deposited, authors=[{"given": "A.", "family": "Smith"}], year=2010)
        ref = _ref(title=title, authors="Smith, A.", year=2010)
        assert await _crossref_match(ref, [item]) == "10.1000/exact"

    async def test_match_carries_plain_title_and_container(self):
        from bibr.enrich.references import enrich_references

        item = _item(
            title="Wing shape in <i>Drosophila</i> &amp; flies",
            authors=[{"given": "A.", "family": "Smith"}],
            year=2010,
            **{"container-title": ["Genes &amp; Development"]},
        )
        ref = _ref(title="Wing shape in Drosophila & flies", authors="Smith, A.", year=2010)
        await enrich_references(
            [ref], crossref_client=_crossref([item]), settings=GlobalSettings(_env_file=None)
        )
        match = ref.match[MatchSource.CROSSREF]
        assert (match.title, match.container) == (
            "Wing shape in Drosophila & flies",
            "Genes & Development",
        )

    def test_resolver_candidate_markup_is_stripped_from_the_match(self):
        from bibr.enrich.references import _build_match_from_candidate

        match = _build_match_from_candidate(
            {"title": "Global C\n  <sub>2</sub>\n  H\n  <sub>6</sub>\n  maps", "doi": "10.1/x"},
            100.0,
        )
        assert match.title == "Global C2H6 maps"


def _resolver_settings(**overrides):
    values = {
        "sources": ["crossref"],
        "fallback_sources": ["openalex"],
        "fallback_search_concurrency": 2,
        "fallback_timeout": 30.0,
        "authoritative": True,
    }
    values.update(overrides)
    return GlobalSettings(_env_file=None, resolver=ResolverOptions(_env_file=None, **values))


class TestAuthoritativeResolverPrefetchFailure:
    """clients-external-enrich-7: a failed resolver prefetch left every
    title-searchable reference unmatched without asking Crossref."""

    async def test_a_failed_prefetch_falls_through_to_crossref(self):
        from bibr.enrich.references import enrich_references

        ref = _ref(authors="Wansink, B., & Sobal, J.")
        crossref = _crossref([_item()])
        resolver = mock.AsyncMock()
        resolver.search_many = mock.AsyncMock(side_effect=RuntimeError("search down"))

        report = await enrich_references(
            [ref],
            crossref_client=crossref,
            resolver_client=resolver,
            settings=_resolver_settings(fallback_sources=[]),
        )

        crossref.search.assert_awaited_once()
        assert ref.match[MatchSource.CROSSREF].doi == "10.1000/exact"
        assert (report.matched, report.failed) == (1, 0)

    def test_search_concurrency_below_one_is_rejected_at_load(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResolverOptions(_env_file=None, search_concurrency=0)


class TestResolverFallbackKeepsFinishedWork:
    """clients-external-enrich-3: the fallback deadline threw away every search
    that had already answered."""

    async def test_answered_searches_are_applied_before_the_deadline(self):
        from bibr.enrich.references import enrich_references

        refs = [
            _ref(bib_id=1, title="The first title answers quickly"),
            _ref(bib_id=2, title="The second title answers quickly"),
            _ref(bib_id=3, title="The third title never answers"),
        ]
        resolver = mock.AsyncMock()
        resolver.search_many = mock.AsyncMock(
            side_effect=lambda queries, **kw: [[] for _ in queries]
        )

        async def search(title, year, limit, **kwargs):
            if "never" in title:
                await asyncio.Event().wait()
            return [{"title": title, "year": year, "source": "openalex", "id": title[4:9]}]

        resolver.search = mock.AsyncMock(side_effect=search)

        report = await enrich_references(
            refs,
            crossref_client=_crossref([]),
            resolver_client=resolver,
            settings=_resolver_settings(fallback_timeout=0.2, fallback_search_concurrency=3),
        )

        assert [MatchSource.OPENALEX in ref.match for ref in refs] == [True, True, False]
        assert [(d.code, d.message) for d in report.details] == [
            (
                WarningCode.RESOLVER_FALLBACK_TIMEOUT,
                "resolver fallback timed out after 0.2s with 1 of 3 refs unanswered",
            )
        ]


class TestMalformedResolverCandidates:
    """clients-external-enrich-8: one candidate with ``"authors": null`` aborted
    the fallback for every remaining reference."""

    @pytest.mark.parametrize("malformed", [None, "not a list", [None, 3]])
    async def test_malformed_authors_cost_nothing_and_later_refs_still_match(self, malformed):
        from bibr.enrich.references import enrich_references

        refs = [
            _ref(bib_id=1, title="The first title that is long enough", authors="Smith, J."),
            _ref(bib_id=2, title="The second title that is long enough", authors="Smith, J."),
        ]
        resolver = mock.AsyncMock()
        resolver.search_many = mock.AsyncMock(
            side_effect=lambda queries, **kw: [[] for _ in queries]
        )

        async def search(title, year, limit, **kwargs):
            authors = malformed if title.startswith("The first") else [{"family": "Smith"}]
            return [{"title": title, "year": year, "authors": authors, "source": "openalex"}]

        resolver.search = mock.AsyncMock(side_effect=search)

        report = await enrich_references(
            refs,
            crossref_client=_crossref([]),
            resolver_client=resolver,
            settings=_resolver_settings(),
        )

        assert [MatchSource.OPENALEX in ref.match for ref in refs] == [True, True]
        assert report.details == ()

    async def test_an_error_on_one_reference_costs_only_that_reference(self, monkeypatch):
        from bibr.enrich import references
        from bibr.enrich.references import enrich_references

        real = references._best_resolver_match

        def flaky(title, candidates, **kwargs):
            if candidates and title.startswith("The first"):
                raise ValueError("unexpected candidate shape")
            return real(title, candidates, **kwargs)

        monkeypatch.setattr(references, "_best_resolver_match", flaky)
        refs = [
            _ref(bib_id=1, title="The first title that is long enough"),
            _ref(bib_id=2, title="The second title that is long enough"),
        ]
        resolver = mock.AsyncMock()
        resolver.search_many = mock.AsyncMock(
            side_effect=lambda queries, **kw: [[] for _ in queries]
        )
        resolver.search = mock.AsyncMock(
            side_effect=lambda title, year, limit, **kw: [
                {"title": title, "year": year, "source": "openalex"}
            ]
        )

        report = await enrich_references(
            refs,
            crossref_client=_crossref([]),
            resolver_client=resolver,
            settings=_resolver_settings(),
        )

        assert [MatchSource.OPENALEX in ref.match for ref in refs] == [False, True]
        assert report.failed == 0
        assert [(d.code, d.message) for d in report.details] == [
            (
                WarningCode.RESOLVER_FALLBACK_FAILED,
                "bib_id=1 resolver fallback failed: unexpected candidate shape",
            )
        ]
