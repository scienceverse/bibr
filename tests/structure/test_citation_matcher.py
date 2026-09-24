"""Tests for bibr.structure.citation_matcher — reference-anchored Tier 2."""

import pytest

from bibr.models import PaperReference
from bibr.paper_contents import PaperSentence
from bibr.structure.citation_matcher import (
    _family_keys_for,
    detect_author_year_xrefs,
    extract_families,
    match_with_candidates,
)


def _ref(bib_id, authors, year=2020, year_suffix=None, is_in_press=False):
    return PaperReference(
        bib_id=bib_id,
        title="",
        first_page=None,
        volume=None,
        authors=authors,
        year=year,
        container=None,
        year_suffix=year_suffix,
        is_in_press=is_in_press,
    )


def _sent(text_id, text):
    return PaperSentence(text_id=text_id, text=text, section_id=1, paragraph_id=1)


class TestExtractFamilies:
    def test_apa_single(self):
        assert extract_families("Smith, J.") == ["Smith"]

    def test_apa_multi(self):
        assert extract_families("Smith, J. A., Jones, B. C., & Brown, D.") == [
            "Smith",
            "Jones",
            "Brown",
        ]

    def test_multi_word_family(self):
        assert extract_families("Della Sala, S.") == ["Della Sala"]

    def test_lowercase_particle(self):
        assert extract_families("van der Vyver, J. M.") == ["van der Vyver"]

    def test_continuation_dots(self):
        out = extract_families("Booij, L., Van der Does, W., . . Van der Kloot, W. A")
        assert "Booij" in out
        assert "Van der Kloot" in out

    def test_hyphenated(self):
        assert extract_families("Fei-Fei, L.") == ["Fei-Fei"]


class TestFamilyKeys:
    def test_simple(self):
        assert _family_keys_for("Smith") == {"smith"}

    def test_possessive_stripped(self):
        assert _family_keys_for("Perruchet's") == {"perruchet"}

    def test_multi_word_indexes_last(self):
        assert _family_keys_for("van der Toorn") == {"van der toorn", "toorn"}

    def test_della_sala(self):
        assert _family_keys_for("Della Sala") == {"della sala", "sala"}


class TestRefIndexKeySets:
    def test_family_key_sets_precomputed(self):
        from bibr.structure.citation_matcher import _build_ref_index

        refs, _, _ = _build_ref_index([_ref(1, "Smith, J., & van der Toorn, J.", year=2020)])
        assert refs[0].family_key_sets == (
            frozenset(_family_keys_for("Smith")),
            frozenset(_family_keys_for("van der Toorn")),
        )


class TestThreePlusAuthorParen:
    """Current Tier 2 regex caps at 2 authors before '&'. New matcher handles N."""

    def test_three_authors(self):
        refs = [_ref(1, "Patton, G., Stanford, S., & Barratt, B.", year=1995)]
        sents = [
            _sent(0, "We used the Barratt Impulsiveness Scale (Patton, Stanford, & Barratt, 1995).")
        ]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1
        assert out[0].xref_id == 1
        assert out[0].text_id == 0

    def test_five_authors(self):
        refs = [
            _ref(
                1,
                "Basso, A., Capitani, E., Della Sala, S., Laiacona, M., & Spinnler, H.",
                year=1987,
            )
        ]
        sents = [
            _sent(
                0,
                "Lesion studies (Basso, Capitani, Della Sala, Laiacona, & Spinnler, 1987) confirm.",
            )
        ]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1 and out[0].xref_id == 1


class TestEtAlDisambiguation:
    """A bare 'et al.' must not manufacture certainty from list length."""

    def test_et_al_abstains_for_same_first_family_and_year(self):
        from bibr.structure import citation_matcher as cm

        refs = [
            PaperReference.model_validate(
                {
                    "bib_id": 23,
                    "title": "t",
                    # Non-APA: parsed as 3 families (Pollnac, Richard B., John J. Poggie)
                    "authors": "Pollnac, Richard B., and John J. Poggie",
                    "year": 2008,
                    "first_page": None,
                    "volume": None,
                    "container": None,
                }
            ),
            PaperReference.model_validate(
                {
                    "bib_id": 26,
                    "title": "t",
                    "authors": "Pollnac, Richard B., Susan Abbott-Jamieson, Courtland Smith, Marc L. Miller, Patricia M. Clay, and Bryan Oles",
                    "year": 2008,
                    "first_page": None,
                    "volume": None,
                    "container": None,
                }
            ),
        ]
        _r, by_year, ip = cm._build_ref_index(refs)
        cand = next(
            c for c in cm._find_candidates("As Pollnac et al. (2008) showed,", 1) if c.year == 2008
        )
        ref, tied = cm._match(cand, by_year, ip)
        assert ref is None
        assert set(tied) == {23, 26}


class TestSameFirstAuthorYearDisambiguation:
    """Two refs with same first author + year — must use 2nd author to pick."""

    def test_picks_correct_two_author_match(self):
        refs = [
            _ref(1, "Smith, J., & Jones, K.", year=2020),
            _ref(2, "Smith, J., & Williams, L.", year=2020),
        ]
        sents = [_sent(0, "As shown (Smith & Williams, 2020) this works.")]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1
        assert out[0].xref_id == 2

    def test_ambiguous_returns_none(self):
        """When the cite has only the first author and 2 refs match equally,
        prefer no match (let Tier 3 disambiguate)."""
        refs = [
            _ref(1, "Smith, J., & Jones, K.", year=2020),
            _ref(2, "Smith, J., & Williams, L.", year=2020),
        ]
        sents = [_sent(0, "As shown (Smith, 2020) this works.")]
        out = detect_author_year_xrefs(sents, refs)
        assert out == []


class TestYearSuffix:
    def test_suffix_disambiguation(self):
        refs = [
            _ref(1, "Greene, M., & Oliva, A.", year=2009, year_suffix="a"),
            _ref(2, "Greene, M., & Oliva, A.", year=2009, year_suffix="b"),
        ]
        sents = [_sent(0, "Following (Greene & Oliva, 2009b) closely.")]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1
        assert out[0].xref_id == 2

    @staticmethod
    def _match(text, refs):
        xrefs, ambiguous, _ = match_with_candidates([_sent(0, text)], refs)
        return [x.xref_id for x in xrefs], [a.candidate_bib_ids for a in ambiguous]

    @pytest.mark.parametrize(("cite", "bib_id"), [("2020a", 1), ("2020b", 2)])
    def test_suffixed_citation_links_its_own_entry(self, cite, bib_id):
        refs = [
            _ref(1, "Smith, J., & Jones, K.", year_suffix="a"),
            _ref(2, "Smith, J., & Jones, K.", year_suffix="b"),
        ]
        assert self._match(f"As shown (Smith & Jones, {cite}).", refs) == ([bib_id], [])
        assert self._match(f"As shown (Smith, {cite}).", refs) == ([bib_id], [])

    def test_suffixless_citation_links_a_lone_suffixed_entry(self):
        refs = [_ref(1, "Smith, J.", year_suffix="a"), _ref(2, "Jones, K.")]
        assert self._match("As shown (Smith, 2020).", refs) == ([1], [])

    @pytest.mark.parametrize("text", ["(Smith, 2020)", "(Smith & Jones, 2020)"])
    def test_suffixless_citation_leaves_a_and_b_ambiguous(self, text):
        refs = [
            _ref(1, "Smith, J., & Jones, K.", year_suffix="a"),
            _ref(2, "Smith, J., & Jones, K.", year_suffix="b"),
        ]
        assert self._match(f"As shown {text}.", refs) == ([], [[1, 2]])

    def test_suffixed_citation_links_an_entry_without_a_suffix(self):
        refs = [_ref(1, "Smith, J."), _ref(2, "Jones, K.")]
        assert self._match("As shown (Smith, 2020b).", refs) == ([1], [])

    def test_suffixed_citation_falls_back_when_no_entry_of_its_authors_has_it(self):
        # Only another author's entry carries "a"; the author's own entry was
        # printed (or parsed) with a different letter.
        refs = [_ref(1, "Smith, J.", year_suffix="a"), _ref(2, "Jones, K.", year_suffix="b")]
        assert self._match("As shown (Jones, 2020a).", refs) == ([2], [])

    def test_matching_ignores_suffixes_when_no_entry_has_one(self):
        refs = [_ref(1, "Smith, J."), _ref(2, "Smith, J.")]
        assert self._match("As shown (Smith, 2020).", refs) == ([], [[1, 2]])
        assert self._match("As shown (Smith, 2020a).", refs) == ([], [[1, 2]])


class TestMultiYearCite:
    def test_potter_1975_1976(self):
        refs = [
            _ref(1, "Potter, M.", year=1975),
            _ref(2, "Potter, M.", year=1976),
        ]
        sents = [_sent(0, "Within 100 ms the gist is gleaned (Potter, 1975, 1976).")]
        out = detect_author_year_xrefs(sents, refs)
        assert {x.xref_id for x in out} == {1, 2}


class TestNarrativeEtAl:
    """Old regex broke on 'Author et al. (Year)' because of '.' in prefix."""

    def test_simple(self):
        refs = [_ref(1, "Holmes, J., Smith, A., Brown, B.", year=2009)]
        sents = [_sent(0, "Following Holmes et al. (2009) closely we find.")]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1 and out[0].xref_id == 1

    def test_two_in_one_sentence(self):
        refs = [
            _ref(1, "Xu, Y., Smith, A., Brown, B.", year=2001),
            _ref(2, "Kronbichler, M., Smith, A., Brown, B.", year=2009),
        ]
        sents = [_sent(0, "Findings of Xu et al. (2001) and Kronbichler et al. (2009) showed.")]
        out = detect_author_year_xrefs(sents, refs)
        assert {x.xref_id for x in out} == {1, 2}


class TestMultiWordFamily:
    def test_van_der(self):
        """Cite says 'Toorn' (last word); ref family is 'van der Toorn'."""
        refs = [_ref(1, "van der Toorn, J., Nail, L., Liviatan, J., & Jost, J.", year=2014)]
        sents = [
            _sent(0, "Recently, van der Toorn, Nail, Liviatan, and Jost (2014) showed evidence.")
        ]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1 and out[0].xref_id == 1


class TestSentenceBoundary:
    def test_no_leak_across_period(self):
        """A previous sentence ending in capital word must not be picked up as
        the prefix for a narrative cite in the next sentence."""
        refs = [_ref(1, "Smith, J.", year=2020)]
        sents = [_sent(0, "X said Y. Then Smith (2020) found Z.")]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1 and out[0].xref_id == 1


class TestContentsWhitespaceNormalized:
    """xref.contents must collapse line-wrap artifacts (\\r\\n / NBSP) the
    candidate span picked up from a hard-wrapped citation."""

    def test_crlf_wrapped_citation_collapsed(self):
        refs = [_ref(1, "Møller, A. P.", year=2006)]
        sents = [_sent(0, "Bird song complexity (Møller\r\net al., 2006) varies widely.")]
        out = detect_author_year_xrefs(sents, refs)
        assert len(out) == 1
        assert out[0].xref_id == 1
        assert "\r" not in out[0].contents and "\n" not in out[0].contents
        assert out[0].contents == "(Møller et al., 2006)"


class TestNoMatch:
    def test_unknown_author(self):
        refs = [_ref(1, "Smith, J.", year=2020)]
        sents = [_sent(0, "As shown (Unknown, 1999) this is rare.")]
        assert detect_author_year_xrefs(sents, refs) == []

    def test_year_mismatch(self):
        refs = [_ref(1, "Smith, J.", year=2020)]
        sents = [_sent(0, "Smith (2019) said something.")]
        assert detect_author_year_xrefs(sents, refs) == []


def test_long_multicite_paren_not_dropped():
    from bibr.structure import citation_matcher as cm

    # Real 216-char 7-work parenthetical (psych gold 0956797619890619, t19).
    s = (
        "These stereotypes are reflected in the statistics of texts "
        "(Caliskan, Bryson, & Narayanan, 2017), and like many other factors "
        "(Altmann & Steedman, 1988; Bicknell, Elman, Hare, McRae, & Kutas, 2010; "
        "Kuperberg & Jaeger, 2016; Marslen-Wilson, 1975; Nieuwland & Van Berkum, 2006; "
        "Tanenhaus, Spivey-Knowlton, Eberhard, & Sedivy, 1995; Traxler, 2014), "
        "language processing reflects them."
    )
    cands = cm._find_candidates(s, 19)
    fams = {f for c in cands for f in c.families}
    # 'Marslen-Wilson' sits inside the long (>200) paren; with the old cap it is dropped.
    assert "Marslen-Wilson" in fams
    assert any(c.year == 1975 for c in cands)


def test_narrative_year_paren_with_trailing_content():
    from bibr.structure import citation_matcher as cm

    s = "Sample sizes were based on power analyses following Finke et al. (2010; Study 1) and Ruiz-Rizzo et al. (2018; Study 2)."
    cands = cm._find_candidates(s, 63)
    by_year = {c.year: c for c in cands}
    assert 2010 in by_year and "Finke" in by_year[2010].families
    assert 2018 in by_year and "Ruiz-Rizzo" in by_year[2018].families


def test_narrative_multi_year_paren():
    from bibr.structure import citation_matcher as cm

    s = "This was shown by Smith (2010a, 2010b)."
    cands = cm._find_candidates(s, 1)
    years = sorted((c.year, c.year_suffix) for c in cands)
    assert (2010, "a") in years and (2010, "b") in years
    assert all("Smith" in c.families for c in cands)


def test_extra_year_re_ignores_non_year_numbers():
    from bibr.structure import citation_matcher as cm

    cands = cm._find_candidates("Jones (2010; n = 2500) found something.", 2)
    years = {c.year for c in cands}
    assert 2010 in years and 2500 not in years


def test_narrative_year_paren_with_colon_page():
    from bibr.structure import citation_matcher as cm

    cands = cm._find_candidates("Wells (1983: 76) observed that firms sought lower wages.", 1)
    assert any(c.year == 1983 and "Wells" in c.families for c in cands)


def test_narrative_curly_apostrophe_possessive():
    from bibr.structure import citation_matcher as cm

    c1 = cm._find_candidates("Drawing upon Clemens’ (2017) framework, the analysis proceeds.", 1)
    assert any(c.year == 2017 and any("Clemens" in f for f in c.families) for c in c1)
    c2 = cm._find_candidates("I run Andersson’s (2019) data and code to confirm.", 2)
    assert any(c.year == 2019 and any("Andersson" in f for f in c.families) for c in c2)


def test_separated_possessive_work_noun_resolves_unique_reference():
    from bibr.structure.citation_matcher import match_with_candidates

    refs = [_ref(32, "Lau, Ernst", year=1927)]
    sentence = _sent(
        41,
        "Ernst Lau's survey article (1927) on methods of youth psychology shows this.",
    )

    xrefs, ambiguous, candidates = match_with_candidates([sentence], refs)

    assert ambiguous == []
    assert [(xref.text_id, xref.xref_id) for xref in xrefs] == [(41, 32)]
    accepted = [candidate for candidate in candidates if candidate.accepted]
    assert len(accepted) == 1
    assert accepted[0].raw == "Ernst Lau's survey article (1927)"
    assert accepted[0].evidence == (
        "family_match",
        "year_match",
        "separated_possessive",
        "work_noun_bridge",
    )


def test_separated_possessive_same_family_year_remains_ambiguous():
    from bibr.structure.citation_matcher import match_with_candidates

    refs = [
        _ref(32, "Lau, Ernst", year=1927),
        _ref(33, "Lau, Hans", year=1927),
    ]

    xrefs, ambiguous, candidates = match_with_candidates(
        [_sent(41, "Lau's survey article (1927) established the method.")], refs
    )

    assert xrefs == []
    assert len(ambiguous) == 1
    assert ambiguous[0].candidate_bib_ids == [32, 33]
    assert len(candidates) == 1
    assert candidates[0].accepted is False
    assert candidates[0].bib_ids == (32, 33)
    assert candidates[0].rejection_reasons == ("ambiguous_same_surname_year",)
    assert "separated_possessive" in candidates[0].evidence


def test_separated_possessive_unmatched_work_form_is_receipted():
    from bibr.structure.citation_matcher import match_with_candidates

    xrefs, ambiguous, candidates = match_with_candidates(
        [_sent(7, "Meyer’s study (1927) addressed the question.")],
        [_ref(32, "Lau, Ernst", year=1927)],
    )

    assert xrefs == []
    assert ambiguous == []
    assert len(candidates) == 1
    assert candidates[0].accepted is False
    assert candidates[0].rejection_reasons == ("no_reference_match",)
    assert "separated_possessive" in candidates[0].evidence


def test_bare_year_and_non_work_possessive_forms_are_not_candidates():
    from bibr.structure import citation_matcher as cm

    surfaces = (
        "The archive opened (1927).",
        "Lau's laboratory (1927) stood nearby.",
        "The result x (1927) is not an equation citation.",
    )

    assert all(cm._find_candidates(surface, index) == [] for index, surface in enumerate(surfaces))


def test_curly_apostrophe_possessive_normalizes():
    """Curly-apostrophe possessives must normalize to bare family key."""
    from bibr.structure.citation_matcher import _family_keys_for

    # U+2019 trailing alone (e.g. "Clemens’")
    assert _family_keys_for("Clemens’") == {"clemens"}
    # U+2019 possessive ‘s’ (e.g. "Andersson’s")
    assert _family_keys_for("Andersson’s") == {"andersson"}
    # ASCII possessive still works
    assert _family_keys_for("Perruchet’s") == {"perruchet"}


def test_year_less_resolves_only_to_already_cited_ref():
    from bibr.models import PaperReference
    from bibr.structure import citation_matcher as cm

    refs = [
        PaperReference.model_validate(
            {
                "bib_id": 9,
                "title": "t",
                "authors": "Ihmels, M., & Ache, F.",
                "year": 2018,
                "first_page": None,
                "volume": None,
                "container": None,
            }
        )
    ]
    from types import SimpleNamespace

    body = [
        SimpleNamespace(text_id=5, text="As Ihmels and Ache (2018) showed, the effect held."),
        SimpleNamespace(text_id=13, text="As Ihmels and Ache demonstrated in a reanalysis,"),
        SimpleNamespace(text_id=14, text="Some unrelated Nguyen and Park discussion follows."),
    ]
    cited = {9}  # bib 9 was linked with a year by Tier 2
    links = cm.match_year_less(body, refs, cited)
    assert (13, 9) in links  # back-reference recovered
    assert not any(t == 14 for t, _ in links)  # unrelated name NOT linked


def test_nested_group_cite_inner_parens():
    from bibr.structure import citation_matcher as cm

    # Nested group-cite: inner (YYYY) parens sit one level inside an outer group
    # paren "(Salter et al. (1969), Caves (1992), ...)". The narrative depth guard
    # must allow one level of nesting (econ gold ja.2009-14, t29).
    s = (
        "This is because, as mentioned by Oulton (1998), although the fact that labour "
        "productivity varies between plants and companies is well-known (Salter et al. (1969), "
        "Caves (1992), Green and Mayes (1991), Hart and Shipman (1992), and "
        "Lansbury and Mayes (1996a,b)), the shape of the distribution function differs."
    )
    pairs = {(tuple(c.families), c.year) for c in cm._find_candidates(s, 29)}
    assert (("Salter",), 1969) in pairs
    assert (("Caves",), 1992) in pairs
    assert (("Green", "Mayes"), 1991) in pairs
    assert (("Hart", "Shipman"), 1992) in pairs
