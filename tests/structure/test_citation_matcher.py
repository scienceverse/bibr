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


def _rule_xrefs(sents, refs):
    """Tier 2 as a run with no LLM reads it (``rules=True``)."""
    return detect_author_year_xrefs(sents, refs, rules=True)


def _ref(bib_id, authors, year=2020, year_suffix=None, is_in_press=False, title="", **printed):
    return PaperReference(
        bib_id=bib_id,
        title=title,
        authors=authors,
        year=year,
        container=None,
        year_suffix=year_suffix,
        is_in_press=is_in_press,
        **{"first_page": None, "volume": None, **printed},
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

    def test_all_caps_surname_is_not_initials(self):
        assert extract_families("NGUYEN, T., & BERGSTROM, K.", rules=True) == [
            "NGUYEN",
            "BERGSTROM",
        ]
        assert extract_families("KOWALSKI T., LINDQVIST K.", rules=True) == [
            "KOWALSKI",
            "LINDQVIST",
        ]

    def test_all_caps_surname_is_initials_without_rules(self):
        assert extract_families("NGUYEN, T., & BERGSTROM, K.") == ["BERGSTROM"]

    def test_short_caps_runs_stay_initials(self):
        for rules in (False, True):
            assert extract_families("Smith, JA, & Jones, K. B.", rules=rules) == ["Smith", "Jones"]
            assert extract_families("Smith, J. A. B. C.", rules=rules) == ["Smith"]
            assert extract_families("Lindqvist AJB, Okafor CD", rules=rules) == [
                "Lindqvist",
                "Okafor",
            ]


class TestFamilyKeys:
    def test_simple(self):
        assert _family_keys_for("Smith") == {"smith"}

    def test_possessive_stripped(self):
        assert _family_keys_for("Perruchet's") == {"perruchet"}

    def test_multi_word_indexes_last(self):
        assert _family_keys_for("van der Toorn") == {"van der toorn", "toorn"}

    def test_della_sala(self):
        assert _family_keys_for("Della Sala") == {"della sala", "sala"}

    def test_leading_marker_stripped_with_rules(self):
        assert _family_keys_for("*Lindqvist", rules=True) == {"lindqvist"}
        assert _family_keys_for("† Okafor", rules=True) == {"okafor"}
        assert _family_keys_for("(van der Toorn", rules=True) == {"van der toorn", "toorn"}

    def test_leading_marker_kept_without_rules(self):
        assert _family_keys_for("*Lindqvist") == {"*lindqvist"}


def test_reference_with_leading_marker_links():
    refs = [
        _ref(1, "*Lindqvist, K., & †Okafor, C.", year=2018),
        _ref(2, "Brandt, H.", year=2018),
    ]
    sents = [_sent(0, "This holds (Lindqvist & Okafor, 2018) and elsewhere (Brandt, 2018).")]
    assert sorted(x.xref_id for x in _rule_xrefs(sents, refs)) == [1, 2]
    # the exact reading cannot place the marked entry
    assert [x.xref_id for x in detect_author_year_xrefs(sents, refs)] == [2]


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


class TestRulesAreOptIn:
    """Off, the matcher reads names exactly as it always did; ``rules`` adds readings."""

    TITLE = "Gradient methods for sparse signal recovery"

    @pytest.mark.parametrize(
        ("refs", "text", "with_rules"),
        [
            # a work listed twice
            (
                [
                    _ref(1, "Lindqvist, K.", title=TITLE),
                    _ref(2, "Lindqvist, K.", title=TITLE),
                ],
                "As shown (Lindqvist, 2020).",
                [1],
            ),
            # author count settles a tie
            (
                [
                    _ref(1, "Okafor, C., & Lund, K.", title="Two authors"),
                    _ref(2, "Okafor, C., Lund, K., & Brandt, H.", title="Three authors"),
                ],
                "As Okafor et al. (2020) showed, this holds.",
                [2],
            ),
            # a "Family Given" entry reached by its family
            (
                [_ref(1, "Okafor Chidi, Lund Karl, Brandt Hana", title="A")],
                "As shown (Okafor et al., 2020) here.",
                [1],
            ),
            # a misspelt family
            (
                [_ref(1, "Kowalczyk, A.", title="A"), _ref(2, "Brandt, H.", title="B")],
                "As shown (Kowalchyk, 2020) here.",
                [1],
            ),
            # a marker printed before the name
            (
                [_ref(1, "*Lindqvist, K.", title="A"), _ref(2, "Brandt, H.", title="B")],
                "As shown (Lindqvist, 2020) here.",
                [1],
            ),
            # a surname in capitals
            (
                [_ref(1, "NGUYEN, T.", title="A"), _ref(2, "Brandt, H.", title="B")],
                "As shown (Nguyen, 2020) here.",
                [1],
            ),
        ],
    )
    def test_readings_apply_only_with_rules(self, refs, text, with_rules):
        sents = [_sent(0, text)]

        assert detect_author_year_xrefs(sents, refs) == []
        assert detect_author_year_xrefs(sents, refs, rules=False) == []
        assert [x.xref_id for x in detect_author_year_xrefs(sents, refs, rules=True)] == with_rules

    def test_a_reference_both_readings_place_is_linked_either_way(self):
        refs = [_ref(1, "Okafor, C., & Lund, K.", title="A"), _ref(2, "Brandt, H.", title="B")]
        sents = [_sent(0, "As shown (Okafor & Lund, 2020) and (Brandt, 2020).")]

        assert [x.xref_id for x in detect_author_year_xrefs(sents, refs)] == [1, 2]
        assert [x.xref_id for x in detect_author_year_xrefs(sents, refs, rules=True)] == [1, 2]


class TestDuplicateReferences:
    """A work listed twice is one work: citing it is not ambiguous."""

    TITLE = "Gradient methods for sparse signal recovery"

    def test_listed_twice_links_first_entry(self):
        from bibr.structure.citation_matcher import match_with_ambiguous

        refs = [
            _ref(4, "Lindqvist, K.", title=self.TITLE),
            _ref(9, "Brandt, H.", title="Another paper entirely"),
            _ref(15, "Lindqvist, K.", title=self.TITLE),
        ]
        xrefs, ambiguous = match_with_ambiguous(
            [_sent(0, "As shown (Lindqvist, 2020).")], refs, rules=True
        )
        assert [x.xref_id for x in xrefs] == [4]
        assert ambiguous == []

    def test_case_and_diacritics_do_not_hide_a_repeat(self):
        refs = [
            _ref(1, "Müller, K.", title=self.TITLE),
            _ref(2, "MULLER, K.", title=self.TITLE.upper()),
        ]
        sents = [_sent(0, "As shown (Muller, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_different_titles_stay_ambiguous(self):
        from bibr.structure.citation_matcher import match_with_ambiguous

        refs = [
            _ref(1, "Lindqvist, K.", title=self.TITLE),
            _ref(2, "Lindqvist, K.", title="A second study by the same author"),
        ]
        xrefs, ambiguous = match_with_ambiguous(
            [_sent(0, "As shown (Lindqvist, 2020).")], refs, rules=True
        )
        assert xrefs == []
        assert ambiguous[0].candidate_bib_ids == [1, 2]

    def test_entries_without_a_title_are_not_merged(self):
        refs = [_ref(1, "Lindqvist, K."), _ref(2, "Lindqvist, K.")]
        out = _rule_xrefs([_sent(0, "As shown (Lindqvist, 2020).")], refs)
        assert out == []

    def test_repeat_under_another_year_suffix_stays_citable_by_it(self):
        refs = [
            _ref(1, "Lindqvist, K.", year_suffix="a", title=self.TITLE),
            _ref(2, "Lindqvist, K.", year_suffix="b", title=self.TITLE),
        ]
        sents = [
            _sent(0, "As shown (Lindqvist, 2020a) here."),
            _sent(1, "As shown (Lindqvist, 2020b) here."),
        ]
        out = _rule_xrefs(sents, refs)
        assert [(x.text_id, x.xref_id) for x in out] == [(0, 1), (1, 2)]

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            # The first 40 characters are the same; the number that tells the
            # reports apart comes after them.
            (
                "Weekly outbreak situation report number 51",
                "Weekly outbreak situation report number 72",
            ),
            (
                "Surveillance of respiratory infections in the clinic, Part I",
                "Surveillance of respiratory infections in the clinic, Part II",
            ),
            (
                "Surveillance of respiratory infections in the clinic: methods",
                "Surveillance of respiratory infections in the clinic: results",
            ),
        ],
    )
    def test_titles_that_differ_after_the_first_forty_characters_are_two_works(self, first, second):
        from bibr.structure.citation_matcher import match_with_ambiguous

        refs = [_ref(1, "Lindqvist, K.", title=first), _ref(2, "Lindqvist, K.", title=second)]
        xrefs, ambiguous = match_with_ambiguous(
            [_sent(0, "As shown (Lindqvist, 2020).")], refs, rules=True
        )
        assert xrefs == []
        assert ambiguous[0].candidate_bib_ids == [1, 2]

    @pytest.mark.parametrize(
        "printed",
        [
            ({"volume": "12"}, {"volume": "13"}),
            ({"first_page": "101"}, {"first_page": "245"}),
            ({"last_page": "120"}, {"last_page": "260"}),
            ({"doi": "10.1000/abc.1"}, {"doi": "10.1000/abc.2"}),
            ({"edition": "2nd"}, {"edition": "3rd"}),
        ],
    )
    def test_one_title_with_another_volume_or_pages_is_two_works(self, printed):
        from bibr.structure.citation_matcher import match_with_ambiguous

        first, second = printed
        refs = [
            _ref(1, "Lindqvist, K.", title=self.TITLE, **first),
            _ref(2, "Lindqvist, K.", title=self.TITLE, **second),
        ]
        xrefs, ambiguous = match_with_ambiguous(
            [_sent(0, "As shown (Lindqvist, 2020).")], refs, rules=True
        )
        assert xrefs == []
        assert ambiguous[0].candidate_bib_ids == [1, 2]

    def test_a_volume_one_entry_prints_and_the_other_leaves_out_does_not_part_them(self):
        refs = [
            _ref(1, "Lindqvist, K.", title=self.TITLE, volume="12", first_page="101"),
            _ref(2, "Lindqvist, K.", title=self.TITLE),
        ]
        sents = [_sent(0, "As shown (Lindqvist, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_the_same_work_printed_with_other_punctuation_is_one_work(self):
        refs = [
            _ref(1, "Lindqvist, K.", title=self.TITLE, volume="12", first_page="101"),
            _ref(2, "Lindqvist K", title=self.TITLE + ".", volume="12", first_page="101"),
        ]
        sents = [_sent(0, "As shown (Lindqvist, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_index_keeps_every_entry_in_refs_list(self):
        from bibr.structure.citation_matcher import _build_ref_index

        refs, by_year, _ = _build_ref_index(
            [
                _ref(1, "Lindqvist, K.", title=self.TITLE),
                _ref(2, "Lindqvist, K.", title=self.TITLE),
            ],
            rules=True,
        )
        assert [r.bib_id for r in refs] == [1, 2]
        assert [r.bib_id for r in by_year[(2020, None)]] == [1]


class TestTieBreak:
    """Same first family and year: the author count and order can still settle it."""

    def test_et_al_takes_the_reference_with_three_or_more_authors(self):
        refs = [
            _ref(1, "Okafor, C., & Lund, K.", title="Two authors"),
            _ref(2, "Okafor, C., Lund, K., & Brandt, H.", title="Three authors"),
        ]
        sents = [_sent(0, "As Okafor et al. (2020) showed, this holds.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [2]

    def test_et_al_between_two_long_lists_stays_ambiguous(self):
        refs = [
            _ref(1, "Okafor, C., Lund, K., & Brandt, H.", title="Three authors"),
            _ref(2, "Okafor, C., Lund, K., Brandt, H., & Sato, M.", title="Four authors"),
        ]
        sents = [_sent(0, "As Okafor et al. (2020) showed, this holds.")]
        assert _rule_xrefs(sents, refs) == []

    def test_cited_order_follows_listed_order(self):
        refs = [
            _ref(1, "Okafor, C., & Lund, K.", title="Okafor first"),
            _ref(2, "Lund, K., & Okafor, C.", title="Lund first"),
        ]
        sents = [_sent(0, "As shown (Lund & Okafor, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [2]

    def test_equal_fit_stays_ambiguous(self):
        refs = [
            _ref(1, "Okafor, C., & Lund, K.", title="First paper"),
            _ref(2, "Okafor, C., & Lund, K.", title="Second paper"),
        ]
        sents = [_sent(0, "As shown (Okafor & Lund, 2020) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_one_cited_family_needs_a_single_author_reference(self):
        from bibr.structure import citation_matcher as cm

        refs = [
            _ref(1, "Okafor, C., & Lund, K.", title="Two authors"),
            _ref(2, "Okafor, C.", title="One author"),
        ]
        _r, by_year, ip = cm._build_ref_index(refs, rules=True)
        cand = next(c for c in cm._find_candidates("As Okafor (2020) showed,", 1) if c.year == 2020)
        ref, tied = cm._match(cand, by_year, ip, rules=True)
        assert ref is not None and ref.bib_id == 2
        assert tied == []

    def test_the_reference_led_by_the_first_cited_family_wins(self):
        # Both organisation names end in "Education", the key a second cited
        # word shares with each; only one begins with the first cited name.
        refs = [
            _ref(1, "Greenfield Digital Education", title="Lessons from remote teaching"),
            _ref(2, "Harbor Ad Hoc Committee on Learning Education", title="Report"),
        ]
        sents = [_sent(0, "As reported (Greenfield Digital Education, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_two_references_led_by_the_first_cited_family_are_ranked_by_fit(self):
        refs = [
            _ref(1, "Okafor Chidi, Lund Karl, Brandt Hana", title="Three authors"),
            _ref(2, "Okafor Dayo, Lund Karl", title="Two authors"),
        ]
        sents = [_sent(0, "As shown (Okafor & Lund, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [2]

    def test_one_cited_family_with_no_single_author_reference_stays_ambiguous(self):
        refs = [
            _ref(1, "Okafor, C., & Lund, K.", title="Pair one"),
            _ref(2, "Okafor, D., & Brandt, H.", title="Pair two"),
        ]
        sents = [_sent(0, "As Okafor (2020) showed, this holds.")]
        assert _rule_xrefs(sents, refs) == []


class TestAndOthersMarkers:
    """ "et al." is written many ways, and each of them says more than one author."""

    _REFS = [
        _ref(1, "Okafor, C.", year=2016, title="One author"),
        _ref(2, "Okafor, C., Lund, K., & Brandt, H.", year=2016, title="Three authors"),
    ]

    @pytest.mark.parametrize(
        "marker",
        [
            "et al.",
            "et. al.",
            "et.al.",
            "etal",
            "et\u00a0al.",
            "et\nal.",
            "et al",
            "y cols.",
            "y colaboradores",
            "e col.",
            "et coll.",
            "und Kollegen",
            "dkk.",
            "ve ark.",
            "and colleagues",
            "and others",
        ],
    )
    def test_the_marker_takes_the_reference_with_several_authors(self, marker):
        sents = [_sent(0, f"As shown (Okafor {marker}, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, self._REFS)] == [2]

    def test_a_bare_citation_still_takes_the_single_author_reference(self):
        sents = [_sent(0, "As shown (Okafor, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, self._REFS)] == [1]

    def test_a_narrative_citation_still_takes_the_single_author_reference(self):
        sents = [_sent(0, "As Okafor (2016) showed, this holds.")]
        assert [x.xref_id for x in _rule_xrefs(sents, self._REFS)] == [1]

    def test_words_after_the_name_that_are_no_marker_decline_the_single_author_win(self):
        # "Okafor, C." might be a reference to the three-author work or to the
        # one-author work: the initial after the name settles nothing.
        sents = [_sent(0, "As shown (Okafor, C., 2016) here.")]
        assert _rule_xrefs(sents, self._REFS) == []

    def test_the_page_after_the_year_is_no_word_after_the_name(self):
        sents = [_sent(0, "As shown (Okafor, 2016, p. 12) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, self._REFS)] == [1]

    def test_the_words_before_the_name_are_no_word_after_it(self):
        sents = [_sent(0, "As shown (see also Okafor, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, self._REFS)] == [1]

    def test_off_by_default_only_the_literal_et_al_counts(self):
        from bibr.structure import citation_matcher as cm

        (plain,) = cm._find_candidates("As (Okafor etal, 2016).", 1)
        (rules,) = cm._find_candidates("As (Okafor etal, 2016).", 1, rules=True)
        assert (plain.has_et_al, rules.has_et_al) == (False, True)
        (literal,) = cm._find_candidates("As (Okafor et al., 2016).", 1)
        assert literal.has_et_al is True

    def test_et_al_inside_a_word_is_no_marker(self):
        from bibr.structure import citation_matcher as cm

        (cand,) = cm._find_candidates("As (Ostetalova, 2016) said.", 1, rules=True)
        assert cand.has_et_al is False


class TestLooseFamilyMatch:
    """Prefix reading of a family name, tried only when no exact key matches."""

    def test_family_given_entry_is_reached_by_its_family(self):
        refs = [_ref(1, "Okafor Chidi, Lund Karl, Brandt Hana", title="A")]
        sents = [_sent(0, "As shown (Okafor et al., 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_spaces_and_hyphens_do_not_matter(self):
        refs = [_ref(1, "Hernandez Lopez, M.", title="A"), _ref(2, "Brandt, H.", title="B")]
        sents = [_sent(0, "As shown (Hernandez-Lopez, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_prefix_needs_four_characters_on_both_sides(self):
        refs = [_ref(1, "Leeuwen, A.", title="A"), _ref(2, "Lindqvist, K.", title="B")]
        sents = [_sent(0, "As shown (Lee, 2020) and (Li, 2020) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_exact_key_wins_over_a_longer_name(self):
        refs = [_ref(1, "Lundgren, K.", title="A"), _ref(2, "Lund, K.", title="B")]
        sents = [_sent(0, "As shown (Lund, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [2]

    def test_tie_between_loose_matches_links_nothing_and_offers_no_shortlist(self):
        from bibr.structure.citation_matcher import match_with_ambiguous

        refs = [
            _ref(1, "Okafor Chidi, Lund Karl", title="A"),
            _ref(2, "Okafor Dayo, Lund Karl", title="B"),
        ]
        xrefs, ambiguous = match_with_ambiguous(
            [_sent(0, "As shown (Okafor & Lund, 2020).")], refs, rules=True
        )
        assert xrefs == []
        assert ambiguous == []

    def test_author_count_still_breaks_a_loose_tie(self):
        refs = [
            _ref(1, "Okafor Chidi, Lund Karl", title="A"),
            _ref(2, "Okafor Dayo", title="B"),
        ]
        sents = [_sent(0, "As Okafor (2020) showed, this holds.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [2]


class TestFuzzyFamilyMatch:
    """A misspelt family links when one reference of the exact year is close."""

    def test_close_spelling_links(self):
        refs = [_ref(1, "Kowalczyk, A.", title="A"), _ref(2, "Brandt, H.", title="B")]
        sents = [_sent(0, "As shown (Kowalchyk, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_year_must_be_exact(self):
        refs = [_ref(1, "Kowalczyk, A.", year=2019, title="A")]
        sents = [_sent(0, "As shown (Kowalchyk, 2020) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_short_names_are_never_fuzzy(self):
        refs = [_ref(1, "Smith, A.", title="A")]
        sents = [_sent(0, "As shown (Smyth, 2020) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_dissimilar_names_do_not_link(self):
        refs = [_ref(1, "Kowalczyk, A.", title="A")]
        sents = [_sent(0, "As shown (Kowalski, 2020) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_two_close_references_link_nothing_and_offer_no_shortlist(self):
        from bibr.structure.citation_matcher import match_with_ambiguous

        refs = [_ref(1, "Nakamura, K.", title="A"), _ref(2, "Nakamora, T.", title="B")]
        xrefs, ambiguous = match_with_ambiguous(
            [_sent(0, "As shown (Nakamara, 2020).")], refs, rules=True
        )
        assert xrefs == []
        assert ambiguous == []

    def test_a_better_exact_candidate_is_not_overridden(self):
        refs = [_ref(1, "Nakamura, K.", title="A"), _ref(2, "Nakamora, T.", title="B")]
        sents = [_sent(0, "As shown (Nakamura, 2020) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]


class TestLooseLinksNeedAWholeName:
    """A name that opens, ends or looks like another's is another's."""

    @pytest.mark.parametrize(
        ("cited", "listed"),
        [
            ("John", "Johnson, A."),
            ("Johnson", "John, A."),
            ("Park", "Parker, S."),
            ("Rose", "Rosenberg, T."),
            ("March", "Marchant, T."),
            ("Gold", "Goldberg, T."),
            ("Kant", "Kanter, T."),
        ],
    )
    def test_a_name_is_not_the_opening_letters_of_a_longer_one(self, cited, listed):
        refs = [_ref(1, listed, year=2016, title="A")]
        sents = [_sent(0, f"As shown ({cited}, 2016) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_a_first_author_with_et_al_is_not_the_opening_letters_of_a_name(self):
        refs = [_ref(1, "John, A., Lee, B., & Kim, C.", year=2016, title="A")]
        sents = [_sent(0, "As shown (Johnson et al., 2016) here.")]
        assert _rule_xrefs(sents, refs) == []

    @pytest.mark.parametrize(
        ("cited", "listed"),
        [
            ("European Commission", "Commission on Human Rights"),
            ("Health Council", "Ministry of Health"),
            ("Education Commission", "Department of Education"),
        ],
    )
    def test_a_body_is_not_the_last_words_of_another(self, cited, listed):
        refs = [_ref(1, listed, year=2016, title="A")]
        sents = [_sent(0, f"As shown ({cited}, 2016) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_a_listed_family_given_entry_is_reached_by_its_whole_family(self):
        refs = [_ref(1, "Berg Anna, Lund Karl", year=2016, title="A")]
        sents = [_sent(0, "As shown (Berg & Lund, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_a_name_printed_with_other_spaces_or_hyphens_is_the_same_name(self):
        refs = [_ref(1, "Hernandez Lopez, M.", year=2016, title="A")]
        sents = [_sent(0, "As shown (Hernandez-Lopez, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_a_double_surname_is_reached_by_either_part_and_reaches_it(self):
        listed_whole = [_ref(1, "Hedman-Lagerlof, E.", year=2016, title="A")]
        listed_part = [_ref(1, "Lagerlof, E.", year=2016, title="A")]
        assert [
            x.xref_id for x in _rule_xrefs([_sent(0, "As (Lagerlof, 2016) said.")], listed_whole)
        ] == [1]
        assert [
            x.xref_id
            for x in _rule_xrefs([_sent(0, "As (Hedman-Lagerlof, 2016) said.")], listed_part)
        ] == [1]

    def test_a_short_part_does_not_stand_for_a_double_surname(self):
        refs = [_ref(1, "Kim, A.", year=2016, title="A")]
        sents = [_sent(0, "As shown (Kim-Lee, 2016) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_a_domain_name_is_reached_by_its_first_label(self):
        refs = [_ref(1, "Calculator.net", year=2016, title="")]
        sents = [_sent(0, "As shown (Calculator, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [1]

    def test_off_by_default_none_of_the_loose_readings_apply(self):
        refs = [_ref(1, "Berg Anna", year=2016, title="A")]
        sents = [_sent(0, "As shown (Berg, 2016) here.")]
        assert detect_author_year_xrefs(sents, refs) == []


class TestLooseLinkEvidence:
    """A link read loosely says so, and vouches for itself less."""

    @staticmethod
    def _candidate(text, refs):
        _xrefs, _amb, candidates = match_with_candidates([_sent(0, text)], refs, rules=True)
        (candidate,) = [c for c in candidates if c.accepted]
        return candidate

    def test_exact_link_keeps_its_evidence_and_confidence(self):
        refs = [_ref(1, "Okafor, C.", year=2016, title="A")]
        candidate = self._candidate("As shown (Okafor, 2016) here.", refs)
        assert candidate.evidence == ("family_match", "year_match")
        assert candidate.confidence == 0.95

    def test_prefix_link_has_its_own_evidence_and_a_lower_confidence(self):
        refs = [_ref(1, "Berg Anna", year=2016, title="A")]
        candidate = self._candidate("As shown (Berg, 2016) here.", refs)
        assert candidate.evidence == ("family_prefix_match", "year_match")
        assert candidate.confidence == 0.8

    def test_fuzzy_link_has_its_own_evidence_and_the_lowest_confidence(self):
        refs = [_ref(1, "Kowalczyk, A.", year=2016, title="A")]
        candidate = self._candidate("As shown (Kowalchyk, 2016) here.", refs)
        assert candidate.evidence == ("family_fuzzy_match", "year_match")
        assert candidate.confidence == 0.7

    def test_tie_break_link_says_so(self):
        refs = [
            _ref(1, "Okafor, C., & Lund, K.", year=2016, title="Two authors"),
            _ref(2, "Okafor, C., Lund, K., & Brandt, H.", year=2016, title="Three authors"),
        ]
        candidate = self._candidate("As Okafor et al. (2016) showed, this holds.", refs)
        assert candidate.evidence == ("family_match", "year_match", "author_list_tie_break")
        assert candidate.confidence == 0.9

    def test_a_name_listed_as_cited_under_another_year_is_not_guessed_at(self):
        # "Kowalchyk" is an author of the 2015 entry as spelt: the 2016 entry
        # is not that author's work with a typo.
        refs = [
            _ref(1, "Kowalchyk, B.", year=2015, title="A"),
            _ref(2, "Kowalczyk, A.", year=2016, title="B"),
        ]
        sents = [_sent(0, "As shown (Kowalchyk, 2016) here.")]
        assert _rule_xrefs(sents, refs) == []

    def test_a_second_cited_author_vouches_for_the_spelling(self):
        refs = [
            _ref(1, "Kowalchyk, B.", year=2015, title="A"),
            _ref(2, "Kowalczyk, A., & Lund, K.", year=2016, title="B"),
        ]
        sents = [_sent(0, "As shown (Kowalchyk & Lund, 2016) here.")]
        assert [x.xref_id for x in _rule_xrefs(sents, refs)] == [2]


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


class TestResolveCitationsByRules:
    """Tier 3 without an LLM: citation texts the sentence-level matcher left over."""

    def _resolve(self, refs, *texts):
        from bibr.structure.citation_matcher import resolve_citations_by_rules

        return resolve_citations_by_rules(list(enumerate(texts, start=10)), refs)

    def test_bare_author_year_without_parentheses(self):
        refs = [_ref(1, "Lindqvist, K., & Okafor, C.", year=2018, title="A")]
        assert self._resolve(refs, "Lindqvist & Okafor, 2018") == [
            (10, "Lindqvist & Okafor, 2018", (1,))
        ]

    def test_bracketed_and_narrative_forms(self):
        refs = [_ref(1, "Lindqvist, K.", year=2018, title="A"), _ref(2, "Brandt, H.", year=2019)]
        out = self._resolve(refs, "[Lindqvist, 2018]", "As Brandt (2019) found")
        assert [bib_ids for _t, _c, bib_ids in out] == [(1,), (2,)]

    def test_several_years_link_several_references(self):
        refs = [
            _ref(1, "Okafor, C.", year=2018, title="A"),
            _ref(2, "Okafor, C.", year=2019, title="B"),
            _ref(3, "Okafor, C.", year=2020, title="C"),
        ]
        assert self._resolve(refs, "Okafor, 2018, 2019")[0][2] == (1, 2)

    def test_secondary_source_links_the_work_read(self):
        refs = [
            _ref(1, "Mendoza, F., Valdivia, S., & Rojas, S.", year=2015, title="A"),
            _ref(2, "Harlow, C.", year=2010, title="B"),
        ]
        assert self._resolve(refs, "(Harlow, 2010, citado en Mendoza et al., 2015)")[0][2] == (1,)
        assert self._resolve(refs, "(Harlow, 2010, as cited in Mendoza et al., 2015)")[0][2] == (1,)

    def test_a_later_author_links_when_no_first_author_fits(self):
        from bibr.structure.citation_matcher import detect_author_year_xrefs

        refs = [_ref(1, "Okafor, C., & Brandt, H.", year=2018, title="A")]
        assert detect_author_year_xrefs([_sent(0, "As Brandt (2018) found.")], refs) == []
        assert self._resolve(refs, "Brandt (2018)")[0][2] == (1,)

    def test_a_later_author_shared_by_two_references_links_nothing(self):
        refs = [
            _ref(1, "Okafor, C., & Brandt, H.", year=2018, title="A"),
            _ref(2, "Lund, K., & Brandt, H.", year=2018, title="B"),
        ]
        assert self._resolve(refs, "Brandt (2018)")[0][2] == ()

    def test_organisation_cited_by_its_initials(self):
        refs = [
            _ref(1, "Bureau of Coastal Survey", year=2022, title="A"),
            _ref(2, "Council for Reef Studies", year=2022, title="B"),
            _ref(3, "Okafor, C.", year=2022, title="C"),
        ]
        out = self._resolve(refs, "(BCS, 2022)", "e.g., CfRS, 2022", "CRS (2022)")
        assert [bib_ids for _t, _c, bib_ids in out] == [(1,), (2,), (2,)]

    def test_initials_must_read_as_initials_and_be_unique(self):
        refs = [
            _ref(1, "Nguyen Giang, Pham Binh", year=2022, title="A"),
            _ref(2, "Bureau of Coastal Survey", year=2022, title="B"),
            _ref(3, "Board of Cartographic Studies", year=2022, title="C"),
            _ref(4, "Okafor", year=2022, title="D"),
        ]
        # lower-case "Ng", an ambiguous "BCS", and a one-word author link nothing
        out = self._resolve(refs, "Ng (2022)", "BCS (2022)", "OK (2022)")
        assert [bib_ids for _t, _c, bib_ids in out] == [(), (), ()]

    def test_unknown_citation_links_nothing(self):
        refs = [_ref(1, "Okafor, C.", year=2018, title="A")]
        assert self._resolve(refs, "Sato (2018)", "Okafor (2012)")[0][2] == ()
        assert self._resolve(refs, "Sato (2018)", "Okafor (2012)")[1][2] == ()

    def test_same_family_same_year_without_other_evidence_links_nothing(self):
        refs = [
            _ref(1, "Okafor, C.", year=2018, title="A"),
            _ref(2, "Okafor, C.", year=2018, title="B"),
        ]
        assert self._resolve(refs, "Okafor (2018)")[0][2] == ()

    def test_no_references_and_no_citations(self):
        assert self._resolve([], "Okafor (2018)") == [(10, "Okafor (2018)", ())]
        assert self._resolve([_ref(1, "Okafor, C.", year=2018)]) == []
