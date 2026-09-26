"""Context guards on the numeric tiers and constraints on the Tier-3 LLM step."""

import pytest

from bibr.exceptions import ProcessingError
from bibr.paper import PaperReference
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence
from bibr.schemas import CitationMatch
from bibr.structure.citation_linker import (
    cited_reference_numbers,
    detect_bib_xrefs_with_receipt,
    strip_citation_superscripts,
)


def _ref(bib_id, authors=None, year=2020, text_id=None):
    return PaperReference(
        bib_id=bib_id,
        title=f"Title {bib_id}",
        first_page=None,
        volume=None,
        authors=authors,
        year=year,
        container=None,
        text_id=text_id,
    )


def _sections():
    return [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="Body",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.INTRODUCTION,
        ),
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]


def _body(texts):
    return [
        PaperSentence(text_id=index, text=text, section_id=1, paragraph_id=1)
        for index, text in enumerate(texts)
    ]


async def _link_numbered(texts, count, *, missing=(), row="{n}. "):
    """Link *texts* against a printed, numbered reference list of *count* rows."""
    numbers = [n for n in range(1, count + 1) if n not in missing]
    refs = [_ref(n, authors="Smith AB", text_id=10_000 + n) for n in numbers]
    rows = [
        PaperSentence(
            text_id=10_000 + n,
            text=f"{row.format(n=n)}Smith AB. Printed reference {n}.",
            section_id=2,
            paragraph_id=2,
        )
        for n in numbers
    ]
    sentences = _body(texts) + rows
    xrefs, receipt = await detect_bib_xrefs_with_receipt(sentences, _sections(), refs)
    return xrefs, receipt, sentences


def _links(xrefs, tier=None):
    return sorted(
        (xref.text_id, xref.xref_id) for xref in xrefs if tier is None or xref.tier == tier
    )


class RecordingLLM:
    """Fake Tier-3 client: records each request and answers from a script."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash="x"):
        self.calls.append(list(ambiguous_citations))
        return self.answer(list(ambiguous_citations))


def _every(bib_ids):
    """Answer every offered citation with each of *bib_ids*."""

    def answer(batch):
        return [
            CitationMatch(text_id=text_id, citation_text=text, bib_id=bib_id)
            for text_id, text in batch
            for bib_id in bib_ids
        ]

    return answer


async def _link_author_year(texts, refs, llm):
    return await detect_bib_xrefs_with_receipt(_body(texts), _sections()[:2], refs, llm_client=llm)


# ---------------------------------------------------------------------------
# Statistics, equation references and intervals
# ---------------------------------------------------------------------------


async def test_statistic_degrees_of_freedom_are_not_parenthetical_citations():
    xrefs, receipt, _ = await _link_numbered(
        [
            "Prior work [1] and [2] exists.",
            "Deficiency of vitamin D (12) is common.",
            "Earlier trials (13, 14) agree.",
            "A cohort study (15) found the same.",
            "The main effect of Group was significant, F(3, 84) = 4.49, p < .01.",
            "Scores differed between conditions, t(45) = 2.10, p = .04.",
            "The association was weak, r(58) = .21.",
            "The interaction was not, F (1, 35) = 2.85.",
            "Substituting Eq. (5) into the bound gives the rate.",
            "Equation (6) then follows.",
            "Both effects were small, Fs(1, 44) and more.",
            "The model had df(12) there.",
            "The estimates (13) \u2264 0.5 held.",
            "Using Eqs. (7) and (8) we bound it.",
            "Equations (9)-(10) give it.",
            "Prior trials(16) agree.",
            "It is HIV(17) related.",
            "The path gave \u03b2(18) and more.",
        ],
        100,
    )

    assert _links(xrefs, "paren-numeric") == [
        (1, 12),
        (2, 13),
        (2, 14),
        (3, 15),
        (15, 16),
        (16, 17),
    ]
    rejected = {
        (candidate.text_id, candidate.raw): candidate.rejection_reasons
        for candidate in receipt.candidates
        if candidate.style == "paren-numeric"
        and candidate.text_id >= 4
        and candidate.text_id not in (15, 16)
    }
    assert rejected == {
        (4, "(3, 84)"): ("statistic_context",),
        (5, "(45)"): ("statistic_context",),
        (6, "(58)"): ("statistic_context",),
        (7, "(1, 35)"): ("statistic_context",),
        (8, "(5)"): ("equation_tag",),
        (9, "(6)"): ("equation_tag",),
        (10, "(1, 44)"): ("statistic_context",),
        (11, "(12)"): ("statistic_context",),
        (12, "(13)"): ("statistic_context",),
        (13, "(7)"): ("equation_tag",),
        (13, "(8)"): ("equation_tag",),
        (14, "(9)"): ("equation_tag",),
        (14, "(10)"): ("equation_tag",),
        (17, "(18)"): ("statistic_context",),
    }


async def test_bracketed_statistic_needs_symbol_and_comparison():
    xrefs, receipt, _ = await _link_numbered(
        [
            "Prior work [1] exists.",
            "Later work [2] agrees.",
            "Reviews [3] summarise it.",
            "Data were analysed by ANOVA (F[2, 9] = 5.20, p < .01).",
            "Thiamin 1.26 a[5]; b[6] were reported.",
            "An omega statistic [7, 8] \u2265 0.70 was reached.",
        ],
        40,
        row="[{n}] ",
    )

    assert _links(xrefs) == [(0, 1), (1, 2), (2, 3), (4, 5), (4, 6), (5, 7), (5, 8)]
    [statistic] = [c for c in receipt.candidates if c.raw == "[2, 9]"]
    assert statistic.rejection_reasons == ("statistic_context",)


async def test_wide_two_number_bracket_is_a_citation_in_a_bracket_citing_paper():
    xrefs, receipt, _ = await _link_numbered(
        [
            "Prior work [1] exists.",
            "Later work [2] agrees.",
            "Reviews [3] summarise it.",
            "They had comparable risk perception [11, 33].",
            "The 95% CI [12, 45] excluded zero.",
            "Participants were adults (range [18, 45]).",
            "The effect held, F[1, 44] = 5.92.",
        ],
        58,
        row="[{n}] ",
    )

    assert _links(xrefs) == [(0, 1), (1, 2), (2, 3), (3, 11), (3, 33)]
    [lifted] = [c for c in receipt.candidates if c.raw == "[11, 33]"]
    assert (lifted.accepted, lifted.bib_ids, lifted.confidence) == (True, (11, 33), 1.0)
    assert "bracket_citation_style" in lifted.evidence
    rejected = {c.raw: c.rejection_reasons for c in receipt.candidates if c.text_id in (4, 5, 6)}
    assert rejected == {
        "[12, 45]": ("numeric_interval_guard",),
        "[18, 45]": ("numeric_interval_guard",),
        "[1, 44]": ("numeric_interval_guard", "statistic_context"),
    }
    assert cited_reference_numbers(receipt) == {1, 2, 3, 11, 33}


async def test_wide_two_number_bracket_stays_an_interval_without_bracket_citations():
    xrefs, receipt, _ = await _link_numbered(
        ["The estimate was [4, 37] in the first wave.", "Prior work [1] exists."],
        40,
        row="[{n}] ",
    )

    assert _links(xrefs) == [(1, 1)]
    [interval] = [c for c in receipt.candidates if c.raw == "[4, 37]"]
    assert interval.rejection_reasons == ("numeric_interval_guard",)


async def test_bracket_style_counts_sentences_with_accepted_markers():
    # Three accepted markers in one sentence, and three sentences whose only
    # brackets are wide pairs, do not make a bracket-citing paper.
    xrefs, receipt, _ = await _link_numbered(
        [
            "Prior work [1], [2] and [3] exists.",
            "The estimate was [4, 37] early.",
            "It was [5, 38] later.",
            "It ended at [6, 39].",
        ],
        40,
        row="[{n}] ",
    )

    assert _links(xrefs) == [(0, 1), (0, 2), (0, 3)]
    assert [c.rejection_reasons for c in receipt.candidates if c.text_id > 0] == [
        ("numeric_interval_guard",)
    ] * 3


# ---------------------------------------------------------------------------
# Superscript markers
# ---------------------------------------------------------------------------


async def test_superscript_group_with_a_dropped_reference_links_like_a_bracket():
    xrefs, receipt, sentences = await _link_numbered(
        [
            "Bracket form [3-5] here.",
            "Superscript form^{3-5} here.",
            "Beyond the list^{41} here.",
            "Across the end^{39-41} here.",
            "From zero^{0-2} here.",
        ],
        40,
        missing=(4,),
        row="[{n}] ",
    )

    assert _links(xrefs) == [(0, 3), (0, 5), (1, 3), (1, 5)]
    rejected = {c.raw: c.rejection_reasons for c in receipt.candidates if c.text_id >= 2}
    assert rejected == {
        "^{41}": ("unknown_bib_id",),
        "^{39-41}": ("unknown_bib_id",),
        "^{0-2}": ("unknown_bib_id",),
    }
    assert cited_reference_numbers(receipt) == {3, 4, 5, 39, 40, 41}

    strip_citation_superscripts(sentences, _sections(), receipt)
    assert [s.text for s in sentences[:5]] == [
        "Bracket form [3-5] here.",
        "Superscript form here.",
        "Beyond the list^{41} here.",
        "Across the end^{39-41} here.",
        "From zero^{0-2} here.",
    ]


async def test_unit_exponents_are_not_citations_and_stay_in_the_text():
    xrefs, receipt, sentences = await _link_numbered(
        [
            "Each plot measured 25 cm^{2} in area.",
            "Density was 3 g cm $ ^{3} $ overall.",
            "Prior work^{4} agrees.",
            "Flow was 75 beats/min^{5} at rest.",
        ],
        10,
    )

    assert _links(xrefs) == [(2, 4), (3, 5)]
    reasons = {c.raw: c.rejection_reasons for c in receipt.candidates if c.text_id in (0, 1)}
    assert reasons == {"^{2}": ("math_superscript",), "$ ^{3} $": ("math_superscript",)}

    strip_citation_superscripts(sentences, _sections(), receipt)
    assert [s.text for s in sentences[:4]] == [
        "Each plot measured 25 cm^{2} in area.",
        "Density was 3 g cm $ ^{3} $ overall.",
        "Prior work agrees.",
        "Flow was 75 beats/min at rest.",
    ]


# ---------------------------------------------------------------------------
# Flattened superscripts
# ---------------------------------------------------------------------------


async def test_float_and_equation_abbreviations_do_not_switch_on_flattened_style():
    xrefs, receipt, _ = await _link_numbered(
        [
            "Prior work [1] and [2] exists.",
            "As shown in Fig.3 the curve saturates.",
            "The bound follows from Eq.5 directly.",
        ],
        30,
        row="[{n}] ",
    )

    assert sorted((x.text_id, x.xref_id, x.tier) for x in xrefs) == [
        (0, 1, "numeric"),
        (0, 2, "numeric"),
    ]
    locators = [c for c in receipt.candidates if c.style == "flattened-superscript"]
    assert [(c.raw, "locator_abbreviation" in c.rejection_reasons) for c in locators] == [
        ("3", True),
        ("5", True),
    ]


async def test_locator_abbreviations_are_not_flattened_citations():
    xrefs, _receipt, _ = await _link_numbered(
        [
            "Prior evidence.1,2 established this.",
            "Later reports3,4 confirmed it.",
            "Results are listed in Tab.2 and Exp.1 of the report.",
            "The value in Vol.12 was reported on pp.14-16 of it.",
            "It is the state of the art.5 today.",
            "Clone No.6 was used.",
        ],
        30,
    )

    assert _links(xrefs, "flattened-superscript") == [(0, 1), (0, 2), (1, 3), (1, 4), (4, 5)]


# ---------------------------------------------------------------------------
# Tier 3: what is offered, and what may be accepted
# ---------------------------------------------------------------------------


async def test_citation_linked_by_the_matcher_is_not_offered_again():
    llm = RecordingLLM(_every([2]))
    xrefs, _receipt = await _link_author_year(
        [
            "In Smith (2020), the effect was large.",
            "Following Brown (2020) we test this.",
            "Sanchez, Mayo, and Rodriguez (2012) found it too.",
            "Indeed, De Vos (2019) agreed.",
            "The effect was large (see [Smith, 2020] for details).",
        ],
        [
            _ref(1, "Smith, J.", 2020),
            _ref(2, "Brown, K.", 2020),
            _ref(3, "Sanchez, A.; Mayo, B.; Rodriguez, C.", 2012),
            _ref(4, "De Vos, D.", 2019),
            _ref(5, "Rodriguez, E.", 2012),
        ],
        llm,
    )

    assert llm.calls == []
    assert sorted((x.text_id, x.xref_id, x.tier) for x in xrefs) == [
        (0, 1, "author-year"),
        (1, 2, "author-year"),
        (2, 3, "author-year"),
        (3, 4, "author-year"),
        (4, 1, "author-year"),
    ]


@pytest.mark.parametrize("hill_year", [1938, 1966])
async def test_works_inside_a_linked_multi_work_parenthetical_are_offered(hill_year):
    # The matcher links the whole parenthetical to Gordon; it names another
    # year, so it holds another work, even when that work shares the year.
    llm = RecordingLLM(
        lambda batch: [
            CitationMatch(text_id=text_id, citation_text=text, bib_id=2 if "Hill" in text else 1)
            for text_id, text in batch
        ]
    )
    xrefs, _receipt = await _link_author_year(
        [
            f"Force (per the length [Gordon et al., 1966] and velocity [Hill, {hill_year}] curves) fell."
        ],
        [_ref(1, "Gordon, A.; Huxley, A.; Julian, F.", 1966), _ref(2, None, hill_year)],
        llm,
    )

    assert llm.calls == [[(0, "[Gordon et al., 1966]"), (0, f"[Hill, {hill_year}]")]]
    assert sorted((x.text_id, x.xref_id, x.tier) for x in xrefs) == [
        (0, 1, "author-year"),
        (0, 2, "llm"),
    ]


_TIED_REFS = [
    _ref(1, "Smith, J., & Jones, K.", 2020),
    _ref(2, "Smith, J., & Williams, L.", 2020),
    _ref(3, "Taylor, M.", 2015),
]


@pytest.mark.parametrize(
    "pick",
    [_ref(3, "Taylor, M.", 2015), _ref(3, "Smith, M.", 2015), _ref(3, "Taylor, M.", 2020)],
    ids=["other-surname-other-year", "same-surname-other-year", "other-surname-same-year"],
)
async def test_llm_pick_outside_a_same_surname_tie_is_rejected(pick):
    llm = RecordingLLM(_every([3]))
    xrefs, receipt = await _link_author_year(
        ["As shown (Smith, 2020) this works."], [*_TIED_REFS[:2], pick], llm
    )

    assert xrefs == []
    [candidate] = receipt.candidates
    assert (candidate.accepted, candidate.bib_ids, candidate.rejection_reasons) == (
        False,
        (1, 2),
        ("ambiguous_same_surname_year", "llm_outside_shortlist"),
    )
    assert candidate.evidence == ("family_match", "year_match")


async def test_llm_hedge_between_tied_references_links_nothing():
    llm = RecordingLLM(_every([1, 2]))
    xrefs, receipt = await _link_author_year(
        ["As shown (Smith, 2020) this works."], _TIED_REFS, llm
    )

    assert xrefs == []
    [candidate] = receipt.candidates
    assert (candidate.accepted, candidate.rejection_reasons) == (
        False,
        ("ambiguous_same_surname_year", "llm_ambiguous"),
    )


async def test_llm_hedge_on_one_unmatched_work_links_nothing():
    # The parsed years are wrong, so the matcher finds no reference for Kim 2017.
    llm = RecordingLLM(_every([1, 2]))
    xrefs, receipt = await _link_author_year(
        ["Prior work (Kim, 2017) agrees."],
        [_ref(1, "Kim, J.", 2015), _ref(2, "Kim, S.", 2016)],
        llm,
    )

    assert xrefs == []
    [candidate] = receipt.candidates
    assert candidate.rejection_reasons == ("no_reference_match", "llm_ambiguous")


async def test_llm_picks_off_the_shortlist_are_dropped_from_the_link():
    llm = RecordingLLM(_every([2, 4]))
    xrefs, receipt = await _link_author_year(
        ["As shown (Smith, 2020) this works."], [*_TIED_REFS, _ref(4, "Taylor, M.", 2020)], llm
    )

    assert [(x.xref_id, x.tier) for x in xrefs] == [(2, "llm")]
    assert [(c.accepted, c.bib_ids) for c in receipt.candidates] == [(True, (2,))]


async def test_llm_pick_inside_a_same_surname_tie_is_accepted():
    llm = RecordingLLM(_every([2]))
    xrefs, receipt = await _link_author_year(
        ["As shown (Smith, 2020) this works."], _TIED_REFS, llm
    )

    assert [(x.text_id, x.xref_id, x.tier) for x in xrefs] == [(0, 2, "llm")]
    [candidate] = receipt.candidates
    assert (candidate.accepted, candidate.bib_ids, candidate.evidence) == (
        True,
        (2,),
        (
            "family_match",
            "year_match",
            "llm_overrode:ambiguous_same_surname_year",
            "llm_resolution",
        ),
    )


async def test_llm_may_pick_a_same_surname_year_entry_the_tie_left_out():
    # Parsed references without their year suffix tie on author counts, which
    # leaves the cited "2011b" entry (bib 2) off the shortlist.
    refs = [
        _ref(1, "Oddo, C.; Controzzi, M.; Beccai, L.; Cipriani, C.; Carrozza, M.", 2011),
        _ref(
            2, "Oddo, C.; Beccai, L.; Wessberg, J.; Wasling, H.; Mattioli, F.; Carrozza, M.", 2011
        ),
        _ref(3, "Oddo, C.; Controzzi, M.; Beccai, L.; Cipriani, C.; Carrozza, M.", 2011),
    ]
    llm = RecordingLLM(_every([2]))
    xrefs, receipt = await _link_author_year(
        ["Recordings were made in humans (Oddo et al., 2011b)."], refs, llm
    )

    assert [(x.text_id, x.xref_id, x.tier) for x in xrefs] == [(0, 2, "llm")]
    assert [(c.accepted, c.bib_ids) for c in receipt.candidates] == [(True, (2,))]


async def test_unresolved_works_of_a_group_are_offered_and_linked_one_by_one():
    llm = RecordingLLM(
        lambda batch: [
            CitationMatch(text_id=text_id, citation_text=text, bib_id=2 if "Jones" in text else 3)
            for text_id, text in batch
        ]
    )
    xrefs, _receipt = await _link_author_year(
        ["Prior work (Smith, 2020; Jones, 2019; Brown, 2016) agrees."],
        # The parsed years of Jones and Brown are wrong, so the matcher links Smith only.
        [_ref(1, "Smith, J.", 2020), _ref(2, "Jones, K.", 2018), _ref(3, "Brown, L.", 2017)],
        llm,
    )

    assert llm.calls == [[(0, "Jones, 2019"), (0, "Brown, 2016")]]
    assert sorted((x.xref_id, x.tier, x.contents) for x in xrefs) == [
        (1, "author-year", "Smith, 2020"),
        (2, "llm", "Jones, 2019"),
        (3, "llm", "Brown, 2016"),
    ]


@pytest.mark.parametrize(
    "citation",
    ["[Jones 2019; Brown 2016]", "[Jones 2019; Brown, in press]", "[Jones et al., 2019a, b]"],
)
async def test_every_llm_match_for_one_citation_text_is_kept(citation):
    llm = RecordingLLM(_every([2, 3]))
    xrefs, receipt = await _link_author_year(
        [f"Both were shown before {citation}."],
        [_ref(1, "Smith, J.", 2020), _ref(2, "Jones, K.", 2019), _ref(3, "Brown, L.", 2016)],
        llm,
    )

    assert llm.calls == [[(0, citation)]]
    assert sorted((x.text_id, x.xref_id, x.tier) for x in xrefs) == [(0, 2, "llm"), (0, 3, "llm")]
    assert [(c.style, c.bib_ids, c.accepted) for c in receipt.candidates] == [("llm", (2, 3), True)]


def _many_unresolved(count):
    names = [f"Na{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(count)]
    texts = [f"Prior work ({name}, 2001) agrees." for name in names]
    # Mis-parsed years: the matcher finds no reference, so each citation goes to Tier 3.
    refs = [_ref(index + 1, f"{name}, A.", 1999) for index, name in enumerate(names)]
    return texts, refs


async def test_tier3_requests_are_batched_and_a_failed_batch_loses_only_itself():
    texts, refs = _many_unresolved(100)

    def answer(batch):
        if any(text_id == 0 for text_id, _text in batch):
            raise RuntimeError("truncated structured output")
        return [
            CitationMatch(text_id=text_id, citation_text=text, bib_id=text_id + 1)
            for text_id, text in batch
        ]

    llm = RecordingLLM(answer)
    xrefs, _receipt = await _link_author_year(texts, refs, llm)

    assert [len(call) for call in llm.calls] == [40, 40, 20]
    assert _links(xrefs, "llm") == [(text_id, text_id + 1) for text_id in range(40, 100)]


async def test_processing_error_in_one_tier3_batch_still_propagates():
    texts, refs = _many_unresolved(50)

    def answer(batch):
        if any(text_id == 45 for text_id, _text in batch):
            raise ProcessingError("rate limit exhausted")
        return []

    with pytest.raises(ProcessingError, match="rate limit exhausted"):
        await _link_author_year(texts, refs, RecordingLLM(answer))
