"""Extra coverage for bibr.structure.citation_linker — Tier 3 bracket scan."""

from bibr.models import PaperReference
from bibr.paper_contents import CanonicalSection, PaperSection, PaperSentence
from bibr.structure.citation_linker import (
    _STRIP_CITE_SUP_RE,
    _is_citation_superscript,
    detect_bib_xrefs,
)


async def test_bracket_scan_runs_on_partially_covered_sentence():
    """Sentence with both numeric (resolved) and author-year brackets must
    still feed the author-year bracket to Tier 3."""

    sentences = [
        PaperSentence(
            text_id=1,
            text="Smith et al. [14] and [Jones, 2019] both reported.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
        )
    ]
    sections = [PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0)]
    references = [
        PaperReference(
            bib_id=14,
            title="A study",
            first_page=None,
            volume=None,
            authors="Smith",
            year=2020,
            container=None,
        ),
        PaperReference(
            bib_id=99,
            title="Another study",
            first_page=None,
            volume=None,
            authors="Jones",
            year=2019,
            container=None,
        ),
    ]

    captured: dict = {"deduped": None}

    class FakeLLM:
        async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash="x"):
            captured["deduped"] = list(ambiguous_citations)
            return []  # don't resolve — we only care that Tier 3 sees the bracket

    await detect_bib_xrefs(
        sentences=sentences,
        sections=sections,
        references=references,
        llm_client=FakeLLM(),
    )
    assert captured["deduped"], "Tier 3 must receive at least one candidate"
    assert any("[Jones, 2019]" in ct for _tid, ct in captured["deduped"]), (
        f"author-year bracket missed; got {captured['deduped']}"
    )


async def test_resolved_map_round_trips_with_normalization():
    """LLM-returned `citation_text` may have whitespace differences from the
    cite_text the orchestrator stored (the bracket path appends raw text
    without normalization).  The resolved_map must reconcile both sides
    through `_normalize_citation_text`, otherwise the lookup misses."""
    from bibr.schemas import CitationMatch

    # Bracket with internal whitespace (OCR line-wrap artifact).  The bracket
    # scan stores it raw; a real LLM almost always normalizes whitespace in
    # its returned `citation_text`, so the two strings won't match by ==.
    sentences = [
        PaperSentence(
            text_id=10,
            text="Per [Smith\n  et al.,\n  2020] this is true.",
            section_id=1,
            paragraph_id=1,
            page_number=1,
        )
    ]
    sections = [PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0)]
    references = [
        PaperReference(
            bib_id=42,
            title="A study",
            first_page=None,
            volume=None,
            authors="Smith",
            year=2020,
            container=None,
        ),
    ]

    class FakeLLM:
        async def resolve_citations(self, ambiguous_citations, reference_summary, file_hash="x"):
            tid, _cite_text = ambiguous_citations[0]
            # LLM normalizes whitespace as it round-trips the citation.
            return [
                CitationMatch(
                    text_id=tid,
                    citation_text="[Smith et al., 2020]",
                    bib_id=42,
                )
            ]

    out = await detect_bib_xrefs(
        sentences=sentences,
        sections=sections,
        references=references,
        llm_client=FakeLLM(),
    )
    assert any(x.xref_id == 42 and x.text_id == 10 for x in out), (
        f"expected resolved xref to bib 42 despite whitespace round-trip, got {out}"
    )


def test_reversed_numeric_range_recovered():
    from bibr.structure.citation_linker import _expand_numeric_range

    assert _expand_numeric_range("5-3") == [3, 4, 5]
    # Single-element ranges still work.
    assert _expand_numeric_range("7-7") == [7]


def test_pos_zero_does_not_wrap_to_last_char():
    # Text MUST end in an alphabetic character.  Without the `pos > 0` guard,
    # `text[pos - 1]` at pos=0 wraps to `text[-1]` (the last char); if that
    # char is alpha the function would wrongly preserve the marker.  An
    # ending like "." would mask the regression because "." is not alpha.
    text = "^{1} introduces a section starting with the marker text"
    assert text[-1].isalpha(), "test text must end in an alpha char to catch the wraparound bug"
    matches = list(_STRIP_CITE_SUP_RE.finditer(text))
    assert matches, "regex should match the leading marker"
    assert matches[0].start() == 0, "match must be at pos 0 to exercise the guard"
    out = _is_citation_superscript(matches[0])
    # At pos 0, there is no preceding char — the function should return ""
    assert out == "", f"expected '' at pos 0, got {out!r} (likely from text[-1] wraparound)"


async def test_equation_tags_in_display_formulas_not_cited():
    """Equation number tags inside display formulas must not become bib xrefs.

    Mined from a live run on 10.1515_econ-2022-0125: a numbered 39-entry
    bibliography makes tags "(1)"–"(11)" all valid bib ids, the tags sit
    whitespace-separated (no maths-adjacency rejection), and many formula
    sentences satisfy the style gate — 13 false "(N)" citations anchored
    on "[equation]" placeholder sentences.
    """
    sections = [PaperSection(section_id=1, header="Model", level=1, parent_section_id=0)]
    sentences = [
        PaperSentence(
            text_id=tid,
            text=rf"w = P F'(N_{tid}) , \quad ({n})",
            section_id=1,
            paragraph_id=tid,
            page_number=1,
            is_display_formula=True,
        )
        for tid, n in [(10, 2), (20, 3), (30, 4)]
    ]
    references = [
        PaperReference(
            bib_id=i,
            title=f"Study {i}",
            first_page=None,
            volume=None,
            authors="Smith",
            year=2000 + i,
            container=None,
        )
        for i in range(1, 6)
    ]
    out = await detect_bib_xrefs(
        sentences=sentences, sections=sections, references=references, llm_client=None
    )
    assert out == [], f"equation tags must not be linked as citations, got {out}"


async def test_parenthetical_numeric_citations_in_prose_still_linked():
    """The formula guard must not affect genuine prose (N) citations."""
    sections = [
        PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]
    sentences = [
        PaperSentence(
            text_id=tid,
            text=f"This point was established early ({n}).",
            section_id=1,
            paragraph_id=tid,
            page_number=1,
        )
        for tid, n in [(10, 2), (20, 3), (30, 4)]
    ]
    references = [
        PaperReference(
            bib_id=i,
            title=f"Study {i}",
            first_page=None,
            volume=None,
            authors="Smith",
            year=2000 + i,
            container=None,
            text_id=100 + i,
        )
        for i in range(1, 6)
    ]
    sentences.extend(
        PaperSentence(
            text_id=100 + i,
            text=f"[{i}] Smith AB. Printed reference {i}.",
            section_id=2,
            paragraph_id=100 + i,
        )
        for i in range(1, 6)
    )
    out = await detect_bib_xrefs(
        sentences=sentences, sections=sections, references=references, llm_client=None
    )
    assert {(x.text_id, x.xref_id) for x in out} == {(10, 2), (20, 3), (30, 4)}


def _author_year_refs():
    return [
        PaperReference(
            bib_id=i,
            title=f"Study {i}",
            first_page=None,
            volume=None,
            authors=author,
            year=year,
            container=None,
        )
        for i, (author, year) in enumerate(
            [
                ("Cartelier", 2018),
                ("Smith", 2011),
                ("Jones", 2012),
                ("Brown", 2013),
                ("Davis", 2014),
                ("Evans", 2015),
            ],
            start=1,
        )
    ]


async def test_equation_numbers_in_prose_not_cited_in_author_year_paper():
    """Prose mentions of equation numbers must not become bib xrefs when the
    paper cites author-year style.

    Mined from a live run on 10.1515_econ-2022-0125: the display-formula
    guard removes the equation tags themselves, but prose like "summing
    budget constraints (1) and (2)" still matches the parenthetical-numeric
    tier — the internal bib_ids are always a dense 1..N run, and three such
    sentences satisfy the style gate.  Papers don't mix parenthetical-numeric
    with author-year citations, so author-year evidence must suppress the
    numeric-fallback tiers.
    """
    sections = [PaperSection(section_id=1, header="Model", level=1, parent_section_id=0)]
    prose = [
        (10, "Cartelier (2018) develops this monetary framework."),
        (20, "Smith (2011) reached a similar conclusion."),
        (30, "Jones (2012) formalized the argument."),
        (40, "This can be verified by summing budget constraints (1) and (2)."),
        (50, "Substituting (2) into (3) yields the equilibrium condition."),
        (60, "Condition (4) then follows immediately."),
    ]
    sentences = [
        PaperSentence(text_id=tid, text=text, section_id=1, paragraph_id=tid, page_number=1)
        for tid, text in prose
    ]
    out = await detect_bib_xrefs(
        sentences=sentences, sections=sections, references=_author_year_refs(), llm_client=None
    )
    equation_hits = [x for x in out if x.text_id in (40, 50, 60)]
    assert equation_hits == [], (
        f"equation-number prose must not be linked as citations, got {equation_hits}"
    )
    author_year_hits = {(x.text_id, x.xref_id) for x in out}
    assert author_year_hits == {(10, 1), (20, 2), (30, 3)}, (
        f"author-year citations must survive, got {author_year_hits}"
    )


async def test_equation_tag_collisions_rejected_even_without_author_year_evidence():
    """Prose "(N)" where N is a known equation tag must not link, veto or not.

    Mined from the xref FP-miner residual (10.5018_economics-ejournal.ja.2009-39,
    52 surviving FPs): equation-heavy theory papers can carry too few author-year
    sentences to trip the style veto, yet their prose is saturated with equation
    mentions ("Plugging (3) into (1)", "Equation (2) is..."). The display
    formulas themselves carry the tags, so a candidate number that collides
    with a detected equation tag is rejected.
    """
    sections = [PaperSection(section_id=1, header="Model", level=1, parent_section_id=0)]
    formulas = [
        PaperSentence(
            text_id=tid,
            text=rf"U(C, T) = C^{{1-\eta}} e^{{-\beta T}} , \quad ({n})",
            section_id=1,
            paragraph_id=tid,
            page_number=1,
            is_display_formula=True,
        )
        for tid, n in [(5, 1), (6, 2), (7, 3)]
    ]
    prose = [
        (10, "There is not much difference between (1) and (2) for small values."),
        (20, "Plugging (3) into (1) and (2), one obtains the result."),
        (30, "The additive formulation (2) does not trivialize the welfare impacts."),
        (40, "This finding was reported before (4)."),
    ]
    sentences = formulas + [
        PaperSentence(text_id=tid, text=text, section_id=1, paragraph_id=tid, page_number=1)
        for tid, text in prose
    ]
    out = await detect_bib_xrefs(
        sentences=sentences, sections=sections, references=_author_year_refs(), llm_client=None
    )
    tag_hits = [x for x in out if x.xref_id in (1, 2, 3)]
    assert tag_hits == [], f"equation-tag collisions must be rejected, got {tag_hits}"
    # "(4)" collides with no tag: with tag-mentions rejected, the remaining
    # candidates span too few sentences for the style gate, so it drops too.
    assert out == [], f"expected no xrefs at all (style gate), got {out}"


async def test_tag_guard_does_not_ungate_flattened_superscript_tier():
    """Equation-tag rejection emptying Tier 1c must not let Tier 1d fire.

    Found in the round-2 corpus replay (10.5018_economics-ejournal.ja.2013-11):
    the tag guard emptied Tier 1c, which un-gated the flattened-superscript
    tier, and it emitted 5 footnote-marker FPs ("pension system.1").  The
    1c→1d cascade preference is decided by the paper's unguarded
    parenthetical-numeric surface signal; the guard only filters what
    Tier 1c emits.
    """
    sections = [PaperSection(section_id=1, header="Model", level=1, parent_section_id=0)]
    formulas = [
        PaperSentence(
            text_id=tid,
            text=rf"c_{{1t}} + s_t = w_t , \quad ({n})",
            section_id=1,
            paragraph_id=tid,
            page_number=1,
            is_display_formula=True,
        )
        for tid, n in [(5, 1), (6, 2), (7, 3)]
    ]
    prose = [
        (10, "Under the budget constraint (1), the allocation is optimal."),
        (20, "Eq. (2) becomes the following equation."),
        (30, "H corresponds to the ratio in (3)."),
        (40, "This reshaped the pension system.1"),
        (50, "The effect holds across OECD countries.2"),
        (60, "Participation rates kept falling.3"),
    ]
    sentences = formulas + [
        PaperSentence(text_id=tid, text=text, section_id=1, paragraph_id=tid, page_number=1)
        for tid, text in prose
    ]
    out = await detect_bib_xrefs(
        sentences=sentences, sections=sections, references=_author_year_refs(), llm_client=None
    )
    assert out == [], f"flattened tier must stay gated behind parenthetical signal, got {out}"


async def test_sparse_author_year_hits_do_not_suppress_numeric_fallback():
    """Below the style bar, author-year evidence must not veto the numeric tier.

    A numeric-citation paper can mention one or two author-year works in
    passing (e.g. in a methods aside); that must not disable genuine
    parenthetical-numeric linking.
    """
    sections = [
        PaperSection(section_id=1, header="Intro", level=1, parent_section_id=0),
        PaperSection(
            section_id=2,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]
    prose = [
        (10, "Cartelier (2018) develops this monetary framework."),
        (20, "Smith (2011) reached a similar conclusion."),
        (40, "This point was established early (2)."),
        (50, "Later work confirmed the effect (3)."),
        (60, "A replication followed (4)."),
    ]
    sentences = [
        PaperSentence(text_id=tid, text=text, section_id=1, paragraph_id=tid, page_number=1)
        for tid, text in prose
    ]
    references = _author_year_refs()
    for ref in references:
        ref.text_id = 100 + ref.bib_id
        sentences.append(
            PaperSentence(
                text_id=ref.text_id,
                text=f"[{ref.bib_id}] {ref.authors}. Printed reference.",
                section_id=2,
                paragraph_id=ref.text_id,
            )
        )
    out = await detect_bib_xrefs(
        sentences=sentences, sections=sections, references=references, llm_client=None
    )
    hits = {(x.text_id, x.xref_id) for x in out}
    assert {(40, 2), (50, 3), (60, 4)} <= hits, (
        f"numeric citations must survive sparse author-year mentions, got {hits}"
    )
