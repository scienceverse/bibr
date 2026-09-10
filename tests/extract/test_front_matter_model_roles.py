"""Front-role classifier scores as evidence in front-matter ownership."""

from __future__ import annotations

from bibr.extract.front_matter import (
    FrontRolePolicy,
    collect_front_matter_candidates,
    resolve_front_matter,
)
from bibr.extract.front_role import FrontRolePredictions, RoleScores
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

TITLE_BBOX = (90.0, 60.0, 500.0, 100.0)
BYLINE_BBOX = (90.0, 110.0, 500.0, 160.0)
AFF_BBOX = (90.0, 170.0, 500.0, 220.0)
ABSTRACT_BBOX = (90.0, 240.0, 500.0, 400.0)

# A separator-rich byline (Springer house style). The lexical byline shape
# refuses separator lists until ownership selection proves the record, so on
# its own this row never gets the byline role.
EIGHTEEN_AUTHORS = " · ".join(f"Given{i} Middle{i}. Surname{i}" for i in range(18))


def _scores(**probs: float) -> RoleScores:
    full = dict.fromkeys(
        (
            "title",
            "doi_line",
            "byline",
            "affiliation",
            "keywords",
            "abstract",
            "ref_header",
            "heading",
            "masthead",
            "body",
            "other",
        ),
        0.0,
    )
    full.update(probs)
    rest = 1.0 - sum(full.values())
    full["other"] += max(rest, 0.0)
    top = max(full, key=full.get)
    return RoleScores(probs=full, top=top, confidence=full[top])


def _sentence(text_id, text, *, paragraph_id, bbox, label="text", section_id=0, page=1):
    return PaperSentence(
        text_id=text_id,
        text=text,
        section_id=section_id,
        paragraph_id=paragraph_id,
        page_number=page,
        provenance=[Provenance(page_no=page, bbox=bbox)],
        region_meta={"region_type": label, "font_size": 9.0, "font_bold": False},
    )


def _summary(index, label, bbox, *, section_id=0, page=1):
    return RegionSummary(page=page, index=index, label=label, bbox=bbox, section_id=section_id)


def _section(section_id, header, section_type=CanonicalSection.UNKNOWN, *, level=None):
    return PaperSection(
        section_id=section_id,
        header=header,
        level=(0 if section_id == 0 else 1) if level is None else level,
        parent_section_id=None,
        section_type=section_type,
        provenance=[],
    )


def _contents(sentences, summaries, *, sections=None, predictions=None):
    sections = sections or [_section(0, "Root")]
    return PaperContents(
        sentences=sentences,
        sections=sections,
        tables=[],
        links=[],
        sections_text={s.section_id: "" for s in sections},
        region_summaries=summaries,
        front_role_predictions=predictions,
    )


def _page(predictions_by_index: dict[int, RoleScores]) -> FrontRolePredictions:
    return FrontRolePredictions(
        {(1, index): scores for index, scores in predictions_by_index.items()},
        model_version="test",
    )


def _consortium_contents(predictions=None):
    sentences = [
        _sentence(
            1,
            "Deep Phenotyping Of Rare Disease Cohorts",
            paragraph_id=1,
            bbox=TITLE_BBOX,
            label="doc_title",
        ),
        _sentence(2, EIGHTEEN_AUTHORS, paragraph_id=2, bbox=BYLINE_BBOX),
        _sentence(
            3,
            "Department of Genetics, University of Somewhere, City, Country",
            paragraph_id=3,
            bbox=AFF_BBOX,
        ),
    ]
    summaries = [
        _summary(0, "doc_title", TITLE_BBOX),
        _summary(1, "text", BYLINE_BBOX),
        _summary(2, "text", AFF_BBOX),
    ]
    return _contents(sentences, summaries, predictions=predictions)


def test_heuristics_alone_reject_the_eighteen_author_byline():
    candidates = collect_front_matter_candidates(_consortium_contents())
    byline = next(c for c in candidates if c.raw_text == EIGHTEEN_AUTHORS)
    assert "byline" not in byline.roles
    assert byline.model_roles == frozenset()


def test_model_byline_admits_the_long_byline_and_is_recorded():
    predictions = _page({1: _scores(byline=0.91)})
    candidates = collect_front_matter_candidates(_consortium_contents(predictions))
    byline = next(c for c in candidates if c.raw_text == EIGHTEEN_AUTHORS)
    assert "byline" in byline.roles
    assert byline.model_roles == frozenset({"byline"})
    assert byline.model_scores[0] == ("byline", 0.91)
    resolution, _ = resolve_front_matter(_consortium_contents(predictions))
    assert "front_role_model" in resolution.reason_flags
    assert resolution.selected_block_id is not None


def test_model_below_threshold_is_ignored():
    predictions = _page({1: _scores(byline=0.4)})
    candidates = collect_front_matter_candidates(_consortium_contents(predictions))
    byline = next(c for c in candidates if c.raw_text == EIGHTEEN_AUTHORS)
    assert "byline" not in byline.roles


def test_policy_threshold_is_honoured():
    predictions = _page({1: _scores(byline=0.4)})
    candidates = collect_front_matter_candidates(
        _consortium_contents(predictions), policy=FrontRolePolicy(min_confidence=0.3)
    )
    byline = next(c for c in candidates if c.raw_text == EIGHTEEN_AUTHORS)
    assert "byline" in byline.roles


def test_model_title_seeds_a_non_latin_title():
    title = "Влияние температуры на рост микроводорослей"
    sentences = [
        _sentence(1, title, paragraph_id=1, bbox=TITLE_BBOX, label="text"),
        _sentence(2, "Иванов И. И., Петров П. П.", paragraph_id=2, bbox=BYLINE_BBOX),
        _sentence(
            3, "Московский государственный университет, Москва", paragraph_id=3, bbox=AFF_BBOX
        ),
    ]
    summaries = [
        _summary(0, "text", TITLE_BBOX),
        _summary(1, "text", BYLINE_BBOX),
        _summary(2, "text", AFF_BBOX),
    ]
    without = collect_front_matter_candidates(_contents(sentences, summaries))
    assert "title" not in next(c for c in without if c.raw_text == title).roles
    predictions = _page(
        {0: _scores(title=0.95), 1: _scores(byline=0.9), 2: _scores(affiliation=0.9)}
    )
    with_model = collect_front_matter_candidates(
        _contents(sentences, summaries, predictions=predictions)
    )
    seeded = next(c for c in with_model if c.raw_text == title)
    assert "title" in seeded.roles and "title" in seeded.model_roles
    assert "byline" in next(c for c in with_model if c.raw_text.startswith("Иванов")).roles
    resolution, _ = resolve_front_matter(_contents(sentences, summaries, predictions=predictions))
    assert resolution.selected_block_id is not None
    assert resolution.selection_method == "unique_block"


def test_confident_masthead_cannot_root_a_record():
    masthead = "Advances In Applied Sciences Quarterly Review"
    sentences = [
        _sentence(
            1, masthead, paragraph_id=1, bbox=(90.0, 20.0, 500.0, 50.0), label="paragraph_title"
        ),
        _sentence(
            2,
            "Deep Phenotyping Of Rare Disease Cohorts",
            paragraph_id=2,
            bbox=TITLE_BBOX,
            label="doc_title",
        ),
        _sentence(3, "A. B. Smith,1 C. D. Jones2", paragraph_id=3, bbox=BYLINE_BBOX),
        _sentence(
            4, "Department of Genetics, University of Somewhere", paragraph_id=4, bbox=AFF_BBOX
        ),
    ]
    summaries = [
        _summary(0, "paragraph_title", (90.0, 20.0, 500.0, 50.0)),
        _summary(1, "doc_title", TITLE_BBOX),
        _summary(2, "text", BYLINE_BBOX),
        _summary(3, "text", AFF_BBOX),
    ]
    without = collect_front_matter_candidates(_contents(sentences, summaries))
    assert "title" in next(c for c in without if c.raw_text == masthead).roles
    predictions = _page({0: _scores(masthead=0.9)})
    with_model = collect_front_matter_candidates(
        _contents(sentences, summaries, predictions=predictions)
    )
    assert "title" not in next(c for c in with_model if c.raw_text == masthead).roles
    # A doc_title layout label is trusted over the veto.
    predictions = _page({1: _scores(masthead=0.95)})
    trusted = collect_front_matter_candidates(
        _contents(sentences, summaries, predictions=predictions)
    )
    assert "title" in next(c for c in trusted if c.raw_text.startswith("Deep")).roles


def test_model_byline_rescues_a_probation_paragraph_on_page_one():
    """A byline the section classifier typed as ACKNOWLEDGMENTS is admitted on
    the model's word even when the lexical byline shape fails."""
    byline = EIGHTEEN_AUTHORS
    sections = [
        _section(0, "Root"),
        _section(1, "Acknowledgments", CanonicalSection.ACKNOWLEDGMENT),
    ]
    sentences = [
        _sentence(
            1,
            "Deep Phenotyping Of Rare Disease Cohorts",
            paragraph_id=1,
            bbox=TITLE_BBOX,
            label="doc_title",
        ),
        _sentence(2, byline, paragraph_id=2, bbox=BYLINE_BBOX, section_id=1),
        _sentence(
            3, "Department of Genetics, University of Somewhere", paragraph_id=3, bbox=AFF_BBOX
        ),
    ]
    summaries = [
        _summary(0, "doc_title", TITLE_BBOX),
        _summary(1, "text", BYLINE_BBOX, section_id=1),
        _summary(2, "text", AFF_BBOX),
    ]
    without = collect_front_matter_candidates(_contents(sentences, summaries, sections=sections))
    assert not any(c.raw_text == byline for c in without)
    predictions = _page({1: _scores(byline=0.85)})
    with_model = collect_front_matter_candidates(
        _contents(sentences, summaries, sections=sections, predictions=predictions)
    )
    admitted = next(c for c in with_model if c.raw_text == byline)
    assert "byline" in admitted.roles


def _lone_title_contents(predictions=None):
    """One title-bearing row, labelled ``paragraph_title`` — no doc_title shield."""
    title = "Deep Phenotyping Of Rare Disease Cohorts"
    sentences = [
        _sentence(1, title, paragraph_id=1, bbox=TITLE_BBOX, label="paragraph_title"),
        _sentence(2, "A. B. Smith,1 C. D. Jones2", paragraph_id=2, bbox=BYLINE_BBOX),
        _sentence(
            3, "Department of Genetics, University of Somewhere", paragraph_id=3, bbox=AFF_BBOX
        ),
    ]
    summaries = [
        _summary(0, "paragraph_title", TITLE_BBOX),
        _summary(1, "text", BYLINE_BBOX),
        _summary(2, "text", AFF_BBOX),
    ]
    return _contents(sentences, summaries, predictions=predictions), title


def test_model_masthead_cannot_erase_the_only_title_candidate():
    """The classifier is evidence, never a veto of last resort.

    A confident masthead score kills the title seed on a row layout did not
    label ``doc_title``. When that row is the page's only title, dropping it
    makes front matter abstain and the paper loses a title the heuristics had —
    which is what every title regression in the 2026-09-02 validation replay
    was (7 of 192 papers, 6 of them abstaining outright). Falling back to the
    heuristic-only pass keeps it.
    """
    contents, title = _lone_title_contents()
    assert (
        "title"
        in next(c for c in collect_front_matter_candidates(contents) if c.raw_text == title).roles
    )

    contents, title = _lone_title_contents(_page({0: _scores(masthead=0.95)}))
    candidates = collect_front_matter_candidates(contents)

    assert "title" in next(c for c in candidates if c.raw_text == title).roles
    # The fallback pass consulted no scores, so nothing is attributed to the model.
    assert all(candidate.model_roles == frozenset() for candidate in candidates)
    resolution, _ = resolve_front_matter(contents)
    assert resolution.selected_block_id is not None


def test_model_evidence_survives_when_a_title_remains():
    """The fallback is conditional: a page that still has a title keeps the scores."""
    predictions = _page({1: _scores(byline=0.91)})
    candidates = collect_front_matter_candidates(_consortium_contents(predictions))

    assert any("title" in candidate.roles for candidate in candidates)
    byline = next(c for c in candidates if c.raw_text == EIGHTEEN_AUTHORS)
    assert "byline" in byline.model_roles


def _candidate(order, text, roles, *, model_roles=(), bbox=TITLE_BBOX):
    from bibr.extract.front_matter import FrontMatterCandidate

    return FrontMatterCandidate(
        candidate_id=f"front-matter-candidate-{order}",
        source_kind="paragraph",
        reading_order=order,
        page=1,
        bbox=bbox,
        region_label="text",
        font_size=9.0,
        font_bold=False,
        section_id=0,
        text_ids=(order,),
        paragraph_id=order,
        raw_text=text,
        normalized_text=text.casefold(),
        roles=frozenset(roles),
        model_roles=frozenset(model_roles),
        model_scores=(),
    )


def _blocks(*groups):
    from bibr.extract.front_matter import _make_block

    return tuple(_make_block(i, list(g)) for i, g in enumerate(groups, start=1))


RECORD = (
    _candidate(1, "Deep Phenotyping Of Rare Disease Cohorts", ["title"]),
    _candidate(2, "A. B. Smith, C. D. Jones", ["byline"], bbox=BYLINE_BBOX),
    _candidate(3, "https://doi.org/10.1234/abcd", ["doi"], bbox=AFF_BBOX),
)


def test_a_model_only_byline_does_not_veto_the_dominant_record():
    """A model byline is evidence, not an independently developed record.

    Splitting on one makes a second block, and a competitor holding *any*
    byline used to be enough to keep dominance fail-closed — which is how six
    papers in the 2026-09-02 validation replay went from ``unique_block`` to
    ``multiple_plausible_blocks`` and lost a title the heuristics had found.
    """
    from bibr.extract.front_matter import _select_dominant_coherent_block

    competitor = _candidate(4, "Received 12 March 2024", ["byline"], model_roles=["byline"])
    blocks = _blocks(RECORD, (competitor,))
    by_id = {c.candidate_id: c for c in (*RECORD, competitor)}

    dominant = _select_dominant_coherent_block(blocks, by_id)
    assert dominant is not None and dominant.block_id == blocks[0].block_id


def test_a_heuristic_byline_in_a_competitor_still_fails_closed():
    """Two genuinely developed records stay fail-closed, model or no model."""
    from bibr.extract.front_matter import _select_dominant_coherent_block

    competitor = _candidate(4, "E. F. Brown, G. H. Green", ["byline"])
    blocks = _blocks(RECORD, (competitor,))
    by_id = {c.candidate_id: c for c in (*RECORD, competitor)}

    assert _select_dominant_coherent_block(blocks, by_id) is None


# The Elsevier/Wiley first page: title, byline, then a boxed "A R T I C L E I N
# F O" / "Correspondence" header that the heading+TITLE seed admits as a title
# and the classifier types as a heading at 1.00, followed by the abstract.
SPLIT_PAGE = (
    _candidate(1, "Harmonic Enhancement To Optimize Ocular Activity Decoding", ["title"]),
    _candidate(
        2, "C. Demirel, L. Regus, H. Kose", ["byline"], model_roles=["byline"], bbox=BYLINE_BBOX
    ),
    _candidate(3, "A R T I C L E I N F O", ["title", "heading"], bbox=AFF_BBOX),
    _candidate(
        4,
        "Intelligent robotic systems for patients with motor impairments are",
        ["abstract"],
        bbox=ABSTRACT_BBOX,
    ),
)


def _root_indices(candidates, marked: set[int] = frozenset()):
    from bibr.extract.front_matter import MODEL_NON_TITLE_SEED_ROLE, _record_title_indices

    rows = tuple(
        c
        if i not in marked
        else _candidate(
            c.reading_order, c.raw_text, [*c.roles, MODEL_NON_TITLE_SEED_ROLE], bbox=c.bbox
        )
        for i, c in enumerate(candidates)
    )
    return rows, _record_title_indices(rows, allow_byline_only=True)


def test_a_model_denied_title_seed_cannot_root_a_second_record():
    """The byline is *correct* here — bounding it would throw away the win.

    What the model's byline does is complete the first record's anatomy, which
    promotes the boxed header below it into a second record root and cuts the
    real title away from the abstract. The header keeps its title role and
    loses only the right to root, so the page stays one block.
    """
    from bibr.extract.front_matter import group_front_matter_blocks

    unmarked, roots = _root_indices(SPLIT_PAGE)
    assert roots == frozenset({0, 2})
    assert len(group_front_matter_blocks(unmarked)) == 2

    marked, roots = _root_indices(SPLIT_PAGE, marked={2})
    assert roots == frozenset({0})
    (block,) = group_front_matter_blocks(marked)
    assert marked[0].candidate_id in block.title_candidate_ids


def test_a_model_denied_seed_keeps_its_title_role():
    """Evidence, never a decision: the veto is scoped to record rooting, so a
    page whose only title seed is model-denied still reports that title."""
    from bibr.extract.front_matter import group_front_matter_blocks

    marked, roots = _root_indices(SPLIT_PAGE[2:], marked={0})
    assert roots == frozenset()
    (block,) = group_front_matter_blocks(marked)
    assert "title" in marked[0].roles
    assert marked[0].candidate_id in block.title_candidate_ids


def test_the_classifier_denies_a_confident_heading_but_not_a_confident_title():
    from bibr.extract.front_matter import MODEL_NON_TITLE_SEED_ROLE, _model_denies_title_seed

    policy = FrontRolePolicy()
    assert _model_denies_title_seed(_scores(heading=1.0), policy) is True
    assert _model_denies_title_seed(_scores(title=1.0), policy) is False
    # Unsure is not denial: the veto needs the classifier to be confident.
    assert _model_denies_title_seed(_scores(heading=0.55, title=0.45), policy) is False
    assert _model_denies_title_seed(None, policy) is False
    # 1.0 turns the veto off.
    assert (
        _model_denies_title_seed(_scores(heading=1.0), FrontRolePolicy(record_root_confidence=1.01))
        is False
    )
    assert MODEL_NON_TITLE_SEED_ROLE == "model_non_title_seed"


def _boxed_header_contents(predictions=None):
    """Title, byline, a boxed section header the seed reads as a title, abstract."""
    title = "Harmonic Enhancement To Optimize Ocular Activity Decoding"
    header = "A R T I C L E I N F O"
    sentences = [
        _sentence(1, title, paragraph_id=1, bbox=TITLE_BBOX, label="paragraph_title"),
        _sentence(2, EIGHTEEN_AUTHORS, paragraph_id=2, bbox=BYLINE_BBOX),
        _sentence(3, header, paragraph_id=3, bbox=AFF_BBOX, label="paragraph_title"),
        _sentence(
            4,
            "Intelligent robotic systems for patients with motor impairments rely on "
            "ocular activity decoded from electrooculography recordings.",
            paragraph_id=4,
            bbox=ABSTRACT_BBOX,
            label="abstract",
        ),
    ]
    summaries = [
        _summary(0, "paragraph_title", TITLE_BBOX),
        _summary(1, "text", BYLINE_BBOX),
        _summary(2, "paragraph_title", AFF_BBOX),
        _summary(3, "abstract", ABSTRACT_BBOX),
    ]
    return _contents(sentences, summaries, predictions=predictions), title, header


def test_a_confident_heading_score_marks_the_seed_it_scored():
    """The wiring: real scores, through ``collect_front_matter_candidates``."""
    from bibr.extract.front_matter import MODEL_NON_TITLE_SEED_ROLE

    predictions = _page({1: _scores(byline=0.97), 2: _scores(heading=1.0)})
    contents, title, header = _boxed_header_contents(predictions)
    candidates = collect_front_matter_candidates(contents)

    marked = next(c for c in candidates if c.raw_text == header)
    assert MODEL_NON_TITLE_SEED_ROLE in marked.roles
    # Scoped to rooting: the row keeps whatever title role it had.
    real = next(c for c in candidates if c.raw_text == title)
    assert "title" in real.roles
    assert MODEL_NON_TITLE_SEED_ROLE not in real.roles


def _selected_titles(resolution):
    selected = next(b for b in resolution.blocks if b.block_id == resolution.selected_block_id)
    by_id = {c.candidate_id: c for c in resolution.candidates}
    return [by_id[cid].raw_text for cid in selected.title_candidate_ids]


def test_the_page_the_model_used_to_split_stays_one_record():
    """End to end, both arms: the model may add the byline it alone can see
    without costing the page the title the heuristics already had."""
    contents, title, _ = _boxed_header_contents()
    heuristic, _ = resolve_front_matter(contents)
    assert len(heuristic.blocks) == 1
    assert title in _selected_titles(heuristic)

    predictions = _page({1: _scores(byline=0.97), 2: _scores(heading=1.0)})
    contents, title, _ = _boxed_header_contents(predictions)
    resolution, _ = resolve_front_matter(contents)

    byline = next(c for c in resolution.candidates if c.raw_text == EIGHTEEN_AUTHORS)
    assert "byline" in byline.model_roles
    assert len(resolution.blocks) == 1
    assert title in _selected_titles(resolution)
