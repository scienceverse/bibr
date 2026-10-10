"""Page-level front-matter model: contract, parsing and record selection.

All titles, names and geometry below are invented. The selection fixtures are
pages the heuristics abstain on (two complete records) or select wrongly, with
a canned model answer standing in for the served model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bibr.extract import front_page_model as fpm
from bibr.extract.front_matter import (
    PAGE_MODEL_BLOCK_ID,
    _apply_page_model,
    collect_front_matter_candidates,
    group_front_matter_blocks,
    resolve_front_matter,
)
from bibr.extract.front_page_model import (
    PageAffiliation,
    PageAuthor,
    PageRecordPrediction,
    page_regions_from_ocr,
    parse_prediction,
    render_region_listing,
    render_target,
)
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    Provenance,
    RegionSummary,
)

# -- contract ---------------------------------------------------------------


def test_render_target_round_trips_through_the_parser():
    text = render_target(
        target=[(1, 4), (1, 2), (1, 3)],
        articles_on_page=2,
        title="A Title",
        authors=[PageAuthor("Ana Example", ("1", "*")), PageAuthor("Ben Sample", ("2",))],
        affiliations=[PageAffiliation("1", "Univ A"), PageAffiliation("2", "Univ B")],
        doi="10.1000/x.1",
    )

    assert json.loads(text)["target_regions"] == ["p1r2", "p1r3", "p1r4"]
    prediction = parse_prediction(text, known=frozenset({(1, 2), (1, 3), (1, 4)}))

    assert prediction is not None
    assert prediction.target == frozenset({(1, 2), (1, 3), (1, 4)})
    assert prediction.articles_on_page == 2
    assert prediction.authors[0] == PageAuthor("Ana Example", ("1", "*"))
    assert prediction.affiliations[1] == PageAffiliation("2", "Univ B")
    assert prediction.unknown_region_ids == ()


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[]", '{"target_regions": "p1r1", "articles_on_page": 1}', '{"x": 1}'],
)
def test_malformed_answers_are_rejected(text):
    assert parse_prediction(text) is None


def test_region_ids_the_page_does_not_have_are_reported():
    text = '{"articles_on_page":1,"target_regions":["p1r1","p9r9","oops"]}'

    prediction = parse_prediction(text, known=frozenset({(1, 1)}))

    assert prediction is not None
    assert prediction.target == frozenset({(1, 1)})
    assert prediction.unknown_region_ids == ("p9r9", "oops")


def test_region_listing_numbers_pages_from_one_and_clips_text():
    pages = [
        [
            {"index": 0, "label": "header", "bbox_2d": [10, 5, 990, 30], "content": "J. Inv."},
            {"index": 1, "label": "image", "bbox_2d": [10, 40, 500, 300], "content": ""},
            {"index": 2, "label": "text", "bbox_2d": [10, 310, 990, 900], "content": "x " * 400},
        ],
        [{"index": 0, "label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": "page two"}],
        [{"index": 0, "label": "text", "bbox_2d": [0, 0, 1000, 1000], "content": "page three"}],
    ]

    regions = page_regions_from_ocr(pages, max_pages=2)
    listing = render_region_listing(regions)

    assert [r.region_id for r in regions] == ["p1r0", "p1r1", "p1r2", "p2r0"]
    assert "[p1r0] header (10,5,990,30): J. Inv." in listing
    assert "[p1r1] image (10,40,500,300)" in listing.splitlines()
    assert "page three" not in listing
    assert max(len(line) for line in listing.splitlines()) < fpm.MAX_REGION_CHARS + 60


def test_client_posts_schema_constrained_request(monkeypatch):
    sent = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"articles_on_page":1,"target_regions":["p1r0"]}'}}
                ]
            }

    client = fpm.FrontPageModelClient(
        "http://localhost:8000/v1",
        model="bibr-front-page",
        timeout=5,
        max_pages=2,
        image_max_side=512,
        send_images=False,
    )
    monkeypatch.setattr(
        client._client, "post", lambda url, json: sent.update(url=url, body=json) or _Response()
    )

    prediction = client.predict([[{"index": 0, "label": "doc_title", "content": "T"}]])

    assert sent["url"] == "http://localhost:8000/v1/chat/completions"
    assert sent["body"]["response_format"]["json_schema"]["schema"] == fpm.RESPONSE_SCHEMA
    assert sent["body"]["temperature"] == 0.0
    assert prediction is not None and prediction.target == frozenset({(1, 0)})
    client.close()


def test_predict_front_page_is_off_without_a_url():
    settings = SimpleNamespace(ml=SimpleNamespace(front_page_model_mode="primary"))

    assert fpm.load_front_page_model(settings) is None
    assert fpm.predict_front_page(settings, [[{"content": "x"}]]) is None


# -- selection --------------------------------------------------------------

LONELINESS = "Loneliness Among Older Immigrants in Northern Cities"
STUDENTS = "Mental Health of University Students During the Pandemic"


def _settings(mode: str, min_share: float = 0.8):
    return SimpleNamespace(
        ml=SimpleNamespace(
            front_page_model_mode=mode,
            front_page_model_min_share=min_share,
            front_role_min_confidence=0.5,
            front_role_masthead_confidence=0.8,
            front_role_record_root_confidence=0.9,
        )
    )


def _page(rows: list[tuple[int, str, float, str]], *, detected_title: str | None = None):
    """Rows of (region index, text, y, layout label) on page 1, each its own region."""

    sentences = []
    summaries = []
    for index, text, y, label in rows:
        bbox = (60.0, y, 460.0, y + 30.0)
        sentences.append(
            PaperSentence(
                text_id=index + 1,
                text=text,
                section_id=0,
                paragraph_id=index + 1,
                page_number=1,
                provenance=[Provenance(page_no=1, bbox=bbox)],
                region_meta={"region_type": label, "font_size": 9.0, "font_bold": False},
            )
        )
        summaries.append(RegionSummary(page=1, index=index, label=label, bbox=bbox, section_id=0))
    section = PaperSection(
        section_id=0,
        header="Root",
        level=0,
        parent_section_id=None,
        section_type=CanonicalSection.UNKNOWN,
        provenance=[],
    )
    return PaperContents(
        sentences=sentences,
        sections=[section],
        tables=[],
        links=[],
        sections_text={0: ""},
        detected_title=detected_title,
        region_summaries=summaries,
    )


def _two_articles(detected_title: str | None = None) -> PaperContents:
    return _page(
        [
            (0, LONELINESS, 60.0, "doc_title"),
            (1, "Mei Lin, Jordan Smith", 100.0, "text"),
            (2, "We interviewed older immigrants about loneliness.", 140.0, "abstract"),
            (3, STUDENTS, 300.0, "doc_title"),
            (4, "Anna Example, Ben Sample", 340.0, "text"),
            (5, "We surveyed students about their mental health.", 380.0, "abstract"),
        ],
        detected_title=detected_title,
    )


def _predict(contents: PaperContents, *keys: tuple[int, int], count: int = 2) -> PaperContents:
    contents.front_page_prediction = PageRecordPrediction(
        target=frozenset(keys), articles_on_page=count
    )
    return contents


def _selected_text(resolution) -> list[str]:
    block = next(b for b in resolution.blocks if b.block_id == resolution.selected_block_id)
    by_id = {c.candidate_id: c for c in resolution.candidates}
    return [by_id[cid].raw_text for cid in block.candidate_ids]


def test_heuristics_alone_abstain_on_two_records():
    resolution, issues = resolve_front_matter(
        _two_articles(STUDENTS), target_required=True, settings=_settings("off")
    )

    assert resolution.selected_block_id is None
    assert [issue.code for issue in issues] == ["VAL_METADATA_MULTI_ITEM"]


@pytest.mark.parametrize("mode", ["arbiter", "primary"])
def test_model_selects_the_target_record_the_heuristics_abstain_on(mode):
    contents = _predict(_two_articles(STUDENTS), (1, 3), (1, 4), (1, 5))

    resolution, issues = resolve_front_matter(
        contents, target_required=True, settings=_settings(mode)
    )

    assert issues == ()
    assert resolution.selection_method == "page_model"
    assert _selected_text(resolution)[0] == STUDENTS
    assert "multiple_plausible_blocks" not in resolution.reason_flags
    assert "front_page_model" in resolution.reason_flags


def test_mode_off_ignores_a_prediction():
    contents = _predict(_two_articles(STUDENTS), (1, 3), (1, 4), (1, 5))

    resolution, _ = resolve_front_matter(contents, target_required=True, settings=_settings("off"))

    assert resolution.selected_block_id is None
    assert "front_page_model" not in resolution.reason_flags


def _one_article() -> PaperContents:
    return _page(
        [
            (0, STUDENTS, 60.0, "doc_title"),
            (1, "Anna Example, Ben Sample", 100.0, "text"),
            (2, "We surveyed students about their mental health.", 140.0, "abstract"),
        ]
    )


def test_arbiter_never_overrides_a_heuristic_selection():
    contents = _predict(_one_article(), (1, 2), count=1)

    resolution, _ = resolve_front_matter(contents, settings=_settings("arbiter"))

    assert resolution.selection_method == "unique_block"


def test_primary_agreement_keeps_the_record_and_says_so():
    contents = _predict(_one_article(), (1, 0), (1, 1), (1, 2), count=1)

    resolution, issues = resolve_front_matter(
        contents, target_required=True, settings=_settings("primary")
    )

    assert issues == ()
    assert resolution.selection_method == "page_model"
    assert "page_model_agrees:unique_block" in resolution.reason_flags


def test_a_target_outside_every_candidate_leaves_the_heuristics_alone():
    contents = _one_article()
    contents.front_page_prediction = PageRecordPrediction(
        target=frozenset({(1, 9)}), articles_on_page=1
    )

    resolution, _ = resolve_front_matter(contents, settings=_settings("primary"))

    assert resolution.selection_method == "unique_block"
    assert "page_model_no_overlap" in resolution.reason_flags


@pytest.mark.parametrize(("mode", "expected"), [("primary", None), ("arbiter", "block-1")])
def test_primary_disagreement_with_the_heuristics_abstains(mode, expected):
    candidates = collect_front_matter_candidates(_two_articles(STUDENTS))
    blocks = group_front_matter_blocks(candidates)
    by_id = {c.candidate_id: c for c in candidates}
    assert len(blocks) == 2
    prediction = PageRecordPrediction(target=frozenset({(1, 3), (1, 4)}), articles_on_page=2)

    selected, method, _, flags = _apply_page_model(
        mode=mode,
        prediction=prediction,
        blocks=blocks,
        by_id=by_id,
        selected=blocks[0],
        method="coherent_dominance",
        min_share=0.8,
    )

    if expected is None:
        assert selected is None and method == "abstained"
        assert "page_model_disagreement:coherent_dominance" in flags
    else:
        assert selected is blocks[0] and method == "coherent_dominance"


def test_unknown_region_ids_disable_the_prediction():
    contents = _two_articles(STUDENTS)
    contents.front_page_prediction = PageRecordPrediction(
        target=frozenset({(1, 3)}), articles_on_page=2, unknown_region_ids=("p4r1",)
    )

    resolution, _ = resolve_front_matter(contents, settings=_settings("primary"))

    assert resolution.selected_block_id is None
    assert "page_model_unknown_regions" in resolution.reason_flags


def test_model_trims_a_block_holding_a_second_title():
    # A title with no byline of its own (a notice, a previous item): the grouping
    # keeps both titles in one block, and the model names only the second.
    contents = _page(
        [
            (0, "Coping With Grief in Rural Family Clinics", 40.0, "doc_title"),
            (1, STUDENTS, 120.0, "doc_title"),
            (2, "Anna Example, Ben Sample", 160.0, "text"),
            (3, "We surveyed students about their mental health.", 200.0, "abstract"),
        ]
    )
    _predict(contents, (1, 1), (1, 2), (1, 3), count=1)

    resolution, _ = resolve_front_matter(contents, settings=_settings("primary"))

    assert resolution.selection_method == "page_model"
    assert resolution.selected_block_id == PAGE_MODEL_BLOCK_ID
    assert _selected_text(resolution) == [
        STUDENTS,
        "Anna Example, Ben Sample",
        "We surveyed students about their mental health.",
    ]


def test_expected_identity_outranks_the_model():
    from bibr.pipeline.identity import ExpectedIdentity

    contents = _predict(_two_articles(STUDENTS), (1, 0), (1, 1), (1, 2))
    identity = ExpectedIdentity(queue_record_id="q1", expected_title=STUDENTS)

    resolution, _ = resolve_front_matter(
        contents, expected_identity=identity, settings=_settings("primary")
    )

    assert resolution.selection_method == "expected_title"
    assert _selected_text(resolution)[0] == STUDENTS
