"""The feature contract is shared between bibr and the trainer, so it has to be
stable in exactly two places: the key set, and the page frame.

OCR hands out two boxes for the same region — ``bbox_2d`` (0..1000, top-down)
and ``bbox_pdf_pts`` (PDF points, bottom-up). Training reads the first, the
pipeline reads the second. If they do not land in the same frame the model is
fed upside-down geometry at inference and nothing raises.
"""

from bibr.extract.front_role_features import (
    PROD_FEATURE_KEYS,
    FrontRegion,
    region_features,
)

PAGE_W, PAGE_H = 595.28, 841.89


def _region(**kw) -> FrontRegion:
    base = {
        "page": 0,
        "index": 0,
        "label": "text",
        "text": "x",
        "x0": 0.1,
        "y0": 0.1,
        "x1": 0.9,
        "y1": 0.2,
    }
    base.update(kw)
    return FrontRegion(**base)


def test_the_two_ocr_boxes_land_in_the_same_frame():
    # The same region as OCR emits it both ways (a real page-1 title row).
    image = FrontRegion.from_image_bbox(
        [54, 138, 784, 183], page=0, index=1, label="doc_title", text="T"
    )
    pdf = FrontRegion.from_pdf_bbox(
        [32.14, 687.82, 466.70, 725.71],
        page_w=PAGE_W,
        page_h=PAGE_H,
        page=0,
        index=1,
        label="doc_title",
        text="T",
    )
    for got, want in (
        (pdf.x0, image.x0),
        (pdf.y0, image.y0),
        (pdf.x1, image.x1),
        (pdf.y1, image.y1),
    ):
        assert abs(got - want) < 0.005, (got, want)


def test_pdf_conversion_flips_the_y_axis():
    # y-up in, y-down out: a box near the top of the page must have a small y0.
    top = FrontRegion.from_pdf_bbox(
        [0, 800, 100, 830], page_w=PAGE_W, page_h=PAGE_H, page=0, index=0, label="", text=""
    )
    assert top.y0 < 0.1


def test_feature_keys_are_the_declared_contract():
    feats = region_features([_region()])
    assert set(feats[0]) == PROD_FEATURE_KEYS


def test_font_is_scored_relative_to_the_page():
    # Absolute point sizes vary per journal; "biggest type on the page" does not.
    feats = region_features(
        [
            _region(index=0, text="Journal of Things", font_size=6.0),
            _region(index=1, text="A Title", font_size=12.0),
        ]
    )
    assert feats[0]["font_rel_max"] == 0.5
    assert feats[1]["font_rel_max"] == 1.0


def test_neighbour_deltas_do_not_cross_a_page_break():
    feats = region_features(
        [
            _region(page=0, index=0, y0=0.8, y1=0.9, font_size=6.0),
            _region(page=1, index=0, y0=0.1, y1=0.2, font_size=12.0),
        ]
    )
    assert feats[1]["dy_prev"] == 0.0
    assert feats[1]["dfont_prev"] == 0.0
    assert feats[1]["is_first_on_page"] is True


def test_byline_shaped_text_scores_differently_from_prose():
    byline, prose = region_features(
        [
            _region(index=0, text="A. B. Smith,1 C. D. Jones,2 and E. F. Wu3*"),
            _region(index=1, text="We present a load side management scheme for networks."),
        ]
    )
    assert byline["initial_ratio"] > prose["initial_ratio"]
    assert byline["n_marks"] == 1
    assert byline["comma_per_token"] > prose["comma_per_token"]


def test_the_decoy_cues_fire_on_their_own_rows():
    cite, contrib, corresp = region_features(
        [
            _region(index=0, text="Citation: Ashraf M, Khan AR (2021) PLOS ONE 16(12)."),
            _region(index=1, text="Maria Ashraf and Abdul Khan contributed equally to this work."),
            _region(index=2, text="* Corresponding author: maria@neduet.edu.pk"),
        ]
    )
    assert cite["has_citation_cue"] and not cite["has_contrib_cue"]
    assert contrib["has_contrib_cue"] and not contrib["has_citation_cue"]
    assert corresp["has_corresp_cue"] and corresp["has_email"]
