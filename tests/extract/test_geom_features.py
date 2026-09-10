from bibr.extract.geom_features import PROD_FEATURE_KEYS, line_features
from bibr.ocr.ref_geometry import LineRecord


def _ln(text, x0, y_top=700.0, page=1, fs=10.0):
    return LineRecord(text, page, x0, y_top, x0 + 400.0, y_top - 10.0, fs)


def test_feature_keys_are_exactly_prod_keys():
    feats = line_features([_ln("Aknin, L. B. (2013). Title.", 72.0)])
    assert set(feats[0]) == set(PROD_FEATURE_KEYS)
    assert "region_label" not in feats[0]  # region features dropped
    assert "outside_ref_region" not in feats[0]


def test_hanging_indent_dx_prev():
    lines = [_ln("Aknin, L. B. (2013). Title.", 72.0), _ln("continued journal info", 108.0)]
    feats = line_features(lines)
    assert feats[0]["dx_prev"] == 0.0  # first line, no prev
    assert feats[1]["dx_prev"] > 0  # indented continuation


def test_author_date_and_numbered_starts():
    feats = line_features([_ln("Aknin, L. B. (2013).", 72.0), _ln("[12] Smith, J. 2009.", 72.0)])
    assert feats[0]["starts_author_date"] is True
    assert feats[1]["starts_numbered"] is True
    assert feats[0]["contains_year"] is True


def test_early_year_paren_fires_on_geometry_blind_starts():
    # Reference starts the author-date regex misses (corporate authors with no
    # comma, diacritic surnames) but which all carry an early (year)/(n.d.) cue —
    # the geom psych merge failure mode.
    starts = [
        "Social and Behavioral Sciences Team. (2015). Foo.",
        "United Nations. (n.d.). Report.",
        "U.S. Social Security Administration. (2023). Title.",
        "Bürkner, P.-C. (2017). Title.",
    ]
    for s in starts:
        f = line_features([_ln(s, 263.0)])[0]
        assert f["early_year_paren"] is True, s
        assert f["starts_author_date"] is False or s.startswith("Bürkner")  # regex misses these


def test_early_year_paren_silent_on_continuations():
    conts = [
        "of risky decisions. Cognition, 68(2), 23-49.",
        "Journal of Experimental Psychology: General, 144(2),",
        "https://doi.org/10.1037/a0036577",
    ]
    for s in conts:
        f = line_features([_ln(s, 263.0)])[0]
        assert f["early_year_paren"] is False, s
