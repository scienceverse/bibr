import pickle

from bibr.paper_contents import PaperContents


def _empty_contents(**kw):
    return PaperContents(sentences=[], sections=[], tables=[], links=[], sections_text={}, **kw)


def test_ref_line_geometry_defaults_none():
    assert _empty_contents().ref_line_geometry is None


def test_ref_line_geometry_survives_pickle_round_trip():
    geo = [
        {
            "text": "Aknin, L.",
            "page": 5,
            "x0": 72.0,
            "y_top": 710.0,
            "x1": 180.0,
            "y_bottom": 700.0,
            "font_size": 10.0,
        }
    ]
    payload = pickle.dumps(_empty_contents(ref_line_geometry=geo))  # nosemgrep
    back = pickle.loads(payload)  # nosemgrep
    assert back.ref_line_geometry == geo
