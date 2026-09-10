from bibr.config import GlobalSettings
from bibr.local.cli import _build_parser


def test_geom_is_a_valid_seg_strategy():
    s = GlobalSettings(REF_SEG_STRATEGY="GEOM")  # validator lowercases
    assert s.REF_SEG_STRATEGY == "geom"


def test_geom_settings_defaults(monkeypatch):
    # conftest pins REF_SEG_STRATEGY=llm; delete it to see the real field default.
    monkeypatch.delenv("REF_SEG_STRATEGY", raising=False)
    s = GlobalSettings()
    assert s.REF_SEG_STRATEGY == "geom"  # geom is the default segmenter
    assert s.REF_GEOM_SEG_MODEL_ID == "scienceverse/bibr-geom-segmenter-v1"
    # Pinned to the multilingual feature-port bundle commit.
    assert s.REF_GEOM_SEG_REVISION == "4d1702e2c766d96c8887bd4b30ef56637aa9b32c"
    assert s.REF_GEOM_SEG_CASCADE_THRESHOLD == 0.90


def test_cli_ref_seg_flag_parses():
    args = _build_parser().parse_args(["chew", "x.pdf", "--ref-seg", "geom"])
    assert args.ref_seg == "geom"
