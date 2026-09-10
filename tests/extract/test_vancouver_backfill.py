"""Vancouver/medical journal-tail backfill.

The NER ref parser drops ``year``/``container`` — and tags the whole numeric tail
``O``, or lumps it into one span — on Vancouver-style refs whose tail reads
``<Container> <YEAR>;<VOL>(<ISS>):<PAGES>.``. These recover those fields verbatim
from the printed segment. The ``<YEAR>[;:]`` anchor is Vancouver-specific — APA
parenthesizes the year and never puts ``;``/``:`` right after it — so the backfill
cannot misfire on non-Vancouver styles."""

import pytest

from bibr.extract.ref_extractor import (
    _backfill_vancouver_tail,
    _finalize_reference_fields,
    _normalize_vol_issue,
    _split_vol_issue,
)


def test_year_and_container_from_volume_anchored_tail():
    seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. J Synth Garden Res 2021; 23: e18773."
    fields = {"year": None, "container": None, "volume": "23"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2021
    assert fields["container"] == "J Synth Garden Res"


def test_year_only_when_container_already_parsed():
    seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Synth Studies 2020; 75: 1080."
    fields = {"year": None, "container": "Synth Studies", "volume": "75"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2020
    # An already-parsed container is never overwritten.
    assert fields["container"] == "Synth Studies"


def test_container_only_when_year_already_parsed():
    seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. J Synth Garden Res 2013; 15: e146."
    fields = {"year": 2013, "container": None, "volume": "15"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2013
    assert fields["container"] == "J Synth Garden Res"


def test_volume_less_tail_uses_colon_page_anchor():
    seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Synth Open 2018: e019663."
    fields = {"year": None, "container": None, "volume": None}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2018
    assert fields["container"] == "Synth Open"


def test_multiword_abbreviated_journal():
    seg = "1. Example AB, Sample CD, Reader EF, et al. What can garden observations teach us? J Synth Soc Plant Sci Educ 2008; 59: 938-955."
    fields = {"year": None, "container": None, "volume": "59"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2008
    assert fields["container"] == "J Synth Soc Plant Sci Educ"


def test_container_after_question_mark_title():
    seg = "1. Example AB, Sample CD, Reader EF, et al. What can garden observations teach us? Synth Field Res 2023; 23: 770-780."
    fields = {"year": None, "container": None, "volume": "23"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2023
    assert fields["container"] == "Synth Field Res"


def test_ampersand_journal_name():
    seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Plants & Seasons 2008; 22: 187-200."
    fields = {"year": None, "container": None, "volume": "22"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 2008
    assert fields["container"] == "Plants & Seasons"


def test_does_not_override_existing_year():
    seg = "Smith J. A study. Lancet 2020; 395: 1054."
    fields = {"year": 1999, "container": None, "volume": "395"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] == 1999  # untouched


def test_apa_parenthesized_year_does_not_backfill():
    # APA: year is parenthesized and never followed by ';'/':'. Must not fire.
    seg = "Smith, J., & Doe, A. (2020). A study of things. Journal of Things, 12(3), 45-67."
    fields = {"year": None, "container": None, "volume": "12"}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] is None
    assert fields["container"] is None


def test_in_press_ref_not_backfilled():
    seg = "Smith J. A forthcoming study. J Synth Garden Res in press."
    fields = {"year": None, "container": None, "volume": None}
    _backfill_vancouver_tail(fields, seg)
    assert fields["year"] is None


def test_container_requires_title_boundary_period():
    # A capitalized fragment inside the title (no sentence-ending period right
    # before it) must NOT be mistaken for the container.
    seg = "Doe J. Studying the Human Genome Project 2020; 5: 3."
    fields = {"year": None, "container": None, "volume": "5"}
    _backfill_vancouver_tail(fields, seg)
    # Year is still a safe volume-anchored fill...
    assert fields["year"] == 2020
    # ...but no clean title→journal boundary, so container stays null.
    assert fields["container"] is None


def test_finalize_reference_fields_wires_backfill_and_infers_journal_type():
    # End-to-end through the shared finalize step: backfill runs, and the
    # now-present container drives bib_type inference to journal_article.
    seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. J Synth Plant Sci 2016; 51: 833-842."
    fields = {
        "bib_id": 1,
        "title": "A synthetic study of garden observations",
        "authors": "Example AB, Sample CD, Reader EF, et al.",
        "container": None,
        "year": None,
        "volume": "51",
        "issue": None,
        "first_page": "833",
        "last_page": "842",
        "doi": None,
    }
    out = _finalize_reference_fields(fields, seg)
    assert out["year"] == 2016
    assert out["container"] == "J Synth Plant Sci"
    assert out["bib_type"] == "journal_article"


class TestVancouverNumericTail:
    """Recover YEAR;VOL(ISS):PAGES fields from synthetic references when a parser drops or combines numeric spans."""

    def test_lumped_issue_span_is_decomposed(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Synth Nature. 1994;368(6469):339-342."
        fields = {
            "year": 1994,
            "container": "Synth Nature",
            "volume": None,
            "issue": "368(6469):339-342",
            "first_page": None,
            "last_page": None,
        }
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "368"
        assert fields["issue"] == "6469"
        assert fields["first_page"] == "339"
        assert fields["last_page"] == "342"

    def test_single_page_tail(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. N Synth J Sci. 2015;373(9):880."
        fields = {"year": 2015, "container": "N Synth J Sci", "volume": None, "issue": None}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "373"
        assert fields["issue"] == "9"
        assert fields["first_page"] == "880"
        assert fields.get("last_page") is None

    def test_issue_less_tail_with_lumped_last_page(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Observation. 2018;125:118–25. https://doi.org/10.5555/example.reference."
        fields = {
            "year": 2018,
            "container": "Observation",
            "volume": None,
            "issue": None,
            "first_page": None,
            "last_page": "118–25",
            "title": "A synthetic study of garden observations",
        }
        out = _finalize_reference_fields(fields, seg)
        assert out["volume"] == "125"
        assert out["issue"] is None
        assert out["first_page"] == "118"
        assert out["last_page"] == "125"

    def test_supplement_issue_and_letter_prefixed_page(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Synth Botanist. 2016;56(Suppl 2):S163-166. https://doi.org/10.5555/example.reference."
        fields = {
            "year": 2016,
            "container": "Synth Botanist",
            "volume": None,
            "issue": "Suppl",
            "first_page": None,
            "last_page": "166",
        }
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "56"
        assert fields["first_page"] == "S163"
        # Already-populated spans are never rewritten.
        assert fields["issue"] == "Suppl"
        assert fields["last_page"] == "166"

    def test_article_id_page(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. J Synth Garden Assoc. 2020;9(21):e015981. https://doi.org/10.5555/example.reference."
        fields = {"year": 2020, "container": "J Synth Garden Assoc", "volume": None, "issue": None}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "9"
        assert fields["issue"] == "21"
        assert fields["first_page"] == "e015981"
        assert fields.get("last_page") is None

    def test_spacing_noise_inside_page_range(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Spat Synth Analysis. 2018;25:1– 9. https://doi.org/10.5555/example.reference."
        fields = {"year": 2018, "container": "Spat Synth Analysis", "volume": None}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "25"
        assert fields["first_page"] == "1"
        assert fields["last_page"] == "9"

    def test_fill_only_populated_pages_survive(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Plant Growth Studies. 2018;23(4):483-489."
        fields = {
            "year": 2018,
            "container": "Plant Growth Studies",
            "volume": None,
            "issue": None,
            "first_page": "483",
            "last_page": "489",
        }
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "23"
        assert fields["issue"] == "4"
        assert fields["first_page"] == "483"
        assert fields["last_page"] == "489"

    def test_fill_only_populated_volume_survives_unchanged(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Garden Res. 2010;59(6):409-414."
        fields = {"year": 2010, "container": "Garden Res", "volume": "59", "issue": "6"}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "59"
        assert fields["issue"] == "6"
        assert fields["first_page"] == "409"
        assert fields["last_page"] == "414"

    def test_disagreeing_parsed_volume_blocks_the_fill(self):
        # A parsed volume that no printed tail carries means the anchor is not
        # this reference's tail — fail closed rather than guess.
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Garden Res. 2010;59(6):409-414."
        fields = {"year": 2010, "container": "Garden Res", "volume": "12", "issue": None}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] == "12"
        assert fields["issue"] is None
        assert fields.get("first_page") is None

    def test_apa_segment_never_fills_numbers(self):
        seg = "Smith, J., & Doe, A. (2020). A study of things. Journal of Things, 12(3), 45-67."
        fields = {"year": None, "container": None, "volume": None, "issue": None}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] is None
        assert fields["issue"] is None
        assert fields.get("first_page") is None

    def test_bare_year_semicolon_page_span_is_not_a_volume(self):
        # Book-chapter style: "…; 2019; 45–52." has no issue parens and no
        # ":pages" delimiter, so the digits after ';' are not a volume.
        seg = "Doe J. A chapter. In: Roe A, editor. A Book. Springer; 2019; 45–52."
        fields = {"year": 2019, "container": None, "volume": None, "issue": None}
        _backfill_vancouver_tail(fields, seg)
        assert fields["volume"] is None
        assert fields.get("first_page") is None

    def test_finalize_wires_numeric_tail_end_to_end(self):
        seg = "1. Example AB, Sample CD, Reader EF, et al. A synthetic study of garden observations. Propagation. 2010;122(18 Suppl 3):S640-656. https://doi.org/10.5555/example.reference."
        volume, issue = _normalize_vol_issue("2010;122(18", "Suppl", seg)
        out = _finalize_reference_fields(
            {
                "title": "A synthetic study of garden observations",
                "container": "Propagation",
                "year": 2010,
                "volume": volume,
                "issue": issue,
                "first_page": None,
                "last_page": None,
            },
            seg,
        )
        assert out["volume"] == "122"
        assert out["issue"] == "18"
        assert out["first_page"] == "S640"
        assert out["last_page"] == "656"


class TestSplitVolIssueYearPrefix:
    """A leading ``YEAR;`` (and the truncated ``N(M`` it leaves behind) is the
    parser swallowing the Vancouver delimiter into the volume span."""

    @pytest.mark.parametrize(
        ("volume", "issue", "expected"),
        [
            ("2010;122(18", "Suppl", ("122", "18")),
            ("2021;236(6", None, ("236", "6")),
            # Year prefix with a complete "N(M)".
            ("2021;236(6)", None, ("236", "6")),
            # Year prefix, bare volume.
            ("2021;236", None, ("236", None)),
            # A volume that merely looks like a year is left alone.
            ("2021", None, ("2021", None)),
            # A dangling "(" with a separately parsed issue keeps that issue.
            ("21(", "4", ("21", "4")),
        ],
    )
    def test_year_prefix_stripped(self, volume, issue, expected):
        assert _split_vol_issue(volume, issue) == expected
