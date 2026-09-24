"""GROBID TEI -> bibr export conversion, on synthetic GROBID-shaped TEI.

Every name, title, journal and DOI below is invented; the element structure
follows what GROBID 0.9 writes.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from bibr.export.models import PaperExport
from evaluation.evaluate import extract_comparable_from_json, load_expected_ids, score_paper
from evaluation.grobid_tei import (
    CONVERTER_NAME,
    CONVERTER_VERSION,
    FAILED_SUFFIX,
    MANIFEST_NAME,
    TEI_SUFFIX,
    TeiError,
    conform_doi,
    convert_directory,
    main,
    paper_id_for,
    read_tei,
    tei_to_export,
)
from evaluation.validation_metrics import _get_ref_author, _ref_surname_tokens

PDF_SOURCE = {"file_name": "paper-a.pdf", "sha256": None, "input_format": "pdf"}


def tei(
    *,
    title: str = "Placeholder Effects in Synthetic Samples",
    header: str = "",
    profile: str = "",
    body: str = "",
    back: str = "",
    refs: str = "",
    md5: str = "0123456789ABCDEF0123456789ABCDEF",
) -> bytes:
    """A GROBID-shaped TEI document around the given fragments."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xml:space="preserve" xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader xml:lang="en">
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">{title}</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct status="extracted">
          <analytic>{header}</analytic>
          <monogr><imprint><date/></imprint></monogr>
          <idno type="MD5">{md5}</idno>
          <idno type="DOI">10.5555/Synthetic.2026.001</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <encodingDesc>
      <appInfo>
        <application version="0.9.1" ident="GROBID" when="2026-08-05T09:30+0000">
          <label type="revision">0.9.1</label>
          <label type="parameters">consolidateCitations=0, consolidateHeader=0</label>
        </application>
      </appInfo>
    </encodingDesc>
    <profileDesc>{profile}</profileDesc>
  </teiHeader>
  <text xml:lang="en">
    <body>{body}</body>
    <back>{back}
      <div type="references">
        <listBibl>{refs}</listBibl>
      </div>
    </back>
  </text>
</TEI>""".encode()


def convert(data: bytes, source: dict | None = None) -> dict:
    return tei_to_export(read_tei(data), source=source or PDF_SOURCE, converter_build_sha="abc123")


def bib(*refs: str) -> list[dict]:
    return convert(tei(refs="".join(refs)))["bib"]


def one_ref(fragment: str) -> dict:
    return bib(f'<biblStruct xml:id="b0">{fragment}</biblStruct>')[0]


JOURNAL_ARTICLE = """
  <analytic>
    <title level="a" type="main">Invented findings about placeholder tasks</title>
    <author><persName><forename type="first">Ada</forename><forename type="middle">B</forename>
      <surname>Quill</surname></persName></author>
    <author><persName><forename type="first">Bram</forename><surname>Otter</surname></persName></author>
    <idno type="DOI">10.5555/jps.2019.042</idno>
  </analytic>
  <monogr>
    <title level="j">Journal of Placeholder Studies</title>
    <imprint>
      <biblScope unit="volume">12</biblScope>
      <biblScope unit="issue">3</biblScope>
      <biblScope unit="page" from="101" to="117" />
      <date type="published" when="2019-05-01">May 1, 2019</date>
    </imprint>
  </monogr>
"""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


class TestHeader:
    HEADER = """
      <author role="corresp">
        <persName><roleName>Dr</roleName><forename type="first">Ada</forename>
          <forename type="middle">B</forename><surname>Quill</surname><genName>Jr</genName></persName>
        <email>ada.quill@example.org</email>
        <idno type="ORCID">0000-0002-1825-0097</idno>
        <affiliation key="aff0">
          <note type="raw_affiliation"><label>1</label> Department of Examples, Placeholder
            University, Exampletown, Nowhere</note>
          <orgName type="department">Department of Examples</orgName>
          <orgName type="institution">Placeholder University</orgName>
          <address><settlement>Exampletown</settlement><country key="NW">Nowhere</country></address>
        </affiliation>
      </author>
      <author>
        <persName><forename type="first">Bram</forename><surname>van der Otter</surname></persName>
        <affiliation key="aff0"/>
        <affiliation key="aff1">
          <orgName type="laboratory">Invented Lab</orgName>
          <orgName type="institution">Sample Institute</orgName>
          <address><settlement>Mocksville</settlement></address>
        </affiliation>
      </author>
      <author><orgName type="collaboration">Synthetic Consortium</orgName></author>
      <author>
        <affiliation key="aff2">
          <orgName type="institution">Unattached Institute</orgName>
        </affiliation>
      </author>
    """

    def test_title_doi_and_provenance(self):
        out = convert(tei(header=self.HEADER))
        assert out["metadata"]["title"] == "Placeholder Effects in Synthetic Samples"
        assert out["metadata"]["doi"] == "10.5555/synthetic.2026.001"
        assert out["paper_id"] == "paper-a"
        assert out["schema_version"] == "12.0"
        extraction = out["extraction"]
        assert extraction["producer"] == {"name": "grobid", "version": "0.9.1", "build_sha": None}
        assert extraction["converter"] == {
            "name": CONVERTER_NAME,
            "version": CONVERTER_VERSION,
            "build_sha": "abc123",
        }
        assert extraction["completed_at"] == "2026-08-05T09:30:00Z"
        assert extraction["ocr"] is None and extraction["llm"] is None
        PaperExport.model_validate(out)

    def test_authors_keep_every_forename_contact_and_group(self):
        authors = convert(tei(header=self.HEADER))["author"]
        # The name-less <author> is an affiliation GROBID could not attach, not a person.
        assert [a["author_id"] for a in authors] == [1, 2, 3]
        ada, bram, group = authors
        assert (ada["given"], ada["family"], ada["suffix"]) == ("Ada B", "Quill", "Jr")
        assert ada["email"] == "ada.quill@example.org"
        assert ada["orcid"] == "https://orcid.org/0000-0002-1825-0097"
        assert ada["corresponding"] is True
        assert (bram["given"], bram["family"], bram["corresponding"]) == (
            "Bram",
            "van der Otter",
            False,
        )
        assert group["literal"] == "Synthetic Consortium"
        assert group["given"] is None and group["family"] is None

    def test_affiliations_prefer_raw_text_and_link_shared_keys(self):
        out = convert(tei(header=self.HEADER))
        rows = {row["affiliation_id"]: row for row in out["affiliation"]}
        assert rows[1]["text"] == (
            "Department of Examples, Placeholder University, Exampletown, Nowhere"
        )
        assert rows[1]["author_ids"] == [1, 2]
        assert rows[1]["institution"] == "Placeholder University"
        assert rows[1]["country"] == "Nowhere"
        # Without includeRawAffiliations the parts are joined in printed order.
        assert rows[2]["text"] == "Invented Lab, Sample Institute, Mocksville"
        assert rows[2]["author_ids"] == [2]
        assert rows[3]["text"] == "Unattached Institute"
        assert rows[3]["author_ids"] == []
        # The evaluator rebuilds each author's affiliation string from this table.
        extracted = extract_comparable_from_json(out)
        assert extracted["authors"][1]["affiliation"] == (
            "Department of Examples, Placeholder University, Exampletown, Nowhere; "
            "Invented Lab, Sample Institute, Mocksville"
        )

    def test_unconformable_orcid_is_dropped_with_a_warning(self):
        header = """<author><persName><forename type="first">Ada</forename>
          <surname>Quill</surname></persName><idno type="ORCID">12</idno></author>"""
        out = convert(tei(header=header))
        assert out["author"][0]["orcid"] is None
        assert [w["code"] for w in out["extraction"]["warnings"]] == [
            "BIBR_GROBID_TEI_ORCID_DROPPED"
        ]

    def test_abstract_joins_blocks_without_gluing_and_keeps_loose_text(self):
        profile = """<abstract><div>
            <ref type="bibr" target="#b0">Quill (2019)</ref> proposed a placeholder.
            <p>First paragraph ends here.</p><p>Second paragraph starts.</p>
          </div><div><head>Methods</head><p>Invented method text.</p></div></abstract>"""
        out = convert(tei(profile=profile))
        assert out["metadata"]["abstract"] == (
            "Quill (2019) proposed a placeholder. First paragraph ends here. "
            "Second paragraph starts. Methods Invented method text."
        )
        abstract = [s for s in out["section"] if s["section_type"] == "abstract"]
        assert len(abstract) == 1
        rows = [t["text"] for t in out["text"] if t["section_id"] == abstract[0]["section_id"]]
        assert " ".join(rows) == out["metadata"]["abstract"]

    def test_keywords_from_terms_or_unsplit_text(self):
        terms = """<textClass><keywords><term>placeholder</term>
            <term>synthetic data</term></keywords></textClass>"""
        assert convert(tei(profile=terms))["metadata"]["keywords"] == [
            "placeholder",
            "synthetic data",
        ]
        raw = "<textClass><keywords>placeholder; synthetic data</keywords></textClass>"
        assert convert(tei(profile=raw))["metadata"]["keywords"] == ["placeholder; synthetic data"]


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


class TestReferences:
    def test_journal_article_fields(self):
        row = one_ref(JOURNAL_ARTICLE)
        assert row["title"] == "Invented findings about placeholder tasks"
        assert row["container"] == "Journal of Placeholder Studies"
        assert row["authors"] == "Quill, Ada B, Otter, Bram"
        assert row["doi"] == "10.5555/jps.2019.042"
        assert (row["volume"], row["issue"]) == ("12", "3")
        assert (row["first_page"], row["last_page"]) == ("101", "117")
        assert (row["year"], row["published_date"]) == (2019, "2019-05-01")

    def test_scorer_reads_every_reference_field(self):
        extracted = extract_comparable_from_json(
            convert(tei(refs=f"<biblStruct>{JOURNAL_ARTICLE}</biblStruct>"))
        )
        ref = extracted["references"][0]
        assert ref == {
            "title": "Invented findings about placeholder tasks",
            "doi": "10.5555/jps.2019.042",
            "author": "Quill, Ada B, Otter, Bram",
            "authors": "Quill, Ada B, Otter, Bram",
            "year": "2019",
            "container": "Journal of Placeholder Studies",
            "volume": "12",
            "first_page": "101",
            "last_page": "117",
        }
        assert _get_ref_author(ref) == "quill"
        # Given names reach the scorer: a gold list printing them can be recalled.
        assert set(_ref_surname_tokens(ref)) == {"quill", "ada", "otter", "bram"}

    def test_chapter_doi_at_monogr_level_editors_kept_apart(self):
        row = one_ref("""
          <analytic>
            <title level="a" type="main">A chapter on invented things</title>
            <author><persName><forename type="first">Ada</forename><surname>Quill</surname></persName></author>
          </analytic>
          <monogr>
            <title level="m">The Handbook of Placeholders</title>
            <editor><persName><forename type="first">Cleo</forename><surname>Marsh</surname></persName></editor>
            <idno type="DOI">10.5555/HANDBOOK.7</idno>
            <imprint><publisher>Example Press</publisher>
              <biblScope unit="page" from="5" to="29" />
              <date type="published" when="2021">2021</date></imprint>
          </monogr>""")
        assert row["title"] == "A chapter on invented things"
        assert row["container"] == "The Handbook of Placeholders"
        assert row["authors"] == "Quill, Ada"
        assert row["editors"] == "Marsh, Cleo"
        assert row["doi"] == "10.5555/handbook.7"
        assert row["publisher"] == "Example Press"

    def test_book_without_analytic_title_is_the_work(self):
        row = one_ref("""
          <monogr>
            <title level="m" type="main">Placeholders: A Primer</title>
            <author><persName><forename type="first">Bram</forename><surname>Otter</surname></persName></author>
            <imprint><date type="published" when="2001">2001</date></imprint>
          </monogr>""")
        assert (row["title"], row["container"]) == ("Placeholders: A Primer", None)
        assert row["authors"] == "Otter, Bram"

    def test_journal_without_article_title_is_only_the_container(self):
        row = one_ref("""
          <analytic><title/><idno type="DOI">10.5555/untitled.1</idno></analytic>
          <monogr><title level="j">Journal of Placeholder Studies</title><imprint/></monogr>""")
        assert (row["title"], row["container"]) == (None, "Journal of Placeholder Studies")

    def test_series_is_the_container_of_last_resort(self):
        row = one_ref("""
          <analytic><title level="a" type="main">An invented working paper</title></analytic>
          <monogr><title level="s">Placeholder Working Papers</title><imprint/></monogr>""")
        assert row["container"] == row["series"] == "Placeholder Working Papers"

    @pytest.mark.parametrize(
        ("scope", "first", "last"),
        [
            ('<biblScope unit="page">e1234</biblScope>', "e1234", None),
            ('<biblScope unit="page">101-117</biblScope>', "101", "117"),
            ('<biblScope unit="page">S5 – S9</biblScope>', "S5", "S9"),
            ('<biblScope unit="page" from="7" />', "7", None),
        ],
    )
    def test_page_forms(self, scope, first, last):
        row = one_ref(f"<monogr><title level='j'>J</title><imprint>{scope}</imprint></monogr>")
        assert (row["first_page"], row["last_page"]) == (first, last)

    @pytest.mark.parametrize(
        ("idno", "expected"),
        [
            ("https://doi.org/10.5555/ABC.1", "10.5555/abc.1"),
            ("doi:10.5555/abc.2.", "10.5555/abc.2"),
            (".org/10.5555/abc.3", "10.5555/abc.3"),
            ("ISSN1234-5678.DOI10.5555/abc.4", "10.5555/abc.4"),
            ("https://publisher.example/article?id=10.5555/abc.5", "10.5555/abc.5"),
        ],
    )
    def test_typed_doi_is_reduced_to_the_bare_doi(self, idno, expected):
        row = one_ref(f'<analytic><idno type="DOI">{idno}</idno></analytic><monogr/>')
        assert row["doi"] == expected

    def test_unconformable_typed_doi_is_reported(self):
        out = convert(
            tei(refs='<biblStruct><analytic><idno type="DOI">10.55</idno></analytic></biblStruct>')
        )
        assert out["bib"][0]["doi"] is None
        warning = out["extraction"]["warnings"][0]
        assert warning["code"] == "BIBR_GROBID_TEI_DOI_DROPPED"
        assert "reference 1" in warning["message"]

    def test_doi_fallbacks_untyped_idno_and_doi_link(self):
        untyped = one_ref("<analytic><idno>10.5555/untyped.9</idno></analytic><monogr/>")
        assert untyped["doi"] == "10.5555/untyped.9"
        link = one_ref(
            '<analytic><ptr target="https://doi.org/10.5555%2Flinked.8"/></analytic><monogr/>'
        )
        assert link["doi"] == "10.5555/linked.8"
        report_number = one_ref("<analytic><idno>WP/19/137</idno></analytic><monogr/>")
        assert report_number["doi"] is None

    def test_typed_doi_wins_over_fallbacks(self):
        row = one_ref("""<analytic><idno type="DOI">10.5555/typed.1</idno>
            <ptr target="https://doi.org/10.5555/other.2"/></analytic><monogr/>""")
        assert row["doi"] == "10.5555/typed.1"

    def test_names_particles_forename_only_groups_and_stray_punctuation(self):
        out = convert(
            tei(
                refs="""<biblStruct><analytic>
            <author><persName><forename type="first">;</forename><forename type="middle">L</forename>
              <surname>van der Berg</surname></persName></author>
            <author><persName><forename type="first">Otter</forename></persName></author>
            <author><persName><forename type="first">.</forename></persName></author>
            <author><persName><forename type="first">Cleo</forename><surname>Marsh</surname></persName>
              <affiliation><orgName type="collaboration">Placeholder Team</orgName></affiliation></author>
            <author><persName><forename type="first">Dan</forename><surname>Reed</surname></persName>
              <affiliation><orgName type="collaboration">Placeholder Team</orgName></affiliation></author>
            <author><orgName type="collaboration">Example Core Team.</orgName></author>
          </analytic><monogr/></biblStruct>"""
            )
        )
        # A forename-only person is kept (GROBID often tags a surname that way);
        # a lone "." or ";" is not a name. Group names follow once each.
        assert out["bib"][0]["authors"] == (
            "van der Berg, L, Otter, Marsh, Cleo, Reed, Dan, Example Core Team., Placeholder Team"
        )
        ref = extract_comparable_from_json(out)["references"][0]
        assert _get_ref_author(ref) == "van"
        assert {"placeholder", "team", "example", "core"} <= set(_ref_surname_tokens(ref))

    def test_first_author_key_is_the_first_surname(self):
        extracted = extract_comparable_from_json(
            convert(
                tei(
                    refs="""<biblStruct><analytic>
                      <author><persName><surname>St</surname></persName></author>
                      <author><persName><forename type="first">Ada</forename><surname>Quill</surname></persName></author>
                    </analytic><monogr/></biblStruct>"""
                )
            )
        )
        # Commas, not semicolons, between names: the scorer's key is "st", not "st;".
        assert _get_ref_author(extracted["references"][0]) == "st"

    def test_monogr_authors_used_when_analytic_has_none(self):
        row = one_ref("""<analytic><title level="a" type="main">T</title></analytic>
          <monogr><author><persName><surname>Otter</surname></persName></author></monogr>""")
        assert row["authors"] == "Otter"

    def test_raw_reference_becomes_a_linked_references_row(self):
        out = convert(
            tei(
                refs=f"""<biblStruct xml:id="b0">{JOURNAL_ARTICLE}
                  <note type="raw_reference">Quill, A. B., &amp; Otter, B. (2019). Invented
                    findings. Journal of Placeholder Studies, 12(3), 101-117.</note>
                </biblStruct><biblStruct xml:id="b1"><monogr/></biblStruct>"""
            )
        )
        first, second = out["bib"]
        row = next(t for t in out["text"] if t["text_id"] == first["text_id"])
        assert row["text"].startswith("Quill, A. B., & Otter, B. (2019). Invented findings.")
        section = next(s for s in out["section"] if s["section_id"] == row["section_id"])
        assert section["section_type"] == "references"
        assert second["text_id"] is None

    def test_every_biblstruct_is_a_row(self):
        rows = bib(*["<biblStruct><monogr/></biblStruct>"] * 3)
        assert [r["bib_id"] for r in rows] == [1, 2, 3]
        assert all(r["title"] is None and r["authors"] is None for r in rows)

    def test_year_from_when_only(self):
        row = one_ref(
            "<monogr><imprint><date type='published' when='2020'>2020a</date></imprint></monogr>"
        )
        assert (row["year"], row["published_date"]) == (2020, "2020")
        undated = one_ref("<monogr><imprint><date>n.d.</date></imprint></monogr>")
        assert undated["year"] is None


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


class TestBody:
    BODY = """
      <div><head n="1.">Introduction</head><p>Intro text with a
        <ref type="bibr" target="#b0">(Quill, 2019)</ref> citation.</p></div>
      <div><head n="1.1">Detail</head><p>Detail text.</p>
        <formula xml:id="formula_0">y = a + b</formula></div>
      <figure xml:id="fig_0"><head>Fig. 1</head><label>1</label>
        <figDesc>Fig. 1. An invented figure.</figDesc></figure>
      <figure type="table" xml:id="tab_0"><head>Table 1</head><label>1</label>
        <figDesc>An invented table.</figDesc>
        <table><row><cell>a</cell><cell>b</cell></row><row><cell>1</cell><cell>2</cell></row></table>
      </figure>
      <note place="foot" n="2"><p>An invented footnote.</p></note>
    """
    BACK = """
      <div type="acknowledgement"><div><head>Acknowledgments</head><p>Thanks to nobody.</p></div></div>
      <div type="conflict"><div><head>Declaration</head><p>No conflicts.</p></div></div>
    """

    def test_sections_text_and_captions(self):
        out = convert(tei(body=self.BODY, back=self.BACK))
        sections = {s["header"]: s for s in out["section"]}
        intro, detail = sections["Introduction"], sections["Detail"]
        assert intro["section_type"] == detail["section_type"] == "unknown"
        assert (detail["level"], detail["parent_section_id"]) == (2, intro["section_id"])
        assert sections["Acknowledgments"]["section_type"] == "acknowledgment"
        assert sections["Declaration"]["section_type"] == "coi"
        texts = {t["text"]: t for t in out["text"]}
        assert (
            texts["Intro text with a (Quill, 2019) citation."]["section_id"] == intro["section_id"]
        )
        assert texts["[equation]"]["formatted"] == "y = a + b"
        figure, table = out["figure"][0], out["table"][0]
        assert figure["label"] == "1" and figure["caption"] == "Fig. 1. An invented figure."
        assert texts[figure["caption"]]["section_id"] is None
        assert table["contents"] == [["a", "b"], ["1", "2"]]
        assert out["footnote"] == [
            {"footnote_id": 1, "label": "2", "text_id": texts["An invented footnote."]["text_id"]}
        ]
        # Caption and footnote rows follow the body text.
        body_ids = [t["text_id"] for t in out["text"] if t["section_id"] is not None]
        assert max(body_ids) < texts[figure["caption"]]["text_id"]

    def test_section_benchmark_reads_the_text(self):
        from evaluation.evaluate import extract_sections_from_json

        grouped = extract_sections_from_json(convert(tei(body=self.BODY, back=self.BACK)))
        assert "Intro text with a (Quill, 2019) citation." in grouped["unknown"]
        assert grouped["acknowledgment"] == "Thanks to nobody."
        assert grouped["figure"] == "Fig. 1. An invented figure."


# ---------------------------------------------------------------------------
# Scoring end to end
# ---------------------------------------------------------------------------


def test_converted_output_scores_like_a_bibr_export():
    header = """<author><persName><forename type="first">Ada</forename>
        <surname>Quill</surname></persName></author>"""
    profile = "<abstract><div><p>An invented abstract about placeholders.</p></div></abstract>"
    prediction = extract_comparable_from_json(
        convert(
            tei(header=header, profile=profile, refs=f"<biblStruct>{JOURNAL_ARTICLE}</biblStruct>")
        )
    )
    gold = {
        "title": "Placeholder Effects in Synthetic Samples",
        "doi_printed": "10.5555/synthetic.2026.001",
        "abstract": "An invented abstract about placeholders.",
        "authors": [{"given": "Ada", "family": "Quill"}],
        "reference_count": 1,
        "references": [
            {
                "authors": "Quill, Ada B., & Otter, Bram",
                "year": "2019",
                "title": "Invented findings about placeholder tasks",
                "container": "Journal of Placeholder Studies",
                "volume": "12",
                "first_page": "101",
                "last_page": "117",
                "doi": "10.5555/jps.2019.042",
            }
        ],
    }
    scores = score_paper(prediction, gold)
    for metric in (
        "title_soft",
        "doi_match",
        "abstract_rouge_l",
        "authors_fullname_f1",
        "ref_matching_f1",
        "ref_title_acc",
        "ref_year_acc",
        "ref_doi_recall",
        "ref_author_acc",
        "ref_journal_acc",
        "ref_volume_acc",
        "ref_pages_acc",
    ):
        assert scores[metric] == 1.0, metric


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


class TestMalformed:
    @pytest.mark.parametrize(
        ("data", "message"),
        [
            (b"", "empty"),
            (b"   \n", "empty"),
            (b"<TEI><teiHeader>", "not well-formed"),
            (b"<html><body>Service unavailable</body></html>", "not TEI"),
        ],
    )
    def test_unusable_files_raise(self, data, message):
        with pytest.raises(TeiError, match=message):
            read_tei(data)

    def test_empty_tei_converts_to_an_empty_prediction(self):
        out = convert(b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text/></TEI>')
        assert out["metadata"]["title"] is None and out["metadata"]["doi"] is None
        assert out["author"] == [] and out["bib"] == [] and out["text"] == []
        assert out["extraction"]["producer"]["version"] == "unknown"
        extracted = extract_comparable_from_json(out)
        assert extracted["reference_count"] == 0 and extracted["abstained"] is False


def test_conform_doi_rejects_non_dois():
    assert conform_doi("10.5555/ok") == "10.5555/ok"
    assert conform_doi("not a doi") is None
    assert conform_doi("") is None
    assert conform_doi("see 10.5555/inside") is None
    assert conform_doi("see 10.5555/inside", search=True) == "10.5555/inside"


def test_paper_id_for_strips_only_tei_suffixes(tmp_path):
    assert paper_id_for(tmp_path / f"10.5555_x.v2{TEI_SUFFIX}") == "10.5555_x.v2"
    assert paper_id_for(tmp_path / "paper.tei.xml") == "paper"
    assert paper_id_for(tmp_path / "10.5555_x.v2.xml") == "10.5555_x.v2"


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def _manifest(tei_dir, papers):
    (tei_dir / MANIFEST_NAME).write_text(
        json.dumps({"ids": [p["paper_id"] for p in papers], "papers": papers})
    )


class TestConvertDirectory:
    def test_runner_manifest_names_the_pdf_and_keeps_failures(self, tmp_path):
        tei_dir, out = tmp_path / "tei", tmp_path / "json"
        tei_dir.mkdir()
        pdf = b"%PDF-1.7 synthetic"
        md5, sha = (
            hashlib.md5(pdf, usedforsecurity=False).hexdigest(),
            hashlib.sha256(pdf).hexdigest(),
        )
        (tei_dir / f"ok{TEI_SUFFIX}").write_bytes(tei(md5=md5.upper()))
        (tei_dir / f"leftover{TEI_SUFFIX}").write_bytes(tei())
        (tei_dir / f"down{FAILED_SUFFIX}").write_text("{}")
        _manifest(
            tei_dir,
            [
                {"paper_id": "ok", "pdf": "ok.pdf", "sha256": sha, "md5": md5, "status": "ok"},
                {"paper_id": "down", "pdf": "down.pdf", "status": "failed", "error": "HTTP 500"},
            ],
        )
        summary = convert_directory(tei_dir, out).as_dict("abc123")
        assert sorted(p.name for p in out.iterdir()) == ["ok.json"]
        payload = json.loads((out / "ok.json").read_text())
        assert payload["source"] == {"file_name": "ok.pdf", "sha256": sha, "input_format": "pdf"}
        assert payload["extraction"]["warnings"] == []
        assert summary["ids"] == ["down", "ok"]
        assert summary["failed"] == [{"paper_id": "down", "reason": "GROBID run failed: HTTP 500"}]
        assert summary["ignored"] == ["leftover"]
        assert load_expected_ids(str(tei_dir / MANIFEST_NAME)) == {"down", "ok"}

    def test_md5_mismatch_is_a_failure_not_a_prediction(self, tmp_path):
        tei_dir, out = tmp_path / "tei", tmp_path / "json"
        tei_dir.mkdir()
        (tei_dir / f"p{TEI_SUFFIX}").write_bytes(tei(md5="FFFF"))
        _manifest(tei_dir, [{"paper_id": "p", "pdf": "p.pdf", "md5": "0000", "status": "ok"}])
        summary = convert_directory(tei_dir, out)
        assert summary.converted == [] and "MD5" in summary.failed["p"]

    def test_pdf_dir_supplies_name_and_digest(self, tmp_path):
        tei_dir, pdf_dir, out = tmp_path / "tei", tmp_path / "pdf", tmp_path / "json"
        tei_dir.mkdir()
        pdf_dir.mkdir()
        (pdf_dir / "10.5555_p.v1.pdf").write_bytes(b"%PDF synthetic")
        md5 = hashlib.md5(b"%PDF synthetic", usedforsecurity=False).hexdigest()
        (tei_dir / "10.5555_p.v1.xml").write_bytes(tei(md5=md5))
        (tei_dir / "no-pdf.xml").write_bytes(tei())
        (tei_dir / "broken.xml").write_bytes(b"<TEI")
        summary = convert_directory(tei_dir, out, pdf_dirs=[pdf_dir])
        found = json.loads((out / "10.5555_p.v1.json").read_text())
        assert found["source"]["sha256"] == hashlib.sha256(b"%PDF synthetic").hexdigest()
        inferred = json.loads((out / "no-pdf.json").read_text())
        assert inferred["source"] == {
            "file_name": "no-pdf.pdf",
            "sha256": None,
            "input_format": "pdf",
        }
        assert inferred["extraction"]["warnings"][0]["code"] == "BIBR_GROBID_TEI_SOURCE_INFERRED"
        assert set(summary.failed) == {"broken"}

    def test_refuses_to_mix_with_existing_predictions(self, tmp_path):
        (tmp_path / "json").mkdir()
        (tmp_path / "json" / "stale.json").write_text("{}")
        with pytest.raises(FileExistsError):
            convert_directory(tmp_path, tmp_path / "json")

    def test_cli_writes_predictions_and_summary(self, tmp_path, capsys):
        tei_dir = tmp_path / "tei"
        tei_dir.mkdir()
        (tei_dir / f"p{TEI_SUFFIX}").write_bytes(
            tei(refs=f"<biblStruct>{JOURNAL_ARTICLE}</biblStruct>")
        )
        summary_path = tmp_path / "summary.json"
        code = main(
            [
                "--tei-dir",
                str(tei_dir),
                "--out",
                str(tmp_path / "json"),
                "--summary",
                str(summary_path),
            ]
        )
        assert code == 0
        summary = json.loads(summary_path.read_text())
        assert summary["ids"] == ["p"] and summary["converted"] == 1
        assert summary["producers"][0]["grobid_version"] == "0.9.1"
        assert "GROBID 0.9.1" in capsys.readouterr().out
        frame = pd.DataFrame(
            [extract_comparable_from_json(json.loads((tmp_path / "json" / "p.json").read_text()))]
        )
        assert frame.loc[0, "file_name"] == "p.pdf"
