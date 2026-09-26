"""Tests for bibr.input.jats_native.JatsParser."""

import re
from pathlib import Path

import pytest

from bibr.input.jats_native import JatsParser
from bibr.paper_contents import CanonicalSection

EUROPEPMC_FIXTURE = Path(__file__).parent / "fixtures" / "jats" / "PMC4383902.xml"

# ---------------------------------------------------------------------------
# Fixtures (inline JATS XML bytes)
# ---------------------------------------------------------------------------

FULL_JATS = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta>
      <journal-title-group><journal-title>Journal of Testing</journal-title></journal-title-group>
      <issn pub-type="ppub">1234-5678</issn>
      <issn pub-type="epub">8765-4321</issn>
      <publisher><publisher-name>Test Press</publisher-name></publisher>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="doi">10.1234/test.2020</article-id>
      <title-group><article-title>A <italic>Study</italic> of Things</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author" corresp="yes">
          <contrib-id contrib-id-type="orcid">0000-0002-1234-5678</contrib-id>
          <name><surname>Smith</surname><given-names>Jane</given-names></name>
          <email>jane@test.org</email>
          <xref ref-type="aff" rid="a1"/>
        </contrib>
        <contrib contrib-type="author">
          <name><surname>Doe</surname><given-names>John</given-names></name>
          <xref ref-type="aff" rid="a2"/>
        </contrib>
        <aff id="a1"><label>1</label>University of Test</aff>
        <aff id="a2"><label>2</label>Test Institute</aff>
      </contrib-group>
      <pub-date pub-type="epub"><day>15</day><month>3</month><year>2020</year></pub-date>
      <volume>10</volume><issue>2</issue><fpage>100</fpage><lpage>110</lpage>
      <permissions><license xlink:href="http://creativecommons.org/licenses/by/4.0/">
        <license-p>CC BY</license-p></license></permissions>
      <abstract><title>Abstract</title><p>This is the abstract text.</p></abstract>
      <kwd-group><kwd>testing</kwd><kwd>science</kwd></kwd-group>
    </article-meta>
  </front>
  <body>
    <sec sec-type="intro"><title>Introduction</title>
      <p>First sentence here. Second one too, citing [1].</p></sec>
    <sec sec-type="methods"><title>Methods</title>
      <p>We did things.</p>
      <sec><title>Participants</title><p>Nested subsection text.</p></sec>
      <disp-formula><tex-math>E = mc^2</tex-math></disp-formula>
      <table-wrap><label>Table 1</label><caption><p>A table.</p></caption>
        <table><thead><tr><th>A</th><th>B</th></tr></thead>
          <tbody><tr><td>1</td><td>2</td></tr></tbody></table>
      </table-wrap>
      <fig><label>Figure 1</label><caption><p>A figure.</p></caption>
        <graphic xlink:href="fig1.png"/></fig>
    </sec>
  </body>
  <back>
    <ack><p>Thanks to everyone.</p></ack>
    <sec><title>Data Availability</title><p>Data available on request.</p></sec>
    <ref-list><title>References</title>
      <ref id="r1"><element-citation publication-type="journal">
        <person-group person-group-type="author">
          <name><surname>Jones</surname><given-names>A B</given-names></name>
          <name><surname>Lee</surname><given-names>C</given-names></name>
        </person-group>
        <year>2018</year><article-title>Prior work</article-title><source>Nature</source>
        <volume>5</volume><issue>3</issue><fpage>20</fpage><lpage>30</lpage>
        <pub-id pub-id-type="doi">10.1/prior</pub-id></element-citation></ref>
      <ref id="r2"><element-citation publication-type="book">
        <person-group person-group-type="author">
          <name><surname>King</surname><given-names>D</given-names></name></person-group>
        <year>2001</year><source>A Big Book</source>
        <publisher-name>Book Co</publisher-name></element-citation></ref>
    </ref-list>
  </back>
</article>"""

NAMESPACED_JATS = b"""<?xml version="1.0"?>
<article xmlns="https://jats.nlm.nih.gov/ns" xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <article-id pub-id-type="doi">10.5/ns.1</article-id>
      <title-group><article-title>Namespaced Paper</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>Ng</surname><given-names>Wei</given-names></name>
        </contrib>
      </contrib-group>
    </article-meta>
  </front>
  <body><sec><title>Intro</title><p>Body text here.</p></sec></body>
  <back>
    <ref-list>
      <ref><mixed-citation>Ng, W. (2019). A paper. <italic>Journal</italic>, 1, 2-3.</mixed-citation></ref>
    </ref-list>
  </back>
</article>"""

MIXED_CITATION_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Mixed Cites</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
  <back><ref-list>
    <ref><mixed-citation>Alpha, A. (2010). First. Journal A, 1, 1-2.</mixed-citation></ref>
    <ref><mixed-citation>Beta, B. (2011). Second. Journal B, 2, 3-4.</mixed-citation></ref>
  </ref-list></back>
</article>"""


def _parse(xml: bytes) -> JatsParser:
    p = JatsParser(xml)
    contents = p.parse()
    p._contents = contents  # stash for tests
    return p


def _segment(p: JatsParser):
    """Run deferred segmentation (identity split) + content sections."""
    contents = p._contents
    segs = [[e.text] for e in p.assembler.entries if e.needs_segmentation]
    p.apply_segmentation(contents, segs)
    p.create_content_sections(contents)
    return contents


# ---------------------------------------------------------------------------
# Front-matter metadata
# ---------------------------------------------------------------------------


class TestFrontMatter:
    def test_self_identity_fields(self):
        m = _parse(FULL_JATS)._contents.preparsed_metadata
        assert m.doi == "10.1234/test.2020"
        assert m.title == "A Study of Things"
        assert m.journal == "Journal of Testing"
        assert m.issn == "1234-5678"  # print preferred over electronic
        assert m.publisher == "Test Press"
        assert m.volume == "10"
        assert m.issue == "2"
        assert m.first_page == "100"
        assert m.last_page == "110"
        assert m.published == "2020-03-15"
        assert m.license == "http://creativecommons.org/licenses/by/4.0/"

    def test_detected_title_set(self):
        c = _parse(FULL_JATS)._contents
        assert c.detected_title == "A Study of Things"

    def test_abstract_excludes_heading(self):
        m = _parse(FULL_JATS)._contents.preparsed_metadata
        assert m.abstract == "This is the abstract text."

    def test_keywords(self):
        m = _parse(FULL_JATS)._contents.preparsed_metadata
        assert m.keywords == ["testing", "science"]

    def test_authors_with_affiliations_and_corresponding(self):
        m = _parse(FULL_JATS)._contents.preparsed_metadata
        assert len(m.authors) == 2
        a1, a2 = m.authors
        assert (a1.author_id, a1.given, a1.family) == (1, "Jane", "Smith")
        assert a1.affiliation == "University of Test"  # rid resolved, label stripped
        assert a1.corresponding is True
        assert a1.email == "jane@test.org"
        assert a1.orcid == "https://orcid.org/0000-0002-1234-5678"
        assert (a2.author_id, a2.family) == (2, "Doe")
        assert a2.affiliation == "Test Institute"
        assert a2.corresponding is False

    def test_issn_falls_back_to_electronic(self):
        xml = FULL_JATS.replace(b'<issn pub-type="ppub">1234-5678</issn>', b"")
        m = _parse(xml)._contents.preparsed_metadata
        assert m.issn == "8765-4321"

    def test_published_year_only(self):
        xml = FULL_JATS.replace(
            b'<pub-date pub-type="epub"><day>15</day><month>3</month><year>2020</year></pub-date>',
            b'<pub-date pub-type="epub"><year>2020</year></pub-date>',
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert m.published == "2020"

    def test_license_falls_back_to_text(self):
        xml = FULL_JATS.replace(
            b'<license xlink:href="http://creativecommons.org/licenses/by/4.0/">',
            b"<license>",
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert m.license == "CC BY"


# ---------------------------------------------------------------------------
# Body structure
# ---------------------------------------------------------------------------


class TestBody:
    def test_section_tree_and_sec_type_mapping(self):
        c = _parse(FULL_JATS)._contents
        by_header = {s.header: s for s in c.sections if s.level > 0}
        assert by_header["Introduction"].section_type == CanonicalSection.INTRODUCTION
        assert by_header["Methods"].section_type == CanonicalSection.METHODS
        # Nested subsection has depth 2 and parents to Methods
        part = by_header["Participants"]
        assert part.level == 2
        assert part.parent_section_id == by_header["Methods"].section_id

    def test_paragraphs_become_deferred_texts(self):
        p = _parse(FULL_JATS)
        deferred = [t for t in p._deferred_texts if t[3]]  # needs_segmentation
        joined = " ".join(t[0] for t in deferred)
        assert "First sentence here." in joined
        assert "citing [1]" in joined  # inline xref text preserved
        assert "Nested subsection text." in joined

    def test_display_formula_is_non_segmented_formula(self):
        p = _parse(FULL_JATS)
        formulas = [t for t in p._deferred_texts if t[4]]  # is_formula
        assert len(formulas) == 1
        assert formulas[0][3] is False  # needs_segmentation False
        assert "mc^2" in formulas[0][0]

    def test_table_to_dataframe_and_html(self):
        p = _parse(FULL_JATS)
        c = _segment(p)
        assert len(c.tables) == 1
        tbl = c.tables[0]
        assert list(tbl.df.columns) == ["A", "B"]
        assert tbl.df.values.tolist() == [["1", "2"]]
        assert "<table" in tbl.tbl_html
        assert tbl.caption == "Table 1 A table."

    def test_figure_caption_no_image(self):
        p = _parse(FULL_JATS)
        c = _segment(p)
        assert len(c.figures) == 1
        assert c.figures[0].image_b64 is None
        assert c.figures[0].caption == "Figure 1 A figure."

    def test_ack_and_backmatter_sections(self):
        c = _parse(FULL_JATS)._contents
        headers = {s.header: s for s in c.sections if s.level > 0}
        assert headers["Acknowledgments"].section_type == CanonicalSection.ACKNOWLEDGMENT
        assert "Data Availability" in headers  # back <sec> preserved


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


class TestReferences:
    def test_element_citation_builds_native_references(self):
        c = _parse(FULL_JATS)._contents
        assert c.native_ref_strings is None
        refs = c.native_references
        assert refs is not None and len(refs) == 2
        r1 = refs[0]
        assert r1.bib_id == 1
        assert r1.title == "Prior work"
        assert r1.authors == "Jones, A B; Lee, C"
        assert r1.year == 2018
        assert r1.container == "Nature"
        assert r1.volume == "5"
        assert r1.issue == "3"
        assert r1.first_page == "20"
        assert r1.last_page == "30"
        assert r1.doi == "10.1/prior"
        assert r1.bib_type == "journal_article"
        r2 = refs[1]
        assert r2.bib_id == 2
        assert r2.container == "A Big Book"
        assert r2.publisher == "Book Co"
        assert r2.bib_type == "book"

    def test_thesis_publication_type_maps_to_thesis(self):
        # JATS spells it "thesis"; the taxonomy now has a value for it, so the
        # deposit's own answer survives instead of collapsing into "other".
        xml = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <article-id pub-id-type="doi">10.9/thesis</article-id>
    <title-group><article-title>T</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Body.</p></sec></body>
  <back><ref-list>
    <ref><element-citation publication-type="thesis">
      <person-group person-group-type="author">
        <name><surname>Roy</surname><given-names>P</given-names></name></person-group>
      <year>2015</year><source>A Doctoral Study</source>
      <publisher-name>Univ Press</publisher-name></element-citation></ref>
  </ref-list></back>
</article>"""
        refs = _parse(xml)._contents.native_references
        assert refs is not None and len(refs) == 1
        assert refs[0].bib_type == "thesis"

    def test_references_section_created(self):
        c = _parse(FULL_JATS)._contents
        ref_secs = [s for s in c.sections if s.section_type == CanonicalSection.REFERENCES]
        assert len(ref_secs) == 1
        assert ref_secs[0].header == "References"

    def test_mixed_citation_yields_ref_strings(self):
        c = _parse(MIXED_CITATION_JATS)._contents
        assert c.native_references is None
        assert c.native_ref_strings == [
            "Alpha, A. (2010). First. Journal A, 1, 1-2.",
            "Beta, B. (2011). Second. Journal B, 2, 3-4.",
        ]

    def test_mixed_falls_back_when_any_ref_unstructured(self):
        # One element-citation + one mixed-citation → strings for ALL (invariant:
        # either full structured coverage, or ref strings covering every ref).
        xml = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Hybrid</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
  <back><ref-list>
    <ref><element-citation publication-type="journal">
      <person-group><name><surname>A</surname><given-names>X</given-names></name></person-group>
      <year>2000</year><article-title>Structured</article-title><source>Jrnl</source>
    </element-citation></ref>
    <ref><mixed-citation>King, D. (2001). A Big Book.</mixed-citation></ref>
  </ref-list></back>
</article>"""
        c = _parse(xml)._contents
        assert c.native_references is None
        assert c.native_ref_strings is not None
        assert len(c.native_ref_strings) == 2

    def test_ref_strings_survive_native_seg_branch(self):
        # The pre-segmented strings feed the extractor's "native" branch.
        c = _parse(MIXED_CITATION_JATS)._contents
        assert c.native_ref_strings  # non-empty → seg is skipped downstream


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class TestNamespaces:
    def test_namespaced_document_parses(self):
        c = _parse(NAMESPACED_JATS)._contents
        m = c.preparsed_metadata
        assert m.doi == "10.5/ns.1"
        assert m.title == "Namespaced Paper"
        assert len(m.authors) == 1
        assert m.authors[0].family == "Ng"
        # mixed-citation under a default namespace still routes to strings
        assert c.native_ref_strings is not None
        assert "Ng, W. (2019)" in c.native_ref_strings[0]


# ---------------------------------------------------------------------------
# Contract parity with DocxParser
# ---------------------------------------------------------------------------


class TestContract:
    def test_parse_leaves_sentences_empty(self):
        c = _parse(FULL_JATS)._contents
        assert c.sentences == []  # deferred until apply_segmentation
        assert c.sections[0].header == "Root"
        assert c.sections[0].section_id == 0

    def test_apply_segmentation_populates_sentences(self):
        p = _parse(FULL_JATS)
        c = _segment(p)
        assert len(c.sentences) > 0
        # Each reference is one atomic sentence in the REFERENCES section.
        ref_sec_ids = {
            s.section_id for s in c.sections if s.section_type == CanonicalSection.REFERENCES
        }
        ref_sents = [s for s in c.sentences if s.section_id in ref_sec_ids]
        assert len(ref_sents) == 2


# ---------------------------------------------------------------------------
# DTD named entities (&alpha; etc.)
# ---------------------------------------------------------------------------

# A DOCTYPE plus named entities from the ISO sets JATS pulls in. The parser
# never fetches the DTD, so libxml2 leaves these as unresolved entity nodes;
# jats_native resolves them itself after the parse.
ENTITY_JATS = b"""<?xml version="1.0"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Publishing DTD v1.2 20190208//EN"
 "JATS-journalpublishing1.dtd">
<article>
  <front><article-meta>
    <title-group><article-title>Effects of &alpha;-synuclein &amp; A&beta; at 37&deg;C</article-title></title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>M&uuml;ller</surname><given-names>J&oacute;n</given-names></name>
      </contrib>
    </contrib-group>
    <abstract><p>Levels rose by 5&#37; &mdash; the &alpha; band was &lt;10&nbsp;Hz.</p></abstract>
  </article-meta></front>
  <body><sec><title>Results</title>
    <p>The &beta;&gamma; dimer bound <italic>in&nbsp;vitro</italic> at 25&deg;C.</p></sec></body>
  <back><ref-list>
    <ref><mixed-citation>Sch&ouml;n, K. (2019). The &alpha; problem. Journal, 1, 2-3.</mixed-citation></ref>
  </ref-list></back>
</article>"""


class TestNamedEntities:
    def test_title_resolves_entities(self):
        m = _parse(ENTITY_JATS)._contents.preparsed_metadata
        assert m.title == "Effects of α-synuclein & Aβ at 37°C"

    def test_author_names_resolve_entities(self):
        m = _parse(ENTITY_JATS)._contents.preparsed_metadata
        assert m.authors[0].family == "Müller"
        assert m.authors[0].given == "Jón"

    def test_body_text_resolves_entities_around_inline_markup(self):
        # Entities before, inside the tail of, and after a child element.
        c = _segment(_parse(ENTITY_JATS))
        body = " ".join(s.text for s in c.sentences)
        assert "βγ dimer" in body
        assert "in vitro" in body  # &nbsp; resolves, then collapses
        assert "25°C" in body

    def test_reference_string_resolves_entities(self):
        c = _parse(ENTITY_JATS)._contents
        assert "Schön" in c.native_ref_strings[0]
        assert "The α problem" in c.native_ref_strings[0]

    def test_no_raw_entity_markup_survives_anywhere(self):
        p = _parse(ENTITY_JATS)
        c = _segment(p)
        m = c.preparsed_metadata
        blob = " ".join(
            [m.title or "", m.abstract or "", *c.native_ref_strings]
            + [f"{a.given} {a.family}" for a in m.authors]
            + [s.text for s in c.sentences]
        )
        # A bare "&" is legitimate (from &amp;); "&name;" markup is not.
        assert re.search(r"&[A-Za-z][A-Za-z0-9]*;", blob) is None

    def test_predefined_and_numeric_references_still_work(self):
        m = _parse(ENTITY_JATS)._contents.preparsed_metadata
        assert "5%" in (m.abstract or "")  # &#37;
        assert "<10" in (m.abstract or "")  # &lt;
        assert "—" in (m.abstract or "")  # &mdash;


# ---------------------------------------------------------------------------
# Body-located <ref-list> (EuropePMC fullTextXML shape)
# ---------------------------------------------------------------------------

BODY_REFLIST_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Body Bibliography</article-title></title-group>
  </article-meta></front>
  <body>
    <sec><title>Intro</title><p>Body text citing [1].</p></sec>
    <sec sec-type="ref-list" disp-level="1"><title>REFERENCES</title>
      <sec disp-level="2">
        <ref-list>
          <ref><mixed-citation>Alpha, A. (2010). First. Journal A, 1, 1-2.</mixed-citation></ref>
          <ref><mixed-citation>Beta, B. (2011). Second. Journal B, 2, 3-4.</mixed-citation></ref>
        </ref-list>
      </sec>
    </sec>
  </body>
</article>"""


class TestBodyRefList:
    """EuropePMC's fullTextXML emits the bibliography in <body>, not <back>."""

    def test_body_ref_list_is_ingested(self):
        c = _parse(BODY_REFLIST_JATS)._contents
        assert c.native_ref_strings == [
            "Alpha, A. (2010). First. Journal A, 1, 1-2.",
            "Beta, B. (2011). Second. Journal B, 2, 3-4.",
        ]

    def test_refs_land_in_a_references_section(self):
        p = _parse(BODY_REFLIST_JATS)
        c = _segment(p)
        ref_ids = {
            s.section_id for s in c.sections if s.section_type == CanonicalSection.REFERENCES
        }
        assert ref_ids
        in_refs = [s.text for s in c.sentences if s.section_id in ref_ids]
        assert len(in_refs) == 2
        assert in_refs[0].startswith("Alpha, A.")

    def test_the_enclosing_sec_is_reused_not_duplicated(self):
        c = _parse(BODY_REFLIST_JATS)._contents
        refs_secs = [s for s in c.sections if s.section_type == CanonicalSection.REFERENCES]
        assert len(refs_secs) == 1

    def test_a_back_ref_list_still_wins(self):
        """A document with both keeps <back> — the authoritative location."""
        xml = BODY_REFLIST_JATS.replace(
            b"</body>",
            b"</body><back><ref-list>"
            b"<ref><mixed-citation>Gamma, G. (2012). Canonical.</mixed-citation></ref>"
            b"</ref-list></back>",
        )
        c = _parse(xml)._contents
        assert c.native_ref_strings == ["Gamma, G. (2012). Canonical."]

    def test_europepmc_fixture_yields_its_references(self):
        """The tracked PMC4383902 fixture carries its 17 refs inside <body>."""
        c = _parse(EUROPEPMC_FIXTURE.read_bytes())._contents
        assert len(c.native_ref_strings or c.native_references or []) == 17


# ---------------------------------------------------------------------------
# <collab> (consortium) authors
# ---------------------------------------------------------------------------

COLLAB_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Group Authorship</article-title></title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>Smith</surname><given-names>Jane</given-names></name>
      </contrib>
      <contrib contrib-type="author">
        <collab>The Genome Sequencing Consortium</collab>
      </contrib>
      <contrib contrib-type="author">
        <xref ref-type="aff" rid="a1"/>
      </contrib>
    </contrib-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
</article>"""


class TestCollabAuthors:
    def test_collab_name_is_kept(self):
        m = _parse(COLLAB_JATS)._contents.preparsed_metadata
        names = [(a.given, a.family) for a in m.authors]
        assert ("", "The Genome Sequencing Consortium") in names

    def test_nameless_contrib_is_dropped_not_emitted_blank(self):
        m = _parse(COLLAB_JATS)._contents.preparsed_metadata
        assert all(a.given or a.family for a in m.authors)
        assert len(m.authors) == 2

    def test_author_ids_stay_contiguous(self):
        m = _parse(COLLAB_JATS)._contents.preparsed_metadata
        assert [a.author_id for a in m.authors] == [1, 2]

    def test_collab_is_marked_as_an_organization(self):
        # The export writes an organization author's name to author[].literal.
        from bibr.models import ORGANIZATION_ROLE

        m = _parse(COLLAB_JATS)._contents.preparsed_metadata
        assert [a.role for a in m.authors] == [[], [ORGANIZATION_ROLE]]


# ---------------------------------------------------------------------------
# Text flattening — markup that implies a word boundary
# ---------------------------------------------------------------------------

BOUNDARY_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Cognitive load<break/>and recall</article-title></title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>Smith</surname><given-names>Jane</given-names></name>
        <xref ref-type="aff" rid="a1"/>
      </contrib>
      <aff id="a1"><label>1</label><institution>Department of Psychology</institution>\
<institution>Utrecht University</institution><addr-line>Heidelberglaan 1</addr-line>\
<city>Utrecht</city><country>Netherlands</country></aff>
      <aff id="a2"><institution>Institute of Testing</institution>, <city>Leiden</city></aff>
    </contrib-group>
    <abstract><title>Abstract</title><p>First paragraph.</p><p>Second paragraph.</p></abstract>
  </article-meta></front>
  <body><sec><title>Intro</title>
    <p>Water is H<sub>2</sub>O and <italic>very</italic>wet.</p>
    <fig><label>Figure 1</label><caption><title>Overview</title><p>The design.</p></caption>
      <graphic/></fig>
  </sec></body>
</article>"""


class TestTextBoundaries:
    def test_break_separates_words(self):
        m = _parse(BOUNDARY_JATS)._contents.preparsed_metadata
        assert m.title == "Cognitive load and recall"

    def test_structured_aff_fields_are_separated(self):
        m = _parse(BOUNDARY_JATS)._contents.preparsed_metadata
        assert m.authors[0].affiliation == (
            "Department of Psychology Utrecht University Heidelberglaan 1 Utrecht Netherlands"
        )

    def test_existing_source_punctuation_is_not_doubled(self):
        p = _parse(BOUNDARY_JATS)
        assert p._aff_map["a2"] == "Institute of Testing, Leiden"

    def test_abstract_paragraphs_are_separated(self):
        m = _parse(BOUNDARY_JATS)._contents.preparsed_metadata
        assert m.abstract == "First paragraph. Second paragraph."

    def test_inline_markup_is_not_separated(self):
        p = _parse(BOUNDARY_JATS)
        c = _segment(p)
        texts = [s.text for s in c.sentences]
        assert "Water is H2O and verywet." in texts

    def test_caption_title_and_body_are_separated(self):
        c = _parse(BOUNDARY_JATS)._contents
        assert c.figures[0].caption == "Figure 1 Overview The design."


def test_declared_identifiers_and_language_reach_the_metadata():
    xml = b"""
    <article xml:lang="en"><front><article-meta>
      <article-id pub-id-type="doi">10.1234/x</article-id>
      <article-id pub-id-type="pmid">31234567</article-id>
      <article-id pub-id-type="pmc">6543210</article-id>
      <title-group><article-title>Identified</article-title></title-group>
    </article-meta></front><body><sec><title>Intro</title><p>Text.</p></sec></body></article>
    """
    meta = JatsParser(xml).parse().preparsed_metadata
    assert meta is not None
    assert (meta.doi, meta.pmid, meta.pmcid, meta.language) == (
        "10.1234/x",
        "31234567",
        "PMC6543210",
        "en",
    )


FOOTNOTE_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Notes</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
  <back>
    <sec sec-type="fn-group"><title>Footnotes</title>
      <fn-group><fn id="FN1"><label>&#8224;</label><p>Printed under a heading.</p></fn></fn-group>
    </sec>
    <fn-group><fn id="FN2"><p>A back-matter note.</p></fn></fn-group>
  </back>
</article>"""


def test_footnotes_become_synthetic_footnote_sections_with_their_label():
    """Every <fn> becomes a footnote (a synthetic section the export turns into
    a footnote row), whether its <fn-group> sits in back matter or under a
    heading of its own; the printed <label> is kept apart."""
    contents = _segment(_parse(FOOTNOTE_JATS))
    notes = [s for s in contents.sections if s.synthetic_kind == "footnote"]
    assert [s.footnote_label for s in notes] == ["\u2020", None]
    held = [t for t in contents.sentences if t.section_id in {n.section_id for n in notes}]
    assert [t.text for t in held] == ["\u2020 Printed under a heading.", "A back-matter note."]


def test_floats_take_their_label_from_the_label_element():
    """``<label>`` without the word is the float's label; the caption keeps it
    glued on as before. Mentions resolve by the label, not the order."""
    xml = b"""
    <article><front><article-meta>
      <title-group><article-title>Labelled</article-title></title-group>
    </article-meta></front><body><sec><title>Results</title>
      <p>Table S1 and Table 2 agree with Fig. 3.</p>
      <table-wrap><label>Table 2</label><caption><p>Main.</p></caption>
        <table><tr><th>A</th></tr><tr><td>1</td></tr></table></table-wrap>
      <table-wrap><label>S1</label><caption><p>Extra.</p></caption>
        <table><tr><th>A</th></tr><tr><td>1</td></tr></table></table-wrap>
      <fig><label>Fig. 3</label><caption><p>A figure.</p></caption><graphic/></fig>
      <fig><caption><p>Figure 4. No label element.</p></caption><graphic/></fig>
      <fig><label>Scheme 1</label><caption><p>Not a figure label.</p></caption><graphic/></fig>
    </sec></body></article>
    """
    c = _segment(_parse(xml))
    assert [(t.label, t.caption) for t in c.tables] == [("2", "Table 2 Main."), ("S1", "S1 Extra.")]
    assert [f.label for f in c.figures] == ["3", "4", None]
    table_xrefs = [(x.xref_type, x.xref_id, x.tier) for x in c.xrefs if x.xref_type != "figure"]
    assert table_xrefs == [("table", 2, "label"), ("table", 1, "label")]
    assert [(x.xref_id, x.tier) for x in c.xrefs if x.xref_type == "figure"] == [(1, "label")]


def test_jats_sentences_are_not_ocr_text_and_keep_their_prose():
    """Late cleanup assumed OCR input and fused "a 2 x 2 design" into "a2x2"
    and dropped the underscores from ``age_group`` and email addresses."""
    prose = "Participants completed a 2 x 2 x 3 design; age_group was coded 1 2 3."
    xml = (
        b'<?xml version="1.0"?><article><front><article-meta><title-group>'
        b"<article-title>T</article-title></title-group></article-meta></front>"
        b"<body><sec><title>Method</title><p>" + prose.encode() + b"</p></sec></body>"
        b"<back><fn-group><fn><p>Contact john_smith@uni.edu.</p></fn></fn-group></back>"
        b"</article>"
    )
    contents = _segment(_parse(xml))

    contents.finalize_text()
    texts = [s.text for s in contents.sentences]
    assert prose in texts
    assert any("john_smith@uni.edu" in text for text in texts)
    assert all(sentence.from_ocr is False for sentence in contents.sentences)


def test_jats_captions_are_not_ocr_text():
    """Captions are document text too: the late cleanup's spaced-run collapse
    turned "items 1 2 3" into "items 123"."""
    xml = (
        b'<?xml version="1.0"?><article><front><article-meta><title-group>'
        b"<article-title>T</article-title></title-group></article-meta></front>"
        b"<body><sec><title>Results</title><p>Body text.</p>"
        b"<table-wrap><label>Table 1</label><caption><p>Items 1 2 3 by age_group.</p></caption>"
        b"<table><tr><th>A</th></tr><tr><td>1</td></tr></table></table-wrap>"
        b"<fig><label>Figure 1</label><caption><p>Scores on items 1 2 3 by age_group.</p>"
        b"</caption><graphic/></fig></sec></body></article>"
    )
    contents = _segment(_parse(xml))

    contents.finalize_text()
    texts = [s.text for s in contents.sentences]
    assert "Table 1 Items 1 2 3 by age_group." in texts
    assert "Figure 1 Scores on items 1 2 3 by age_group." in texts
    assert all(sentence.from_ocr is False for sentence in contents.sentences)


# Inline MathML as PLOS and eLife pretty-print it: whitespace between elements.
# Operators and punctuation close up, as a renderer shows them and as the same
# formula reads from a publisher that writes no whitespace.
MATHML_OPERATORS = [
    (
        '<mml:msub><mml:mover accent="true"><mml:mi>y</mml:mi> <mml:mo>¯</mml:mo></mml:mover> '
        "<mml:mrow><mml:mi>t</mml:mi> <mml:mo>-</mml:mo> <mml:mn>1</mml:mn></mml:mrow></mml:msub>",
        "y¯t-1",
    ),
    (
        "<mml:mi>t</mml:mi> <mml:mo>(</mml:mo> <mml:mn>28</mml:mn> <mml:mo>)</mml:mo> "
        "<mml:mo>=</mml:mo> <mml:mn>2</mml:mn> <mml:mo>.</mml:mo> <mml:mn>1</mml:mn>",
        "t(28)=2.1",
    ),
    ("<mml:mi>SD</mml:mi>\n  <mml:mo>=</mml:mo>\n  <mml:mn>1.2</mml:mn>", "SD=1.2"),
    (
        "<mml:msub><mml:mi>a</mml:mi><mml:mrow><mml:mi>i</mml:mi><mml:mi>j</mml:mi></mml:mrow>"
        "</mml:msub> <mml:mo>=</mml:mo> <mml:mn>5</mml:mn>",
        "aij=5",
    ),
    (
        "<mml:mi>β</mml:mi> <mml:mo>∼</mml:mo> <mml:mtext>Cauchy</mml:mtext> <mml:mo>(</mml:mo> "
        "<mml:mn>0</mml:mn> <mml:mo>,</mml:mo> <mml:mn>2</mml:mn> <mml:mo>.</mml:mo> "
        "<mml:mn>5</mml:mn> <mml:mo>)</mml:mo>",
        "β∼Cauchy(0,2.5)",
    ),
    ("<mml:mi>x</mml:mi> <mml:mi>y</mml:mi>", "xy"),
    (
        "<mml:mi>f</mml:mi> <mml:mo>(</mml:mo> <mml:mi>x</mml:mi> <mml:mo>,</mml:mo> "
        "<mml:mi>y</mml:mi> <mml:mo>)</mml:mo>",
        "f(x,y)",
    ),
    (
        "<mml:mi>cos</mml:mi> <mml:mo>&#x2061;</mml:mo> <mml:mo>(</mml:mo> "
        "<mml:mi>q</mml:mi> <mml:mo>)</mml:mo>",
        "cos\u2061(q)",
    ),
]
# Where a renderer spaces words by other means, or the source spells a word
# one letter per element, the whitespace keeps them apart.
MATHML_WORDS = [
    (
        "<mml:mi>b</mml:mi> <mml:mo>⋅</mml:mo> <mml:mi>ln</mml:mi> <mml:mi>dbh</mml:mi>",
        "b⋅ln dbh",
    ),
    ("<mml:mn>0.93</mml:mn> <mml:mtext>GeV</mml:mtext>", "0.93 GeV"),
    ("<mml:mn>2</mml:mn>\n  <mml:mtext>s</mml:mtext>", "2 s"),
    (
        "".join(f'<mml:mi mathvariant="normal">{c}</mml:mi>' for c in "direct")
        + " "
        + "".join(f'<mml:mi mathvariant="normal">{c}</mml:mi>' for c in "effect"),
        "direct effect",
    ),
    (
        "<mml:mi>A</mml:mi><mml:mo>,</mml:mo> <mml:mi>B</mml:mi><mml:mo>,</mml:mo> "
        "<mml:mtext>and</mml:mtext> <mml:mi>C</mml:mi>",
        "A,B, and C",
    ),
    # A renderer spaces a function name from a bare argument.
    ("<mml:mi>sin</mml:mi> <mml:mo>&#x2061;</mml:mo> <mml:mi>x</mml:mi>", "sin\u2061 x"),
]
# Whitespace between two numbers stays: a renderer shows them apart (the parts
# of a fraction, a coefficient and its root, a base and its scripts).
MATHML_NUMBERS = [
    (
        "<mml:mi>M</mml:mi> <mml:mo>=</mml:mo> "
        "<mml:mfrac><mml:mn>1</mml:mn> <mml:mn>2</mml:mn></mml:mfrac> "
        "<mml:mo>(</mml:mo> <mml:mi>s</mml:mi> <mml:mo>)</mml:mo>",
        "M=1 2(s)",
    ),
    ("<mml:mn>3</mml:mn> <mml:msqrt><mml:mn>2</mml:mn></mml:msqrt>", "3 2"),
    ("<mml:mroot><mml:mn>27</mml:mn> <mml:mn>3</mml:mn></mml:mroot>", "27 3"),
    (
        "<mml:msubsup><mml:mi>g</mml:mi> <mml:mn>1</mml:mn> <mml:mn>1</mml:mn></mml:msubsup>",
        "g1 1",
    ),
    ("<mml:msup><mml:mn>10</mml:mn> <mml:mn>3</mml:mn></mml:msup>", "10 3"),
    (
        "<mml:msup><mml:mi>Σ</mml:mi> <mml:mrow><mml:mo>-</mml:mo> <mml:mn>1</mml:mn></mml:mrow>"
        "</mml:msup> <mml:mn>1</mml:mn>",
        "Σ-1 1",
    ),
]
# Matrix cells and <mspace> separate the text around them, whitespace or not.
MATHML_SEPARATORS = [
    (
        "<mml:mi>J</mml:mi> <mml:mo>=</mml:mo> <mml:mrow><mml:mo>[</mml:mo> <mml:mtable>"
        "<mml:mtr><mml:mtd><mml:mn>0</mml:mn></mml:mtd> <mml:mtd><mml:mn>1</mml:mn></mml:mtd>"
        "</mml:mtr> <mml:mtr><mml:mtd><mml:mn>10</mml:mn></mml:mtd> "
        "<mml:mtd><mml:mn>20</mml:mn></mml:mtd></mml:mtr></mml:mtable> <mml:mo>]</mml:mo></mml:mrow>",
        "J=[ 0 1 10 20]",
    ),
    # E_{t-1} \quad 0 < λ ≤ 1, as in journal.pone.0278264.
    (
        "<mml:msub><mml:mi>E</mml:mi> <mml:mrow><mml:mi>t</mml:mi> <mml:mo>-</mml:mo> "
        '<mml:mn>1</mml:mn></mml:mrow></mml:msub> <mml:mspace width="8pt"/> <mml:mn>0</mml:mn> '
        "<mml:mo>&lt;</mml:mo> <mml:mi>λ</mml:mi> <mml:mo>≤</mml:mo> <mml:mn>1</mml:mn>",
        "Et-1 0<λ≤1",
    ),
    (
        " ".join(f"<mml:mi>{c}</mml:mi>" for c in "naive")
        + ' <mml:mspace width="1em"/> '
        + " ".join(f"<mml:mi>{c}</mml:mi>" for c in "seasonality"),
        "naive seasonality",
    ),
    # A negative space pulls its neighbours together.
    (
        '<mml:mi>a</mml:mi> <mml:mspace width="negativethinmathspace"/> <mml:mi>b</mml:mi>',
        "ab",
    ),
]


def _mathml_jats(paragraph: str) -> bytes:
    return (
        '<?xml version="1.0"?><article xmlns:mml="http://www.w3.org/1998/Math/MathML">'
        "<front><article-meta><title-group><article-title>T</article-title></title-group>"
        f"</article-meta></front><body><sec><title>Method</title><p>{paragraph}</p></sec>"
        "</body></article>"
    ).encode()


def _inline_text(math: str) -> str:
    inline = (
        '<inline-formula><mml:math display="inline"><mml:mrow>'
        f"{math}</mml:mrow></mml:math></inline-formula>"
    )
    (entry,) = _parse(_mathml_jats(f"We used {inline} here.")).assembler.entries
    return entry.text


@pytest.mark.parametrize(
    ("math", "expected"),
    MATHML_OPERATORS,
    ids=[
        "index",
        "statistic",
        "identifier",
        "subscript",
        "text",
        "variables",
        "arguments",
        "function-bracket",
    ],
)
def test_whitespace_between_mathml_elements_is_dropped(math, expected):
    """Kept as text it split "2.1" into "2 . 1"; the late clean-up used to
    fuse some of those runs back, until it stopped touching document text."""
    assert _inline_text(math) == f"We used {expected} here."
    assert _inline_text(re.sub(r">\s+<", "><", math)) == f"We used {expected} here."


@pytest.mark.parametrize(
    ("math", "expected"),
    MATHML_WORDS,
    ids=[
        "function-names",
        "unit-mtext",
        "unit-newline",
        "spelled-words",
        "text-after-comma",
        "function-argument",
    ],
)
def test_whitespace_between_mathml_elements_keeps_words_apart(math, expected):
    assert _inline_text(math) == f"We used {expected} here."


@pytest.mark.parametrize(
    ("math", "expected"),
    MATHML_NUMBERS,
    ids=["fraction", "coefficient-root", "root-index", "scripts", "power", "different-rows"],
)
def test_whitespace_between_mathml_numbers_is_kept(math, expected):
    """Dropped, it wrote one number for two: one half read "12"."""
    assert _inline_text(math) == f"We used {expected} here."


@pytest.mark.parametrize(
    ("math", "expected"),
    MATHML_SEPARATORS,
    ids=["matrix", "mspace-number", "mspace-words", "negative-mspace"],
)
def test_mathml_matrix_cells_and_spaces_separate_the_text(math, expected):
    assert _inline_text(math) == f"We used {expected} here."
    assert _inline_text(re.sub(r">\s+<", "><", math)) == f"We used {expected} here."


def test_mathml_fraction_in_a_table_cell_keeps_its_numbers_apart():
    fraction = (
        "<inline-formula><mml:math><mml:mfrac><mml:mn>3</mml:mn> <mml:mn>4</mml:mn></mml:mfrac>"
        "</mml:math></inline-formula>"
    )
    xml = _mathml_jats(
        'Text.</p><table-wrap id="t1"><label>Table 1</label><caption><p>Shares.</p></caption>'
        f"<table><tr><th>share</th><th>n</th></tr><tr><td>{fraction}</td><td>12</td></tr></table>"
        "</table-wrap><p>More."
    )
    contents = _segment(_parse(xml))

    assert contents.tables[0].df.values.tolist() == [["3 4", "12"]]


def test_mathml_whitespace_next_to_prose_keeps_the_words_apart():
    xml = _mathml_jats(
        'the value<inline-formula><mml:math display="inline"> <mml:mi>x</mml:mi> '
        "</mml:math></inline-formula>is small, <inline-formula><mml:math>"
        "<mml:mi>y</mml:mi> </mml:math></inline-formula>, too."
    )
    (entry,) = _parse(xml).assembler.entries
    assert entry.text == "the value x is small, y, too."


def test_table_wrap_without_a_table_is_kept_with_its_caption():
    """A table printed as an image has a <graphic> and no <table>. It was
    dropped with its caption, so "Table 2" resolved nowhere; it is kept with
    empty contents, as the HTML parser keeps a captioned image-only table. A
    table-wrap with neither a table nor a label or caption is still dropped."""
    xml = b"""
    <article><front><article-meta>
      <title-group><article-title>Image table</article-title></title-group>
    </article-meta></front><body><sec><title>Results</title>
      <p>Table 2 lists the values.</p>
      <table-wrap><label>Table 1</label><caption><p>Counts.</p></caption>
        <table><tr><th>N</th></tr><tr><td>12</td></tr></table></table-wrap>
      <table-wrap><label>Table 2</label><caption><p>Scanned values.</p></caption>
        <graphic xlink:href="t2.png" xmlns:xlink="http://www.w3.org/1999/xlink"/></table-wrap>
      <table-wrap><graphic/></table-wrap>
    </sec></body></article>
    """
    c = _segment(_parse(xml))

    assert [(t.label, t.caption, t.contents, t.tbl_html) for t in c.tables] == [
        ("1", "Table 1 Counts.", [["N"], ["12"]], c.tables[0].tbl_html),
        ("2", "Table 2 Scanned values.", [], ""),
    ]
    assert [(x.xref_type, x.xref_id) for x in c.xrefs] == [("table", 2)]


# ---------------------------------------------------------------------------
# Multiple / nested <ref-list>s (audit input-parsers-5)
# ---------------------------------------------------------------------------

MULTI_REFLIST_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Two lists</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
  <back>
    <ref-list><title>References</title>
      <ref><element-citation publication-type="journal">
        <person-group><name><surname>A</surname><given-names>X</given-names></name></person-group>
        <year>2000</year><article-title>Title 1</article-title><source>Jrnl</source>
      </element-citation></ref>
      <ref><element-citation publication-type="journal">
        <person-group><name><surname>B</surname><given-names>Y</given-names></name></person-group>
        <year>2001</year><article-title>Title 2</article-title><source>Jrnl</source>
      </element-citation></ref>
      <ref><element-citation publication-type="journal">
        <person-group><name><surname>C</surname><given-names>Z</given-names></name></person-group>
        <year>2002</year><article-title>Title 3</article-title><source>Jrnl</source>
      </element-citation></ref>
    </ref-list>
    <ref-list><title>Data references</title>
      <ref><element-citation publication-type="dataset">
        <person-group><name><surname>D</surname><given-names>W</given-names></name></person-group>
        <year>2003</year><article-title>Title 9</article-title><source>Repo</source>
      </element-citation></ref>
    </ref-list>
  </back>
</article>"""

NESTED_REFLIST_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Nested lists</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Text.</p></sec></body>
  <back><ref-list><title>References</title>
    <ref-list><title>Primary</title>
      <ref><mixed-citation>Alpha, A. (2010). First.</mixed-citation></ref>
      <ref><mixed-citation>Beta, B. (2011). Second.</mixed-citation></ref>
    </ref-list>
  </ref-list></back>
</article>"""


class TestMultipleRefLists:
    def test_later_list_extends_instead_of_replacing(self):
        c = _parse(MULTI_REFLIST_JATS)._contents
        assert c.native_references is not None
        assert [(r.bib_id, r.title) for r in c.native_references] == [
            (1, "Title 1"),
            (2, "Title 2"),
            (3, "Title 3"),
            (4, "Title 9"),
        ]

    def test_each_list_keeps_its_references_section(self):
        c = _parse(MULTI_REFLIST_JATS)._contents
        ref_secs = [s for s in c.sections if s.section_type == CanonicalSection.REFERENCES]
        assert [s.header for s in ref_secs] == ["References", "Data references"]

    def test_mixed_union_falls_back_to_strings_for_all(self):
        # Structured main list + mixed data list: the invariant is decided
        # over the union, so every ref (not just the mixed one) is a string.
        data_ref = (
            b'<ref><element-citation publication-type="dataset">\n'
            b"        <person-group><name><surname>D</surname><given-names>W</given-names>"
            b"</name></person-group>\n"
            b"        <year>2003</year><article-title>Title 9</article-title>"
            b"<source>Repo</source>\n      </element-citation></ref>"
        )
        assert data_ref in MULTI_REFLIST_JATS
        xml = MULTI_REFLIST_JATS.replace(
            data_ref,
            b"<ref><mixed-citation>Datum, D. (2003). Title 9.</mixed-citation></ref>",
        )
        c = _parse(xml)._contents
        assert c.native_references is None
        assert c.native_ref_strings is not None
        assert len(c.native_ref_strings) == 4
        assert c.native_ref_strings[-1] == "Datum, D. (2003). Title 9."

    def test_nested_ref_lists_yield_refs_and_rows(self):
        p = _parse(NESTED_REFLIST_JATS)
        c = _segment(p)
        assert c.native_references is None
        assert c.native_ref_strings == [
            "Alpha, A. (2010). First.",
            "Beta, B. (2011). Second.",
        ]
        ref_ids = {
            s.section_id for s in c.sections if s.section_type == CanonicalSection.REFERENCES
        }
        assert [s.text for s in c.sentences if s.section_id in ref_ids] == [
            "Alpha, A. (2010). First.",
            "Beta, B. (2011). Second.",
        ]


# ---------------------------------------------------------------------------
# <alternatives> (audit input-parsers-7)
# ---------------------------------------------------------------------------

ALTERNATIVES_JATS = (
    b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
    b"<front><article-meta><title-group><article-title>T</article-title></title-group>"
    b"</article-meta></front><body><sec><title>R</title>"
    b"<p>We set the significance level to <inline-formula><alternatives>"
    b"<tex-math>\\documentclass[12pt]{minimal}\\usepackage{amsmath}"
    b"\\begin{document}$$\\alpha = 0.05$$\\end{document}</tex-math>"
    b'<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML">'
    b"<mml:mi>\xce\xb1</mml:mi><mml:mo>=</mml:mo><mml:mn>0.05</mml:mn></mml:math>"
    b'<graphic xlink:href="g.png"/></alternatives></inline-formula> for all tests.</p>'
    b"</sec></body></article>"
)


class TestAlternatives:
    def test_inline_alternatives_emits_mathml_only(self):
        c = _segment(_parse(ALTERNATIVES_JATS))
        assert [s.text for s in c.sentences] == [
            "We set the significance level to \u03b1=0.05 for all tests."
        ]

    def test_tex_only_alternatives_drops_the_preamble(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>R</title><p>With <inline-formula><alternatives><tex-math>"
            b"\\documentclass{minimal}\\usepackage{amsmath}"
            b"\\begin{document}$$y = a + b$$\\end{document}</tex-math>"
            b"</alternatives></inline-formula> inside.</p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [s.text for s in c.sentences] == ["With y = a + b inside."]

    def test_disp_formula_alternatives_is_one_entry(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:mml="http://www.w3.org/1998/Math/MathML">'
            b"<front><article-meta><title-group>"
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>R</title>"
            b"<disp-formula><alternatives><tex-math>\\documentclass{minimal}"
            b"\\begin{document}$$y = a + b$$\\end{document}</tex-math>"
            b"<mml:math><mml:mi>y=a</mml:mi></mml:math></alternatives><label>(1)</label>"
            b"</disp-formula></sec></body></article>"
        )
        p = _parse(xml)
        formulas = [(t[0], t[4]) for t in p._deferred_texts]
        assert formulas == [("y=a (1)", True)]

    def test_graphic_only_alternatives_keeps_the_tail(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>R</title>"
            b"<p>See <inline-formula><alternatives>"
            b'<graphic xlink:href="g.png"/></alternatives></inline-formula> here.</p>'
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [s.text for s in c.sentences] == ["See here."]

    def test_bare_tex_math_preamble_is_trimmed(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>R</title>"
            b"<p>With <inline-formula><tex-math>\\documentclass[12pt]{minimal}"
            b"\\usepackage{amsmath}\\begin{document}$$y = a + b$$\\end{document}"
            b"</tex-math></inline-formula> inside.</p>"
            b"<disp-formula><tex-math>\\documentclass{minimal}"
            b"\\begin{document}$$E = mc^2$$\\end{document}</tex-math></disp-formula>"
            b"</sec></body></article>"
        )
        p = _parse(xml)
        c = _segment(p)
        assert [s.text for s in c.sentences] == ["With y = a + b inside.", "E = mc^2"]
        assert [(t[0], t[4]) for t in p._deferred_texts if t[4]] == [("E = mc^2", True)]

    def test_plain_tex_math_is_untouched(self):
        # No preamble, no delimiters: the formula text passes through as-is.
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>R</title>"
            b"<disp-formula><tex-math>E = mc^2</tex-math></disp-formula>"
            b"</sec></body></article>"
        )
        p = _parse(xml)
        assert [(t[0], t[4]) for t in p._deferred_texts] == [("E = mc^2", True)]


# ---------------------------------------------------------------------------
# Blocks nested inside <p> (audit input-parsers-8)
# ---------------------------------------------------------------------------

BLOCKS_IN_P_JATS = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>Nested blocks</article-title></title-group>
  </article-meta></front>
  <body><sec><title>R</title>
    <p>Text before <fig><label>Figure 1</label><caption><p>A cap.</p></caption>
      <graphic xlink:href="f.png"/></fig> after.</p>
    <p>Eq: <disp-formula><label>(2)</label><tex-math>x</tex-math></disp-formula> done.</p>
    <p>See table <table-wrap><label>Table 1</label><caption><p>Tcap</p></caption>
      <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
      </table-wrap></p>
    <p>A plain paragraph stays one entry.</p>
  </sec></body>
</article>"""


class TestBlocksInsideParagraph:
    def test_fig_inside_p_is_registered_not_merged(self):
        p = _parse(BLOCKS_IN_P_JATS)
        c = _segment(p)
        assert [f.caption for f in c.figures] == ["Figure 1 A cap."]
        texts = [s.text for s in c.sentences]
        assert "Text before Figure 1 A cap. after." not in texts
        assert "Text before" in texts
        assert "after." in texts

    def test_table_wrap_inside_p_is_registered_not_merged(self):
        p = _parse(BLOCKS_IN_P_JATS)
        c = _segment(p)
        assert [(t.label, t.caption) for t in c.tables] == [("1", "Table 1 Tcap")]
        assert c.tables[0].df.values.tolist() == [["1", "2"]]
        texts = [s.text for s in c.sentences]
        assert "See table Table 1 Tcap A B 1 2" not in texts

    def test_disp_formula_inside_p_is_a_formula_entry(self):
        p = _parse(BLOCKS_IN_P_JATS)
        formulas = [t for t in p._deferred_texts if t[4]]
        assert len(formulas) == 1
        assert formulas[0][0] == "(2) x"
        texts = [t[0] for t in p._deferred_texts if t[3]]
        assert "Eq: (2)x done." not in texts

    def test_formula_label_does_not_fuse_with_the_following_word(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>R</title>"
            b"<p>The model is <disp-formula><label>(2)</label><tex-math>y=x</tex-math>"
            b"</disp-formula>where y is the outcome.</p>"
            b"</sec></body></article>"
        )
        p = _parse(xml)
        texts = [t[0] for t in p._deferred_texts if t[3]]
        assert "(2)where" not in " ".join(texts)
        assert "where y is the outcome." in texts

    def test_plain_paragraph_stays_one_entry(self):
        p = _parse(BLOCKS_IN_P_JATS)
        texts = [t[0] for t in p._deferred_texts if t[3]]
        assert "A plain paragraph stays one entry." in texts

    def test_table_footnotes_are_kept_with_their_links(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>R</title>"
            b"<p>Values <table-wrap><label>Table 1</label><caption><p>Tcap</p></caption>"
            b"<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
            b"<table-wrap-foot><p>Source data at "
            b'<ext-link ext-link-type="uri" xlink:href="https://example.org/src">'
            b"https://example.org/src</ext-link>.</p></table-wrap-foot>"
            b"</table-wrap></p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [(t.label, t.caption) for t in c.tables] == [("1", "Table 1 Tcap")]
        assert "Source data at https://example.org/src." in [s.text for s in c.sentences]
        assert [(link.url, link.link_text) for link in c.links] == [
            ("https://example.org/src", "https://example.org/src")
        ]

    def test_caption_sentence_of_in_paragraph_fig_keeps_its_url(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>R</title>"
            b"<p>Shown <fig><label>Figure 1</label><caption><p>Rates at "
            b'<ext-link ext-link-type="uri" xlink:href="https://example.org/rates">'
            b"https://example.org/rates</ext-link>.</p></caption>"
            b'<graphic xlink:href="f.png"/></fig> here.</p>'
            b"<fig><label>Figure 2</label><caption><p>Plain cap at "
            b'<ext-link ext-link-type="uri" xlink:href="https://example.org/plain">'
            b"https://example.org/plain</ext-link>.</p></caption>"
            b'<graphic xlink:href="g.png"/></fig>'
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [f.caption for f in c.figures] == [
            "Figure 1 Rates at https://example.org/rates.",
            "Figure 2 Plain cap at https://example.org/plain.",
        ]
        urls = [link.url for link in c.links]
        # The in-paragraph caption keeps its regex link; a sibling float's
        # caption never had one and still does not.
        assert "https://example.org/rates" in urls
        assert "https://example.org/plain" not in urls

    def test_table_foot_fn_paragraphs_are_kept_with_their_links(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>R</title>"
            b"<p>Values <table-wrap><label>Table 1</label><caption><p>Tcap</p></caption>"
            b"<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
            b"<table-wrap-foot><fn><p>Model from "
            b'<ext-link ext-link-type="uri" xlink:href="https://example.org/mm5">'
            b"https://example.org/mm5</ext-link>.</p></fn></table-wrap-foot>"
            b"</table-wrap></p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert "Model from https://example.org/mm5." in [s.text for s in c.sentences]
        assert [(link.url, link.link_text) for link in c.links] == [
            ("https://example.org/mm5", "https://example.org/mm5")
        ]

    def test_multi_table_wrap_inside_p_keeps_every_table(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>R</title><p>Intro "
            b"<table-wrap><label>Table 1</label><caption><p>Curriculum</p></caption>"
            b"<table><tr><th>Year</th></tr><tr><td>First year Neck ultrasound session</td></tr></table>"
            b"<table><tr><th>Year</th></tr><tr><td>Second year Cardiac ultrasound session</td></tr></table>"
            b"</table-wrap> tail.</p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert len(c.tables) == 1
        grid = c.tables[0].df.values.tolist()
        flat = " ".join(str(v) for row in grid for v in row)
        assert "Neck ultrasound session" in flat
        assert "Cardiac ultrasound session" in flat

    def test_statement_labels_and_disp_quote_attrib_inside_p_are_kept(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title>"
            b"<p>As one author put it <disp-quote><p>Men do not diet.</p>"
            b"<attrib>(Moi 1991, p. 1030)</attrib></disp-quote> which we test.</p>"
            b"<p>Intro <statement><label>Study 1</label><title>Case 1</title>"
            b"<p>A patient presented.</p></statement> tail.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        texts = [s.text for s in c.sentences]
        assert "Study 1" in texts
        assert "Case 1" in texts
        assert "(Moi 1991, p. 1030)" in texts
        assert texts.count("(Moi 1991, p. 1030)") == 1

    def test_boxed_text_sec_inside_p_keeps_its_depth(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>Parent</title><p>Body.</p>"
            b"<p>Para <boxed-text><sec><title>Box in p</title><p>Words.</p></sec>"
            b"</boxed-text> tail.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Box in p"].level == 2
        assert by_header["Box in p"].parent_section_id == by_header["Parent"].section_id

    def test_floats_group_figs_and_tables_are_kept(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group>"
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<floats-group><fig><label>Figure 1</label><caption><p>Cap.</p></caption>"
            b'<graphic xlink:href="f.png"/></fig>'
            b"<table-wrap><label>Table 1</label><caption><p>Tcap</p></caption>"
            b"<table><tr><th>A</th></tr><tr><td>1</td></tr></table></table-wrap>"
            b"</floats-group></article>"
        )
        c = _segment(_parse(xml))
        assert [f.caption for f in c.figures] == ["Figure 1 Cap."]
        assert [(t.label, t.caption) for t in c.tables] == [("1", "Table 1 Tcap")]

    def test_list_inside_p_is_kept(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title><p>Steps <list><list-item><p>First step.</p>"
            b"</list-item><list-item><p>Second step.</p></list-item></list> done.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        texts = [s.text for s in c.sentences]
        assert "First step." in texts
        assert "Second step." in texts

    def test_boxed_text_caption_title_beside_p_is_emitted_once(self):
        # A wrapper's <caption><title> must not double: the heading pass used
        # to emit it and the recursed <caption> emitted it again.
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title>"
            b"<boxed-text><label>Box 1.</label>"
            b"<caption><title>Key points</title></caption>"
            b"<p>Point one.</p></boxed-text><p>After.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        texts = [s.text for s in c.sentences]
        assert texts.count("Key points") == 1
        assert texts == ["Box 1.", "Key points", "Point one.", "After."]

    def test_boxed_text_caption_title_inside_p_is_emitted_once(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title>"
            b"<p>See the box. <boxed-text><caption><title>Practice points</title>"
            b"<p>Sleep more.</p></caption></boxed-text> after.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        texts = [s.text for s in c.sentences]
        assert texts.count("Practice points") == 1

    def test_supplementary_material_caption_title_inside_p_is_emitted_once(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title>"
            b"<p>See <supplementary-material><caption><title>Raw data</title>"
            b"<p>File.</p></caption></supplementary-material> after.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        texts = [s.text for s in c.sentences]
        assert texts.count("Raw data") == 1
        assert "File." in texts

    def test_labelled_list_items_do_not_emit_label_only_sentences(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title>"
            b"<list><list-item><label>1.</label><p>First item.</p></list-item>"
            b"<list-item><label>2.</label><p>Second item.</p></list-item></list>"
            b"<p>After.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [s.text for s in c.sentences] == ["First item.", "Second item.", "After."]

    def test_table_foot_fn_label_is_not_a_standalone_sentence(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>S</title>"
            b"<table-wrap><label>Table 1</label><caption><p>Cap.</p></caption>"
            b"<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
            b"<table-wrap-foot><fn><label>a</label><p>Adjusted for age.</p></fn>"
            b"</table-wrap-foot></table-wrap><p>After.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        texts = [s.text for s in c.sentences]
        assert "a" not in texts
        assert "Adjusted for age." in texts
        assert "After." in texts

    def test_caption_sentence_of_in_paragraph_table_keeps_its_url(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>R</title>"
            b"<p>Shown <table-wrap><label>Table 1</label><caption><p>Rates at "
            b'<ext-link ext-link-type="uri" xlink:href="https://example.org/rates">'
            b"https://example.org/rates</ext-link>.</p></caption>"
            b"<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
            b"</table-wrap> here.</p>"
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert "https://example.org/rates" in [link.url for link in c.links]


# ---------------------------------------------------------------------------
# Back-matter notes / app-group / glossary and body wrappers (input-parsers-4)
# ---------------------------------------------------------------------------

BACK_MATTER_JATS = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>Back matter</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Body.</p>
    <boxed-text><sec><title>Box</title><p>Boxed words here.</p></sec></boxed-text>
    <fig-group><fig><label>Figure 9</label><caption><p>Grouped cap.</p></caption>
      <graphic xlink:href="g.png"/></fig></fig-group>
  </sec></body>
  <back>
    <ack><p>Thanks.</p></ack>
    <notes notes-type="data-availability"><title>Data Availability</title>
      <p>Data are at https://osf.io/abcd.</p></notes>
    <notes notes-type="COI-statement"><title>Competing interests</title>
      <p>None declared.</p></notes>
    <app-group><app id="a1"><title>Appendix A</title><p>Appendix text here.</p></app></app-group>
    <glossary><title>Glossary</title><def-list><def-item><term>Word</term>
      <def><p>Meaning here.</p></def></def-item></def-list></glossary>
    <ref-list><title>References</title>
      <ref><mixed-citation>Doe J. 2020. X.</mixed-citation></ref></ref-list>
  </back>
</article>"""

SEC_WRAPPED_REFLIST_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Wrapped bibliography</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Intro</title><p>Body.</p></sec></body>
  <back><sec><title>References</title>
    <ref-list>
      <ref><mixed-citation>Doe J. 2020. X.</mixed-citation></ref>
    </ref-list>
  </sec></back>
</article>"""


class TestBackMatterSections:
    def test_notes_become_typed_sections_with_their_text(self):
        c = _segment(_parse(BACK_MATTER_JATS))
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Data Availability"].section_type == CanonicalSection.OPEN_DATA
        assert by_header["Competing interests"].section_type == CanonicalSection.COI
        texts = [s.text for s in c.sentences]
        assert "Data are at https://osf.io/abcd." in texts
        assert "None declared." in texts

    def test_untyped_notes_keep_their_text_as_unknown(self):
        xml = BACK_MATTER_JATS.replace(b'notes-type="COI-statement"', b'notes-type="other"')
        c = _segment(_parse(xml))
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Competing interests"].section_type == CanonicalSection.UNKNOWN
        assert "None declared." in [s.text for s in c.sentences]

    def test_app_group_and_glossary_are_kept(self):
        c = _segment(_parse(BACK_MATTER_JATS))
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Appendix A"].section_type == CanonicalSection.APPENDIX
        texts = [s.text for s in c.sentences]
        assert "Appendix text here." in texts
        assert "Meaning here." in texts

    def test_boxed_text_and_fig_group_are_kept(self):
        c = _segment(_parse(BACK_MATTER_JATS))
        texts = [s.text for s in c.sentences]
        assert "Boxed words here." in texts
        assert [f.caption for f in c.figures] == ["Figure 9 Grouped cap."]

    def test_sec_wrapped_ref_list_reuses_its_section(self):
        c = _parse(SEC_WRAPPED_REFLIST_JATS)._contents
        ref_secs = [s for s in c.sections if s.section_type == CanonicalSection.REFERENCES]
        assert len(ref_secs) == 1
        assert ref_secs[0].header == "References"
        # No leftover UNKNOWN twin of the same heading.
        assert [s.header for s in c.sections if s.section_id].count("References") == 1
        assert c.native_ref_strings == ["Doe J. 2020. X."]

    def test_ref_list_in_a_substantive_back_sec_opens_its_own_section(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b'<back><sec sec-type="COI-statement"><title>Competing interests</title>'
            b"<p>None</p><ref-list><title>References</title>"
            b"<ref><mixed-citation>Doe J. 2020. X.</mixed-citation></ref>"
            b"</ref-list></sec></back></article>"
        )
        c = _parse(xml)._contents
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Competing interests"].section_type != CanonicalSection.REFERENCES
        ref_secs = [s for s in c.sections if s.section_type == CanonicalSection.REFERENCES]
        assert len(ref_secs) == 1
        assert ref_secs[0].header == "References"

    def test_notes_without_a_title_get_a_human_header(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b'<back><notes notes-type="data-availability"><p>Data here.</p></notes>'
            b"</back></article>"
        )
        c = _parse(xml)._contents
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Data availability"].section_type == CanonicalSection.OPEN_DATA

    def test_financial_disclosure_notes_are_typed_funding(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b'<back><notes notes-type="financial-disclosure">'
            b"<title>Funding</title><p>Grant X.</p></notes></back></article>"
        )
        c = _parse(xml)._contents
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert by_header["Funding"].section_type == CanonicalSection.FUNDING

    def test_bio_becomes_a_section_with_its_text(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<back><bio><title>Biography</title><p>Jane wrote this.</p></bio></back></article>"
        )
        c = _segment(_parse(xml))
        by_header = {s.header: s for s in c.sections if s.section_id}
        assert "Biography" in by_header
        assert "Jane wrote this." in [s.text for s in c.sentences]


# ---------------------------------------------------------------------------
# Affiliations: IDREFS rids and group-level <aff> (audit input-parsers-6)
# ---------------------------------------------------------------------------

IDREFS_AFF_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Affs</article-title></title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>A</surname><given-names>Ann</given-names></name>
        <xref ref-type="aff" rid="aff1 aff2"/></contrib>
    </contrib-group>
    <aff id="aff1">Uni One</aff><aff id="aff2">Uni Two</aff>
  </article-meta></front>
  <body><sec><title>I</title><p>Body.</p></sec></body>
</article>"""

GROUP_AFF_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Group aff</article-title></title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>A</surname><given-names>Ann</given-names></name></contrib>
      <contrib contrib-type="author">
        <name><surname>B</surname><given-names>Bob</given-names></name></contrib>
      <aff>Department of Psychology, Utrecht University</aff>
    </contrib-group>
  </article-meta></front>
  <body><sec><title>I</title><p>Body.</p></sec></body>
</article>"""


class TestAffiliationFallbacks:
    def test_idrefs_rid_resolves_every_aff(self):
        m = _parse(IDREFS_AFF_JATS)._contents.preparsed_metadata
        assert [(a.family, a.affiliation) for a in m.authors] == [("A", "Uni One; Uni Two")]

    def test_group_level_aff_applies_to_member_contribs(self):
        m = _parse(GROUP_AFF_JATS)._contents.preparsed_metadata
        assert [a.affiliation for a in m.authors] == [
            "Department of Psychology, Utrecht University",
            "Department of Psychology, Utrecht University",
        ]

    def test_single_article_meta_aff_is_a_last_resort(self):
        xml = GROUP_AFF_JATS.replace(
            b"<aff>Department of Psychology, Utrecht University</aff>", b""
        ).replace(
            b"</contrib-group>",
            b"</contrib-group><aff>Lone Institute</aff>",
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert [a.affiliation for a in m.authors] == ["Lone Institute", "Lone Institute"]

    def test_several_article_meta_affs_do_not_attach(self):
        xml = GROUP_AFF_JATS.replace(
            b"<aff>Department of Psychology, Utrecht University</aff>", b""
        ).replace(
            b"</contrib-group>",
            b"</contrib-group><aff>First Institute</aff><aff>Second Institute</aff>",
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert [a.affiliation for a in m.authors] == ["", ""]

    def test_consortium_does_not_inherit_a_person_affiliation(self):
        from bibr.models import ORGANIZATION_ROLE

        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group><contrib-group>"
            b'<contrib contrib-type="author"><name><surname>A</surname>'
            b"<given-names>Ann</given-names></name></contrib>"
            b'<contrib contrib-type="author"><collab>The Consortium</collab></contrib>'
            b"<aff>Uni One</aff></contrib-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body></article>"
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert [(a.family, a.affiliation, a.role) for a in m.authors] == [
            ("A", "Uni One", []),
            ("The Consortium", "", [ORGANIZATION_ROLE]),
        ]

    def test_xref_less_author_inherits_only_unclaimed_group_affs(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group><contrib-group>"
            b'<contrib contrib-type="author"><name><surname>A</surname>'
            b'<given-names>Ann</given-names></name><xref ref-type="aff" rid="a1"/>'
            b"</contrib>"
            b'<contrib contrib-type="author"><name><surname>B</surname>'
            b"<given-names>Bob</given-names></name></contrib>"
            b'<aff id="a1">Institute of Place</aff>'
            b'<aff id="a2">Institute of Elsewhere</aff>'
            b"</contrib-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body></article>"
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert [(a.family, a.affiliation) for a in m.authors] == [
            ("A", "Institute of Place"),
            ("B", "Institute of Elsewhere"),
        ]

    def test_aff_claimed_by_another_group_is_not_inherited(self):
        # An <aff> claimed by an author of a later <contrib-group> must not
        # leak to an xref-less author of an earlier group.
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group><contrib-group>"
            b'<contrib contrib-type="author"><name><surname>Goldacre</surname>'
            b"<given-names>Ben</given-names></name></contrib>"
            b'<aff id="a1">Institute of Elsewhere</aff></contrib-group>'
            b"<contrib-group>"
            b'<contrib contrib-type="author"><name><surname>Other</surname>'
            b'<given-names>Olive</given-names></name><xref ref-type="aff" rid="a1"/>'
            b"</contrib></contrib-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body></article>"
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert [(a.family, a.affiliation) for a in m.authors] == [
            ("Goldacre", ""),
            ("Other", "Institute of Elsewhere"),
        ]

    def test_address_email_and_orcid_reach_the_author(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group><contrib-group>"
            b'<contrib contrib-type="author" corresp="yes">'
            b"<name><surname>Keizer</surname><given-names>Ron</given-names></name>"
            b"<address><email>ron.keizer@example.org</email></address>"
            b'<contrib-id contrib-id-type="orcid">0000-0002-1234-5678</contrib-id>'
            b"</contrib></contrib-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body></article>"
        )
        (author,) = _parse(xml)._contents.preparsed_metadata.authors
        assert author.email == "ron.keizer@example.org"
        assert author.orcid == "https://orcid.org/0000-0002-1234-5678"

    def test_name_alternatives_author_is_kept_with_stable_ids(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group><contrib-group>"
            b'<contrib contrib-type="author"><name><surname>Keizer</surname>'
            b"<given-names>Ron</given-names></name></contrib>"
            b'<contrib contrib-type="author"><name-alternatives>'
            b"<name><surname>Wang</surname><given-names>Wei</given-names></name>"
            b"</name-alternatives></contrib>"
            b'<contrib contrib-type="author"><name><surname>Goldacre</surname>'
            b"<given-names>Ben</given-names></name></contrib>"
            b"</contrib-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body></article>"
        )
        authors = _parse(xml)._contents.preparsed_metadata.authors
        assert [(a.author_id, a.family) for a in authors] == [
            (1, "Keizer"),
            (2, "Wang"),
            (3, "Goldacre"),
        ]


# ---------------------------------------------------------------------------
# ext-link/uri targets (audit input-parsers-10)
# ---------------------------------------------------------------------------


class TestBodyLinks:
    def test_ext_link_target_is_recorded_with_its_display_text(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>All data and code are available on the "
            b'<ext-link ext-link-type="uri" xlink:href="https://osf.io/x7k2q/">'
            b"Open Science Framework</ext-link>.</p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [(link.url, link.link_text) for link in c.links] == [
            ("https://osf.io/x7k2q/", "Open Science Framework")
        ]

    def test_uri_target_is_recorded(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b'<p>See <uri xlink:href="https://example.org/data">the dataset</uri>.</p>'
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [(link.url, link.link_text) for link in c.links] == [
            ("https://example.org/data", "the dataset")
        ]

    def test_bare_url_still_links_by_regex_without_display_text(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Data are at https://osf.io/abcd.</p></sec>"
            b"</body></article>"
        )
        c = _segment(_parse(xml))
        assert [(link.url, link.link_text) for link in c.links] == [("https://osf.io/abcd", None)]

    def test_link_shown_as_its_own_url_is_recorded_once(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>Data are at "
            b'<ext-link ext-link-type="uri" xlink:href="https://osf.io/abcd">'
            b"https://osf.io/abcd</ext-link>.</p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [(link.url, link.text_id) for link in c.links] == [("https://osf.io/abcd", 1)]

    def test_url_in_first_sentence_is_recorded_once_at_the_right_sentence(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>Data are at "
            b'<ext-link ext-link-type="uri" xlink:href="https://github.com/x/y">'
            b"https://github.com/x/y</ext-link>. Second sentence follows here.</p>"
            b"</sec></body></article>"
        )
        p = _parse(xml)
        c = p._contents
        p.apply_segmentation(
            c,
            [["Data are at https://github.com/x/y.", "Second sentence follows here."]],
        )
        assert [(link.url, link.text_id) for link in c.links] == [("https://github.com/x/y", 1)]

    def test_named_link_resolves_to_the_sentence_holding_its_text(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>All data are on the "
            b'<ext-link ext-link-type="uri" xlink:href="https://osf.io/x7k2q/">'
            b"Open Science Framework</ext-link>. Second sentence follows here.</p>"
            b"</sec></body></article>"
        )
        p = _parse(xml)
        c = p._contents
        p.apply_segmentation(
            c,
            [["All data are on the Open Science Framework.", "Second sentence follows here."]],
        )
        assert [(link.url, link.text_id) for link in c.links] == [("https://osf.io/x7k2q/", 1)]

    def test_short_anchor_text_resolves_to_the_sentence_holding_it(self):
        # 'here' is a substring of 'There': only the anchor's offset tells the
        # first-sentence mention apart from the second-sentence anchor.
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>There are more results here and there. Download "
            b'<ext-link ext-link-type="uri" xlink:href="https://x.example.org/d">'
            b"here</ext-link>.</p>"
            b"</sec></body></article>"
        )
        p = _parse(xml)
        c = p._contents
        p.apply_segmentation(
            c,
            [["There are more results here and there.", "Download here."]],
        )
        assert [(link.url, link.text_id) for link in c.links] == [("https://x.example.org/d", 2)]

    def test_bare_doi_href_becomes_a_doi_org_url(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>See "
            b'<ext-link ext-link-type="doi" xlink:href="10.1016/j.cell.2010.01.001">'
            b"the paper</ext-link>.</p></sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [(link.url, link.link_text) for link in c.links] == [
            ("https://doi.org/10.1016/j.cell.2010.01.001", "the paper")
        ]

    def test_accession_href_is_not_exported_as_a_url(self):
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front><body><sec><title>I</title>"
            b"<p>Gene "
            b'<ext-link ext-link-type="gen" xlink:href="EU598807">EU598807</ext-link>.</p>'
            b"</sec></body></article>"
        )
        c = _segment(_parse(xml))
        assert [link.url for link in c.links] == []


# ---------------------------------------------------------------------------
# Reference row text (audit input-parsers-11)
# ---------------------------------------------------------------------------

CITATION_ALTERNATIVES_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Rows</article-title></title-group>
  </article-meta></front>
  <body><sec><title>I</title><p>Body.</p></sec></body>
  <back><ref-list><title>References</title>
    <ref id="r1"><label>1</label><citation-alternatives>
      <element-citation publication-type="journal">
        <person-group person-group-type="author">
          <name><surname>Smith</surname><given-names>J</given-names></name></person-group>
        <article-title>Sleep and memory</article-title><source>Psychol Sci</source>
        <year>2020</year><volume>31</volume><fpage>1</fpage><lpage>9</lpage>
      </element-citation>
      <mixed-citation>Smith J. Sleep and memory. Psychol Sci. 2020;31:1-9.</mixed-citation>
    </citation-alternatives></ref>
  </ref-list></back>
</article>"""

ELEMENT_ONLY_REF_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Rows</article-title></title-group>
  </article-meta></front>
  <body><sec><title>I</title><p>Body.</p></sec></body>
  <back><ref-list><title>References</title>
    <ref id="r1"><element-citation publication-type="journal">
      <person-group person-group-type="author">
        <name><surname>Smith</surname><given-names>J</given-names></name></person-group>
      <article-title>A title</article-title><source>J Psych</source>
      <year>2020</year><volume>31</volume>
    </element-citation></ref>
  </ref-list></back>
</article>"""


class TestReferenceRows:
    def test_row_prefers_mixed_citation_over_fused_fields(self):
        p = _parse(CITATION_ALTERNATIVES_JATS)
        rows = [e.text for e in p.assembler.entries if not e.needs_segmentation]
        assert rows == ["1 Smith J. Sleep and memory. Psychol Sci. 2020;31:1-9."]

    def test_element_only_row_is_separated_not_fused(self):
        p = _parse(ELEMENT_ONLY_REF_JATS)
        rows = [e.text for e in p.assembler.entries if not e.needs_segmentation]
        assert rows == ["Smith J A title J Psych 2020 31"]

    def test_mixed_only_rows_keep_their_exact_text(self):
        c = _parse(MIXED_CITATION_JATS)._contents
        assert c.native_ref_strings == [
            "Alpha, A. (2010). First. Journal A, 1, 1-2.",
            "Beta, B. (2011). Second. Journal B, 2, 3-4.",
        ]

    def test_element_row_keeps_comment_access_notes_and_urls(self):
        # PLOS-style element-citation: the URL lives in a <comment> the
        # structured parse does not model. The row must keep it (and the
        # link it yields), only separated instead of fused.
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<back><ref-list><title>References</title>"
            b'<ref id="r2"><label>2</label>'
            b'<element-citation publication-type="book">'
            b"<collab>World Health Organization</collab><year>2006</year>"
            b"<article-title>Weighted average prices.</article-title>"
            b"<comment>Available: "
            b'<ext-link ext-link-type="uri" xlink:href="http://www.who.int/entity/x.pdf">'
            b"http://www.who.int/entity/x.pdf</ext-link>. Accessed 2006</comment>"
            b"</element-citation></ref></ref-list></back></article>"
        )
        p = _parse(xml)
        rows = [e.text for e in p.assembler.entries if not e.needs_segmentation]
        assert rows == [
            "2 World Health Organization 2006 Weighted average prices. "
            "Available: http://www.who.int/entity/x.pdf. Accessed 2006"
        ]
        c = _segment(p)
        assert [(link.url, link.link_text) for link in c.links] == [
            ("http://www.who.int/entity/x.pdf", None)
        ]

    def test_ref_note_beside_mixed_citation_is_kept_with_its_urls(self):
        # ACS-style <ref>: the access note (with its URLs) sits beside the
        # mixed-citation, not inside it.
        xml = (
            b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b"<front><article-meta><title-group><article-title>T</article-title>"
            b"</title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<back><ref-list><title>References</title>"
            b'<ref id="r39"><label>39</label>'
            b"<mixed-citation>Roy, R. (1993). Program. Physica B, 192, 55.</mixed-citation>"
            b"<note><p>Remarks: the program is at "
            b'<uri xlink:href="http://www.ill.eu/sites/fullprof/">'
            b"http://www.ill.eu/sites/fullprof/</uri></p></note>"
            b"</ref></ref-list></back></article>"
        )
        p = _parse(xml)
        rows = [e.text for e in p.assembler.entries if not e.needs_segmentation]
        assert rows == [
            "39 Roy, R. (1993). Program. Physica B, 192, 55. "
            "Remarks: the program is at http://www.ill.eu/sites/fullprof/"
        ]
        c = _segment(p)
        assert [link.url for link in c.links] == ["http://www.ill.eu/sites/fullprof/"]

    def test_ref_with_two_mixed_citations_keeps_both(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<back><ref-list><title>References</title>"
            b'<ref id="r62"><label>62</label>'
            b"<mixed-citation>Liu X. Tetrahedron 2006, 62, 11039.</mixed-citation>"
            b"<mixed-citation>Ludley P. Tetrahedron 2006, 62, 11043.</mixed-citation>"
            b"</ref></ref-list></back></article>"
        )
        c = _parse(xml)._contents
        assert c.native_ref_strings == [
            "62 Liu X. Tetrahedron 2006, 62, 11039. Ludley P. Tetrahedron 2006, 62, 11043."
        ]

    def test_ref_note_stays_between_its_citations_in_document_order(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<back><ref-list><title>References</title>"
            b'<ref id="r62"><label>62</label>'
            b"<mixed-citation>Liu X. Tetrahedron 2006, 62, 11039.</mixed-citation>"
            b"<note>See also:</note>"
            b"<mixed-citation>Ludley P. Tetrahedron 2006, 62, 11043.</mixed-citation>"
            b"</ref></ref-list></back></article>"
        )
        c = _parse(xml)._contents
        assert c.native_ref_strings == [
            "62 Liu X. Tetrahedron 2006, 62, 11039. See also: "
            "Ludley P. Tetrahedron 2006, 62, 11043."
        ]

    def test_nested_ref_list_inside_a_back_sec_is_not_duplicated(self):
        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group></article-meta></front>"
            b"<body><sec><title>I</title><p>Body.</p></sec></body>"
            b"<back><sec><title>References</title><ref-list>"
            b"<ref><mixed-citation>Doe J. 2020. X.</mixed-citation></ref>"
            b"<ref-list><ref><mixed-citation>Roe J. 2021. Y.</mixed-citation></ref></ref-list>"
            b"</ref-list></sec></back></article>"
        )
        c = _parse(xml)._contents
        assert c.native_ref_strings == ["Doe J. 2020. X.", "Roe J. 2021. Y."]


# ---------------------------------------------------------------------------
# Consortium with nested members (audit input-parsers-12)
# ---------------------------------------------------------------------------

CONSORTIUM_JATS = b"""<?xml version="1.0"?>
<article>
  <front><article-meta>
    <title-group><article-title>Consortium</article-title></title-group>
    <contrib-group>
      <contrib contrib-type="author">
        <name><surname>Alpha</surname><given-names>Ann</given-names></name>
      </contrib>
      <contrib contrib-type="author">
        <collab>The Big Consortium<contrib-group><contrib>
          <name><surname>Member</surname><given-names>M</given-names></name>
        </contrib></contrib-group></collab>
      </contrib>
    </contrib-group>
  </article-meta></front>
  <body><sec><title>I</title><p>Body.</p></sec></body>
</article>"""


class TestConsortiumMembers:
    def test_consortium_appears_once_and_members_are_kept_once(self):
        from bibr.models import ORGANIZATION_ROLE

        m = _parse(CONSORTIUM_JATS)._contents.preparsed_metadata
        assert [(a.author_id, a.given, a.family, a.role) for a in m.authors] == [
            (1, "Ann", "Alpha", []),
            (2, "", "The Big Consortium", [ORGANIZATION_ROLE]),
            (3, "M", "Member", []),
        ]

    def test_nested_member_keeps_its_own_affiliation(self):
        from bibr.models import ORGANIZATION_ROLE

        xml = (
            b'<?xml version="1.0"?><article><front><article-meta><title-group>'
            b"<article-title>T</article-title></title-group><contrib-group>"
            b'<contrib contrib-type="author"><name><surname>A</surname>'
            b"<given-names>Ann</given-names></name></contrib>"
            b'<contrib contrib-type="author"><collab>The Consortium<contrib-group>'
            b'<contrib contrib-type="author"><name><surname>Member</surname>'
            b'<given-names>M</given-names></name><xref ref-type="aff" rid="mem1"/>'
            b"</contrib></contrib-group></collab></contrib>"
            b"<aff>Outer Uni</aff></contrib-group>"
            b'<aff id="mem1">Member Lab</aff></article-meta></front>'
            b"<body><sec><title>I</title><p>Body.</p></sec></body></article>"
        )
        m = _parse(xml)._contents.preparsed_metadata
        assert [(a.family, a.affiliation, a.role) for a in m.authors] == [
            ("A", "Outer Uni", []),
            ("The Consortium", "", [ORGANIZATION_ROLE]),
            ("Member", "Member Lab", []),
        ]
