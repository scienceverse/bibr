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
    ids=["index", "statistic", "identifier", "subscript", "text", "variables", "arguments"],
)
def test_whitespace_between_mathml_elements_is_dropped(math, expected):
    """Kept as text it split "2.1" into "2 . 1"; the late clean-up used to
    fuse some of those runs back, until it stopped touching document text."""
    assert _inline_text(math) == f"We used {expected} here."
    assert _inline_text(re.sub(r">\s+<", "><", math)) == f"We used {expected} here."


@pytest.mark.parametrize(
    ("math", "expected"),
    MATHML_WORDS,
    ids=["function-names", "unit-mtext", "unit-newline", "spelled-words", "text-after-comma"],
)
def test_whitespace_between_mathml_elements_keeps_words_apart(math, expected):
    assert _inline_text(math) == f"We used {expected} here."


def test_mathml_whitespace_next_to_prose_keeps_the_words_apart():
    xml = _mathml_jats(
        'the value<inline-formula><mml:math display="inline"> <mml:mi>x</mml:mi> '
        "</mml:math></inline-formula>is small, <inline-formula><mml:math>"
        "<mml:mi>y</mml:mi> </mml:math></inline-formula>, too."
    )
    (entry,) = _parse(xml).assembler.entries
    assert entry.text == "the value x is small, y, too."
