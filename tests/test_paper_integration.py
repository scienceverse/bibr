from unittest.mock import MagicMock

import pytest

from bibr.paper import BibAuthor, Paper, PaperMetadata, ProcessingStatus


@pytest.fixture
def mock_contents():
    contents = MagicMock()
    import pandas as pd

    from bibr.paper_contents import (
        CanonicalSection,
        PaperFigure,
        PaperSection,
        PaperSentence,
        PaperTable,
        PaperURLLink,
        PaperXref,
    )

    contents.sentences = [
        PaperSentence(
            text_id=1, text="Test sentence", section_id=0, paragraph_id=1, page_number=None
        ),
        # Footnote text now lives in the text table under a footnote section
        PaperSentence(
            text_id=2, text="A footnote.", section_id=2, paragraph_id=2, page_number=None
        ),
    ]
    contents.text_df = pd.DataFrame(
        {
            "text_id": [1, 2],
            "section_id": [0, 2],
            "paragraph_id": [1, 2],
            "text": ["Test sentence", "A footnote."],
            "page_number": [None, None],
        }
    )
    contents.links = [
        PaperURLLink(url="https://example.com", section_id=0, paragraph_id=1, text_id=1),
    ]
    contents.sentences_df = pd.DataFrame(
        {
            "text_id": [1],
            "section_id": [0],
            "paragraph_id": [1],
            "text": ["Test sentence"],
            "section_name": ["Title"],
        }
    )
    contents.tables = [
        PaperTable(
            table_id=1,
            df=pd.DataFrame({"col1": [1], "col2": [2]}),
            tbl_html="<table><tr><td>col1</td><td>col2</td></tr><tr><td>1</td><td>2</td></tr></table>",
            section_id=0,
            caption="Table 1. Summary of results.",
        )
    ]
    contents.sections = [
        PaperSection(section_id=1, header="Title", level=1, parent_section_id=None),
        PaperSection(
            section_id=2,
            header="Footnote 1",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.FOOTNOTE,
        ),
    ]
    contents.sections_text = {1: "Test section text"}
    contents.figures = [
        PaperFigure(
            figure_id=1,
            section_id=0,
            image_b64=None,
            caption="Figure 1. Sample plot.",
        )
    ]
    # Footnote xref: links footnote section (section_id=2) to the referencing sentence
    contents.xrefs = [
        PaperXref(xref_id=2, xref_type="foot", contents="1", text_id=1),
    ]
    contents.equations = []
    contents.studies = []
    return contents


@pytest.fixture
def mock_metadata():
    return PaperMetadata(doi="10.1234/test", title="Test Paper", authors=[], references=[])


def test_paper_export_to_json(mock_contents, mock_metadata):
    from bibr.input.file import InputFile

    inp_file = InputFile(path="test.pdf")
    inp_file.file_hash = "hash123"
    inp_file.sha256 = "0123456789abcdef" * 4
    inp_file.file_name = "test.pdf"
    inp_file.input_format = "pdf"

    paper = Paper(
        input_file=inp_file,
        contents=mock_contents,
        metadata=mock_metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()
    assert result is not None

    # Check paper_id: the input file's stem, not the DOI
    assert result["paper_id"] == "test"

    # Check metadata / source / root version
    assert result["schema_version"] == "12.0"
    metadata = result["metadata"]
    assert metadata["title"] == "Test Paper"
    assert metadata["doi"] == "10.1234/test"
    assert result["source"]["sha256"] == "0123456789abcdef" * 4
    assert result["source"]["file_name"] == "test.pdf"
    assert metadata["paper_type"] is None
    assert metadata["oecd_l1"] is None
    # v12: classifier confidences are processing facts, not paper metadata.
    assert "paper_type_confidence" not in metadata
    assert "oecd_confidence" not in metadata

    # Check text — footnote text is now in the text table
    text_list = result["text"]
    assert len(text_list) == 2
    assert text_list[0]["text"] == "Test sentence"
    assert text_list[0]["section_id"] is None  # Root section_id=0 remapped to null
    assert text_list[0]["formatted"] is None  # no display math
    assert text_list[1]["text"] == "A footnote."
    assert text_list[1]["section_id"] == 2  # footnote section

    # Check url (inner url→href)
    url_list = result["url"]
    assert len(url_list) == 1
    assert url_list[0]["href"] == "https://example.com"

    # Check section (should exclude Root section_id=0)
    section_list = result["section"]
    assert len(section_list) == 2
    assert section_list[0]["header"] == "Title"
    assert section_list[1]["header"] == "Footnote 1"
    assert section_list[1]["section_type"] == "footnote"

    # Check author (empty)
    assert len(result["author"]) == 0

    # Check bib (empty)
    assert len(result["bib"]) == 0

    # Check xref — footnote xref
    xref_list = result["xref"]
    assert len(xref_list) == 1
    assert xref_list[0]["xref_type"] == "foot"
    assert xref_list[0]["target_id"] == 2
    assert xref_list[0]["text_id"] == 1

    # No footnote column in v8.0
    assert "footnote" not in result

    # Check fig — caption round-trips into the export
    figure_list = result["figure"]
    assert len(figure_list) == 1
    assert figure_list[0]["figure_id"] == 1
    assert figure_list[0]["caption"] == "Figure 1. Sample plot."

    # Check table — caption round-trips into the export
    table_list = result["table"]
    assert len(table_list) == 1
    assert table_list[0]["table_id"] == 1
    assert table_list[0]["contents"] is not None
    assert table_list[0]["contents"][0] == ["col1", "col2"]  # headers
    assert table_list[0]["contents"][1] == ["1", "2"]  # data row
    assert table_list[0]["section_id"] is None  # Root remapped
    assert table_list[0]["page_number"] is None
    assert table_list[0]["caption"] == "Table 1. Summary of results."

    # Check eq (empty)
    assert len(result["eq"]) == 0


def test_paper_export_figure_table_caption_absent(mock_metadata):
    """Figures/tables without a matched caption export the field as null."""
    import pandas as pd

    from bibr.input.file import InputFile
    from bibr.paper_contents import PaperFigure, PaperSection, PaperSentence, PaperTable

    contents = MagicMock()
    contents.sentences = [
        PaperSentence(
            text_id=1, text="Only sentence", section_id=0, paragraph_id=1, page_number=None
        ),
    ]
    contents.text_df = pd.DataFrame(
        {
            "text_id": [1],
            "section_id": [0],
            "paragraph_id": [1],
            "text": ["Only sentence"],
            "page_number": [None],
        }
    )
    contents.links_df = pd.DataFrame()
    contents.links = []
    contents.tables = [
        PaperTable(
            table_id=1,
            df=pd.DataFrame({"col1": [1]}),
            tbl_html="<table><tr><td>col1</td></tr><tr><td>1</td></tr></table>",
            section_id=0,
        )
    ]
    contents.figures = [
        PaperFigure(figure_id=1, section_id=0, image_b64=None, caption=None),
    ]
    contents.sections = [
        PaperSection(section_id=0, header="Intro", level=1, parent_section_id=None)
    ]
    contents.sections_text = {0: "Intro text"}
    contents.xrefs = []
    contents.equations = []
    contents.studies = []

    inp_file = InputFile(path="no_caption.pdf")
    inp_file.file_hash = "nocaptionh"
    inp_file.file_name = "no_caption.pdf"
    inp_file.input_format = "pdf"

    paper = Paper(
        input_file=inp_file,
        contents=contents,
        metadata=mock_metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()

    assert result["table"][0]["caption"] is None
    assert result["figure"][0]["caption"] is None


def test_paper_export_no_contents_raises_error():
    from bibr.input.file import InputFile

    inp = InputFile("f")
    inp.file_hash = "h"
    inp.file_name = "f"
    inp.input_format = "x"

    paper = Paper(
        input_file=inp,
        contents=None,
    )
    with pytest.raises(ValueError, match="Paper has no contents"):
        paper.export_to_json()


def test_paper_export_empty_tables(mock_metadata):
    """Test export when there are no tables, links, authors, refs, or citations."""
    import pandas as pd

    from bibr.input.file import InputFile
    from bibr.paper_contents import PaperSection, PaperSentence

    contents = MagicMock()
    contents.sentences = [
        PaperSentence(
            text_id=1, text="Only sentence", section_id=0, paragraph_id=1, page_number=None
        ),
    ]
    contents.text_df = pd.DataFrame(
        {
            "text_id": [1],
            "section_id": [0],
            "paragraph_id": [1],
            "text": ["Only sentence"],
            "page_number": [None],
        }
    )
    contents.links_df = pd.DataFrame()
    contents.links = []
    contents.tables = []
    contents.sections = [
        PaperSection(section_id=0, header="Intro", level=1, parent_section_id=None)
    ]
    contents.sections_text = {0: "Intro text"}
    contents.figures = []
    contents.xrefs = []
    contents.equations = []
    contents.studies = []

    inp_file = InputFile(path="empty_tables.pdf")
    inp_file.file_hash = "emptyh"
    inp_file.file_name = "empty_tables.pdf"
    inp_file.input_format = "pdf"

    paper = Paper(
        input_file=inp_file,
        contents=contents,
        metadata=mock_metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()

    assert len(result["table"]) == 0
    assert len(result["author"]) == 0
    assert len(result["bib"]) == 0
    assert len(result["text"]) == 1
    assert len(result["figure"]) == 0
    assert "footnote" not in result


def test_paper_export_bib_with_populated_references(mock_contents):
    """Test bib export with populated references and top-level bib_match (v10.1)."""
    from bibr.input.file import InputFile
    from bibr.paper import ExternalMatch, MatchSource, PaperReference

    metadata = PaperMetadata(
        doi="10.1234/test",
        title="Test Paper",
        references=[
            PaperReference(
                bib_id=1,
                title="Referenced Paper",
                first_page="100",
                volume="42",
                authors="Smith, J.",
                year=2020,
                container="Nature",
                doi="10.1000/ref1",
                bib_type="journal_article",
                last_page="115",
                issue="3",
                publisher="Nature Publishing",
                url="https://doi.org/10.1000/ref1",
                match={
                    MatchSource.CROSSREF: ExternalMatch(
                        id="10.1000/ref1",
                        score=100.0,
                        title="Referenced Paper",
                        authors=[BibAuthor(given="J.", family="Smith")],
                        year=2020,
                        container="Nature",
                        volume="42",
                        issue="3",
                        first_page="100",
                        last_page="115",
                        publisher="Nature Publishing",
                        doi="10.1000/ref1",
                        bib_type="journal_article",
                        url="https://doi.org/10.1000/ref1",
                    ),
                },
            )
        ],
    )

    inp_file = InputFile(path="test.pdf")
    inp_file.file_hash = "hash123"
    inp_file.file_name = "test.pdf"
    inp_file.input_format = "pdf"

    paper = Paper(
        input_file=inp_file,
        contents=mock_contents,
        metadata=metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()

    # Verify bib (flat, no nested match)
    bib_list = result["bib"]
    assert len(bib_list) == 1
    row = bib_list[0]
    assert row["title"] == "Referenced Paper"
    assert row["last_page"] == "115"
    assert row["issue"] == "3"
    assert row["publisher"] == "Nature Publishing"
    assert row["container"] == "Nature"
    assert row["url"] == "https://doi.org/10.1000/ref1"
    assert row["bib_type"] == "journal_article"
    assert row["year"] == 2020
    assert row["authors"] == "Smith, J."

    # Verify top-level bib_match
    bib_match_list = result["bib_match"]
    assert len(bib_match_list) == 1
    cr = bib_match_list[0]
    assert cr["bib_id"] == 1
    assert cr["service"] == "crossref"
    assert cr["service_id"] == "10.1000/ref1"
    # Enrichment's 0-100 score is published on the 0-1 scale.
    assert cr["score"] == 1.0
    assert cr["title"] == "Referenced Paper"
    assert cr["container"] == "Nature"
    assert cr["doi"] == "10.1000/ref1"
    assert cr["bib_type"] == "journal_article"

    # Verify version
    assert result["schema_version"] == "12.0"


def test_paper_export_bib_without_matches(mock_contents):
    """Test bib export with references but no enrichment matches."""
    from bibr.input.file import InputFile
    from bibr.paper import PaperReference

    metadata = PaperMetadata(
        doi="10.1234/test",
        title="Test Paper",
        references=[
            PaperReference(
                bib_id=1,
                title="Some Reference",
                first_page=None,
                volume=None,
                authors="Doe, J.",
                year=2021,
                container=None,
            )
        ],
    )

    inp_file = InputFile(path="test.pdf")
    inp_file.file_hash = "hash123"
    inp_file.file_name = "test.pdf"
    inp_file.input_format = "pdf"

    paper = Paper(
        input_file=inp_file,
        contents=mock_contents,
        metadata=metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()

    bib_list = result["bib"]
    assert len(bib_list) == 1
    # v10.1: no nested match key; bib_match is top-level and empty here
    assert "match" not in bib_list[0]
    assert result["bib_match"] == []


def test_paper_export_display_math_formatted():
    """Test that display-math formulas are replaced with [equation] and raw LaTeX is preserved."""
    from unittest.mock import MagicMock

    from bibr.input.file import InputFile
    from bibr.paper_contents import PaperSection, PaperSentence

    contents = MagicMock()
    contents.sentences = [
        PaperSentence(text_id=1, text="Normal sentence.", section_id=1, paragraph_id=1),
        PaperSentence(
            text_id=2,
            text="\\alpha = 2",
            section_id=1,
            paragraph_id=1,
            is_display_formula=True,
        ),
    ]
    contents.sections = [
        PaperSection(section_id=1, header="Methods", level=1, parent_section_id=None),
    ]
    contents.links = []
    contents.tables = []
    contents.figures = []
    contents.xrefs = []
    contents.equations = []
    contents.studies = []

    inp_file = InputFile(path="math.pdf")
    inp_file.file_hash = "mathh"
    inp_file.file_name = "math.pdf"
    inp_file.input_format = "pdf"

    metadata = PaperMetadata(doi="10.1234/math", title="Math Paper")
    paper = Paper(
        input_file=inp_file,
        contents=contents,
        metadata=metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()
    text_list = result["text"]

    assert text_list[0]["text"] == "Normal sentence."
    assert text_list[0]["formatted"] is None
    assert text_list[1]["text"] == "[equation]"
    assert text_list[1]["formatted"] == "\\alpha = 2"


def test_paper_export_xref_split_fks():
    """Test that xrefs use explicit FK columns — footnote_id replaced by section_id."""
    from unittest.mock import MagicMock

    from bibr.input.file import InputFile
    from bibr.paper_contents import PaperSection, PaperSentence, PaperXref

    contents = MagicMock()
    contents.sentences = [
        PaperSentence(text_id=1, text="See [1] and Table 2.", section_id=1, paragraph_id=1),
    ]
    contents.sections = [
        PaperSection(section_id=1, header="Results", level=1, parent_section_id=None),
    ]
    contents.links = []
    contents.tables = []
    contents.figures = []
    contents.xrefs = [
        PaperXref(xref_id=1, xref_type="bib", contents="[1]", text_id=1),
        PaperXref(xref_id=2, xref_type="table", contents="Table 2", text_id=1),
    ]
    contents.equations = []
    contents.studies = []

    inp_file = InputFile(path="xref.pdf")
    inp_file.file_hash = "xrefh"
    inp_file.file_name = "xref.pdf"
    inp_file.input_format = "pdf"

    metadata = PaperMetadata(doi="10.1234/xref", title="Xref Paper")
    paper = Paper(
        input_file=inp_file,
        contents=contents,
        metadata=metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()
    xref_list = result["xref"]

    assert len(xref_list) == 2

    # bib xref
    bib_xref = xref_list[0]
    assert bib_xref["xref_type"] == "bib"
    assert bib_xref["target_id"] == 1

    # table xref
    tbl_xref = xref_list[1]
    assert tbl_xref["xref_type"] == "table"
    assert tbl_xref["target_id"] == 2


def test_paper_export_orcid_canonicalization():
    """Test that ORCIDs are exported in canonical URI form."""
    from unittest.mock import MagicMock

    from bibr.input.file import InputFile
    from bibr.paper import PaperAuthor
    from bibr.paper_contents import PaperSection, PaperSentence

    contents = MagicMock()
    contents.sentences = [
        PaperSentence(text_id=1, text="Test.", section_id=1, paragraph_id=1),
    ]
    contents.sections = [
        PaperSection(section_id=1, header="Intro", level=1, parent_section_id=None),
    ]
    contents.links = []
    contents.tables = []
    contents.figures = []
    contents.xrefs = []
    contents.equations = []
    contents.studies = []

    metadata = PaperMetadata(
        doi="10.1234/orcid",
        title="ORCID Paper",
        authors=[
            PaperAuthor(
                author_id=1,
                given="Jane",
                family="Doe",
                affiliation="MIT",
                orcid="0000-0001-2345-6789",  # bare ID
            ),
            PaperAuthor(
                author_id=2,
                given="John",
                family="Smith",
                affiliation="",
                orcid="https://orcid.org/0000-0002-3456-789X",  # full URI
            ),
        ],
    )

    inp_file = InputFile(path="orcid.pdf")
    inp_file.file_hash = "orcidh"
    inp_file.file_name = "orcid.pdf"
    inp_file.input_format = "pdf"

    paper = Paper(
        input_file=inp_file,
        contents=contents,
        metadata=metadata,
        processing_status=ProcessingStatus(parsed=True),
    )

    result = paper.export_to_json()
    author_list = result["author"]

    # Both should be in canonical URI form
    assert author_list[0]["orcid"] == "https://orcid.org/0000-0001-2345-6789"
    assert author_list[1]["orcid"] == "https://orcid.org/0000-0002-3456-789X"
    # Affiliations live only in the affiliation table; an empty byline string
    # contributes no row.
    assert "affiliation" not in author_list[1]
    assert all(2 not in row["author_ids"] for row in result["affiliation"])


class TestEnforceImradOrder:
    """Tests for enforce_imrad_order — resets duplicate unique-type sections to UNKNOWN."""

    def test_typical_pdf_structure(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Paper Title", 1, None, CanonicalSection.UNKNOWN, 0.0),
            PaperSection(1, "Abstract", 2, None, CanonicalSection.ABSTRACT, 0.95),
            PaperSection(2, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 0.99),
            PaperSection(3, "Methods", 2, None, CanonicalSection.METHODS, 0.98),
            PaperSection(4, "Results", 2, None, CanonicalSection.RESULTS, 0.97),
            PaperSection(5, "Discussion", 2, None, CanonicalSection.DISCUSSION, 0.96),
            PaperSection(6, "References", 2, None, CanonicalSection.REFERENCES, 0.99),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.UNKNOWN
        assert sections[1].section_type == CanonicalSection.ABSTRACT
        assert sections[2].section_type == CanonicalSection.INTRODUCTION
        assert sections[3].section_type == CanonicalSection.METHODS
        assert sections[4].section_type == CanonicalSection.RESULTS
        assert sections[5].section_type == CanonicalSection.DISCUSSION
        assert sections[6].section_type == CanonicalSection.REFERENCES

    def test_subsection_after_discussion_preserved(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 0.99),
            PaperSection(1, "Methods", 2, None, CanonicalSection.METHODS, 0.98),
            PaperSection(2, "Results", 2, None, CanonicalSection.RESULTS, 0.97),
            PaperSection(3, "Discussion", 2, None, CanonicalSection.DISCUSSION, 0.96),
            PaperSection(4, "Results of Our Analysis", 3, 3, CanonicalSection.RESULTS, 0.70),
            PaperSection(5, "References", 2, None, CanonicalSection.REFERENCES, 0.99),
        ]
        enforce_imrad_order(sections)

        assert sections[4].section_type == CanonicalSection.RESULTS
        assert sections[4].classification_score == 0.70
        assert sections[0].section_type == CanonicalSection.INTRODUCTION
        assert sections[3].section_type == CanonicalSection.DISCUSSION
        assert sections[5].section_type == CanonicalSection.REFERENCES

    def test_duplicate_methods_preserved(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 0.99),
            PaperSection(1, "Methods", 2, None, CanonicalSection.METHODS, 0.98),
            PaperSection(2, "Data Collection", 3, 1, CanonicalSection.METHODS, 0.65),
            PaperSection(3, "Results", 2, None, CanonicalSection.RESULTS, 0.97),
        ]
        enforce_imrad_order(sections)

        assert sections[1].section_type == CanonicalSection.METHODS
        assert sections[2].section_type == CanonicalSection.METHODS
        assert sections[3].section_type == CanonicalSection.RESULTS

    def test_all_unknown_unchanged(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Part One", 1, None, CanonicalSection.UNKNOWN, 0.0),
            PaperSection(1, "Part Two", 1, None, CanonicalSection.UNKNOWN, 0.0),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.UNKNOWN
        assert sections[1].section_type == CanonicalSection.UNKNOWN

    def test_duplicate_unique_type_keeps_highest_trust_source(self):
        # A corrupted-header hint section can claim ABSTRACT first in document
        # order; the later clean exact-alias heading must win the dedup.
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(
                1,
                "Ab stract",
                2,
                None,
                CanonicalSection.ABSTRACT,
                0.95,
                classification_source="substring_alias",
            ),
            PaperSection(
                2,
                "Abstract",
                1,
                None,
                CanonicalSection.ABSTRACT,
                1.0,
                classification_source="exact_alias",
            ),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.UNKNOWN
        assert sections[0].classification_score == 0.0
        assert sections[0].classification_source == "imrad_dedup"
        assert sections[1].section_type == CanonicalSection.ABSTRACT
        assert sections[1].classification_source == "exact_alias"

    def test_duplicate_dedup_stamps_provenance_on_tie(self):
        # Equal trust: document order breaks the tie (first wins), and the
        # demoted section records why it lost its type.
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(
                1,
                "References",
                1,
                None,
                CanonicalSection.REFERENCES,
                1.0,
                classification_source="exact_alias",
            ),
            PaperSection(
                2,
                "Bibliography",
                1,
                None,
                CanonicalSection.REFERENCES,
                1.0,
                classification_source="exact_alias",
            ),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.REFERENCES
        assert sections[1].section_type == CanonicalSection.UNKNOWN
        assert sections[1].classification_source == "imrad_dedup"

    def test_flat_h1_structure(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Introduction", 1, None, CanonicalSection.INTRODUCTION, 0.99),
            PaperSection(1, "Methods", 1, None, CanonicalSection.METHODS, 0.98),
            PaperSection(2, "Results", 1, None, CanonicalSection.RESULTS, 0.97),
            PaperSection(3, "Discussion", 1, None, CanonicalSection.DISCUSSION, 0.96),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.INTRODUCTION
        assert sections[1].section_type == CanonicalSection.METHODS
        assert sections[2].section_type == CanonicalSection.RESULTS
        assert sections[3].section_type == CanonicalSection.DISCUSSION

    def test_acknowledgment_before_references(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Discussion", 2, None, CanonicalSection.DISCUSSION, 0.96),
            PaperSection(1, "Acknowledgments", 2, None, CanonicalSection.ACKNOWLEDGMENT, 0.85),
            PaperSection(2, "Funding", 2, None, CanonicalSection.FUNDING, 0.90),
            PaperSection(3, "References", 2, None, CanonicalSection.REFERENCES, 0.99),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.DISCUSSION
        assert sections[1].section_type == CanonicalSection.ACKNOWLEDGMENT
        assert sections[2].section_type == CanonicalSection.FUNDING
        assert sections[3].section_type == CanonicalSection.REFERENCES

    def test_duplicate_abstract_reset(self):
        from bibr.paper import enforce_imrad_order
        from bibr.paper_contents import CanonicalSection, PaperSection

        sections = [
            PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 0.99),
            PaperSection(1, "Introduction", 2, None, CanonicalSection.INTRODUCTION, 0.98),
            PaperSection(2, "Summary", 2, None, CanonicalSection.ABSTRACT, 0.60),
            PaperSection(3, "Methods", 2, None, CanonicalSection.METHODS, 0.97),
        ]
        enforce_imrad_order(sections)

        assert sections[0].section_type == CanonicalSection.ABSTRACT
        assert sections[2].section_type == CanonicalSection.UNKNOWN
        assert sections[2].classification_score == 0.0

    def test_empty_sections(self):
        from bibr.paper import enforce_imrad_order

        enforce_imrad_order([])
