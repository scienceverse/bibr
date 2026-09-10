"""Provider-independent author-list sanitizer in ``_convert_llm_authors``.

Defends downstream code against degenerate LLM author output (blank entries,
affiliation fragments mislabeled as organization authors, mass duplicates, and
runaway repetition loops) regardless of which provider produced it. See
``CoreMetadataExtractor._convert_llm_authors`` and the runaway warning recorded
by ``extract()``.
"""

from unittest import mock

import pandas as pd

from bibr.extract.core_metadata import AUTHOR_ANOMALY_WARNING_PREFIX, CoreMetadataExtractor
from bibr.extract.extractor import MetadataExtractor
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection
from bibr.schemas import AuthorLLM, CoreMetadataLLM


def _convert(authors):
    return CoreMetadataExtractor._convert_llm_authors(authors)


def test_numbered_affiliations_are_reconciled_from_byline_and_blocks():
    authors = _convert(
        [
            AuthorLLM(given="Saskia M", family="Kelders", affiliation="University of Twente"),
            AuthorLLM(given="Hanneke", family="Kip", affiliation="University of Twente"),
            AuthorLLM(
                given="Nienke Beerlage-de",
                family="Jong",
                affiliation="Stichting Mindfit, Thubble",
            ),
            AuthorLLM(given="Nadine", family="Köhle", affiliation=""),
        ]
    )
    meta_df = pd.DataFrame(
        {
            "page_number": [1, 1, 1, 1, 1],
            "text": [
                "Saskia M Kelders1,2, Hanneke Kip1,3, Nienke Beerlage-de Jong4 and Nadine Köhle5",
                "1 Department of Health, Psychology and Technology, University of Twente, "
                "Enschede, The Netherlands",
                "2 Optentia Research Unit, North-West University, Vanderbijlpark, South Africa",
                "3 Department of Research, Transfore, Deventer, The Netherlands "
                "4 Section of Health Technology and Services Research, Technical Medical "
                "Centre, University of Twente, Enschede, The Netherlands",
                "5 Stichting Mindfit, Thubble, Deventer, The Netherlands Corresponding author: "
                "Saskia M Kelders, Department of Health, Psychology and Technology, "
                "University of Twente, Enschede, The Netherlands",
            ],
        }
    )

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, meta_df)

    assert authors[0].affiliation.startswith("Department of Health")
    assert "; Optentia Research Unit" in authors[0].affiliation
    assert authors[1].affiliation.endswith("The Netherlands")
    assert "Department of Research" in authors[1].affiliation
    assert authors[2].affiliation.startswith("Section of Health Technology")
    assert authors[3].affiliation == "Stichting Mindfit, Thubble, Deventer, The Netherlands"


def test_a_page_one_caption_does_not_redefine_an_affiliation_marker():
    """ "<digit> <Capital…>" is also the shape of a figure caption and of a
    publication-history line. With no institution test, adding a caption row
    to correct front matter rewrote author 1's affiliation to the caption."""
    authors = _convert(
        [AuthorLLM(given="Ada", family="Lovelace", affiliation="University of Twente")]
    )
    meta_df = pd.DataFrame(
        {
            "page_number": [1, 1, 1],
            "text": [
                "Ada Lovelace1",
                "1 Department of Health, University of Twente, Enschede, The Netherlands",
                "Figure 1 Study design and participant flow",
            ],
        }
    )

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, meta_df)

    assert authors[0].affiliation.startswith("Department of Health")


def test_publication_history_line_does_not_define_an_affiliation_marker():
    authors = _convert(
        [AuthorLLM(given="Ada", family="Lovelace", affiliation="University of Twente")]
    )
    meta_df = pd.DataFrame(
        {
            "page_number": [1, 1, 1],
            "text": [
                "Ada Lovelace2",
                "2 Department of Health, University of Twente, Enschede, The Netherlands",
                "2 Received 12 March 2019; accepted 5 May 2019",
            ],
        }
    )

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, meta_df)

    assert authors[0].affiliation.startswith("Department of Health")


async def test_affiliation_reconciliation_uses_rows_beyond_metadata_cutoff():
    full_df = pd.DataFrame(
        {
            "section_name": ["Title", "Footnote", "Footnote"],
            "page_number": [1, 1, 1],
            "text": [
                "Saskia M Kelders1,2 and Hanneke Kip1,3",
                "1 Department of Health, Psychology and Technology, University of Twente, "
                "Enschede, The Netherlands 2 Optentia Research Unit, North-West University, "
                "Vanderbijlpark, South Africa",
                "3 Department of Research, Transfore, Deventer, The Netherlands",
            ],
        }
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = full_df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    contents.processing_warnings = []

    locator = mock.MagicMock()
    locator.get_cutoff_index.return_value = 1
    # Simulate the real failure: numbered footnotes sit after the metadata cutoff.
    locator.collect_core_metadata_rows.return_value = full_df.iloc[:1].copy()
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="A Study",
            authors=[
                AuthorLLM(given="Saskia M", family="Kelders", affiliation=None),
                AuthorLLM(given="Hanneke", family="Kip", affiliation=None),
            ],
            keywords=[],
        )
    )
    email_harvester = mock.MagicMock()
    extractor = CoreMetadataExtractor(
        contents,
        llm_client=llm_client,
        locator=locator,
        email_harvester=email_harvester,
    )

    metadata = await extractor.extract()

    assert metadata.authors[0].affiliation.startswith("Department of Health")
    assert "; Optentia Research Unit" in metadata.authors[0].affiliation
    assert metadata.authors[1].affiliation.startswith("Department of Health")
    assert "; Department of Research" in metadata.authors[1].affiliation


def test_numbered_affiliations_use_clean_backmatter_and_normalized_names():
    authors = _convert(
        [
            AuthorLLM(given="María F.", family="Jara-Rizzo", affiliation=""),
            AuthorLLM(given="Juan F.", family="Navas", affiliation=""),
            AuthorLLM(given="Jose A.", family="Rodas", affiliation=""),
            AuthorLLM(given="José C.", family="Perales", affiliation=""),
        ]
    )
    meta_df = pd.DataFrame(
        {
            "page_number": [1, 1, 12, 12, 12, 12, 7],
            "section_type": [
                CanonicalSection.TITLE,
                CanonicalSection.FOOTNOTE,
                CanonicalSection.AUTHOR_CONTRIBUTIONS,
                CanonicalSection.AUTHOR_CONTRIBUTIONS,
                CanonicalSection.AUTHOR_CONTRIBUTIONS,
                CanonicalSection.AUTHOR_CONTRIBUTIONS,
                CanonicalSection.RESULTS,
            ],
            "text": [
                "María F. Jara‑Rizzo1*, Juan F. Navas2, Jose A. Rodas1,3 and José C. Perales4",
                "1 Faculty of Psychology, University of Guayaquil, Ecuador "
                "Full list of author information is available at the end of the article",
                "1 Faculty of Psychology, University of Guayaquil, Guayaquil, Ecuador.",
                "2 Department of Clinical Psychology, Complutense University of Madrid, "
                "Madrid, Spain.",
                "3 School of Psychology, University College Dublin, Dublin, Ireland.",
                "4 Department of Experimental Psychology; Mind, Brain and Behavior Research "
                "Centre, University of Granada, Granada, Spain.",
                "2 This lack of significance is likely due to low power.",
            ],
        }
    )

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, meta_df)

    assert "Full list of author information" not in authors[0].affiliation
    assert authors[0].affiliation.startswith("Faculty of Psychology")
    assert authors[1].affiliation.startswith("Department of Clinical Psychology")
    assert "University College Dublin" in authors[2].affiliation
    assert "Mind, Brain and Behavior Research Centre" in authors[3].affiliation


def _bjpsych_authors():
    return _convert(
        [
            AuthorLLM(given="Hannah Louise", family="Belcher", affiliation=""),
            AuthorLLM(given="Lois", family="Parri", affiliation=""),
            AuthorLLM(given="Imogen", family="Kilcoyne", affiliation=""),
            AuthorLLM(given="Joanne", family="Evans", affiliation=""),
            AuthorLLM(given="Caroline Da Cunha", family="Lewin", affiliation=""),
            AuthorLLM(given="Robin", family="Lau", affiliation=""),
            AuthorLLM(given="Nicola", family="Bond", affiliation=""),
            AuthorLLM(given="Conor", family="D’Arcy", affiliation=""),
            AuthorLLM(given="Melissa", family="Hatch", affiliation=""),
            AuthorLLM(given="Til", family="Wykes", affiliation=""),
        ]
    )


def test_repeated_name_affiliation_block_resolves_all_authors():
    authors = _bjpsych_authors()
    line = (
        "Hannah Louise Belcher, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK; "
        "Lois Parri, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK; "
        "Imogen Kilcoyne, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK; "
        "Joanne Evans, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK; "
        "Caroline Da Cunha Lewin, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK; "
        "Robin Lau, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK; "
        "Nicola Bond, The Money and Mental Health Policy Institute, "
        "The Policy Institute at King’s, London, UK; "
        "Conor D’Arcy, The Money and Mental Health Policy Institute, "
        "The Policy Institute at King’s, London, UK; "
        "Melissa Hatch, Department of Health Service & Population Research, "
        "Citizens Advice, London, UK; "
        "Til Wykes, Institute of Psychiatry, Psychology & Neuroscience, "
        "King’s College London, London, UK"
    )
    # The reconciler reads the paper's *complete* sentence frame and only
    # considers rows printed past the front page, so the fixture carries a
    # front-page row alongside the later affiliation block.
    meta_df = pd.DataFrame(
        {"page_number": [1, 8], "text": ["Autistic adults and financial harm", line]}
    )

    CoreMetadataExtractor._reconcile_repeated_name_affiliations(authors, meta_df)

    assert all(author.affiliation for author in authors)
    assert authors[0].affiliation.startswith("Institute of Psychiatry")
    assert authors[6].affiliation.startswith("The Money and Mental Health Policy Institute")
    assert authors[8].affiliation.startswith("Department of Health Service")


def test_repeated_name_affiliations_ignore_contribution_prose():
    authors = _convert(
        [
            AuthorLLM(given="Ada", family="Lovelace", affiliation=""),
            AuthorLLM(given="Alan", family="Turing", affiliation=""),
        ]
    )
    meta_df = pd.DataFrame(
        {
            "page_number": [8],
            "text": ["Ada Lovelace, designed the study; Alan Turing, analysed the results"],
        }
    )

    CoreMetadataExtractor._reconcile_repeated_name_affiliations(authors, meta_df)

    assert [author.affiliation for author in authors] == ["", ""]


def test_repeated_name_affiliations_require_two_exact_author_names():
    authors = _bjpsych_authors()
    meta_df = pd.DataFrame(
        {
            "page_number": [8],
            "text": ["Hannah Louise Belcher, Institute of Psychiatry, London, UK"],
        }
    )

    CoreMetadataExtractor._reconcile_repeated_name_affiliations(authors, meta_df)

    assert [author.affiliation for author in authors] == [""] * 10


class TestSanitizeAuthors:
    def test_blank_author_dropped(self):
        out = _convert(
            [
                AuthorLLM(given="", family=""),
                AuthorLLM(given="Ada", family="Lovelace"),
            ]
        )
        assert [(a.given, a.family) for a in out] == [("Ada", "Lovelace")]

    def test_organization_fragment_dropped(self):
        out = _convert(
            [
                AuthorLLM(given="", family="University of Testing", role=["organization"]),
                AuthorLLM(given="Ada", family="Lovelace"),
            ]
        )
        assert [(a.given, a.family) for a in out] == [("Ada", "Lovelace")]

    def test_named_organization_author_preserved(self):
        # A genuine organization/consortium author with a name must survive —
        # only the empty-given affiliation fragments are fragments.
        out = _convert([AuthorLLM(given="", family="The WHO Study Group")])
        assert [(a.given, a.family) for a in out] == [("", "The WHO Study Group")]

    def test_exact_duplicates_collapsed_order_preserving(self):
        out = _convert(
            [
                AuthorLLM(given="Ada", family="Lovelace"),
                AuthorLLM(given="Alan", family="Turing"),
                AuthorLLM(given="Ada", family="Lovelace"),
            ]
        )
        assert [(a.given, a.family) for a in out] == [("Ada", "Lovelace"), ("Alan", "Turing")]
        assert [a.author_id for a in out] == [1, 2]

    def test_legit_eight_author_list_untouched(self):
        incoming = [AuthorLLM(given=f"Given{i}", family=f"Family{i}") for i in range(8)]
        out = _convert(incoming)
        assert len(out) == 8
        assert [a.author_id for a in out] == list(range(1, 9))

    def test_forty_author_consortium_untouched(self):
        incoming = [AuthorLLM(given=f"Given{i}", family=f"Family{i}") for i in range(40)]
        out = _convert(incoming)
        assert len(out) == 40
        assert [a.author_id for a in out] == list(range(1, 41))

    def test_runaway_distinct_list_trimmed_to_cap(self):
        incoming = [AuthorLLM(given=f"Given{i}", family=f"Family{i}") for i in range(200)]
        out = _convert(incoming)
        assert len(out) == 64
        assert [a.author_id for a in out] == list(range(1, 65))


def _build_extractor(llm_metadata: CoreMetadataLLM):
    df = pd.DataFrame(
        {
            "section_name": ["Abstract", "1. Introduction"],
            "text": ["Some abstract text.", "Some intro text."],
            "page_number": [1, 1],
        }
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = df
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = [
        PaperSection(0, "Abstract", 2, None, CanonicalSection.ABSTRACT, 1.0),
        PaperSection(1, "1. Introduction", 2, None, CanonicalSection.INTRODUCTION, 1.0),
    ]
    contents.sentences = []
    contents.processing_warnings = []
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(return_value=llm_metadata)
    return MetadataExtractor(contents, llm_client=llm_client), contents


class TestRunawayWarning:
    async def test_runaway_author_list_trimmed_and_warned(self):
        incoming = [AuthorLLM(given=f"First{i}", family=f"Last{i}") for i in range(200)]
        ext, contents = _build_extractor(
            CoreMetadataLLM(title="A Study", authors=incoming, keywords=["k"])
        )

        await ext.extract_core_metadata()

        assert ext.metadata is not None
        assert len(ext.metadata.authors) == 64
        assert any(AUTHOR_ANOMALY_WARNING_PREFIX in w for w in contents.processing_warnings)

    async def test_clean_byline_records_no_anomaly(self):
        incoming = [AuthorLLM(given=f"First{i}", family=f"Last{i}") for i in range(5)]
        ext, contents = _build_extractor(
            CoreMetadataLLM(title="A Study", authors=incoming, keywords=["k"])
        )

        await ext.extract_core_metadata()

        assert len(ext.metadata.authors) == 5
        assert not any(AUTHOR_ANOMALY_WARNING_PREFIX in w for w in contents.processing_warnings)
