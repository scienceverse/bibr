"""Affiliation values after the reconcilers: cleaning, grounded re-join, folded dedup.

The author LLM returns one "; "-joined affiliation string per author, and the
numbered reconciler overwrites it with the printed definition. Both copy what
the page prints around the institution, and "; " is also a printed character.
The cases below are modelled on real tester and development papers (synthetic
institutions where the real text is not needed).
"""

import pandas as pd
import pytest

from bibr.extract.core_metadata import CoreMetadataExtractor, _clean_affiliation_value
from bibr.extract.research_integrity import affiliation_key, collect_affiliations
from bibr.paper import PaperAuthor


def _author(author_id: int, given: str, family: str, affiliation: str = "") -> PaperAuthor:
    return PaperAuthor(author_id=author_id, given=given, family=family, affiliation=affiliation)


def _frame(*rows: tuple[int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_id": list(range(len(rows))),
            "page_number": [page for page, _ in rows],
            "text": [text for _, text in rows],
        }
    )


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        # JNS/AMA house style: the list's connective before the next marker.
        (
            "Department of Neurological Surgery, Houston\r\nMethodist Hospital, Houston, "
            "Texas; and",
            "Department of Neurological Surgery, Houston Methodist Hospital, Houston, Texas",
        ),
        # A footnote definition ending in the corresponding author's address.
        (
            "Usher Institute, University of\r\nEdinburgh, Edinburgh, UK. E-mail: "
            "someone@example.ac.uk",
            "Usher Institute, University of Edinburgh, Edinburgh, UK",
        ),
        (
            "Department of Psychology, Example University, 00185 Rome, Italy; name@example.com",
            "Department of Psychology, Example University, 00185 Rome, Italy",
        ),
        (
            "Max Planck Institute for Example Research, Tel.: +49 89 1234",
            "Max Planck Institute for Example Research",
        ),
        (
            "Icahn School of Medicine \r\nat Mount Sinai, New York, NY, USA.",
            "Icahn School of Medicine at Mount Sinai, New York, NY, USA",
        ),
        # BMC / Springer pointers to the end-of-article list.
        (
            "Example University of Norway, Tromsø, Norway\r\nFull list of author information "
            "is available at the end of the article",
            "Example University of Norway, Tromsø, Norway",
        ),
        (
            "Department of Health, Example University; Extended author information available "
            "on the last page of the article",
            "Department of Health, Example University",
        ),
        (
            "School of Health Sciences, Example University, UK |",
            "School of Health Sciences, Example University, UK",
        ),
        ("Example Systems †", "Example Systems"),
        # Markers the LLM copied.
        (
            "1 Department of Radiology, Example Hospital",
            "Department of Radiology, Example Hospital",
        ),
        (
            "2. NIHR Example Research Centre, Bristol, UK.",
            "NIHR Example Research Centre, Bristol, UK",
        ),
        ("1Department of Physics, Example University", "Department of Physics, Example University"),
        ("a School of Chemistry, Example University", "School of Chemistry, Example University"),
        ("and", ""),
        # Must survive as printed.
        (
            "Tel Aviv-Yaffo Academic College, Yaffo, Israel",
            "Tel Aviv-Yaffo Academic College, Yaffo, Israel",
        ),
        ("Department of X & Y, University Z", "Department of X & Y, University Z"),
        ("3M Company, St Paul, MN, USA", "3M Company, St Paul, MN, USA"),
        ("Example University, Boston, MA, U.S.A.", "Example University, Boston, MA, U.S.A."),
    ],
)
def test_affiliation_values_lose_what_is_printed_around_the_institution(raw, clean):
    assert _clean_affiliation_value(raw) == clean


def test_a_printed_inner_semicolon_is_not_a_second_affiliation():
    """One printed affiliation "..., Cambridge, MA, USA; Basel, Switzerland": the
    "; " split made "Basel, Switzerland" a row of its own for 11 authors."""
    printed = (
        "1 Disease Area Research, Example Biomedical Research, Cambridge, MA, USA; "
        "Basel, Switzerland"
    )
    frame = _frame((1, "Ann Lee1, Bo Chen1"), (1, printed))
    copied = (
        "Disease Area Research, Example Biomedical Research, Cambridge, MA, USA; Basel, Switzerland"
    )
    authors = [_author(1, "Ann", "Lee", copied), _author(2, "Bo", "Chen", copied)]

    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    texts, author_ids = collect_affiliations(authors)
    assert texts == [
        "Disease Area Research, Example Biomedical Research, Cambridge, MA, USA, Basel, Switzerland"
    ]
    assert author_ids == [[1, 2]]


def test_separately_printed_affiliations_stay_separate():
    """ "Example Systems" is not an institution word, but it was printed on its
    own, so it is the author's second affiliation and not a fragment."""
    frame = _frame(
        (1, "Ann Lee1,2"),
        (1, "1 Institute of Example Models, Example City, USA"),
        (1, "2 Example Systems"),
    )
    authors = [
        _author(1, "Ann", "Lee", "Institute of Example Models, Example City, USA; Example Systems")
    ]

    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[0].affiliation == (
        "Institute of Example Models, Example City, USA; Example Systems"
    )


def test_connective_and_contact_tails_never_become_affiliations():
    """World Neurosurgery and EMBO Reports styles, through the numbered reconciler:
    the export had an affiliation "and", and an e-mail in another author's row."""
    frame = _frame(
        (1, "Ann Lee1 and Bo Chen1,2"),
        (
            1,
            "From the 1 Department of Neurological Surgery, Houston\r\nMethodist Hospital, "
            "Houston, Texas; and 2 Department of Neurological Surgery, Example College of "
            "Medicine, Houston, Texas.",
        ),
    )
    authors = [_author(1, "Ann", "Lee", "x"), _author(2, "Bo", "Chen", "y")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    texts, author_ids = collect_affiliations(authors)
    assert texts == [
        "Department of Neurological Surgery, Houston Methodist Hospital, Houston, Texas",
        "Department of Neurological Surgery, Example College of Medicine, Houston, Texas",
    ]
    assert author_ids == [[1, 2], [2]]

    frame = _frame(
        (1, "Ann Lee1 and Bo Chen2"),
        (
            1,
            "1 Department of Example Studies, Example University, Edinburgh, UK. "
            "2 Usher Institute, University of\r\nEdinburgh, Edinburgh, UK. E-mail: "
            "someone@example.ac.uk",
        ),
    )
    authors = [_author(1, "Ann", "Lee"), _author(2, "Bo", "Chen")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert (
        authors[0].affiliation == "Department of Example Studies, Example University, Edinburgh, UK"
    )
    assert authors[1].affiliation == "Usher Institute, University of Edinburgh, Edinburgh, UK"


def test_llm_markers_emails_and_marker_only_parts_are_dropped():
    frame = _frame(
        (1, "Ann Lee1,2"),
        (1, "1 Department of Psychology, Example University, Rome, Italy; name@example.com"),
    )
    authors = [
        _author(
            1,
            "Ann",
            "Lee",
            "1 Department of Psychology, Example University, Rome, Italy; name@example.com; "
            "2; 3; 2 Department of Psychology, Example University, Rome, Italy.",
        )
    ]

    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[0].affiliation == "Department of Psychology, Example University, Rome, Italy"


def test_variants_of_one_institution_are_one_row():
    """Exact-string dedup listed the same institution twice: once as the
    reconciler copied it (line breaks, "LondonUK"), once as the LLM wrote it."""
    authors = [
        _author(1, "A", "B", "Department of Epidemiology, University College London,\r\nLondonUK"),
        _author(2, "C", "D", "Department of Epidemiology, University College London, London, UK"),
        _author(3, "E", "F", "Department of Public Health & Primary Care, Universität Bern"),
        _author(4, "G", "H", "Department of Public Health and Primary Care, Universitat Bern"),
    ]

    texts, author_ids = collect_affiliations(authors)

    assert texts == [
        "Department of Epidemiology, University College London, London, UK",
        "Department of Public Health & Primary Care, Universität Bern",
    ]
    assert author_ids == [[1, 2], [3, 4]]


def test_the_folded_key_keeps_different_institutions_and_other_scripts_apart():
    assert affiliation_key("Department of Nursing, University A") != affiliation_key(
        "Department of Nursing, University B"
    )
    assert affiliation_key("東京大学医学部") == "東京大学医学部"


# ── grounding: affiliation text the paper does not print ──────────────────


def test_an_affiliation_printed_nowhere_in_the_paper_is_dropped():
    """RSC-style: the definitions were page-1 footnotes outside the LLM's
    context, and it answered with a university the paper never prints."""
    frame = _frame(
        (1, "Ann Lee a and Bo Chen b"),
        (1, "a School of Mechanical Engineering, Qinghai University, Xining 810016, China"),
        (1, "b Faculty of Materials, Example University of Technology, Beijing, China"),
    )
    authors = [
        _author(
            1,
            "Ann",
            "Lee",
            "a School of Materials Science and Engineering, Liaocheng University, "
            "Liaocheng 252059, P. R. China",
        ),
        _author(2, "Bo", "Chen", "Faculty of Materials, Example University of Technology, Beijing"),
    ]

    dropped = CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[0].affiliation == ""
    assert authors[1].affiliation == (
        "Faculty of Materials, Example University of Technology, Beijing"
    )
    assert dropped == [
        (
            1,
            "School of Materials Science and Engineering, Liaocheng University, "
            "Liaocheng 252059, P. R. China",
        )
    ]


def test_an_unprinted_component_is_pruned_from_a_printed_affiliation():
    """World-knowledge completion: "University at Buffalo" became "University at
    Buffalo, State University of New York", which the paper does not print."""
    frame = _frame(
        (1, "Ann Lee3"),
        (
            10,
            "3 Department of Epidemiology, School of Public Health, University at Buffalo, "
            "Buffalo, NY, USA.",
        ),
    )
    authors = [
        _author(
            1,
            "Ann",
            "Lee",
            "Department of Epidemiology, School of Public Health, University at Buffalo, "
            "State University of New York, Buffalo, NY, USA",
        )
    ]

    dropped = CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[0].affiliation == (
        "Department of Epidemiology, School of Public Health, University at Buffalo, "
        "Buffalo, NY, USA"
    )
    assert dropped == [(1, "State University of New York")]


def test_ocr_typos_and_text_only_the_llm_saw_are_grounded():
    """The LLM silently corrects an OCR typo ("Universily"); an author table the
    LLM context carried is not in the sentence frame. Neither is an invention."""
    frame = _frame((1, "Ann Lee, Department of Physics, Universily of Exampleton, UK"))
    authors = [
        _author(1, "Ann", "Lee", "Department of Physics, University of Exampleton, UK"),
        _author(2, "Bo", "Chen", "Institute of Table Studies, Example Hospital, Paris"),
    ]

    dropped = CoreMetadataExtractor._normalize_author_affiliations(
        authors, frame, context_text="Bo Chen | Institute of Table Studies, Example Hospital"
    )

    assert dropped == []
    assert authors[0].affiliation == "Department of Physics, University of Exampleton, UK"
    assert authors[1].affiliation == "Institute of Table Studies, Example Hospital, Paris"


async def test_the_extractor_warns_about_dropped_affiliations():
    from unittest import mock

    from bibr.paper_contents import PaperContents
    from bibr.schemas import AuthorLLM, CoreMetadataLLM

    frame = pd.DataFrame(
        {
            "section_name": ["Title", "Title"],
            "page_number": [1, 1],
            "text": ["A Study of Things", "Ann Lee and Bo Chen, Example University, Utrecht"],
        }
    )
    contents = mock.Mock(spec=PaperContents)
    contents.sentences_df = frame
    contents.detected_headers = []
    contents.detected_footers = []
    contents.layout_hints = []
    contents.sections = []
    contents.sentences = []
    contents.processing_warnings = []
    locator = mock.MagicMock()
    locator.get_cutoff_index.return_value = 2
    locator.collect_core_metadata_rows.return_value = frame.copy()
    llm_client = mock.MagicMock()
    llm_client.extract_core_metadata = mock.AsyncMock(
        return_value=CoreMetadataLLM(
            title="A Study of Things",
            authors=[
                AuthorLLM(given="Ann", family="Lee", affiliation="Example University, Utrecht"),
                AuthorLLM(given="Bo", family="Chen", affiliation="Invented Polder University"),
            ],
            keywords=[],
        )
    )
    extractor = CoreMetadataExtractor(
        contents, llm_client=llm_client, locator=locator, email_harvester=mock.MagicMock()
    )

    metadata = await extractor.extract()

    assert [author.affiliation for author in metadata.authors] == [
        "Example University, Utrecht",
        "",
    ]
    issues = [i for i in extractor.validation_issues if i.code == "VAL_AFFILIATION_UNGROUNDED"]
    assert len(issues) == 1
    assert issues[0].evidence_ids == ("author:2",)
    assert "Invented Polder University" in issues[0].message


# ── numbered reconciler: byline markers and end-of-article lists ─────────


def _sectioned_frame(*rows: tuple[int, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_id": list(range(len(rows))),
            "page_number": [page for page, _, _ in rows],
            "section_type": [section for _, section, _ in rows],
            "text": [text for _, _, text in rows],
        }
    )


def test_symbols_between_name_and_number_do_not_block_the_marker():
    """medRxiv style "Shin#*1", "Cravedi%5*": those authors kept the LLM value
    while their co-authors got the printed one."""
    frame = _frame(
        (1, "Ann Shin#*1, Bo Weiss#2,4, Cy Moreau%1 and Di Cravedi%5*"),
        (1, "1 Department of Example Research, Example Institute, Cambridge, MA, USA"),
        (1, "2 Preclinical Unit, Example Pharma, Basel, Switzerland"),
        (1, "4 Fellowship Program, Example Pharma, Basel, Switzerland"),
        (1, "5 Department of Medicine, Example School of Medicine, New York, NY, USA"),
    )
    authors = [
        _author(1, "Ann", "Shin", "llm"),
        _author(2, "Bo", "Weiss", "llm"),
        _author(3, "Cy", "Moreau", "llm"),
        _author(4, "Di", "Cravedi", "llm"),
    ]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)

    assert [author.affiliation for author in authors] == [
        "Department of Example Research, Example Institute, Cambridge, MA, USA",
        "Preclinical Unit, Example Pharma, Basel, Switzerland; "
        "Fellowship Program, Example Pharma, Basel, Switzerland",
        "Department of Example Research, Example Institute, Cambridge, MA, USA",
        "Department of Medicine, Example School of Medicine, New York, NY, USA",
    ]


@pytest.mark.parametrize(
    ("given", "family", "byline"),
    [
        # The LLM drops the period the byline prints.
        ("Robyn A", "Frankel", "Robyn A. Frankel1,2 and Bo Chen1"),
        # A degree between the name and the numbers.
        ("Robyn", "Frankel", "Robyn Frankel MSci1,2*, Bo Chen PhD1"),
        ("Robyn", "Frankel", "Robyn Frankel, MD,1,2 and Bo Chen, MD1"),
        # BMJ: an ORCID icon was dropped and left a comma before the numbers.
        ("Robyn", "Frankel", "Robyn Frankel ,1,2 Bo Chen 1"),
    ],
)
def test_name_variants_degrees_and_a_comma_do_not_block_the_marker(given, family, byline):
    frame = _frame(
        (1, byline),
        (1, "1 Department of Psychology, Example University, Toronto, ON, Canada"),
        (1, "2 Example Research Institute, Toronto, ON, Canada"),
    )
    authors = [_author(1, given, family, "llm"), _author(2, "Bo", "Chen", "llm")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)

    assert authors[0].affiliation == (
        "Department of Psychology, Example University, Toronto, ON, Canada; "
        "Example Research Institute, Toronto, ON, Canada"
    )
    assert authors[1].affiliation == (
        "Department of Psychology, Example University, Toronto, ON, Canada"
    )


def test_a_name_that_only_starts_like_the_author_is_not_their_marker():
    frame = _frame(
        (1, "Ann Leeba1 and Bo Chen2"),
        (1, "1 Department of Psychology, Example University, Toronto, ON, Canada"),
        (1, "2 Example Research Institute, Toronto, ON, Canada"),
    )
    authors = [_author(1, "Ann", "Lee", "llm"), _author(2, "Bo", "Chen", "llm")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)

    assert authors[0].affiliation == "llm"


def test_an_end_of_article_list_typed_acknowledgment_fills_only_open_markers():
    """BMC: page 1 defines the corresponding author's markers and points to the
    end of the article; that block was typed acknowledgment, so marker 3 stayed
    open and the LLM's guess was exported."""
    frame = _sectioned_frame(
        (1, "title", "Erik Berg1,2* and Cato Moe3"),
        (
            1,
            "footnote",
            "* Correspondence: someone@example.no 1 Nordland Example Hospital, Bodø, Norway\r\n"
            "2 Example Arctic University of Norway, Tromsø, Norway\r\nFull list of author "
            "information is available at the end of the article",
        ),
        (9, "acknowledgment", "3 Nord Example University, Bodø, Norway."),
        (9, "acknowledgment", "1 Somewhere Else University, Oslo, Norway."),
    )
    authors = [_author(1, "Erik", "Berg", "guess A"), _author(2, "Cato", "Moe", "guess B")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    assert authors[1].affiliation == "Nord Example University, Bodø, Norway"
    assert authors[0].affiliation == (
        "Nordland Example Hospital, Bodø, Norway; Example Arctic University of Norway, "
        "Tromsø, Norway"
    )


def test_the_late_tier_is_not_read_for_body_sections():
    frame = _sectioned_frame(
        (1, "title", "Erik Berg3"),
        (5, "results", "3 Department of Example Results, Example University, Oslo, Norway"),
    )
    authors = [_author(1, "Erik", "Berg", "Example University, Oslo")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)

    assert authors[0].affiliation == "Example University, Oslo"


# ── #107: markers glued to the institution ────────────────────────────────

_GLUED_BYLINE = (
    "Ann Sahoo∗,†,1 , Bo Chen†,1,2 , Cy Pham†,1,3 , Di\r\n"
    "Geuter†,1,4 , Ed Dwivedi1 , Flo Pimpalkhute1 , Gus Elhoushi5 , Hal Thickstun3"
)
_GLUED_DEFINITIONS = (
    "1 Institute of Foundation Models, 2University of Illinois Urbana-Champaign, "
    "3Cornell Tech\r\n4Harvard University 5Cerebras Systems †"
)


def test_glued_markers_bound_each_definition():
    """One spaced "1 " and glued "2University", "3Cornell", ...: definition 1 ran
    to the end of the line and became every marker-1 author's affiliation."""
    frame = _frame((1, _GLUED_BYLINE), (1, _GLUED_DEFINITIONS))
    authors = [
        _author(1, "Ann", "Sahoo"),
        _author(2, "Bo", "Chen"),
        _author(3, "Cy", "Pham"),
        _author(4, "Di", "Geuter"),
        _author(5, "Ed", "Dwivedi"),
        _author(6, "Flo", "Pimpalkhute"),
        _author(7, "Gus", "Elhoushi", "Cerebras Systems"),
        _author(8, "Hal", "Thickstun", "Cornell Tech"),
    ]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)

    texts, author_ids = collect_affiliations(authors)
    assert texts == [
        "Institute of Foundation Models",
        "University of Illinois Urbana-Champaign",
        "Harvard University",
        "Cerebras Systems",
        "Cornell Tech",
    ]
    # "Cornell Tech" carries no institution word, so marker 3 does not resolve
    # and Pham keeps no reconciled value; Thickstun keeps the LLM's.
    assert author_ids == [[1, 2, 4, 5, 6], [2], [4], [7], [8]]


def test_a_definition_that_runs_into_another_never_overwrites_the_llm_value():
    """Backstop: no author uses marker 2, so "2University" does not bound
    definition 1, which then holds two institutions and is not used."""
    frame = _frame(
        (1, "Ann Lee1 and Bo Chen3"),
        (1, "1 Institute of Example Models, 2University of Example, 3 Example College, UK"),
    )
    authors = [
        _author(1, "Ann", "Lee", "Institute of Example Models"),
        _author(2, "Bo", "Chen", "Example College, UK"),
    ]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)

    assert authors[0].affiliation == "Institute of Example Models"
    assert authors[1].affiliation == "Example College, UK"


def test_a_digit_glued_to_a_word_that_is_not_a_marker_stays_text():
    """ "3D Printing" and "3M" are not glued markers even when the byline uses 3."""
    frame = _frame(
        (1, "Ann Lee1 and Bo Chen3"),
        (1, "1 Example University, 3D Printing Laboratory, 3M Company, St Paul, MN, USA"),
        (1, "3 Other College, Oxford, UK"),
    )
    authors = [_author(1, "Ann", "Lee"), _author(2, "Bo", "Chen")]

    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)

    assert authors[0].affiliation == (
        "Example University, 3D Printing Laboratory, 3M Company, St Paul, MN, USA"
    )
    assert authors[1].affiliation == "Other College, Oxford, UK"
