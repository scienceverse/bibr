"""A reference's row (``text_id``) and its text fields after a line break.

Rows from real papers (W3154309365, W3001627138, W2042250854, W2529702590):
two references must not share one row, or the citation linker reads that
row's printed number on both and links neither.
"""

from bibr.extract.ref_extractor import _finalize_reference_fields, _map_bib_text_ids
from bibr.paper import PaperReference
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence


def _contents(rows: list[str]) -> PaperContents:
    sections = [
        PaperSection(section_id=0, header="Root", level=0, parent_section_id=None),
        PaperSection(
            section_id=1,
            header="References",
            level=1,
            parent_section_id=0,
            section_type=CanonicalSection.REFERENCES,
        ),
    ]
    sentences = [
        PaperSentence(text_id=100 + i, text=row, section_id=1, paragraph_id=i)
        for i, row in enumerate(rows)
    ]
    return PaperContents(
        sentences=sentences, sections=sections, tables=[], links=[], sections_text={}
    )


def _refs(*titles: str) -> list[PaperReference]:
    return [
        PaperReference(
            bib_id=i + 1,
            title=t,
            authors=None,
            year=None,
            container=None,
            volume=None,
            first_page=None,
        )
        for i, t in enumerate(titles)
    ]


def test_two_works_of_one_title_get_their_own_rows():
    contents = _contents(
        [
            "2. Perk, J. European guidelines on cardiovascular disease prevention. 2012.",
            "3. Korea Centers for Disease Control and Prevention. Available online: https://a.kr",
            "4. Go, A.S. Heart disease and stroke statistics. Circulation 2013.",
            "5. Korea Centers for Disease Control and Prevention. Available online: http://b.kr",
        ]
    )
    refs = _refs(
        "European guidelines on cardiovascular disease prevention",
        "Korea Centers for Disease Control and Prevention",
        "Heart disease and stroke statistics",
        "Korea Centers for Disease Control and Prevention",
    )
    _map_bib_text_ids(refs, contents)
    assert [r.text_id for r in refs] == [100, 101, 102, 103]


def test_a_short_title_inside_a_longer_one_takes_its_own_row():
    contents = _contents(
        [
            "Beiker S (2012) Legal aspects of autonomous driving. Santa Clara Law Rev 52:1145",
            "Bonnefon J-F (2016) The social dilemma of autonomous vehicles. Science 352:1573",
            "Wachenfeld W (2016) Autonomous driving. Springer, Berlin",
        ]
    )
    refs = _refs(
        "Legal aspects of autonomous driving",
        "The social dilemma of autonomous vehicles",
        "Autonomous driving",
    )
    _map_bib_text_ids(refs, contents)
    assert [r.text_id for r in refs] == [100, 101, 102]


def test_a_title_listed_before_the_longer_title_holding_it():
    contents = _contents(
        [
            "Ministry of Education. (1998). Special Education 2000 update. Wellington.",
            "Ministry of Education. (1997). Special Education 2000. Wellington.",
        ]
    )
    refs = _refs("Special Education 2000", "Special Education 2000 update")
    _map_bib_text_ids(refs, contents)
    assert [r.text_id for r in refs] == [101, 100]


def test_a_title_wrapped_at_a_line_break_matches_its_row():
    contents = _contents(
        [
            "Chua ASM (2003) Production of polyhydroxyalkanoates (PHA) by activated sludge "
            "treating municipal wastewater. Water Sci Technol 47:1",
            "Mino T (2003) Production of polyhydroxyalkanoates by activated sludge treating a "
            "paper mill wastewater. Water Res 12:3",
        ]
    )
    refs = _refs(
        "Production of polyhydroxyalkanoates (PHA) by activated sludge treating municipal "
        "wastewater",
        "Production of polyhydroxyalkanoates by activated sludge\ntreating a paper mill wastewater",
    )
    _map_bib_text_ids(refs, contents)
    assert [r.text_id for r in refs] == [100, 101]


def test_a_sentence_holding_several_citations_is_still_shared():
    contents = _contents(
        [
            "9. Voir : Rapport : Colloques régionaux sur la violence; Politique "
            "d'intervention en matière de violence conjugale.",
        ]
    )
    refs = _refs(
        "Rapport : Colloques régionaux sur la violence",
        "Politique d'intervention en matière de violence conjugale",
    )
    _map_bib_text_ids(refs, contents)
    assert [r.text_id for r in refs] == [100, 100]


def test_text_fields_lose_the_line_break_of_a_wrapped_reference():
    segment = (
        "Fox, N.A. (2010). Error-related responses are enhanced in children with\n"
        "obsessive-compulsive disorder. Social\nCognition and Social Development. Cambridge\n"
        "University Press."
    )
    fields = _finalize_reference_fields(
        {
            "title": "Error-related responses are enhanced in children with\n"
            "obsessive-compulsive disorder",
            "authors": "Fox, N.A.,\nDean, J.",
            "container": "Social\nCognition and Social Development",
            "publisher": "Cambridge\nUniversity Press",
            "year": 2010,
        },
        segment,
    )
    assert fields["title"] == (
        "Error-related responses are enhanced in children with obsessive-compulsive disorder"
    )
    assert fields["authors"] == "Fox, N.A., Dean, J."
    assert fields["container"] == "Social Cognition and Social Development"
    assert fields["publisher"] == "Cambridge University Press"
