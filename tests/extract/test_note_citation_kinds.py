"""Note citations the reference tagger cannot read on its own.

The note texts are short excerpts of dev-set papers whose notes stand in for a
reference list: a French chronicle of copyright case law (TGI, CA, Cass. and CE
decisions with their reporters), a Québec article on family violence (a Court
of Appeal decision cited by the parties, surnames spaced out), a Spanish
humanities paper citing classical works, and an economics bulletin citing a
survey named for its year.
"""

from __future__ import annotations

import re
from unittest import mock
from unittest.mock import AsyncMock

from bibr.config import GlobalSettings
from bibr.extract import footnote_citations as fc
from bibr.models import PaperMetadata, PaperReference
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence


def _contents(notes: list[str]) -> PaperContents:
    """A paper whose body cites in notes: one footnote section per note."""
    sections = [PaperSection(1, "Introduction", 1, None, CanonicalSection.INTRODUCTION, 1.0)]
    sentences = [PaperSentence(1, "Body text citing a note.", 1, 1)]
    for index, note in enumerate(notes, start=2):
        sections.append(
            PaperSection(
                index,
                f"Footnote {index - 1}",
                2,
                None,
                CanonicalSection.FOOTNOTE,
                1.0,
                synthetic_kind="footnote",
            )
        )
        sentences.append(PaperSentence(100 + index, note, index, index))
    return PaperContents(
        sentences=sentences, sections=sections, tables=[], links=[], sections_text={}
    )


def _texts(notes: list[str]) -> list[str]:
    return [citation.text for citation in fc.note_citations(_contents(notes)).citations]


# ---------------------------------------------------------------------------
# Court decisions
# ---------------------------------------------------------------------------

# The French chronicle: commentary before a decision, a second reporter after
# a semicolon, two decisions in one note, a decision named in running text.
_DECISION_NOTES = [
    "7. Reproduction d’un lapin sculpté en chocolat : TGI Laval, 16 févr. 2009, "
    "Propr. intell. 2009, n° 32, p. 260, obs. Bruguière ; RLDI 2009, \r\nn° 50, p. 8, "
    "obs. Fontaine.",
    "11. Voir en ce sens cette décision de la haute Cour des Pays-Bas qui souligne "
    "clairement qu’une création originale n’implique pas de « choix conscients », "
    "Hoge Raad, 30 mai 2008 cité par P.-B. Hugenholtz, « Chronique des Pays-Bas », "
    "RIDA\r\n2010/4, p. 305.",
    "21. Paris, 27 avr. 1934, DH 1934. 385.",
    "32. Cass. 1re civ., 16 mai 2018, n° 15-14.023.",
    "33. Voir par exemple CA Paris, 4e ch., 28 sept. 1988, JurisData no 1988-025669.",
    "34. Civ. 1re, 16 juill. 1998, D. 1999. 306, note Dreyer ; JCP E 2000, p. 77, obs. "
    "Laporte-Legeais ; RIDA 1998, no 178, 241, obs. Kéréver ;\r\nRTD com. 1999. 394, obs. "
    "Françon. – Paris, 1re ch., 18 janv. 2000, D. 2000. 203.",
]


def test_a_court_decision_is_a_citation_of_its_own_without_its_second_reporters():
    assert _texts(_DECISION_NOTES) == [
        "TGI Laval, 16 févr. 2009, Propr. intell. 2009, n° 32, p. 260, obs. Bruguière",
        "Hoge Raad, 30 mai 2008",
        "P.-B. Hugenholtz, « Chronique des Pays-Bas », RIDA 2010/4, p. 305.",
        "Paris, 27 avr. 1934, DH 1934. 385.",
        "Cass. 1re civ., 16 mai 2018, n° 15-14.023.",
        "CA Paris, 4e ch., 28 sept. 1988, JurisData no 1988-025669.",
        "Civ. 1re, 16 juill. 1998, D. 1999. 306, note Dreyer",
        "Paris, 1re ch., 18 janv. 2000, D. 2000. 203.",
    ]
    found = fc.note_citations(_contents(_DECISION_NOTES))
    assert found.citing_notes == 6
    # Every decision is a full citation; the chronicle it is cited from has no year.
    assert [citation.full for citation in found.citations] == [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_a_decision_is_titled_by_its_court_date_number_and_parties():
    cases = {
        "Civ. 1re, 16 juill. 1998, D. 1999. 306, note Dreyer": ("Civ. 1re, 16 juill. 1998", 1998),
        "Cass. 1re civ., 16 mai 2018, n° 15-14.023.": (
            "Cass. 1re civ., 16 mai 2018, n° 15-14.023",
            2018,
        ),
        "CA Paris, Pôle 5, ch. 2, 12 janvier 2018, RG n° 16/19375, SAS Les Éditions du net "
        "c/ Victima H.": (
            "CA Paris, Pôle 5, ch. 2, 12 janvier 2018, RG n° 16/19375, SAS Les Éditions du net "
            "c/ Victima H.",
            2018,
        ),
        "CE, 19 nov. 2001, Titanic, RTD com. 2002. 474, obs. Françon.": (
            "CE, 19 nov. 2001, Titanic",
            2001,
        ),
        "Paris, 27 avr. 1934, DH 1934. 385.": ("Paris, 27 avr. 1934", 1934),
        "TGI Paris, 5 juin 1987, Cah. dr. auteur, janv.1988, p. 14.": (
            "TGI Paris, 5 juin 1987",
            1987,
        ),
        "Cohen v. Wolkoff US District Court, East. District of New York 12 février 2018": (
            "Cohen v. Wolkoff US District Court, East. District of New York 12 février 2018",
            2018,
        ),
        # The Québec article: the parties, the court and the docket number
        # come before the date.
        "R. c. Deschamps, C.A. Québec, n° 500-10-000003-887, 11 mars 1988, JJ. Monet, "
        "McCarthy, Mailhot.": (
            "R. c. Deschamps, C.A. Québec, n° 500-10-000003-887, 11 mars 1988",
            1988,
        ),
    }
    assert {text: fc.legal_decision(text) for text in cases} == cases


def test_a_dated_newspaper_or_event_is_no_decision():
    for text in (
        "V. Duponchelle, « L’art brut gagne ses lettres de noblesse », Le Figaro, 18 déc. 2014.",
        "Le Figaro, 18 déc. 2014.",
        "La Presse, 20 août 1987, p. A-3 « La mort d’Hélène Lizotte »",
        "Projet de loi C-15, adopté le 23 juin 1987, 33e législature (Can.)",
        "présenté à la réunion des ministres, Niagara-on-the-Lake, les 28-30 mai 1984",
        "Paris, 12 mars 2010, colloque de l’Association.",
    ):
        assert fc.legal_decision(text) is None, text


def test_a_decision_in_running_text_is_cut_out_of_it():
    note = (
        "30. Pour une illustration des possibilités limitées de l’intervention judiciaire, "
        "voir R. c. Deschamps, C.A. Québec, n° 500-10-000003-887, \r\n11 mars 1988, JJ. "
        "Monet, McCarthy, Mailhot."
    )
    assert _texts([note]) == [
        "R. c. Deschamps, C.A. Québec, n° 500-10-000003-887, 11 mars 1988, JJ. Monet, "
        "McCarthy, Mailhot."
    ]


# A European migration-law paper cites Court of Justice and Strasbourg
# decisions by the parties, with "See" before some and long party names.
_PARTIES_NOTES = [
    "12 See CJEU, Bundesrepublik Deutschland v. Kaveh Puid, Judgment, Case C-4/11, \r\n"
    "14 November 2013, ECLI:EU:C:2013:740, paras. 30 and 36.",
    "6 CJEU, Abubacarr Jawo v. Bundesrepublik Deutschland, Judgment, Case C-163/17, \r\n"
    "19 March 2019, ECLI:EU:C:2019:218, para. 88; CJEU, Bashar Ibrahim and\xa0Others v. "
    "\r\nBundesrepublik Deutschland and Bundesrepublik Deutschland v. Taus Magamadov, "
    "Judgement, Joined Cases C-297/17, C-318/17, C-319/17 and C-438/17, 19 March 2019, "
    "\r\nECLI:EU:C:2019:219, para. 87.",
    "15 CJEU, N.\xa0S.\xa0v. Secretary of State for the Home Department and M.\xa0E. and "
    "Others v. Refugee Applications Commissioner and Minister for Justice, Equality and Law "
    "Reform, Judgment, \r\nJoined Cases C-411/10 and C-493/10, 21\xa0December 2011, "
    "ECLI:EU:C:2011:865, paras.\xa078 to 81.",
]


def test_a_decision_keeps_its_court_and_all_its_parties_and_drops_its_lead_in():
    texts = _texts(_PARTIES_NOTES)
    assert [text.split(", Judg")[0] for text in texts] == [
        "CJEU, Bundesrepublik Deutschland v. Kaveh Puid",
        "CJEU, Abubacarr Jawo v. Bundesrepublik Deutschland",
        "CJEU, Bashar Ibrahim and Others v. Bundesrepublik Deutschland and Bundesrepublik "
        "Deutschland v. Taus Magamadov",
        # Too long a party list to read as a designation: the citation stays
        # whole rather than start inside it.
        "CJEU, N. S. v. Secretary of State for the Home Department and M. E. and Others v. "
        "Refugee Applications Commissioner and Minister for Justice, Equality and Law Reform",
    ]
    assert fc.legal_decision(texts[0]) == (
        "CJEU, Bundesrepublik Deutschland v. Kaveh Puid, Judgment, Case C-4/11, 14 November 2013",
        2013,
    )


def _fake_ner(self, ref_strings):
    """What the tagger makes of a decision: a title cut at the date, no year."""
    parsed = []
    for text in ref_strings:
        head, *_ = re.split(r"\s\d{1,2}\s", text, maxsplit=1)
        parsed.append(
            PaperReference(
                bib_id=1,
                title=head,
                authors=None,
                year=None,
                container=None,
                volume=None,
                first_page=None,
            )
        )
    return parsed


async def test_the_notes_path_gives_a_decision_its_title_and_year():
    from bibr.extract.extractor import MetadataExtractor

    extractor = MetadataExtractor(
        _contents(_DECISION_NOTES),
        llm_client=mock.MagicMock(),
        ref_parse_strategy="ner",
        settings=GlobalSettings(),
    )
    extractor.extract_core_metadata = AsyncMock()
    extractor.metadata = PaperMetadata(doi="", title="T")

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned", _fake_ner
    ):
        metadata = await extractor.extract_all_metadata()

    assert [(ref.title, ref.year, ref.authors, ref.text_id) for ref in metadata.references] == [
        ("TGI Laval, 16 févr. 2009", 2009, None, 102),
        ("Hoge Raad, 30 mai 2008", 2008, None, 103),
        ("P.-B. Hugenholtz, « Chronique des Pays-Bas », RIDA 2010/4, p. 305.", None, None, 103),
        ("Paris, 27 avr. 1934", 1934, None, 104),
        ("Cass. 1re civ., 16 mai 2018, n° 15-14.023", 2018, None, 105),
        ("CA Paris, 4e ch., 28 sept. 1988", 1988, None, 106),
        ("Civ. 1re, 16 juill. 1998", 1998, None, 107),
        ("Paris, 1re ch., 18 janv. 2000", 2000, None, 107),
    ]


# ---------------------------------------------------------------------------
# Hyphenated initials
# ---------------------------------------------------------------------------


def test_hyphenated_initials_open_a_citation_after_commentary():
    note = (
        "13. Ceci étant posé, on pourrait aller plus loin. Comme nous l’avons relevé à "
        "propos des créations des aborigènes : « La propriété intellectuelle sur ces "
        "peintures n’a donc aucune signification » (J.-M. Bruguière, Droit \r\ndes "
        "propriétés intellectuelles, Ellipses, 2018, p. 17). Il en est peut-être de même ici."
    )
    assert _texts([note]) == [
        "J.-M. Bruguière, Droit des propriétés intellectuelles, Ellipses, 2018, p. 17)"
    ]
    for name in ("J.-M. Bruguière,", "P.-B. Hugenholtz,", "J-M. Martin,", "J.-Ch. Dupont,"):
        assert fc._ONSET_RE.match(name), name


# ---------------------------------------------------------------------------
# Spaced-out surnames
# ---------------------------------------------------------------------------

# The Québec article prints its authors' surnames in small capitals, which the
# text layer spaces out.
_SPACED_NOTES = [
    "8. Cette lacune est déplorée par tous les intervenants — sociaux et judiciaires — et\r\n"
    "ce, partout au Canada. Voir Linda M cLe o d , Pour de vraies amours... Prévenir la\r\n"
    "violence familiale, Conseil consultatif canadien de la situation de la femme, Ottawa, "
    "1987, p. 87.",
    "9. Il en va de même dans les cas d’abus sexuels. Voir Christine Ze l le r , Des\r\n"
    "enfants maltraités au Québec ?, Québec, Publications du Québec, 1987, pp. 47 et 50.",
    "17. L.R.Q., c. P-34.1 : voir J.F. Bo u l a is, Loi annotée sur la protection de la\r\n"
    "jeunesse, Montréal, Soquij, 1986.",
    "20. Voir J. F o r t in , Preuve pénale, Montréal, Les Éditions Thémis, 1984, pp. 116-117.",
    "25. Selon les résultats de l’enquête Bonnemain (Ch. Bo n n e m a in , Le contrôle social "
    "de la déviance, sous la direction de A. D a v id o v itc h et J. P r a d e l, Centre "
    "national de la Recherche scientifique, 1978, p. 90.)",
]


def test_a_spaced_out_surname_is_joined_so_the_citation_opens_on_its_author():
    assert _texts(_SPACED_NOTES) == [
        "Linda McLeod, Pour de vraies amours... Prévenir la violence familiale, Conseil "
        "consultatif canadien de la situation de la femme, Ottawa, 1987, p. 87.",
        "Christine Zeller, Des enfants maltraités au Québec ?, Québec, Publications du "
        "Québec, 1987, pp. 47 et 50.",
        "J.F. Boulais, Loi annotée sur la protection de la jeunesse, Montréal, Soquij, 1986.",
        "J. Fortin, Preuve pénale, Montréal, Les Éditions Thémis, 1984, pp. 116-117.",
        "Ch. Bonnemain, Le contrôle social de la déviance, sous la direction de A. "
        "Davidovitch et J. Pradel, Centre national de la Recherche scientifique, 1978, p. 90.)",
    ]


def test_names_and_particles_that_are_not_spaced_out_stay_apart():
    for text in (
        "Y. Wu et al., A study of things, 2001.",
        "M. De La Cruz, Historia, Madrid, 1990.",
        "J. Li y J. Wang, Title of the work, 2003.",
        "A. Le Goff, La civilisation, Paris, 1964.",
        "C. Ze ller, op. cit., p. 48.",
    ):
        assert fc._join_spaced_surnames(text) == text, text


# ---------------------------------------------------------------------------
# French lead-ins
# ---------------------------------------------------------------------------


def test_lire_and_an_authors_title_hand_over_to_the_citation():
    notes = [
        "18. Telle est la solution préconisée notamment par le professeur Hélène D u m o n t,"
        "\r\nLe contrôle judiciaire de la criminalité familiale, Montréal, Les Éditions Thémis, "
        "1986, \r\n233 pages.",
        "19. À ce sujet, lire Jean P in e a u , La famille, Montréal, Les Presses de l’Université "
        "\r\nde Montréal, 1983, pp. 86 à 91.",
    ]
    assert _texts(notes) == [
        "Hélène Dumont, Le contrôle judiciaire de la criminalité familiale, Montréal, "
        "Les Éditions Thémis, 1986, 233 pages.",
        "Jean Pineau, La famille, Montréal, Les Presses de l’Université de Montréal, 1983, "
        "pp. 86 à 91.",
    ]
    # An article alone is no author's title: a funder stays in its sentence.
    funded = "Ce travail est financé par la Fondation Knut, Stockholm, 2015."
    assert fc._citation_start(funded) == 0


# ---------------------------------------------------------------------------
# Classical works
# ---------------------------------------------------------------------------

# A Spanish humanities paper cites ancient authors by work, book and chapter;
# the tagger reads the author and finds no title in any of them.
_CLASSICAL_NOTES = [
    "29. Plinio, Historia Natural, V, 45. Esquilo, Prometeo encadenado, 454.",
    "30. Diodoro de Sicilia, Biblioteca histórica, III, 32, 4.",
    "31. Estrabón, Geografía, XVI, 17.",
    "32. Homero, Odisea, XI, 11.",
    "33. Estrabón, Geografía, XVI, 4, 17.",
]


def test_an_undated_classical_citation_yields_its_author_and_work():
    assert [fc.classical_work(text) for text in _texts(_CLASSICAL_NOTES)] == [
        ("Plinio", "Historia Natural"),
        ("Esquilo", "Prometeo encadenado"),
        ("Diodoro de Sicilia", "Biblioteca histórica"),
        ("Estrabón", "Geografía"),
        ("Homero", "Odisea"),
        ("Estrabón", "Geografía"),
    ]
    # A dated citation is no classical work, whatever its shape.
    assert fc.classical_work("Stoczkowski, Aux origines, 12, 2001.") is None
    assert fc.classical_work("W. Stoczkowski, “Le peintre”, p. 237.") is None


# Five dated works cited in full, so that the notes stand in for the list.
_DATED_NOTES = [
    "1. W. Stoczkowski, Anthropologie naïve, anthropologie savante, Paris, CNRS, 1994.",
    "2. C. Cohen, L’Homme des origines, Paris, Seuil, 1999.",
    "3. R. Bartra, El mito del salvaje, México, FCE, 2011.",
    "4. G. Cadogan, The amazons in antiquity and modern times, London, Pelican, 1998.",
    "5. M. Bost-Fiévet, L’antiquité dans l’imaginaire contemporain, Paris, Garnier, 2014.",
]


def _author_only_ner(self, ref_strings):
    """The tagger on "Estrabón, Geografía, XVI, 17": the author, no title."""
    parsed = []
    for text in ref_strings:
        authors, title, *_ = text.split(", ")
        dated = re.search(r"\b(19|20)\d\d\b", text)
        parsed.append(
            PaperReference(
                bib_id=1,
                title=title if dated else "",
                authors=authors,
                year=int(dated.group()) if dated else None,
                container=None,
                volume=None,
                first_page=None,
            )
        )
    return parsed


async def test_the_notes_path_titles_a_classical_work_the_tagger_misses():
    from bibr.extract.extractor import MetadataExtractor

    extractor = MetadataExtractor(
        _contents([*_DATED_NOTES, *_CLASSICAL_NOTES]),
        llm_client=mock.MagicMock(),
        ref_parse_strategy="ner",
        settings=GlobalSettings(),
    )
    extractor.extract_core_metadata = AsyncMock()
    extractor.metadata = PaperMetadata(doi="", title="T")

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned",
        _author_only_ner,
    ):
        metadata = await extractor.extract_all_metadata()

    assert [(ref.authors, ref.title, ref.text_id) for ref in metadata.references[5:]] == [
        ("Plinio", "Historia Natural", 107),
        ("Esquilo", "Prometeo encadenado", 107),
        ("Diodoro de Sicilia", "Biblioteca histórica", 108),
        ("Estrabón", "Geografía", 109),
        ("Homero", "Odisea", 110),
    ]


# ---------------------------------------------------------------------------
# Titles that open on a year
# ---------------------------------------------------------------------------

# An economics bulletin cites a survey named for the year it covers; the
# tagger reads that year as the work's and the rest of the title as its
# container.
_SURVEY_NOTE = (
    "7. For example, the 1977 Survey of Consumer Finances found that only 8 percent "
    "of homeowners had refinanced. See Thomas A. Durkin and\r\nGregory E. Elliehausen, "
    "1977 Consumer Credit Survey (Board of\r\nGovernors of the Federal Reserve System, "
    "1978), p. 72."
)


def test_a_title_that_opens_on_a_year_is_read_with_its_imprint_year():
    [citation] = _texts([_SURVEY_NOTE])
    assert fc.year_led_title(citation) == ("1977 Consumer Credit Survey", 1978)
    # A title that does not open on a year, or no bracketed imprint year.
    assert fc.year_led_title("J. Smith, Consumer Credit (Chicago, 1978), p. 2.") is None
    assert fc.year_led_title("J. Smith, 1977 Consumer Credit, Chicago, 1978.") is None


def _year_misread_ner(self, ref_strings):
    """The tagger on the survey: its title's year as the year, the rest as container."""
    parsed = _author_only_ner(self, ref_strings)
    for index, text in enumerate(ref_strings):
        if "Consumer Credit Survey" in text:
            parsed[index] = PaperReference(
                bib_id=1,
                title="",
                authors="Thomas A. Durkin and Gregory E. Elliehausen",
                year=1977,
                container="Consumer Credit Survey (Board of Governors of the Federal "
                "Reserve System, 1978),",
                volume=None,
                first_page=None,
                bib_type="journal_article",
            )
    return parsed


async def test_the_notes_path_gives_back_a_title_the_tagger_read_a_year_out_of():
    from bibr.extract.extractor import MetadataExtractor

    extractor = MetadataExtractor(
        _contents([*_DATED_NOTES, _SURVEY_NOTE]),
        llm_client=mock.MagicMock(),
        ref_parse_strategy="ner",
        settings=GlobalSettings(),
    )
    extractor.extract_core_metadata = AsyncMock()
    extractor.metadata = PaperMetadata(doi="", title="T")

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned",
        _year_misread_ner,
    ):
        metadata = await extractor.extract_all_metadata()

    [survey] = metadata.references[5:]
    assert (survey.authors, survey.title, survey.year, survey.container, survey.text_id) == (
        "Thomas A. Durkin and Gregory E. Elliehausen",
        "1977 Consumer Credit Survey",
        1978,
        None,
        107,
    )
