"""A reference list read from the citations in a paper's notes.

The note texts are short excerpts of the dev-set papers that print no
reference list: a French endnote apparatus, a Federal Reserve Bulletin
article, a Spanish humanities paper citing classical works and a Slovak
history paper in ISO 690 style.
"""

from __future__ import annotations

import re
from unittest import mock
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from bibr.config import GlobalSettings
from bibr.extract import footnote_citations as fc
from bibr.field_states import FieldScope, build_field_states
from bibr.models import PaperMetadata, PaperReference
from bibr.paper_contents import (
    CanonicalSection,
    PaperContents,
    PaperSection,
    PaperSentence,
    ReferenceSegmentationAttempt,
    ReferenceYieldReceipt,
)
from bibr.processing_warnings import DESCRIPTIONS, WarningCode


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


def _ref(title: str, authors: str | None = None, **fields) -> PaperReference:
    base = {"bib_id": 1, "first_page": None, "volume": None, "year": None, "container": None}
    return PaperReference(title=title, authors=authors, **{**base, **fields})


# ---------------------------------------------------------------------------
# Which note text is a citation
# ---------------------------------------------------------------------------


def test_endnotes_yield_one_citation_each_and_drop_ibid_lifespans_and_stamps():
    found = fc.note_citations(
        _contents(
            [
                "1. Une référence à l’ouvrage de François Roustang, La fin de la plainte, "
                "Editions Odile\r\nJacob, 2001.",
                "2. Sylvie Le Pelletier-Beaufond, Abécédaire François Roustang, Odile Jacob, "
                "Paris, 2019.\r\n3. Ibid.",
                "7. La Gestalt-théorie dont les fondateurs sont Max Wertheimer (1880-1943), "
                "Wolfgang\r\nKöhler (1887-1967) et Kurt Koffka (1886-1941).",
                "Collège européen de Gestalt-thérapie | Téléchargé le 23/09/2026 sur "
                "https://shs.cairn.info (IP: 31.151.2.36)",
                "11. Phénoménologie de la perception, Paris Gallimard, 1945.",
            ]
        )
    )

    assert [(c.text, c.text_id) for c in found.citations] == [
        ("François Roustang, La fin de la plainte, Editions Odile Jacob, 2001.", 102),
        (
            "Sylvie Le Pelletier-Beaufond, Abécédaire François Roustang, Odile Jacob, Paris, 2019.",
            103,
        ),
        ("Phénoménologie de la perception, Paris Gallimard, 1945.", 106),
    ]
    assert (found.notes, found.citing_notes, found.repeats) == (6, 3, 1)


def test_citations_separated_by_semicolons_split_but_an_imprint_does_not():
    texts = _texts(
        [
            "1. See Glenn B. Canner, James T. Fergus, and Charles A. Luckett,\r\n‘‘Home Equity "
            "Lines of Credit,’’ Federal Reserve Bulletin, vol. 74\r\n(June 1988), pp. 361–73; "
            "Glenn B. Canner, Charles A. Luckett, and\r\nThomas A. Durkin, ‘‘Home Equity "
            "Lending,’’ Federal Reserve Bulletin, vol. 75 (May 1989), pp. 333–44.",
            "6 E. g. ABRAMS, Lynn. At home and in the Family. In SIMONTON, Deborah (ed.). The "
            "Routledge History of Women since 1700. London; New York: Routledge, 2006, p. 14–53.",
        ]
    )

    assert [text[:36] for text in texts] == [
        "Glenn B. Canner, James T. Fergus, an",
        "Glenn B. Canner, Charles A. Luckett,",
        "ABRAMS, Lynn. At home and in the Fam",
    ]
    assert "London; New York: Routledge, 2006" in texts[2]


def test_a_citation_after_a_locator_is_no_imprint_of_the_one_before():
    # Note 9 of a Québec paper on conjugal violence: "Rapport : Colloques ..."
    # reads like "Place: Publisher", but it follows a repeat's page, not a place.
    assert _texts(
        [
            "9. Voir : L. M c Le o d , op. cit., supra, note 8, p. 7; Rapport : Colloques "
            "régionaux\r\nsur la violence, Gouvernement du Québec, ministère de la Justice, "
            "Direction des \r\ncommunications, 1980, p. 23; Politique d'intervention en matière "
            "de violence conjugale, \r\nQuébec, ministère de la Justice, ministère du Solliciteur "
            "général, 1986, p. 9."
        ]
    ) == [
        "Rapport : Colloques régionaux sur la violence, Gouvernement du Québec, ministère de la "
        "Justice, Direction des communications, 1980, p. 23",
        "Politique d'intervention en matière de violence conjugale, Québec, ministère de la "
        "Justice, ministère du Solliciteur général, 1986, p. 9.",
    ]
    # The Slovak paper's note 12: after a place, the next clause is its imprint.
    [text] = _texts(
        [
            "12 BRADBURY, Bettina. Wife to Widow. Lives, Laws, and Politics in Nineteenth-Century "
            "Montreal. Vancouver; Toronto: UBC Press, 2011. ISBN 978-07-7481-952-7 must be named."
        ]
    )
    assert text.endswith("Montreal. Vancouver; Toronto: UBC Press, 2011")


def test_a_lead_in_after_a_repeat_opens_the_next_work_whatever_its_byline():
    # The Québec paper's notes 10 and 23: "Voir aussi" after a repeat opens a
    # work that starts with its title, or with a letter-spaced name.
    assert _texts(
        [
            "10. Voir pour le Canada en général : L. M c Le o d , op. cit., supra, note 8, p. 3. "
            "Voir \r\naussi Rapport fédéral-provincial territorial sur les femmes battues, présenté "
            "à la réunion \r\ndes ministres responsables de la Condition féminine, "
            "Niagara-on-the-Lake, les 28-30 mai\r\n1984, ministère des Approvisionnements et "
            "Services Canada, 1984; et pour le Québec en\r\nparticulier : Politique d ’intervention "
            "en matière de violence conjugale, op. cit., supra, \r\nnote 9.",
            "23. A. M cGill iv r a y , loc. cit., supra, note 21. Voir aussi, Linda M c Le o d , "
            "La\r\nfemme battue au Canada : un cercle vicieux, Ottawa, Conseil consultatif "
            "canadien de la \r\nsituation de la femme, ministre des Approvisionnements et "
            "Services, 1980, p. 29.",
        ]
    ) == [
        "Rapport fédéral-provincial territorial sur les femmes battues, présenté à la réunion des "
        "ministres responsables de la Condition féminine, Niagara-on-the-Lake, les 28-30 mai "
        "1984, ministère des Approvisionnements et Services Canada, 1984",
        "Linda McLeod, La femme battue au Canada : un cercle vicieux, Ottawa, Conseil "
        "consultatif canadien de la situation de la femme, ministre des Approvisionnements et "
        "Services, 1980, p. 29.",
    ]


@pytest.mark.parametrize(
    ("note", "citation_start"),
    [
        (
            "3 El mito del héroe civilizador es uno de los ejemplos recogidos\r\npor W. "
            "Stoczkowski, Aux origines de l’humanité, Anthologie, Paris, Pocket, 1996,\r\np. 45- 94.",
            "W. Stoczkowski, Aux origines",
        ),
        (
            "38 Delon relaciona el personaje de Conan the Barbarian de John Milius (1982) con los "
            "Hércules cinematográficos, G. Delon, “Avatars du péplum?ˮ, en M. Bost-Fiévet y S. "
            "Provini (éd.), L’antiquité dans l’imaginaire contemporain, p. 65-79 (p. 75).",
            "G. Delon, “Avatars",
        ),
        (
            "4 The topic of female offenders in Victorian England was covered by WILLIAMS, Lucy. "
            "\r\nWayward Women: Female Offending in Victorian England. Barnsley: Pen and Sword "
            "History, \r\n2016.",
            "WILLIAMS, Lucy. Wayward",
        ),
        (
            "5 Véase al respecto la valiosa introducción de M. Bost-Fiévet y S. Provini, “L’Antiquité "
            "gréco-latine dans l’imaginaire contemporainˮ, en M. Bost-Fiévet y S. Provini (éd.), "
            "L’antiquité dans l’imaginaire contemporain, Paris, Classiques Garnier, 2014, p. 15-34.",
            "M. Bost-Fiévet y S. Provini, “L’Antiquité",
        ),
        # A corporate author is no commentary to cut.
        (
            "6. U.S. Department of Housing and Urban Development, U.S.\r\nHousing Market "
            "Conditions, table 29, ‘‘Homeownership Rates by Age\r\nof Householder: 1982–Present’’ "
            "(3rd quarter 1999).",
            "U.S. Department of Housing",
        ),
    ],
)
def test_commentary_before_a_citation_is_cut(note, citation_start):
    [text] = _texts([note])

    assert text.startswith(citation_start)


def test_sentence_breaks_split_citations_but_not_abbreviations():
    assert _texts(
        [
            "21 Plinio, Historia Natural, V, 45. Esquilo, Prometeo encadenado, 454.",
            "6. James Gibson, Approche écologique de la perception visuelle (1979), tr. fr. "
            "Olivier\r\nPutois, Editions Dehors, 2014.",
            "19 J. Bestard y J. Contreras, Bárbaros, paganos y primitivos, Barcelona, Barcanova, "
            "1987, p. 180-193. Estos autores han demostrado la conexión entre los conceptos.",
        ]
    ) == [
        "Plinio, Historia Natural, V, 45",
        "Esquilo, Prometeo encadenado, 454.",
        "James Gibson, Approche écologique de la perception visuelle (1979), tr. fr. Olivier "
        "Putois, Editions Dehors, 2014.",
        # The commentary after the citation is cut off.
        "J. Bestard y J. Contreras, Bárbaros, paganos y primitivos, Barcelona, Barcanova, 1987, "
        "p. 180-193",
    ]


def test_a_conjunction_before_an_iso_690_byline_opens_a_new_work():
    # The Slovak paper's note 13: the second work's byline is a
    # sentence of its own, and the repeat after the last "or" is cut off.
    assert _texts(
        [
            "13 See VAN POPPEL, Frans. Widows, Widowers and Remarriage. In Population Studies, "
            "1995, year 49, no. 3, p. 421–441. ISSN 1477-4747 or BRADBURY, Bettina. Surviving "
            "as a Widow in 19th-century Montreal. In Urban History Review, 1989, \r\nyear 17, "
            "no. 3, p. 148–160. ISSN 0703-0428 or the abovementioned CURRAN, ref. 8."
        ]
    ) == [
        "VAN POPPEL, Frans. Widows, Widowers and Remarriage. In Population Studies, 1995, "
        "year 49, no. 3, p. 421–441. ISSN 1477-4747",
        "BRADBURY, Bettina. Surviving as a Widow in 19th-century Montreal. In Urban History "
        "Review, 1989, year 17, no. 3, p. 148–160",
    ]


def test_commentary_after_a_repeat_hands_over_at_a_colon():
    # The Slovak paper's note 12: "ABRAMS, ref. 6." is a repeat, and the next sentence
    # names the next work after a colon.
    assert _texts(
        [
            "12 Also WINKELHOFER, \r\nRef. 5, who wrote about aristocratic women and ABRAMS, "
            "\r\nref. 6. Form authors dealing with the topic of widowhood: MACHTEMES, Ursula. "
            "Leben \r\nzwischen Trauer und Pathos. Osnabrück: \r\nUniversitätsverlag Rasch, "
            "2001. ISBN 3-934005-98-5 must be mentioned."
        ]
    ) == [
        "MACHTEMES, Ursula. Leben zwischen Trauer und Pathos. Osnabrück: Universitätsverlag "
        "Rasch, 2001"
    ]


def test_commentary_hands_over_at_a_colon_and_at_segun():
    assert _texts(
        [
            # The Slovak paper's note 15
            "15 There are several works which must be mentioned in connection with this research "
            "made \r\nin the Czech Republic: KAZLEPKA, Zdeněk. Ostrov italského vkusu. Brno: "
            "Barrister Principal, 2011.",
            # The Spanish paper's note 22
            "22 Homero, Himnos, XX, 5. Se criaban como fieras en grutas o en bosques, según "
            "Vitruvio, Arquitectura, II, 1, 1-2.",
        ]
    ) == [
        "KAZLEPKA, Zdeněk. Ostrov italského vkusu. Brno: Barrister Principal, 2011.",
        "Homero, Himnos, XX, 5",
        "Vitruvio, Arquitectura, II, 1, 1-2.",
    ]


def test_a_period_inside_a_quoted_title_is_no_sentence_break():
    # The Spanish paper's note 32: the third work's title holds a period ("museums.
    # Exhibiting"), so the second work ends before its byline, not inside it.
    assert _texts(
        [
            "32 Cf. W. Stoczkowski “Homme préhistorique et imagination conditionnéeˮ, en \r\n"
            "G. Lagardère (éd.), Peintres d’un monde disparu. La préhistoire vue par les artistes "
            "de \r\nla fin du xixe\r\n siècle à nos jours, Solutré, Musée de Préhistoire de "
            "Solutré, 1990, p. 61-71 \r\n(p. 56). También R. Bartra, El salvaje en el espejo, "
            "Barcelona, Destino, 1996, p. 40 y 54. \r\nS. Moser, “Representing archaeological "
            "Knowledge in museums. Exhibiting human \r\norigins and strategies for changeˮ, "
            "Public Archaeology, 3 (2003), p. 3-20 (p. 12)."
        ]
    )[1:] == [
        "R. Bartra, El salvaje en el espejo, Barcelona, Destino, 1996, p. 40 y 54",
        "S. Moser, “Representing archaeological Knowledge in museums. Exhibiting human origins "
        "and strategies for changeˮ, Public Archaeology, 3 (2003), p. 3-20 (p. 12).",
    ]


def _paged_contents(notes: list[tuple[str | None, int, str]]) -> PaperContents:
    """Like :func:`_contents`, with each note's printed mark and page: ``(label, page, text)``."""
    contents = _contents([text for _, _, text in notes])
    notes_by_section = dict(enumerate(notes, start=2))
    for section in contents.sections:
        if section.section_id in notes_by_section:
            section.footnote_label = notes_by_section[section.section_id][0]
    for sentence in contents.sentences:
        sentence.page_number = notes_by_section.get(sentence.section_id, (None, 1))[1]
    return contents


def test_a_note_carried_over_to_the_next_page_is_read_as_one():
    # The Slovak paper's note 14 breaks off inside a bracketed translation; the parser
    # makes its rest on the next page a note without a mark.
    found = fc.note_citations(
        _paged_contents(
            [
                (
                    "14",
                    3,
                    "14 If we concentrate only on the 19th century the works of Alice Velková "
                    "must be mentioned: \r\ne. g. VELKOVÁ, Alice. Sebevědomé, nebo zoufalé? Vdovy "
                    "hospodařící na venkovských \r\nusedlostech v první polovině 19. století "
                    "[Confident or Desperate? Widows Farming on Rural",
                ),
                (
                    None,
                    4,
                    "Estates in the First Half of the 19th Century]. In VOJÁČEK, Milan (ed.). "
                    "Reflexe a sebereflexe \r\nženy v české národní elitě 2. poloviny 19. století. "
                    "Praha, 2007, p. 321–340. ISBN 978-80-86712-45-1. In general, the sumarizing "
                    "\r\narticle by SKOŘEPOVÁ, Markéta. Vdovství v tradiční venkovské společnosti. "
                    "In Historická demografie, 2011, year 35, no.1, p. 1–31. ISSN 0323-0937 must "
                    "be mentioned.",
                ),
            ]
        )
    )

    assert (found.notes, found.citing_notes) == (1, 1)
    assert [(c.text_id, c.text[:40]) for c in found.citations] == [
        (102, "VELKOVÁ, Alice. Sebevědomé, nebo zoufalé"),
        (102, "SKOŘEPOVÁ, Markéta. Vdovství v tradiční "),
    ]
    assert "Rural Estates in the First Half" in found.citations[0].text
    assert found.citations[0].text.endswith("Praha, 2007, p. 321–340. ISBN 978-80-86712-45-1")


def test_a_note_breaking_off_in_commentary_stays_apart_from_the_next_page():
    # The Spanish paper's note 34 breaks off in commentary ("... este « mundo hostil »
    # de"); read as one, the commentary would hide the citation on page 18.
    found = fc.note_citations(
        _paged_contents(
            [
                (
                    "34",
                    17,
                    "34 Mosquion, Fr. 6. 3-15. Incluso, uno de los libros de divulgación más "
                    "influyentes en la \r\nconstrucción del imaginario prehistórico, describe e "
                    "ilustra este « mundo hostil » de",
                ),
                (
                    None,
                    18,
                    "lucha « terrible » entre hombres y animales, M. M. L. Figuier y W. F. A. "
                    "Zimmermann, El \r\nmundo antes de la creación del hombre. Origen del hombre. "
                    "Problemas y maravillas de la \r\nnaturaleza, Barcelona, Montaner y Simon, "
                    "1871, p. 134.",
                ),
            ]
        )
    )

    assert found.notes == 2
    assert [(c.text_id, c.text[:30]) for c in found.citations] == [
        (103, "W. F. A. Zimmermann, El mundo ")
    ]


def test_an_undated_iso_690_byline_with_a_locator_is_a_short_citation():
    found = fc.note_citations(
        _contents(
            [
                # The Slovak paper's notes 2 and 95: "year 26" is a volume, not a date.
                "2 BLANCHARD, Rae. Richard Steele and the Status of Women. In Studies in "
                "Philology, year \r\n26, no. 3, p. 322–355. ISSN 0039-3738.",
                "95 MZA, G 169, c. 308, i. n. 233; COLLALTO, Marie Therese. Erlebtes und "
                "geschautes, p. 348.",
                # An archive signature and a byline with no container or locator.
                "29 MZA, G 169, c. 320, i. n. 441.",
                "30 SMITH, John. Personal communication with the author.",
            ]
        )
    )

    assert [(c.text, c.full) for c in found.citations] == [
        (
            "BLANCHARD, Rae. Richard Steele and the Status of Women. In Studies in Philology, "
            "year 26, no. 3, p. 322–355. ISSN 0039-3738.",
            False,
        ),
        ("COLLALTO, Marie Therese. Erlebtes und geschautes, p. 348.", False),
    ]
    # Undated citations never count toward the trigger.
    assert found.citing_notes == 0


def test_a_short_form_without_year_is_dropped_and_the_next_citation_kept():
    [text] = _texts(
        [
            "Cf. W. Stoczkowski, Anthropologie naïve, anthropologie savante. Y también, \r\nW. "
            "Stoczkowski, “Le peintre et l’homme préhistorique: l’hypothèse exprimée au pinceauˮ, "
            "en A. Ducros y J. Ducros (éd.), L’homme préhistorique. Images et imaginaire, p. "
            "233-241 (p. 235)."
        ]
    )

    assert text.startswith("W. Stoczkowski, “Le peintre")


@pytest.mark.parametrize(
    "clause",
    [
        "Ibid., I, 24, 6.",
        "Id., p. 75.",
        "L. McLeod, op. cit., note 8, p. 12.",
        "APPONYI, ref. 24.",
        "Linda MacLeod, précitée, note 8.",
    ],
)
def test_repeat_citations(clause):
    assert fc.is_repeat_citation(clause)


def test_a_full_citation_is_no_repeat():
    assert not fc.is_repeat_citation(
        "P. Payen, “L’Antiquité après l’Antiquité”, Anabases 1 (2002), p. 5-13."
    )


def test_supra_inside_a_hyphenated_title_word_is_no_repeat():
    """A title opening "Supra-national" is a work, not a pointer back; "supra"
    and "infra" standing alone still are."""
    note = "7 Supra-national Law and Its Discontents, Oxford: Clarendon Press, 2001."

    assert [(c.text, c.full) for c in fc.note_citations(_contents([note])).citations] == [
        ("Supra-national Law and Its Discontents, Oxford: Clarendon Press, 2001.", True)
    ]
    assert not fc.is_repeat_citation("Infra-red Astronomy in Practice, Paris: Seuil, 1988.")
    assert fc.is_repeat_citation("L. McLeod, op. cit., supra, note 8, p. 3.")
    assert fc.is_repeat_citation("Voir (supra) note 12.")
    assert fc.is_repeat_citation("Smith, infra, p. 4.")


@pytest.mark.parametrize(
    "clause",
    [
        "Note. All survey data in this and the following tables are weighted, 1998.",
        "Refinancing activity in years preceding 1998 is not fully reflected in this table.",
        "The incidence of refinancing was lower in Board-sponsored household surveys in the "
        "1970s and 1980s.",
        "Kurt Lewin (1890-1947), psychologue américain spécialiste du comportement.",
        "A portion of the funds used for ‘‘home improvement’’ may in fact have been spent in 1999.",
        "Lo mismo ocurre en X-Men 2 de Brian Signer (2003), donde la superheroina Tormenta explica.",
    ],
)
def test_table_notes_and_commentary_are_no_citations(clause):
    assert not fc.looks_like_citation(clause)


# ---------------------------------------------------------------------------
# When the notes stand in for the reference list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("listed", "citing_notes", "expected"),
    [
        (0, 4, False),
        (0, 5, True),
        (1, 14, False),
        (1, 15, True),
        (2, 30, True),
        (3, 100, False),
    ],
)
def test_trigger_thresholds(listed, citing_notes, expected):
    found = fc.NoteCitations(citations=(), notes=100, citing_notes=citing_notes, repeats=0)

    assert fc.notes_replace_list(found, listed) is expected


def test_a_few_citing_notes_beside_a_short_list_do_not_trigger():
    # A scanned 1970s paper: a one-entry list and notes pointing into the text.
    found = fc.note_citations(
        _contents(
            [
                "1 See references in TIWGrPh p. 462.",
                "2 Most recently by Aagesen 1973; see also Hansen, Studies, 1971, p. 12.",
            ]
        )
    )

    assert not fc.notes_replace_list(found, 1)


# ---------------------------------------------------------------------------
# One reference per cited work
# ---------------------------------------------------------------------------


def test_short_forms_fold_into_the_first_full_citation():
    refs = fc.collapse_repeats(
        [
            _ref(
                "Le peintre et l’homme préhistorique: l’hypothèse exprimée au pinceau",
                "W. Stoczkowski",
                container="L’homme préhistorique",
                text_id=5,
            ),
            _ref("Anthropologie naïve, anthropologie savante", "W. Stoczkowski", year=1994),
            _ref("Le peintre et l’homme préhistorique", "W. Stoczkowski", text_id=9),
        ]
    )

    assert [(ref.title[:22], ref.text_id) for ref in refs] == [
        ("Le peintre et l’homme ", 5),
        ("Anthropologie naïve, a", None),
    ]


def test_the_fuller_parse_of_a_work_takes_the_first_citations_place():
    [ref] = fc.collapse_repeats(
        [
            _ref("Bad Hair Days in the Paleolithic", "J. C. Berman", text_id=3),
            _ref(
                "Bad Hair Days in the Paleolithic",
                "J. Berman",
                year=1999,
                container="American Anthropologist",
                volume="101",
                text_id=8,
            ),
        ]
    )

    assert (ref.year, ref.text_id) == (1999, 3)


def test_listed_references_are_kept_and_their_note_citations_dropped():
    listed = [_ref("Home Equity Lending", "Glenn B. Canner")]

    added = fc.collapse_repeats(
        [
            _ref("Home Equity Lending", "Glenn B. Canner, Charles A. Luckett"),
            _ref("Mortgage Refinancing", "Glenn B. Canner"),
        ],
        listed,
    )

    assert [ref.title for ref in added] == ["Mortgage Refinancing"]


def test_a_title_opening_a_later_one_of_another_year_is_another_work():
    # The Federal Reserve Bulletin article's notes 1 and 3: the 1994 survey article is
    # no short form of the 1989 article whose title opens its own.
    refs = fc.collapse_repeats(
        [
            _ref("Home Equity Lending", "Glenn B. Canner, Charles A. Luckett", year=1989),
            _ref(
                "Home Equity Lending: Evidence from Recent Surveys",
                "Glenn B. Canner, Charles A. Luckett, and Thomas A. Durkin",
                year=1994,
            ),
            _ref("Home Equity Lending", "Canner and Luckett"),
        ]
    )

    assert [(ref.title, ref.year) for ref in refs] == [
        ("Home Equity Lending", 1989),
        ("Home Equity Lending: Evidence from Recent Surveys", 1994),
    ]


def test_one_title_by_different_authors_is_two_works():
    refs = fc.collapse_repeats([_ref("Introduction", "A. Smith"), _ref("Introduction", "B. Jones")])

    assert len(refs) == 2


# ---------------------------------------------------------------------------
# Through the extractor
# ---------------------------------------------------------------------------

_NOTES = [
    "1. François Roustang, La fin de la plainte, Editions Odile Jacob, 2001.",
    "2. Sylvie Le Pelletier-Beaufond, Abécédaire François Roustang, Odile Jacob, Paris, 2019.",
    "3. Ibid.",
    "4. James Gibson, Approche écologique de la perception visuelle, Editions Dehors, 2014.",
    "5. Frederick Perls, Manuel de Gestalt-thérapie, ESF, 2003.",
    "6. Olivier Putois, La nature, notes de cours du Collège de France, Seuil, 1995.",
    # A later short form of note 1: folded into its first citation.
    "7. François Roustang, « La fin de la plainte », p. 12.",
]


def _fake_ner(self, ref_strings):
    """Parse "Author, Title, ..., Year." the way the NER parser would."""
    parsed = []
    for text in ref_strings:
        authors, title, *_ = text.split(", ")
        year = re.search(r"\b(19|20)\d\d\b", text)
        parsed.append(_ref(title, authors, year=int(year.group()) if year else None, publisher="P"))
    return parsed


def _extractor(notes: list[str], *, settings: GlobalSettings | None = None):
    from bibr.extract.extractor import MetadataExtractor

    contents = _contents(notes)
    extractor = MetadataExtractor(
        contents,
        llm_client=mock.MagicMock(),
        ref_parse_strategy="ner",
        settings=settings or GlobalSettings(),
    )
    extractor.extract_core_metadata = AsyncMock()
    extractor.metadata = PaperMetadata(doi="", title="T")
    return extractor


def _codes(contents) -> list[str]:
    return [str(warning.code) for warning in contents.processing_warnings]


async def test_without_a_reference_section_the_notes_become_the_reference_list():
    extractor = _extractor(_NOTES)

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned", _fake_ner
    ):
        metadata = await extractor.extract_all_metadata()

    assert [(ref.bib_id, ref.authors, ref.text_id) for ref in metadata.references] == [
        (1, "François Roustang", 102),
        (2, "Sylvie Le Pelletier-Beaufond", 103),
        (3, "James Gibson", 105),
        (4, "Frederick Perls", 106),
        (5, "Olivier Putois", 107),
    ]
    assert _codes(extractor.contents) == [
        WarningCode.REF_FOOTNOTE_CITATIONS,
        WarningCode.REF_SECTION_NOT_FOUND,
    ]
    receipt = extractor.contents.reference_yield_receipt
    assert [(a.strategy, a.selected) for a in receipt.attempts] == [("footnotes", True)]
    assert (receipt.parsed_count, receipt.valid_count) == (6, 5)


# The French endnote apparatus, notes 4 and 5: the tagger finds nothing in
# either citation, whose titles only the quotes mark.
_QUOTED_NOTES = [
    "4. « Au commencement était l’hypnose », Conférences et débats, septembre 2005.",
    "5. « Indifférence au succès » in L’éloge du risque dans le soin psychiatrique, "
    "Marcel Sassolas, ERES, 2006.",
]


def test_a_citation_opening_on_a_quoted_title_yields_that_title_and_its_year():
    assert [fc.quoted_title(text) for text in _texts(_QUOTED_NOTES)] == [
        ("Au commencement était l’hypnose", 2005),
        ("Indifférence au succès", 2006),
    ]
    assert fc.quoted_title("Homero, Himnos, XX, 5") is None
    assert fc.quoted_title("« » 2001") is None


async def test_a_citation_the_tagger_finds_nothing_in_keeps_its_quoted_title():
    def tagger_misses_quoted_titles(self, ref_strings):
        parsed = _fake_ner(self, ref_strings)
        return [
            None if text.startswith("«") else ref
            for text, ref in zip(ref_strings, parsed, strict=True)
        ]

    extractor = _extractor([*_NOTES[:6], *_QUOTED_NOTES])

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned",
        tagger_misses_quoted_titles,
    ):
        metadata = await extractor.extract_all_metadata()

    assert [(ref.title, ref.authors, ref.year, ref.text_id) for ref in metadata.references[5:]] == [
        ("Au commencement était l’hypnose", None, 2005, 108),
        ("Indifférence au succès", None, 2006, 109),
    ]


async def test_with_the_setting_off_a_missing_section_stays_empty():
    settings = GlobalSettings(REF_FOOTNOTE_CITATIONS=False)
    extractor = _extractor(_NOTES, settings=settings)

    metadata = await extractor.extract_all_metadata()

    assert metadata.references == []
    assert _codes(extractor.contents) == [WarningCode.REF_SECTION_NOT_FOUND]


async def test_too_few_citing_notes_keep_the_missing_section_warning():
    extractor = _extractor(_NOTES[:3])

    metadata = await extractor.extract_all_metadata()

    assert metadata.references == []
    assert _codes(extractor.contents) == [WarningCode.REF_SECTION_NOT_FOUND]


# A Federal Reserve Bulletin article citing in its notes, where the locator
# finds no reference list (excerpts of notes 1, 5, 6, 8 and 14).
_BULLETIN_NOTES = [
    "1. See Glenn B. Canner, James T. Fergus, and Charles A. Luckett, ‘‘Home Equity Lines of "
    "Credit,’’ Federal Reserve Bulletin, vol. 74 (June 1988), pp. 361–73.",
    "5. See Dean M. Maki, ‘‘Household Debt and the Tax Reform Act of 1986,’’ American Economic "
    "Review (forthcoming), for an analysis of the substitution of mortgage for consumer debt.",
    "6. U.S. Department of Housing and Urban Development, U.S. Housing Market Conditions, "
    "table 29, ‘‘Homeownership Rates by Age of Householder: 1982–Present’’ (3rd quarter 1999).",
    "8. For additional information, see Glenn B. Canner and Dolores S. Smith, ‘‘Expanded HMDA "
    "Data on Residential Lending: One Year Later,’’ Federal Reserve Bulletin, vol. 78 "
    "(November 1992), pp. 801–24.",
    "14. Tax data for the calculations came from David Campbell and Michael Parisi, "
    "‘‘Individual Income Tax Returns, 1997,’’ Statistics of Income Bulletin (Fall 1999), "
    "pp. 8–45.",
]


async def test_notes_standing_in_for_a_missing_list_keep_the_missing_section_warning():
    """The notes stand in, but no list was located: the paper may print one
    the locator missed, so REF_SECTION_NOT_FOUND stays and says what happened."""
    assert fc.note_citations(_contents(_BULLETIN_NOTES)).citing_notes == 5
    extractor = _extractor(_BULLETIN_NOTES)

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned", _fake_ner
    ):
        metadata = await extractor.extract_all_metadata()

    assert len(metadata.references) == 5
    warnings = extractor.contents.processing_warnings
    assert _codes(extractor.contents) == [
        WarningCode.REF_FOOTNOTE_CITATIONS,
        WarningCode.REF_SECTION_NOT_FOUND,
    ]
    assert warnings[1].message.endswith("; the references were read from the notes")
    # The list is still extracted; both codes qualify it.
    bib = build_field_states(
        present={"bib": True},
        sources={},
        scope=FieldScope(references_source="footnotes"),
        issues=[],
        warnings=warnings,
    )["bib"]
    assert (bib.state, bib.source, bib.issues) == (
        "extracted",
        "footnotes",
        ("REF_FOOTNOTE_CITATIONS", "REF_SECTION_NOT_FOUND"),
    )


async def test_a_failed_notes_parse_keeps_the_missing_section_warning():
    extractor = _extractor(_BULLETIN_NOTES)

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned",
        side_effect=RuntimeError("tagger crashed"),
    ):
        metadata = await extractor.extract_all_metadata()

    assert metadata.references == []
    missing = [
        warning
        for warning in extractor.contents.processing_warnings
        if warning.code == WarningCode.REF_SECTION_NOT_FOUND
    ]
    assert [warning.message.rsplit("; ", 1)[-1] for warning in missing] == [
        "the reference list is empty"
    ]


def test_the_notes_warning_does_not_say_the_paper_prints_no_list():
    """A numbered list typed as notes, or one the locator missed, is exported
    from the notes too: the description names the locator, not the paper."""
    description = DESCRIPTIONS[WarningCode.REF_FOOTNOTE_CITATIONS]

    assert description.startswith("No reference list was located, or ")
    assert "prints no reference list" not in description
    assert (
        "unless the references were read from the notes"
        in (DESCRIPTIONS[WarningCode.REF_SECTION_NOT_FOUND])
    )


async def test_native_metadata_without_a_reference_section_reads_the_notes():
    from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

    contents = _contents(_NOTES)
    ready: list[list[PaperReference]] = []

    with mock.patch(
        "bibr.extract.ref_extractor.ReferenceExtractor._parse_references_ner_aligned", _fake_ner
    ):
        metadata = await _resolve_preparsed_references(
            contents,
            PaperMetadata(doi="", title="T"),
            "deadbeef",
            mock.MagicMock(),
            None,
            "ner",
            settings=GlobalSettings(),
            on_references_ready=ready.append,
        )

    assert [ref.text_id for ref in metadata.references] == [102, 103, 105, 106, 107]
    assert ready == [metadata.references]
    assert _codes(contents) == [
        WarningCode.REF_FOOTNOTE_CITATIONS,
        WarningCode.REF_SECTION_NOT_FOUND,
    ]
    assert contents.processing_warnings[-1].message.endswith(
        "; the references were read from the notes"
    )


async def test_a_reference_list_of_three_entries_is_never_supplemented():
    extractor = _extractor(_NOTES)
    listed = [_ref(f"Listed work {n}", "A. Author", year=2000) for n in (1, 2, 3)]
    extractor.refs.extract = AsyncMock(return_value=listed)
    extractor.refs.extract_from_notes = AsyncMock()

    refs = await extractor._extract_references(pd.DataFrame({"text": ["x"]}))

    assert refs is listed
    extractor.refs.extract_from_notes.assert_not_called()


async def test_a_one_entry_list_is_supplemented_only_past_fifteen_citing_notes():
    extractor = _extractor(_NOTES)
    listed = [_ref("Legal acts", None)]
    extractor.refs.extract = AsyncMock(return_value=listed)

    refs = await extractor._extract_references(pd.DataFrame({"text": ["x"]}))

    # Five citing notes do not outweigh a located list fifteen times.
    assert refs is listed


# ---------------------------------------------------------------------------
# A failure or a runaway note never costs the paper
# ---------------------------------------------------------------------------


def _notes_fail(contents):
    raise RuntimeError("note scan crashed")


async def test_a_failure_reading_the_notes_leaves_the_missing_section_as_before():
    extractor = _extractor(_BULLETIN_NOTES)

    with mock.patch("bibr.extract.ref_extractor.note_citations", _notes_fail):
        metadata = await extractor.extract_all_metadata()

    assert metadata.references == []
    assert _codes(extractor.contents) == [WarningCode.REF_SECTION_NOT_FOUND]
    assert extractor.contents.processing_warnings[0].message.endswith(
        "; the reference list is empty"
    )


async def test_a_failure_reading_the_notes_on_the_native_path_leaves_the_missing_section():
    from bibr.pipeline.stages.post_parse import _resolve_preparsed_references

    contents = _contents(_BULLETIN_NOTES)

    with mock.patch("bibr.extract.ref_extractor.note_citations", _notes_fail):
        metadata = await _resolve_preparsed_references(
            contents,
            PaperMetadata(doi="", title="T"),
            "deadbeef",
            mock.MagicMock(),
            None,
            "ner",
            settings=GlobalSettings(),
        )

    assert metadata.references == []
    assert _codes(contents) == [WarningCode.REF_SECTION_NOT_FOUND]


async def test_a_failure_reading_the_notes_keeps_a_short_located_list():
    extractor = _extractor(_BULLETIN_NOTES)
    listed = [_ref("Legal acts", None)]
    extractor.refs.extract = AsyncMock(return_value=listed)

    with mock.patch("bibr.extract.ref_extractor.note_citations", _notes_fail):
        refs = await extractor._extract_references(pd.DataFrame({"text": ["x"]}))

    assert refs is listed


# One note of commentary running to thousands of characters, citing as it
# goes (the shape of a law-review note): the split scan grows faster than
# the note, so a note this long is not scanned at all.
_COMMENTARY = (
    "The court held that the doctrine applies where the parties agreed, as discussed in the "
    "literature more generally, see J. Kany-Turpin, “Notre passé”, in A. Moser (ed.), Oxford: "
    "Clarendon Press, 2005, p. 7. "
)


def test_a_note_too_long_to_be_a_citation_apparatus_is_not_scanned():
    long_note = "12. " + _COMMENTARY * 60
    assert 8_000 < len(long_note) < 20_000
    scanned: list[int] = []
    split = fc._split_citations

    def spy(note):
        scanned.append(len(note))
        return split(note)

    with mock.patch.object(fc, "_split_citations", spy):
        found = fc.note_citations(_contents([*_BULLETIN_NOTES, long_note]))

    assert max(scanned) < 1_000
    assert len(scanned) == len(_BULLETIN_NOTES)
    # The other notes are read as before, and the long one still counts as read.
    assert (found.citing_notes, found.notes) == (5, 6)
    assert [citation.text_id for citation in found.citations] == [102, 103, 104, 105, 106]


# ---------------------------------------------------------------------------
# Linking: a note mark is no reference number
# ---------------------------------------------------------------------------

# A note that first cites two works leaves its number ambiguous, and most
# notes of the Spanish paper do: the numbering of the notes-derived list then falls
# back to positions, where the note mark ^{3} would name the third work.
_LINKED_NOTES = [
    (
        101,
        "1. A. Smith, “First work”, Revue, 1990, p. 3; B. Jones, “Second work”, London: Press, 1991.",
    ),
    (
        102,
        "2. C. Brown, “Third work”, Paris: Seuil, 1992; D. White, “Fourth work”, Madrid: Cátedra, 1993.",
    ),
    (103, "3. E. Green, “Fifth work”, Oxford: Clarendon, 1994."),
]


async def _link_note_references(references_from_notes: bool) -> list[tuple[int, int]]:
    from bibr.structure.citation_linker import detect_bib_xrefs_with_receipt

    sections = [
        PaperSection(0, "Root", 0, None),
        PaperSection(1, "Introduction", 1, 0, CanonicalSection.INTRODUCTION, 1.0),
        PaperSection(3, "Notes", 1, 0, CanonicalSection.FOOTNOTE, 1.0),
    ]
    body = [
        "The gesture was read as a sign of progress.^{3}",
        "Earlier readings differ.^{1}",
        "Green (1994) disagrees.",
    ]
    sentences = [PaperSentence(index, text, 1, 1) for index, text in enumerate(body, start=1)]
    sentences += [PaperSentence(text_id, text, 3, 2) for text_id, text in _LINKED_NOTES]
    works = [
        ("A. Smith", 1990, "First work", 101),
        ("B. Jones", 1991, "Second work", 101),
        ("C. Brown", 1992, "Third work", 102),
        ("D. White", 1993, "Fourth work", 102),
        ("E. Green", 1994, "Fifth work", 103),
    ]
    references = [
        _ref(title, authors, year=year, bib_id=bib_id, text_id=text_id)
        for bib_id, (authors, year, title, text_id) in enumerate(works, start=1)
    ]
    xrefs, _ = await detect_bib_xrefs_with_receipt(
        sentences, sections, references, references_from_notes=references_from_notes
    )
    return sorted((xref.text_id, xref.xref_id) for xref in xrefs)


async def test_note_marks_do_not_link_to_references_read_from_the_notes():
    # Read as reference numbers, ^{3} links the third work (Brown), not Green.
    assert (1, 3) in await _link_note_references(False)
    # Read from the notes, only the author-year citation links.
    assert await _link_note_references(True) == [(3, 5)]


def test_references_from_notes_is_read_off_the_yield_receipt():
    from bibr.pipeline.stages.post_parse import _references_from_notes

    def receipt(selected: bool) -> ReferenceYieldReceipt:
        attempt = ReferenceSegmentationAttempt(
            strategy="footnotes",
            spans=(),
            credible_starts=6,
            selected=selected,
            reason_flags=("notes_as_reference_list",),
        )
        return ReferenceYieldReceipt(
            credible_source_starts=None,
            attempts=(attempt,),
            selected_spans=(),
            source_character_coverage=None,
            parsed_count=6,
            valid_count=5 if selected else 0,
            duplicate_rate=0.0,
            reason_flags=("notes_as_reference_list",),
        )

    contents = _contents([])
    assert not _references_from_notes(contents)
    contents.reference_yield_receipt = receipt(False)
    assert not _references_from_notes(contents)
    contents.reference_yield_receipt = receipt(True)
    assert _references_from_notes(contents)
