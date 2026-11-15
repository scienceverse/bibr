"""Contact lines printed after an affiliation never stay in its value.

Modelled on Estudos de Psicologia (Campinas) 38, e190088: the page-1 footnote
defining marker 1 runs on into "Correspondence to: M.R. BAZON. E-mails:
<...>", and the corresponding author's affiliation kept that tail.
"""

import pandas as pd
import pytest

from bibr.extract.core_metadata import CoreMetadataExtractor, _clean_affiliation_value
from bibr.paper import PaperAuthor

_INSTITUTION = (
    "Universidade de São Paulo, Faculdade de Filosofia, Ciências e Letras de Ribeirão Preto, "
    "Departamento de Psicologia. Av. Bandeirantes, 3900, 14040-901, Ribeirão Preto, SP, Brasil"
)


@pytest.mark.parametrize(
    "raw",
    [
        f"{_INSTITUTION}. Correspondence to: M.R. BAZON. E-mails",
        f"{_INSTITUTION}. Correspondence to: M.R. BAZON",
        f"{_INSTITUTION}. E-mails: <mbazon@ffclrp.usp.br>, <rn_azevedo@yahoo.com.br>.",
        f"{_INSTITUTION}; Corresponding author: M.R. Bazon",
    ],
)
def test_a_correspondence_or_plural_email_tail_is_cut(raw):
    assert _clean_affiliation_value(raw) == _INSTITUTION


def test_the_numbered_definition_loses_its_correspondence_line():
    footnote = (
        f"1 {_INSTITUTION}. Correspondence to: M.R. BAZON. E-mails: <mbazon@ffclrp.usp.br>, "
        "<rn_azevedo@yahoo.com.br>."
    )
    frame = pd.DataFrame(
        {
            "text_id": [1, 2, 3],
            "page_number": [1, 1, 1],
            "text": [
                "Roberta Noronha AZEVEDO 0000-0002-3893-8112",
                "Marina Rezende BAZON1 0000-0002-8037-8710",
                footnote,
            ],
        }
    )
    authors = [
        PaperAuthor(author_id=1, given="Roberta Noronha", family="AZEVEDO", affiliation=""),
        PaperAuthor(author_id=2, given="Marina Rezende", family="BAZON", affiliation=""),
    ]
    CoreMetadataExtractor._reconcile_numbered_affiliations(authors, frame)
    CoreMetadataExtractor._normalize_author_affiliations(authors, frame)
    assert authors[1].affiliation == _INSTITUTION
