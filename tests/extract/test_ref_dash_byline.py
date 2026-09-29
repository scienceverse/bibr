"""NER reference fields of "SURNAME, Given — Title, Place, Publisher Year" entries.

The texts are entries of a scanned Brazilian law-journal reference list, with
the fields the NER tagger emitted for them. The guards use other dev-set
entries that print a dash near the byline (dash-joined co-author lists, a dash
inside a title), with the fields the rule would need to fire on.
"""

import pytest

from bibr.config import GlobalSettings
from bibr.extract import ref_extractor as ex
from bibr.extract.ref_field_repair import repair_ner_reference_fields

CHAVES = (
    "CHAVES, Antônio — Direito Autoral de Radiodifusão, S. Paulo, Ed. Rev. dos Tribunais "
    "1952, págs. 444-453."
)
# What the tagger found in CHAVES: the source and the pages, no author, no title.
CHAVES_FIELDS = {
    "container": "Rev. dos Tribunais",
    "year": 1952,
    "first_page": "444",
    "last_page": "453",
}


def test_untagged_dash_byline_gives_the_author_and_the_title():
    fields = dict(CHAVES_FIELDS)
    fired = repair_ner_reference_fields(fields, CHAVES)
    assert "dash_byline" in fired
    assert fields["authors"] == "CHAVES, Antônio"
    assert fields["title"] == "Direito Autoral de Radiodifusão"
    assert fields["container"] == "Rev. dos Tribunais"


@pytest.mark.parametrize(
    ("text", "fields", "authors", "title"),
    [
        (
            "FIGUEIREDO, Guilherme — Defesa de alguns pontos capitais do Projeto de lei de "
            "direito autoral, Diário do Congresso Nacional de 18-11-1947, pág. 8157 e segs.",
            {
                "authors": "FIGUEIREDO",
                "title": "Guilherme — Defesa de alguns pontos capitais do Projeto de lei de "
                "direito autoral",
                "container": "Diário do Congresso Nacional de 18-11-1947",
            },
            "FIGUEIREDO, Guilherme",
            "Defesa de alguns pontos capitais do Projeto de lei de direito autoral",
        ),
        (
            "HAMMES, Bruno Jorge — O Domínio Público Remunerado, Estudos Jurídicos, n. 39, "
            "1984, p. 39-52.",
            {"authors": "HAMMES", "container": "Estudos Jurídicos", "year": 1984},
            "HAMMES, Bruno Jorge",
            "O Domínio Público Remunerado",
        ),
        (
            "MOUCHET, Carlos — El Domínio Público Pagante, Buenos Aires, Fondo Nacional de "
            "Ias Artes, 1970, 182 págs.",
            {"authors": "MOUCHET", "year": 1970},
            "MOUCHET, Carlos",
            "El Domínio Público Pagante",
        ),
        (
            "VILBOIS, Jean — Du Domaine Public Payant en Matière de Droit d'Auteur, Paris, "
            "Sirey 1929, 544 págs.",
            {
                "authors": "VILBOIS",
                "title": "Jean — Du Domaine Public Payant en Matière de Droit d'Auteur, "
                "Paris, Sirey",
                "year": 1929,
            },
            "VILBOIS, Jean",
            # The tagger's end stands: only the given name is taken off.
            "Du Domaine Public Payant en Matière de Droit d'Auteur, Paris, Sirey",
        ),
        (
            "VILBOIS, Jean — Du Domaine Public Payant en Matière de Droit d'Auteur, Paris, "
            "Sirey 1929, 544 págs.",
            {"authors": "VILBOIS", "year": 1929},
            "VILBOIS, Jean",
            "Du Domaine Public Payant en Matière de Droit d'Auteur",
        ),
        (
            # A title tagged only on the given name has no end to keep.
            "HAMMES, Bruno Jorge — O Domínio Público Remunerado, Estudos Jurídicos, n. 39, "
            "1984, p. 39-52.",
            {"authors": "HAMMES", "title": "Bruno Jorge", "container": "Estudos Jurídicos"},
            "HAMMES, Bruno Jorge",
            "O Domínio Público Remunerado",
        ),
    ],
    ids=[
        "given-name-in-title",
        "surname-only",
        "no-title",
        "tagged-title-keeps-its-end",
        "untagged-title-ends-at-comma",
        "title-inside-byline",
    ],
)
def test_given_names_before_the_dash_belong_to_the_byline(text, fields, authors, title):
    fired = repair_ner_reference_fields(fields, text)
    assert "dash_byline" in fired
    assert fields["authors"] == authors
    assert fields["title"] == title


@pytest.mark.parametrize(
    ("text", "fields"),
    [
        # The tagger's title starts after the dash: left as it is, even with
        # a comma inside it.
        (
            "DOZORTSEV, V. — Les Editeurs, les Producteurs de Phonogrammes et de "
            "Videogrammes et Ia Protection des Oeuvres du Domaine Public, Doe. "
            "PRS/CPÍ/DP/CEG/I/?, apresentado ao Comitê de Peritos Governamentais",
            {
                "authors": "DOZORTSEV, V",
                "title": "Les Editeurs, les Producteurs de Phonogrammes et de Videogrammes "
                "et Ia Protection des Oeuvres du Domaine Public",
            },
        ),
        # Co-authors joined by dashes.
        (
            "EBEL, Petr – SCHMIDT, Ondřej. Z Trevisa do Brtnice: Příběhy šlechtického rodu "
            "Collalto ukryté v českých archívech (katalog výstavy).",
            {"year": 2019},
        ),
        (
            "BŮŽEK, Václav – KRÁL Pavel. Společnost v zemích habsburské monarchie a její obraz "
            "v pramenech. České Budějovice: JČU, 2006.",
            {"year": 2006},
        ),
        # A period ends the byline; the dash is the title's.
        (
            "BAY, József. Lengyel – Kastélypark [Lengyel – the Chateau Garden]. Budapest: "
            "Tájak-Korok-Múzeumok Kiskönyvtára, 1989.",
            {"authors": "BAY, József", "title": "Lengyel – Kastélypark", "year": 1989},
        ),
        # A tagged author that runs past the byline is another rule's case.
        (
            "LIPSZYC, Delia — Domínio Público, 14 folhas mimeografadas sem indicações.",
            {"authors": "LIPSZYC, Delia", "title": "Domínio Público"},
        ),
        # Surnames not in capitals are not this style.
        (
            "Marguerat, D. – Bourquin, Y. (20094 [1998]), Pour lire les récits bibliques, "
            "Parijs: Cerf; Genève: Labor et fides.",
            {"authors": "Marguerat, D", "year": 1998},
        ),
    ],
    ids=[
        "title-after-dash",
        "co-author-list",
        "co-author-in-capitals",
        "dash-in-title",
        "author-dash-title",
        "mixed-case-surname",
    ],
)
def test_dash_byline_leaves_other_references_alone(text, fields):
    before = dict(fields)
    fired = repair_ner_reference_fields(fields, text)
    assert "dash_byline" not in fired
    assert fields.get("authors") == before.get("authors")
    assert fields.get("title") == before.get("title")


def test_untagged_dash_byline_reference_is_kept_by_the_ner_parse(monkeypatch):
    class _Parser:
        def parse_batch(self, texts, batch_size=32):
            return [dict(CHAVES_FIELDS) for _ in texts]

    monkeypatch.setattr(ex, "_get_ner_parser", lambda settings, memory_mode=None: _Parser())
    extractor = ex.ReferenceExtractor(contents=None, file_hash="x", settings=GlobalSettings())
    refs = extractor._parse_references_ner_aligned([CHAVES])
    assert refs[0] is not None
    assert refs[0].authors == "CHAVES, Antônio"
    assert refs[0].title == "Direito Autoral de Radiodifusão"
    assert refs[0].year == 1952
