"""Shared conservative text filters for printed title candidates."""

import re
import unicodedata

# Reject a composite heading only when every content token belongs to the heading vocabulary,
# using accent-insensitive matching.
_BODY_HEADING_WORDS = frozenset(
    {
        # presentation / introduction
        "apresentacao",
        "presentacion",
        "presentazione",
        "presentation",
        "introducao",
        "introduccion",
        "introduzione",
        "introduction",
        # method / materials
        "metodologia",
        "metodologias",
        "metodologie",
        "methodologie",
        "metodo",
        "metodos",
        "metodi",
        "methode",
        "methodes",
        "materiais",
        "materiales",
        "materiali",
        "materiel",
        "materiels",
        "procedimentos",
        "procedimientos",
        # results / analysis
        "resultado",
        "resultados",
        "resultat",
        "resultats",
        "risultati",
        "risultato",
        "analise",
        "analises",
        "analisis",
        "analisi",
        "analyse",
        "analyses",
        "dados",
        "datos",
        "dati",
        "donnees",
        # discussion / conclusion
        "discussao",
        "discussoes",
        "discusion",
        "discusiones",
        "discussione",
        "discussioni",
        "discussion",
        "conclusao",
        "conclusoes",
        "conclusion",
        "conclusiones",
        "conclusione",
        "conclusioni",
        "conclusions",
        "consideracoes",
        "consideraciones",
        "considerazioni",
        "considerations",
        "sintese",
        "sintesi",
        "synthese",
        "limitacoes",
        "limitaciones",
        "limitazioni",
        "recomendacoes",
        "recomendaciones",
        "raccomandazioni",
        "recommandations",
        "final",
        "finais",
        "finales",
        "finali",
        # framing / literature
        "revisao",
        "revision",
        "revisione",
        "revue",
        "literatura",
        "letteratura",
        "litterature",
        "fundamentacao",
        "fundamentacion",
        "fundamentos",
        "teorica",
        "teorico",
        "teoricos",
        "theorique",
        "objetivo",
        "objetivos",
        "obiettivi",
        "obiettivo",
        "objectif",
        "objectifs",
        "hipotese",
        "hipoteses",
        "hipotesis",
        "ipotesi",
        "hypothese",
        "hypotheses",
        # front/back matter
        "resumo",
        "resumen",
        "riassunto",
        "resume",
        "palavras",
        "palabras",
        "parole",
        "chave",
        "clave",
        "chiave",
        "cle",
        "cles",
        "agradecimentos",
        "agradecimientos",
        "ringraziamenti",
        "remerciements",
        "referencias",
        "riferimenti",
        "bibliografia",
        "bibliografias",
    }
)
# Function words and articles that carry no heading/title signal on their own.
_HEADING_FILLER_WORDS = frozenset(
    {
        "a",
        "ai",
        "al",
        "alla",
        "and",
        "as",
        "com",
        "con",
        "da",
        "das",
        "de",
        "degli",
        "dei",
        "del",
        "della",
        "delle",
        "dello",
        "des",
        "di",
        "do",
        "dos",
        "du",
        "e",
        "ed",
        "el",
        "em",
        "en",
        "et",
        "gli",
        "i",
        "il",
        "in",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "of",
        "os",
        "para",
        "per",
        "pour",
        "the",
        "u",
        "um",
        "uma",
        "un",
        "una",
        "unas",
        "uno",
        "unos",
        "y",
    }
)
_HEADING_TOKEN_RE = re.compile(r"[^\W\d_]+")
# A printed heading is short; beyond this the row is prose, not a heading.
_MAX_HEADING_TOKENS = 8


def _strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", value) if not unicodedata.combining(char)
    )


def _is_ordinary_body_heading(normalized: str) -> bool:
    """Whether a normalized row is a bare body heading rather than a title.

    Complements ``_ORDINARY_HEADING_TEXT``'s exact-membership test: numbering
    and function words are dropped, and the row is a heading only when every
    remaining token is heading vocabulary.
    """

    tokens = _HEADING_TOKEN_RE.findall(_strip_accents(normalized))
    if not tokens or len(tokens) > _MAX_HEADING_TOKENS:
        return False
    content = [token for token in tokens if token not in _HEADING_FILLER_WORDS]
    return bool(content) and all(token in _BODY_HEADING_WORDS for token in content)
