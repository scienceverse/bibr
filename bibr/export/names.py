"""Split a printed author or editor list into structured names.

A reference prints its names in one of a few styles, and
:func:`split_printed_names` recognises them:

- inverted, comma-separated: ``Smith, J. A., Doe, B., & van der Berg, C.``
  (APA, Harvard; initials may lose their last period: ``Smith, J.A``);
- Chicago: the first author inverted, the others given name first:
  ``Smith, John, Jane Doe, and Ann Roe``;
- Vancouver: ``Smith JA, Doe B, van der Berg C`` (also with dotted initials;
  ``Smith JR`` and ``Kim II`` are initials);
- given name first: ``J. A. Smith and B. Doe``, ``John Smith, Jane Doe``;
- one author per ``;``-separated part: ``Smith, J.; Doe, A.``.

A given name never follows "and" or "&", and two full names in a row
("Jane Doe, Ann Roe") are two people, not one inverted name. "et al." (and
"and others") is cut off and reported, a suffix ("Jr.", "III") joins the
name it belongs to, and lowercase particles ("van der", "de la") stay with
the family name. An organization ("World Health Organization", "Open Science
Collaboration") becomes a ``literal`` name. When a list fits none of the
styles, its names come back as one ``literal`` holding the whole list: a
wrong split (initials read as surnames) looks worse in a reference manager
than an unsplit one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "PrintedName",
    "after_repeated_author_rule",
    "is_repeated_author_rule",
    "split_printed_names",
]


@dataclass(frozen=True)
class PrintedName:
    """One name: ``family``/``given``/``suffix`` for a person, ``literal`` for
    an organization or a list that could not be split."""

    family: str | None = None
    given: str | None = None
    suffix: str | None = None
    literal: str | None = None

    def as_dict(self) -> dict[str, str]:
        """The CSL-JSON name object (absent parts left out)."""
        parts = {
            "family": self.family,
            "given": self.given,
            "suffix": self.suffix,
            "literal": self.literal,
        }
        return {key: value for key, value in parts.items() if value}


# "et al." and its relatives, with the separator and any ellipsis before them.
# Everything after them is dropped.
_ET_AL = re.compile(
    r"(?:\s*[,;])?\s*(?:(?:\.\s*){3}|…)?\s*"
    r"(?:\bet\.?\s*al\b\.?|\band\s+others\b|&\s*others\b|\bu\.\s*a\.)",
    re.IGNORECASE,
)
# An editor marker: "(Eds.)", "(Ed.)", "Eds.", "(Hrsg.)", "edited by".
_EDITOR_MARK = re.compile(
    r"\(\s*(?:eds?|editors?|hrsg|hg|coords?|dirs?)\.?\s*\)|"
    r"(?:^|\s)(?:eds?|editors?)\.?(?=\s*$|\s*[,;])|^\s*edited\s+by\b",
    re.IGNORECASE,
)
# The "In" that starts an editor list printed after a chapter title.
_LEADING_IN = re.compile(r"^\s*in\s*:?\s+(?=\S)", re.IGNORECASE)
# APA 7's ellipsis between the 19th and the last author.
_ELLIPSIS = re.compile(r"(?:\.\s*){3}|…")
# A rule ("——") standing for the previous reference's authors.
_RULE = re.compile(r"^\s*[-\u2010-\u2015_]{2,}")
# The rule with the punctuation and "and" that follow it.
_RULE_LEAD = re.compile(r"^\s*[-\u2010-\u2015_]{2,}\s*[.,;]?\s*(?:(?:&|and)\s+)?", re.IGNORECASE)
# Words joining the last two names.
# ("y" and "e" are left out: "Ortega y Gasset" and "Silva e Souza" are one name.)
_AND = re.compile(r"\s+(?:and|und|en|et|og|och)\s+", re.IGNORECASE)
_LEADING_AND = re.compile(r"^(?:and|und|en|et|og|och)\s+", re.IGNORECASE)
_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "2nd", "3rd", "4th"})
# Lowercase words that belong to a family name.
_PARTICLES = frozenset(
    {
        "van",
        "von",
        "der",
        "den",
        "de",
        "del",
        "della",
        "di",
        "da",
        "das",
        "du",
        "la",
        "le",
        "ten",
        "ter",
        "zu",
        "des",
        "do",
        "dos",
        "e",
        "y",
        "los",
        "las",
        "lo",
        "vom",
        "zum",
        "zur",
        "op",
        "het",
        "bin",
        "ben",
        "ibn",
        "al",
        "el",
        "'t",
    }
)
# Words that make a name an organization's.
_ORGANIZATION_WORDS = frozenset(
    {
        "academy",
        "administration",
        "agency",
        "alliance",
        "association",
        "authority",
        "bank",
        "board",
        "bureau",
        "census",
        "center",
        "centre",
        "coalition",
        "collaboration",
        "collaborative",
        "college",
        "commission",
        "committee",
        "company",
        "consortium",
        "corporation",
        "corp",
        "council",
        "department",
        "federation",
        "force",
        "foundation",
        "fund",
        "government",
        "group",
        "hospital",
        "inc",
        "initiative",
        "institute",
        "institution",
        "investigators",
        "laboratory",
        "league",
        "llc",
        "ltd",
        "ministry",
        "nations",
        "network",
        "office",
        "organisation",
        "organization",
        "panel",
        "partnership",
        "press",
        "program",
        "programme",
        "project",
        "registry",
        "service",
        "services",
        "society",
        "statistics",
        "survey",
        "team",
        "trust",
        "union",
        "university",
        "unit",
    }
)
_WORD = re.compile(r"[^\W\d_]+")
# Single-letter initials joined by periods or hyphens, the last period
# optional: "J.A", "J.-P", "J-P.", "A.M.-C.".
_INITIAL_RUN = re.compile(r"[^\W\d_](?:(?:\.[-\u2010\u2011]?|[-\u2010\u2011])[^\W\d_])+\.?")


def is_repeated_author_rule(text: str | None) -> bool:
    """Does *text* open with a rule ("——") standing for the previous
    reference's authors?"""
    return bool(text and _RULE.match(text))


def after_repeated_author_rule(text: str) -> str:
    """What *text* prints after its opening rule: the names added to the
    previous reference's authors ("——, & Roe, R." gives "Roe, R.")."""
    return _RULE_LEAD.sub("", text, count=1).strip()


def _is_initial(token: str) -> bool:
    """An initial or a run of them: "J.", "J.A.", "Th.", "J.-P.", "JA", and a
    dotted run whose last period went with the reference's own ("J.A").

    A dotless all-caps "JR", "SR", "II" or "IV" is a run of initials too
    (Vancouver "Smith JR"); "Jr.", "Sr" and "2nd" are suffixes.
    """
    if not token[:1].isupper():
        return False
    if token.lower() in _SUFFIXES and not (token.isalpha() and token.isupper()):
        return False
    if re.fullmatch(r"(?:[^\W\d_]{1,2}\.)+", token):
        return all(chunk[:1].isupper() for chunk in token.split(".") if chunk)
    if _INITIAL_RUN.fullmatch(token):
        # The letter after a hyphen may be lowercase: "C.-h.", "Y.-t".
        return all(
            c.isupper() for c in re.sub(r"[-\u2010\u2011][^\W\d_]", "-", token) if c.isalpha()
        )
    return bool(re.fullmatch(r"[^\W\d_]{1,3}", token)) and token.isupper()


def _is_particle(token: str) -> bool:
    return token in _PARTICLES or token.lower() in {"d'", "l'"}


def _is_name_word(token: str) -> bool:
    """A capitalized word that can be part of a name: "Smith", "O'Brien",
    "García-López", "d'Alembert", "SMITH"."""
    word = re.sub(r"^(?:[dl]['’])", "", token)
    if not word[:1].isupper():
        return False
    return bool(re.fullmatch(r"[^\W\d_]+(?:['’\-\u2010\u2011][^\W\d_]+)*\.?", word))


def _is_suffix(segment: str) -> bool:
    return segment.lower() in _SUFFIXES


def _looks_corporate(segment: str) -> bool:
    words = segment.split()
    if any(w.lower().strip(".,()") in _ORGANIZATION_WORDS for w in words):
        return True
    # A lowercase word that is not a name particle ("of", "for", "the").
    if any(w[:1].islower() and not _is_particle(w) for w in words):
        return True
    return len(words) >= 5


def _is_family_word(token: str) -> bool:
    return _is_name_word(token) and not _is_initial(token)


def _family_tokens(tokens: list[str]) -> bool:
    """Family name tokens: name words (not initials) with particles before them."""
    if not tokens or not _is_family_word(tokens[-1]):
        return False
    return all(_is_family_word(t) or _is_particle(t) for t in tokens)


def _strip_dot(word: str) -> str:
    """A family name keeps no trailing period ("Smith." at the list's end)."""
    return word[:-1] if word.endswith(".") and not _is_initial(word) else word


def _family(tokens: list[str]) -> str:
    return " ".join([*tokens[:-1], _strip_dot(tokens[-1])])


def _person(family: list[str], given: list[str]) -> PrintedName:
    # Dutch order "Berg, C. van der": particles after the given name join the family name.
    moved: list[str] = []
    while given and _is_particle(given[-1]):
        moved.insert(0, given.pop())
    return PrintedName(family=_family([*moved, *family]), given=" ".join(given) or None)


def _given_tokens(tokens: list[str]) -> bool:
    """A given-name part: initials or written-out names, 1-4 tokens."""
    return 0 < len(tokens) <= 4 and all(
        _is_initial(t) or _is_name_word(t) or _is_particle(t) for t in tokens
    )


def _trailing_suffix(tokens: list[str], *, after_initials: bool) -> str | None:
    """A suffix printed at the end of a name ("King ML Jr", "Martin Luther
    King Jr."), taken off *tokens*.

    With *after_initials* (Vancouver) the suffix must follow the initials
    ("Gates WH III"). Otherwise a dotless all-caps "JR" or "II" right after
    a single word stays initials ("Kim II"), and after two words is a suffix
    ("John Smith II").
    """
    if len(tokens) < 2 or tokens[-1].lower() not in _SUFFIXES:
        return None
    if after_initials:
        if len(tokens) < 3 or not _is_initial(tokens[-2]):
            return None
    elif _is_initial(tokens[-1]) and len(tokens) < 3:
        return None  # "Kim II", "Smith JR": Vancouver initials
    return tokens.pop()


def _given_first(segment: str) -> PrintedName | None:
    """'J. A. Smith', 'John Smith', 'C. van der Berg', 'Smith' alone,
    'Martin Luther King Jr.'."""
    tokens = segment.split()
    suffix = _trailing_suffix(tokens, after_initials=False)
    if not tokens:
        return None
    if len(tokens) == 1:
        if not _is_family_word(tokens[0]):
            return None
        return PrintedName(family=_strip_dot(tokens[0]), suffix=suffix)
    if len(tokens) > 4:
        return None
    # The family name: from a particle after the given name ("Gabriel de
    # Oliveira Ramos"), else the last word with the particles before it.
    first_particle = next(
        (k for k in range(1, len(tokens)) if _is_particle(tokens[k])),
        None,
    )
    if first_particle is not None and _family_tokens(tokens[first_particle:]):
        start = first_particle
    else:
        start = len(tokens) - 1
        while start > 0 and _is_particle(tokens[start - 1]):
            start -= 1
    given, family = tokens[:start], tokens[start:]
    if not given or not _family_tokens(family):
        return None
    if not all(_is_initial(t) or _is_name_word(t) for t in given):
        return None
    return PrintedName(family=_family(family), given=" ".join(given), suffix=suffix)


def _full_name(segment: str) -> bool:
    """Can *segment* be a whole name printed given name first ("John Smith",
    "Christopher F Chabris")? Then, read as one half of an inverted name, it
    may be two people instead."""
    person = _given_first(segment)
    return person is not None and person.given is not None


def _vancouver(segment: str) -> PrintedName | None:
    """'Smith JA', 'van der Berg C', 'Smith J.A.', and a suffix after the
    initials or before them: 'Thorpe RJ Jr', 'Reynolds III CF', 'Thorpe Jr RJ'."""
    tokens = segment.split()
    suffix = _trailing_suffix(tokens, after_initials=True)
    split = len(tokens)
    while split > 0 and _is_initial(tokens[split - 1]):
        split -= 1
    if split == 0 or split == len(tokens):
        return None
    family, given = tokens[:split], tokens[split:]
    if suffix is None and len(given) > 1 and given[0].lower() in _SUFFIXES:
        suffix = given.pop(0)
    elif suffix is None and len(family) > 1 and family[-1].lower() in _SUFFIXES:
        suffix = family.pop()
    if not _family_tokens(family):
        return None
    return PrintedName(family=_family(family), given=" ".join(given), suffix=suffix)


def _with_suffix(name: PrintedName, suffix: str) -> PrintedName:
    return PrintedName(family=name.family, given=name.given, suffix=suffix)


def _organization(segment: str) -> PrintedName:
    return PrintedName(literal=segment.strip())


@dataclass(frozen=True)
class _Segment:
    """One comma-separated part of a name list; *joined* says an "and" or
    "&" came before it."""

    text: str
    joined: bool = False


def _segments(text: str) -> list[_Segment]:
    """Split on commas, semicolons and "&", then on "and" between two names
    (not inside an organization's name)."""
    parts: list[_Segment] = []
    joined = False
    for raw in re.split(r"([,;&])", text):
        if raw == "&":
            joined = True
            continue
        if raw in (",", ";"):
            continue
        part = raw.strip()
        stripped = _LEADING_AND.sub("", part).strip()
        if stripped != part:
            joined = True
        if not stripped:
            continue
        pieces = [p.strip() for p in _AND.split(stripped)]
        if len(pieces) > 1 and not any(_looks_corporate(p) for p in pieces):
            parts.extend(_Segment(p, joined or k > 0) for k, p in enumerate(pieces) if p)
        else:
            parts.append(_Segment(stripped, joined))
        joined = False
    return parts


def _inverted_pair(segments: list[_Segment], i: int) -> PrintedName | None:
    """Segments *i* and *i + 1* read as "Family, Given", or None. A given
    name never follows an "and"; a suffix may end the family part ("Spiro
    III, A.", but not Vancouver's "Kim II, Lee SR")."""
    if i + 1 >= len(segments) or segments[i + 1].joined:
        return None
    family, given = segments[i].text, segments[i + 1].text
    if _looks_corporate(family) or _looks_corporate(given):
        return None
    family_tokens, given_tokens = family.split(), given.split()
    suffix = None
    if (
        len(family_tokens) > 1
        and family_tokens[-1].lower() in _SUFFIXES
        and (not _is_initial(family_tokens[-1]) or all(_is_initial(t) for t in given_tokens))
    ):
        suffix = family_tokens.pop()
    if not (_family_tokens(family_tokens) and _given_tokens(given_tokens)):
        return None
    person = _person(family_tokens, given_tokens)
    return _with_suffix(person, suffix) if suffix else person


def _parse_inverted(segments: list[_Segment]) -> list[PrintedName] | None:
    """'Smith, J. A., Doe, B.', 'Smith, John, and Jane Doe' and Chicago's
    'Smith, John, Jane Doe, Ann Roe'.

    Two full names in a row ("Jane Doe, Ann Roe") read as one inverted name
    would merge two people, so they end the inverted part: the list opening
    with them is not inverted, and after the first name they start the
    given-name-first rest.
    """
    names: list[PrintedName] = []
    given_first_rest = False
    i = 0
    while i < len(segments):
        segment = segments[i].text
        tokens = segment.split()
        if names and _is_suffix(segment) and names[-1].literal is None:
            names[-1] = _with_suffix(names[-1], segment)
            i += 1
            continue
        pair = None if given_first_rest else _inverted_pair(segments, i)
        if pair is not None:
            if _full_name(segment) and _full_name(segments[i + 1].text):
                if not names:
                    return None  # "John Smith, Jane Doe": two people
                given_first_rest = True
            else:
                names.append(pair)
                i += 2
                continue
        if not names:
            return None  # the list must open "Family, Given"
        if _looks_corporate(segment):
            names.append(_organization(segment))
        else:
            # Chicago: the authors after the first are printed given name first.
            person = _given_first(segment) if len(tokens) > 1 else None
            if person is None:
                return None
            names.append(person)
            given_first_rest = True
        i += 1
    return names


def _parse_each(segments: list[str], parse) -> list[PrintedName] | None:
    """Every segment one name, read by *parse*, an organization, or a suffix."""
    names: list[PrintedName] = []
    for segment in segments:
        if names and _is_suffix(segment) and names[-1].literal is None:
            names[-1] = _with_suffix(names[-1], segment)
            continue
        person = parse(segment)
        if person is None:
            if not _looks_corporate(segment) or not _WORD.search(segment):
                return None
            person = _organization(segment)
        names.append(person)
    return names


def _parse_semicolons(text: str) -> list[PrintedName] | None:
    """'Smith, J.; Doe, A.': each part one name, inverted or not."""
    names: list[PrintedName] = []
    for raw in text.split(";"):
        part = _LEADING_AND.sub("", raw.strip()).strip()
        if not part:
            continue
        if "," in part:
            family, _, given = part.partition(",")
            given_tokens = given.split()
            family_tokens = family.split()
            suffix = None
            if "," in given:
                given, _, suffix = given.partition(",")
                given_tokens = given.split()
                suffix = suffix.strip() if _is_suffix(suffix.strip()) else None
                if suffix is None:
                    return None
            if _family_tokens(family_tokens) and _given_tokens(given_tokens):
                person = _person(family_tokens, given_tokens)
                names.append(_with_suffix(person, suffix) if suffix else person)
                continue
            if _looks_corporate(part):
                names.append(_organization(part))
                continue
            return None
        found = _vancouver(part) or _given_first(part)
        if found is None:
            if not _looks_corporate(part):
                return None
            found = _organization(part)
        names.append(found)
    return names or None


def _clean(text: str, *, editors: bool) -> str:
    text = " ".join(text.split())
    if editors:
        text = _LEADING_IN.sub("", text)
        text = _EDITOR_MARK.sub(" ", text)
    text = _ELLIPSIS.sub(",", text)
    # "*" marks a reference included in a meta-analysis.
    return text.strip(" ,;:&*")


def split_printed_names(
    text: str | None, *, editors: bool = False
) -> tuple[list[PrintedName], bool]:
    """``(names, et_al)`` for a printed author (or, with *editors*, editor) list.

    *et_al* says the list was cut short with "et al." or "and others". A list
    that fits no known style comes back as one literal name holding it whole.
    A rule standing for the previous reference's authors ("——") gives no
    names; see :func:`is_repeated_author_rule`.
    """
    if not text or not text.strip() or is_repeated_author_rule(text):
        return [], False
    et_al = False
    match = _ET_AL.search(text)
    if match:
        et_al = True
        text = text[: match.start()]
    text = _clean(text, editors=editors)
    if not text:
        return [], et_al
    if ";" in text:
        names = _parse_semicolons(text)
        return (names, et_al) if names else ([_organization(text)], et_al)
    segments = _segments(text)
    texts = [segment.text for segment in segments]
    if len(texts) == 1 and _looks_corporate(texts[0]):
        return [_organization(texts[0])], et_al
    names = (
        _parse_inverted(segments)
        or _parse_each(texts, _vancouver)
        or _parse_each(texts, _given_first)
    )
    return (names, et_al) if names else ([_organization(text)], et_al)
