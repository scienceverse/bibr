"""Reference-anchored Tier 2 xref matcher.

Replaces the regex-based ``_detect_author_year_citations`` from
``citation_linker``. Instead of matching open patterns and trying to look up
author surnames in a flat string, we:

1. Parse author family names out of every ``PaperReference``.
2. Walk every parenthetical / narrative ``(YYYY)`` candidate span in body text.
3. Match candidates back to refs by ``(year, family overlap)`` with
   disambiguation by:

   - number of family-name matches (more = better)
   - "et al." preferring 3+ author refs
   - year_suffix match
   - smaller author-list size when match counts tie

Candidate matching and ambiguity handling are exercised by the synthetic citation-linking tests.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

from bibr.models import PaperReference
from bibr.paper_contents import CitationCandidate, PaperSentence, PaperXref
from bibr.utils.text import IN_PRESS_LABEL, collapse_ws

# ---------------------------------------------------------------------------
# Author family extraction
# ---------------------------------------------------------------------------

# Capitalized name token — allows hyphens, apostrophes, Latin extended.
_FAMILY_CORE = r"[A-Z][a-zA-ZÀ-ɏ'\-]+"


def _is_initials(s: str, *, caps_surnames: bool = False) -> bool:
    """True if s looks like author initials (e.g., 'J.', 'J. A.', 'JA').

    With *caps_surnames*, an unspaced, undotted all-caps run of four letters or
    more ("NGUYEN") is a surname printed in capitals, not initials: those come
    as "JA", "J. A." or "J.A.".
    """
    s = s.strip().rstrip(".")
    if not s or len(s) > 12:
        return False
    alphas = [c for c in s if c.isalpha()]
    if not alphas:
        return False
    if caps_surnames and len(s) >= 4 and s.isalpha() and s.isupper():
        return False
    upper_ratio = sum(1 for c in alphas if c.isupper()) / len(alphas)
    return len(alphas) <= 6 and upper_ratio >= 0.6 and all(c.isalpha() or c in ". " for c in s)


_DOTTED_INITIALS_RE = re.compile(r"(?:[^\W\d_]\.[\s\-\u2010]*)+")


def _is_dotted_initials(s: str) -> bool:
    """True for initials printed with their dots ("J.", "J. A.", "J.-P.")."""
    s = s.strip()
    return bool(_DOTTED_INITIALS_RE.fullmatch(s)) and re.sub(r"[.\s\-\u2010]", "", s).isupper()


def _is_caps_family(s: str, following: str | None, *, last: bool) -> bool:
    """True for a surname in capitals of any length that dotted initials follow.

    "LEE, J., NGUYEN, T.": the short names "LEE" and "SÁ" read like initials
    ("JA", "LEE" are both three capitals) until the initials after the comma
    show which is which. *last*: ``following`` ends the string, which lost its
    final dot.
    """
    letters = [c for c in s if c.isalpha()]
    if following is None or len(letters) < 2 or not all(c.isupper() for c in letters):
        return False
    if not all(c.isalpha() or c in " -'\u2019" for c in s):
        return False
    return _is_dotted_initials(following) or (last and _is_dotted_initials(following + "."))


def extract_families(authors_str: str | None, *, rules: bool = False) -> list[str]:
    """Best-effort APA-style author parsing — returns family names in order.

    Handles APA "Family, J. A., Family, J., & Family, J." patterns plus
    multi-word names ("Della Sala", "van der Vyver"), hyphenated names
    ("Fei-Fei"), continuation dots ". . ." and "and"/"&" connectors.

    With *rules* (a run with no LLM, see ``match_with_candidates``) a surname
    printed in capitals is read as one (``_is_initials``, ``_is_caps_family``).
    """
    if not authors_str:
        return []
    s = authors_str.strip().rstrip(".")
    s = re.sub(r"\.\s*\.\s*\.?", ",", s)
    s = re.sub(r"\s+&\s+", ", ", s)
    s = re.sub(r"\s+\band\b\s+", ", ", s)

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return []

    # Vancouver: ``Family Initials, Family Initials`` (commas or semicolons).
    # Detect this before the APA alternating ``Family, Initials`` walk.  Every
    # component must carry its own trailing initials so mixed/organizational
    # strings fall back to the established best-effort parser below.
    vancouver_source = re.sub(
        r"(?:[,;]\s*)?(?:(?:\.{2,}|…)\s*)?et\s+al\.?\s*$",
        "",
        authors_str,
        flags=re.IGNORECASE,
    )
    vancouver_parts = [p.strip() for p in re.split(r"[,;]", vancouver_source) if p.strip()]
    vancouver_families: list[str] = []
    for part in vancouver_parts:
        part = re.sub(r"^(?:and|&)\s+", "", part, flags=re.IGNORECASE)
        tokens = part.rstrip(".").split()
        suffix_start = len(tokens)
        # The last word of "Family Initials" is the initials, as long as it
        # looks like some ("Li JAMK"); a word before it in capitals is the family.
        while suffix_start > 0 and _is_initials(
            tokens[suffix_start - 1], caps_surnames=rules and suffix_start < len(tokens)
        ):
            initials = "".join(c for c in tokens[suffix_start - 1] if c.isalpha())
            if not initials or not all(c.isupper() for c in initials):
                break
            suffix_start -= 1
        if suffix_start == len(tokens) or suffix_start == 0:
            vancouver_families = []
            break
        family = " ".join(tokens[:suffix_start]).strip()
        if not family:
            vancouver_families = []
            break
        vancouver_families.append(family)
    if vancouver_families:
        return vancouver_families

    families: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        following = parts[i + 1] if i + 1 < len(parts) else None
        if _is_initials(cur, caps_surnames=rules) and not (
            rules and _is_caps_family(cur, following, last=i + 2 == len(parts))
        ):
            i += 1
            continue
        families.append(cur)
        if i + 1 < len(parts) and _is_initials(parts[i + 1], caps_surnames=rules):
            i += 2
        else:
            i += 1
    return families


# Lowercase surname particles ("van der Toorn", "de la Cruz"). Whitelist —
# "and", "of", "to" must NOT be treated as particles.
_NAME_PARTICLES = frozenset(
    {
        "van",
        "der",
        "den",
        "von",
        "de",
        "del",
        "della",
        "di",
        "da",
        "du",
        "la",
        "le",
        "el",
        "al",
        "ten",
        "ter",
        "bin",
        "ben",
    }
)

# English stop-words that look like family names but aren't.
_STOP_WORDS = frozenset(
    {
        "In",
        "And",
        "The",
        "See",
        "But",
        "For",
        "Also",
        "Cf",
        "However",
        "Although",
        "Findings",
        "Then",
        "When",
        "While",
        "Since",
        "Because",
        "Though",
        "First",
        "Second",
        "Third",
        "Recent",
        "Recently",
        "Indeed",
        "Furthermore",
        "Moreover",
        "Additionally",
        "Specifically",
        "Conversely",
        "By",
        "Of",
        "At",
        "On",
        "Per",
        "Via",
        "Such",
        "These",
        "Those",
        "This",
        "That",
    }
)


# ASCII apostrophe (U+0027), left single quote (U+2018), right single quote (U+2019).
_APOS = chr(0x27) + chr(0x2018) + chr(0x2019)

# Trailing possessive: apostrophe (any variant) followed by optional ‘s’, at end of string.
_POSSESSIVE_RE = re.compile("[" + _APOS + "]s?$")


# Footnote and equal-contribution marks a reference list prints before a name.
_LEADING_NON_WORD_RE = re.compile(r"^\W+")


def _family_key(family: str, *, rules: bool = False) -> str:
    """Normalize a family name for matching.

    Strips trailing possessive ``’s`` / ``’s`` / ``‘s``
    ("Perruchet’s" -> "perruchet", "Andersson’s" -> "andersson"),
    trailing lone apostrophes, trailing punctuation, and lowercases. With
    *rules*, also leading marker characters ("*Berg", "†Berg").
    """
    s = family.strip()
    if rules:
        s = _LEADING_NON_WORD_RE.sub("", s)
    s = s.rstrip(",.").lower()
    s = _POSSESSIVE_RE.sub("", s)
    # Citations often omit reference-list diacritics (Garcia vs García). Keep
    # the displayed family untouched, but compare on a decomposed ASCII key.
    return "".join(
        char for char in unicodedata.normalize("NFKD", s) if not unicodedata.combining(char)
    )


def _family_keys_for(family: str, *, rules: bool = False) -> set[str]:
    """All match-keys for a family name. Multi-word names also index under
    their last token so a candidate ``Toorn`` can match a ref ``van der Toorn``.
    """
    base = _family_key(family, rules=rules)
    if not base:
        return set()
    keys = {base}
    parts = base.split()
    if len(parts) > 1:
        last = parts[-1]
        if last not in _NAME_PARTICLES:
            keys.add(last)
    return keys


# ---------------------------------------------------------------------------
# Reference index
# ---------------------------------------------------------------------------


def _fold(text: str | None) -> str:
    """Lowercase and strip diacritics, for comparing printed strings."""
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", (text or "").lower())
        if not unicodedata.combining(char)
    )


@dataclass(frozen=True)
class _RefIdx:
    bib_id: int
    year: int | None
    year_suffix: str | None
    families: tuple[str, ...]
    is_in_press: bool
    family_key_sets: tuple[frozenset[str], ...]
    # Authors, year and the whole title, folded: two entries with the same key
    # are one work listed twice unless the volume or pages they print differ
    # (``_is_same_work``). None when the entry prints no author or no title, so
    # nothing vouches that it is a repeat.
    duplicate_key: tuple[str, int | None, str] | None = None


_NON_WORD_RE = re.compile(r"[\W_]+")
# What tells two entries with the same authors, year and title apart: the
# issues of a series, the parts of a work, the editions of a book
_WORK_FIELDS = ("volume", "issue", "first_page", "last_page", "doi", "edition", "version")


def _folded_alnum(text: str | None) -> str:
    """The letters and digits of ``text``, lowercased and without diacritics."""
    return _NON_WORD_RE.sub("", _fold(text))


def _is_same_work(a: PaperReference, b: PaperReference) -> bool:
    """Whether two entries that print one title are one work: no field both print differs."""
    for name in _WORK_FIELDS:
        printed_a, printed_b = (
            _folded_alnum(getattr(a, name, None)),
            _folded_alnum(getattr(b, name, None)),
        )
        if printed_a and printed_b and printed_a != printed_b:
            return False
    return True


def _build_ref_index(
    references: list[PaperReference], *, rules: bool = False
) -> tuple[
    list[_RefIdx],
    dict[tuple[int | None, str | None], list[_RefIdx]],
    list[_RefIdx],
]:
    """Build (refs, year_suffix_index, in_press_refs) from PaperReference list.

    With *rules*, a work listed twice (the same authors, year and whole title,
    and no volume or page that differs) is indexed once, under its first
    entry: a citation of it is not ambiguous. Entries that print different
    year suffixes ("2018a", "2018b") stay apart, since a citation names one of
    them by its suffix. Names are
    read as ``extract_families`` and ``_family_key`` read them with *rules*.
    """
    refs: list[_RefIdx] = []
    in_press: list[_RefIdx] = []
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]] = defaultdict(list)
    listed: dict[tuple[tuple[str, int | None, str], str | None], list[PaperReference]] = {}
    for r in references:
        is_in_press = bool(getattr(r, "is_in_press", False) or r.year == 0)
        year = None if r.year in (0, None) else r.year
        families = tuple(extract_families(r.authors, rules=rules))
        authors, title = _folded_alnum(r.authors), _folded_alnum(r.title)
        idx = _RefIdx(
            bib_id=r.bib_id,
            year=year,
            year_suffix=r.year_suffix,
            families=families,
            is_in_press=is_in_press,
            family_key_sets=tuple(frozenset(_family_keys_for(f, rules=rules)) for f in families),
            duplicate_key=(authors, year, title) if rules and authors and title else None,
        )
        refs.append(idx)
        if idx.duplicate_key is not None:
            twins = listed.setdefault((idx.duplicate_key, idx.year_suffix), [])
            if any(_is_same_work(r, twin) for twin in twins):
                continue
            twins.append(r)
        by_year[(idx.year, idx.year_suffix)].append(idx)
        if is_in_press:
            in_press.append(idx)
    return refs, by_year, in_press


# ---------------------------------------------------------------------------
# Candidate span extraction
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(?P<year>\d{4})(?P<suffix>[a-z])?\b")
_PAREN_CHUNK_RE = re.compile(r"\(([^()]{2,500}?)\)")
_YEAR_PAREN_RE = re.compile(r"\(\s*(\d{4})([a-z])?\s*\)")
# Year at the OPEN of a paren, allowing trailing content: "(2010; Study 1)",
# "(2010, p. 5)", "(2010a, 2010b)". Does NOT match "(Smith, 2010)" (author first).
_YEAR_PAREN_OPEN_RE = re.compile(r"\(\s*(\d{4})([a-z])?(?=[\s;,:)])")
# Additional year tokens inside the same paren (e.g. "2010b" in "(2010a, 2010b)").
# Plausible publication years only (19xx/20xx) — avoids harvesting bogus
# 4-digit numbers like sample sizes ("n=2500") inside a citation paren.
_EXTRA_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})([a-z])?\b")
_IN_PRESS_RE = re.compile(r"\b(" + IN_PRESS_LABEL + r")\b", re.IGNORECASE)
_IN_PRESS_PAREN_RE = re.compile(r"\(\s*(" + IN_PRESS_LABEL + r")\s*\)", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+(?=[A-Z])")

# Character class fragment covering Latin-extended letters plus all apostrophe variants.
# NOTE: byte-identical to bibr.utils.text.NAME_CHAR_CLS. Unify with it in the
# deferred Task 1b — left separate for now because this feeds LIVE in-text
# citation matching and there is no xref P/R/F1 gate yet to catch a recall shift.
_NAME_CHAR_CLS = "[a-zA-ZÀ-ɏ" + _APOS + r"\-]"

_PARTICLES_PAT = "|".join(sorted(_NAME_PARTICLES, key=len, reverse=True))
_CAPITALIZED_FAMILY_PAT = "[A-Z]" + _NAME_CHAR_CLS + "+"
_APOSTROPHE_PARTICLE_FAMILY_PAT = "[dDlL][" + _APOS + "][A-Z]" + _NAME_CHAR_CLS + "*"
_FAMILY_HEAD_PAT = "(?:" + _CAPITALIZED_FAMILY_PAT + "|" + _APOSTROPHE_PARTICLE_FAMILY_PAT + ")"
_FAMILY_PHRASE_PAT = (
    "(?:(?:\\b(?:"
    + _PARTICLES_PAT
    + ")\\s+){1,3})?"
    + _FAMILY_HEAD_PAT
    + "(?:\\s+"
    + _CAPITALIZED_FAMILY_PAT
    + ")?"
)
_NARRATIVE_PREFIX_RE = re.compile(
    _FAMILY_PHRASE_PAT
    + "(?:"
    + "(?:[,]\\s+"
    + _FAMILY_PHRASE_PAT
    + ")"
    + "|(?:[,]?\\s+(?:&|and)\\s+"
    + _FAMILY_PHRASE_PAT
    + ")"
    + "|(?:\\s+et\\s+al\\.?)"
    + "){0,6}"
    + "\\s*$"
)
_FAMILY_MULTI_RE = re.compile(_FAMILY_PHRASE_PAT)
_WORK_NOUNS = (
    "account|analysis|article|book|essay|framework|method|model|monograph|paper|"
    "publication|report|review|study|survey|theory|work"
)
_SEPARATED_POSSESSIVE_PREFIX_RE = re.compile(
    r"(?P<name>" + _FAMILY_PHRASE_PAT + r")[" + _APOS + r"]s\s+"
    r"(?P<bridge>(?:(?:[A-Za-zÀ-ɏ-]+)\s+){0,6}(?i:" + _WORK_NOUNS + r"))\s*$"
)


def _extract_families_from_span(span: str) -> list[str]:
    """Pull family-name tokens out of a candidate span (before year)."""
    span = _YEAR_RE.split(span, maxsplit=1)[0]
    raw = _FAMILY_MULTI_RE.findall(span)
    out: list[str] = []
    for t in raw:
        t = t.strip().rstrip(",.")
        if not t:
            continue
        words = t.split()
        while len(words) > 1 and words[0] in _STOP_WORDS:
            words = words[1:]
        if not words:
            continue
        family_words = [
            word
            for word in words
            if word[:1].isupper() or re.match(r"^[dDlL][" + _APOS + r"][A-Z]", word)
        ]
        if not family_words or all(word in _STOP_WORDS for word in family_words):
            continue
        out.append(" ".join(words))
    return out


@dataclass
class _Candidate:
    text_id: int
    start: int
    end: int
    raw: str
    year: int | None  # None means in-press / forthcoming
    year_suffix: str | None
    families: list[str]
    has_et_al: bool
    is_in_press: bool = False
    surface_kind: str = "standard"
    # Whether only the year follows the last family cited, so that nothing
    # says the work has more authors than are named
    bare: bool = True


# "et al." as the papers print it. A citation with this after its family cites
# a work of several authors, and the exact spelling is not the same in any two
# of them.
_ET_AL_RE = re.compile(r"(?<!\w)et[\s.,\-\u2010\u2011]*al(?!\w)", re.IGNORECASE)
# The same said in other words and languages: "and colleagues", "y cols.",
# "e col.", "et coll.", "und Kollegen", "e.a.", "dkk.", "ve ark.", "и др."
_AND_OTHERS_RE = re.compile(
    r"(?<!\w)(?:"
    r"and\s+(?:colleagues|others|co-?authors)"
    r"|y\s+(?:cols?|colaboradores|otros)"
    r"|e\s+(?:coll?s?|colaboradores|collaboratori|outros|altri)"
    r"|et\s+(?:coll?|collab\.?|autres)"
    r"|und\s+(?:kollegen|mitarbeiter|andere)"
    r"|u\.\s?a"
    r"|en\s+(?:collega[\u2019']?s|anderen)"
    r"|e\.\s?a"
    r"|dkk"
    r"|ve\s+ark"
    r"|и\s+др"
    r")(?!\w)",
    re.IGNORECASE,
)


def _has_et_al(text: str, *, rules: bool = False) -> bool:
    """Whether a citation's names are followed by "et al." or its equivalent.

    The literal "et al" only, unless *rules*: "et. al.", "etal", a non-breaking
    space or a line break inside it, "and colleagues" and the same in other
    languages.
    """
    if not rules:
        return "et al" in text.lower()
    return bool(_ET_AL_RE.search(text) or _AND_OTHERS_RE.search(text))


def _is_bare(before_year: str) -> bool:
    """Whether the text before a citation's year ends with its last family.

    "(Smith, 2010)" and "Smith (2010)" do; "(Smith et. al., 2010)", "Smith and
    colleagues (2010)" and "Smith J (2010)" have more words after the name,
    which may say the work has more authors than are named.
    """
    last_end = None
    for match in _FAMILY_MULTI_RE.finditer(before_year):
        last_end = match.end()
    if last_end is None:
        return False
    return re.search(r"[^\W\d_]", before_year[last_end:]) is None


def _find_candidates(text: str, text_id: int, *, rules: bool = False) -> list[_Candidate]:
    """Walk a sentence; return all parenthetical + narrative citation candidates.

    *rules* reads "et al." in more of the ways it is written and notes whether
    a citation is bare (``_Candidate.bare``).
    """
    out: list[_Candidate] = []

    # Parenthetical: split on ";" for multi-cites.
    for m in _PAREN_CHUNK_RE.finditer(text):
        chunk = m.group(1)
        chunk_cursor = 0
        for sub in chunk.split(";"):
            sub = sub.strip()
            if not sub:
                continue

            in_press = bool(_IN_PRESS_RE.search(sub))
            years = list(_YEAR_RE.finditer(sub))
            if not years and not in_press:
                continue
            families = _extract_families_from_span(sub)
            if not families:
                continue
            if ";" in m.group(1):
                relative = m.group(1).find(sub, chunk_cursor)
                chunk_cursor = max(relative, chunk_cursor) + len(sub)
                start = m.start(1) + max(relative, 0)
                end = start + len(sub)
            else:
                start, end = m.start(), m.end()
            raw_span = text[start:end]
            if years:
                for ym in years:
                    out.append(
                        _Candidate(
                            text_id=text_id,
                            start=start,
                            end=end,
                            raw=raw_span,
                            year=int(ym.group("year")),
                            year_suffix=ym.group("suffix"),
                            families=families,
                            has_et_al=_has_et_al(sub, rules=rules),
                            bare=_is_bare(_YEAR_RE.split(sub, maxsplit=1)[0]),
                        )
                    )
            else:
                out.append(
                    _Candidate(
                        text_id=text_id,
                        start=start,
                        end=end,
                        raw=raw_span,
                        year=None,
                        year_suffix=None,
                        families=families,
                        has_et_al=_has_et_al(sub, rules=rules),
                        is_in_press=True,
                        bare=_is_bare(_IN_PRESS_RE.split(sub, maxsplit=1)[0]),
                    )
                )

    # Narrative: walk every "(YYYY...)" paren where the year comes first,
    # allowing trailing content like "(2010; Study 1)" or "(2010a, 2010b)".
    for ym in _YEAR_PAREN_OPEN_RE.finditer(text):
        depth = 0
        for ch in text[: ym.start()]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        # Allow one level of nesting: a "(YYYY" at depth 1 is an inner work of a
        # group cite "(Salter et al. (1969), Caves (1992), ...)". Deeper nesting is
        # pathological, and ordinary asides "(... (n=2500))" carry no "(YYYY" so
        # never reach here.
        if depth > 1:
            continue
        lookback_start = max(0, ym.start() - 120)
        search_region = text[lookback_start : ym.start()]
        last_boundary = -1
        for bm in _SENTENCE_BOUNDARY_RE.finditer(search_region):
            preceding = search_region[: bm.start()].rstrip()
            if preceding.endswith("et al"):
                continue
            last_boundary = bm.end()
        if last_boundary > 0:
            search_region = search_region[last_boundary:]
        pm = _NARRATIVE_PREFIX_RE.search(search_region)
        surface_kind = "standard"
        if pm is None:
            pm = _SEPARATED_POSSESSIVE_PREFIX_RE.search(search_region)
            surface_kind = "separated_possessive"
        if pm is None:
            continue
        prefix = (
            pm.group("name").strip()
            if surface_kind == "separated_possessive"
            else pm.group(0).strip()
        )
        if not prefix:
            continue
        families = _extract_families_from_span(prefix)
        if not families:
            continue
        close = text.find(")", ym.start())
        prefix_start = (
            lookback_start + last_boundary + pm.start()
            if last_boundary > 0
            else lookback_start + pm.start()
        )
        span_end = close + 1 if close != -1 else ym.end()
        paren_body = text[ym.start() : close if close != -1 else ym.end()]
        seen_years: set[tuple[int, str | None]] = set()
        for yr in _EXTRA_YEAR_RE.finditer(paren_body):
            key = (int(yr.group(1)), yr.group(2))
            if key in seen_years:
                continue
            seen_years.add(key)
            out.append(
                _Candidate(
                    text_id=text_id,
                    start=prefix_start,
                    end=span_end,
                    raw=text[prefix_start:span_end],
                    year=int(yr.group(1)),
                    year_suffix=yr.group(2),
                    families=families,
                    has_et_al=_has_et_al(prefix, rules=rules),
                    surface_kind=surface_kind,
                    bare=surface_kind == "standard" and _is_bare(prefix),
                )
            )

    # Narrative in-press: "Smith (in press)".
    for ipm in _IN_PRESS_PAREN_RE.finditer(text):
        depth = 0
        for ch in text[: ipm.start()]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        if depth > 0:
            continue
        lookback_start = max(0, ipm.start() - 120)
        search_region = text[lookback_start : ipm.start()]
        last_boundary = -1
        for bm in _SENTENCE_BOUNDARY_RE.finditer(search_region):
            preceding = search_region[: bm.start()].rstrip()
            if preceding.endswith("et al"):
                continue
            last_boundary = bm.end()
        if last_boundary > 0:
            search_region = search_region[last_boundary:]
        pm = _NARRATIVE_PREFIX_RE.search(search_region)
        if pm is None:
            continue
        prefix = pm.group(0).strip()
        families = _extract_families_from_span(prefix)
        if not families:
            continue
        out.append(
            _Candidate(
                text_id=text_id,
                start=(
                    lookback_start + last_boundary + pm.start()
                    if last_boundary > 0
                    else lookback_start + pm.start()
                ),
                end=ipm.end(),
                raw=text[
                    (
                        lookback_start + last_boundary + pm.start()
                        if last_boundary > 0
                        else lookback_start + pm.start()
                    ) : ipm.end()
                ],
                year=None,
                year_suffix=None,
                families=families,
                has_et_al=_has_et_al(prefix, rules=rules),
                is_in_press=True,
                bare=_is_bare(prefix),
            )
        )

    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _overlap_exact(cited: frozenset[str], listed: frozenset[str]) -> bool:
    """A cited family and a listed author share a match-key."""
    return not cited.isdisjoint(listed)


def _full_key(keys: frozenset[str]) -> str:
    """The key of the whole family name, from a set ``_family_keys_for`` built.

    The key of a multi-word name's last word, which the set also holds, is a
    tail of the whole name's, so the whole name's is the longest.
    """
    return max(keys, key=len, default="")


_NAME_WORD_SPLIT_RE = re.compile(r"[\s\-\u2010\u2011.]+")
# A prefix needs this many letters, so that "Li" never opens "Lindqvist" (it
# needs a word boundary besides, see ``_overlap_prefix``).
_PREFIX_MIN_LEN = 3


def _name_words(key: str) -> list[str]:
    return [word for word in _NAME_WORD_SPLIT_RE.split(key) if word]


def _compact(key: str) -> str:
    return "".join(_name_words(key))


_HYPHEN_RE = re.compile(r"[\-\u2010\u2011]+")
# A part of a hyphenated name needs this many letters to stand for the name.
_HYPHEN_PART_MIN_LEN = 4


def _is_part_of_hyphenated(name: str, whole: str) -> bool:
    """Whether ``name`` is one of the hyphen-joined parts of ``whole``."""
    parts = [part for part in _HYPHEN_RE.split(whole) if part]
    return (
        len(parts) > 1 and " " not in name and len(name) >= _HYPHEN_PART_MIN_LEN and name in parts
    )


def _overlap_prefix(cited: frozenset[str], listed: frozenset[str]) -> bool:
    """The whole cited family is the listed one printed with other spaces or
    hyphens, or its opening words: "Berg" reaches an entry parsed as the
    family "Berg Anna" (a "Family Given" order), and "Hernandez-Lopez" reaches
    "Hernandez Lopez". A double surname joined by a hyphen reaches either of
    its parts and is reached by it ("Hedman-Lagerlof" and "Lagerlof"). Never
    the opening *letters* of a longer name ("John" of "Johnson") and never the
    last word of a name ("Health Council" of "Ministry of Health"): those are
    other people and other bodies.
    """
    cited_key, listed_key = _full_key(cited), _full_key(listed)
    if not cited_key or not listed_key:
        return False
    if _compact(cited_key) == _compact(listed_key):
        return True
    if _is_part_of_hyphenated(cited_key, listed_key) or _is_part_of_hyphenated(
        listed_key, cited_key
    ):
        return True
    cited_words, listed_words = _name_words(cited_key), _name_words(listed_key)
    return (
        len(cited_words) < len(listed_words)
        and listed_words[: len(cited_words)] == cited_words
        and len(_compact(cited_key)) >= _PREFIX_MIN_LEN
    )


# A misspelt or differently transliterated family ("Kowalczyk" / "Kowalchyk"):
# both names this long and at least this similar. Short names differ by one
# letter and still belong to two people.
_FUZZY_MIN_LEN = 6
_FUZZY_MIN_RATIO = 0.86


def _overlap_fuzzy(cited: frozenset[str], listed: frozenset[str]) -> bool:
    """``_overlap_prefix``, or two long names spelt almost alike."""
    if _overlap_prefix(cited, listed):
        return True
    cited_key, listed_key = _compact(_full_key(cited)), _compact(_full_key(listed))
    if min(len(cited_key), len(listed_key)) < _FUZZY_MIN_LEN:
        return False
    matcher = SequenceMatcher(None, cited_key, listed_key)
    return (
        matcher.real_quick_ratio() >= _FUZZY_MIN_RATIO
        and matcher.quick_ratio() >= _FUZZY_MIN_RATIO
        and matcher.ratio() >= _FUZZY_MIN_RATIO
    )


_Overlap = Callable[[frozenset[str], frozenset[str]], bool]

# Capitalised words that open a sentence or title a name and are no part of it:
# a narrative citation read from "As Okafor (2020)" names "Okafor", not
# "As Okafor". The exact reading gets there through the key of the name's last
# word; the looser ones compare whole names and drop these instead.
_OPENER_WORDS = frozenset(word.lower() for word in _STOP_WORDS) | {
    "as",
    "according",
    "following",
    "like",
    "using",
    "from",
    "with",
    "after",
    "before",
    "under",
    "than",
    "versus",
    "vs",
    "dr",
    "prof",
    "professor",
    "mr",
    "mrs",
    "ms",
}


def _whole_name_keys(family: str) -> frozenset[str]:
    """The key of a cited family as printed, less the words that open the sentence."""
    words = _family_key(family).split()
    while len(words) > 1 and words[0] in _OPENER_WORDS:
        words = words[1:]
    return frozenset({" ".join(words)} if words else ())


@dataclass(frozen=True)
class _Reading:
    """A way of comparing a cited family with a listed author."""

    name: str
    # The keys a cited family is compared by
    cited_keys: Callable[[str], frozenset[str]]
    overlaps: _Overlap
    # Whether it places a citation only when one reference of the year fits
    only_one: bool = False


def _exact_keys(family: str) -> frozenset[str]:
    return frozenset(_family_keys_for(family))


_EXACT = _Reading("exact", _exact_keys, _overlap_exact)
# Looser ways to read a family name, tried in turn only when no listed author
# shares a match-key with the citation.
_LOOSE_READINGS = (
    _Reading("prefix", _whole_name_keys, _overlap_prefix),
    _Reading("fuzzy", _whole_name_keys, _overlap_fuzzy, only_one=True),
)


def _score_ref(ref: _RefIdx, cand: _Candidate, reading: _Reading = _EXACT) -> tuple[int, int, int]:
    """Score how well ``ref`` matches ``cand``. Higher is better.

    Returns ``(matched_families, et_al_bonus, neg_extra_authors)`` for tuple
    sort. The third element prefers refs with fewer extra authors when match
    counts tie.
    """
    cited = [reading.cited_keys(f) for f in cand.families]
    ref_key_sets = ref.family_key_sets
    if not ref_key_sets or not any(reading.overlaps(keys, ref_key_sets[0]) for keys in cited):
        return (0, 0, 0)
    matched = sum(
        1 for author in ref_key_sets if any(reading.overlaps(keys, author) for keys in cited)
    )
    et_al_bonus = 0
    if cand.has_et_al:
        # "et al." supports a multi-author ref, but list length is never
        # disambiguating evidence when first family and year collide.
        et_al_bonus = min(len(ref_key_sets), 5) if len(ref_key_sets) >= 3 else -2
    extra = len(ref_key_sets) - matched
    return (matched, et_al_bonus, -extra)


def _tie_key(ref: _RefIdx, cand: _Candidate, reading: _Reading) -> tuple[bool, int, int]:
    """Rank a reference that ties with others on the same family and year.

    Returns ``(count_ok, order_ok, any_author)``. ``count_ok``: the number of
    listed authors fits how the citation names them: "et al." needs three or
    more, one cited family needs a single-author reference (when nothing but
    the year follows it), otherwise the counts are equal. ``order_ok``: cited families that match the listed author
    at the same position. ``any_author``: cited families that match any listed
    author. Names are compared as *reading* compares them, the reading the tie
    arose in.
    """
    cited = [reading.cited_keys(family) for family in cand.families]
    listed = ref.family_key_sets
    if cand.has_et_al:
        count_ok = len(listed) >= 3
    elif len(cited) == 1:
        # A single author is the answer only to a bare "Family (year)": the
        # words after the name may be a way of saying "and the others"
        count_ok = len(listed) == 1 and cand.bare
    else:
        count_ok = len(listed) == len(cited)
    order_ok = sum(
        1 for keys, author in zip(cited, listed, strict=False) if reading.overlaps(keys, author)
    )
    any_author = sum(1 for keys in cited if any(reading.overlaps(keys, a) for a in listed))
    return count_ok, order_ok, any_author


def _break_tie(tied: list[_RefIdx], cand: _Candidate, reading: _Reading) -> _RefIdx | None:
    """The one reference of ``tied`` that fits the citation best, if any.

    A reference whose first author is the first family cited wins when it is
    the only one ("Stanford Digital Education" cited against two entries that
    both end in "Education"). Otherwise the best-ranked reference wins if it
    fits the author count and strictly out-ranks every other on ``_tie_key``;
    two references that fit equally stay tied (a bare "et al." does not tell a
    three-author reference from a six-author one).
    """
    if cand.families:
        first_cited = _whole_name_keys(cand.families[0])
        first = [
            ref
            for ref in tied
            if ref.family_key_sets and _overlap_prefix(first_cited, ref.family_key_sets[0])
        ]
        if len(first) == 1:
            return first[0]
        tied = first or tied
    ranked = sorted(
        ((_tie_key(ref, cand, reading), ref) for ref in tied), key=lambda p: p[0], reverse=True
    )
    best_key, best = ranked[0]
    if best_key[0] and best_key > ranked[1][0]:
        return best
    return None


def _match_pool(
    pool: list[_RefIdx],
    cand: _Candidate,
    reading: _Reading,
    *,
    break_ties: bool = True,
) -> tuple[_RefIdx | None, list[int], bool]:
    """Score ``cand`` against the references of its year; see ``_match``.

    Returns ``(ref, tied_bib_ids, by_tie_break)``. With the reading's
    ``only_one``, a reference is linked only when no other reference of the
    year fits the citation, or when ``_break_tie`` picks it from those that do;
    a higher score is not enough. Without ``break_ties`` references that tie
    stay tied.
    """
    # A bare first-family/year citation cannot distinguish references that
    # share that first family and year by the author list alone: a coauthor or
    # year suffix must disambiguate. The one thing the count of authors does
    # settle is a single-author reference against multi-author ones, because a
    # work by several authors is not cited by its first author alone; an
    # "et al." citation separates three or more authors from one or two, never
    # three from six (``_break_tie``).
    if len(cand.families) == 1 and cand.year_suffix is None:
        candidate_keys = reading.cited_keys(cand.families[0])
        same_first = [
            ref
            for ref in pool
            if ref.family_key_sets and reading.overlaps(candidate_keys, ref.family_key_sets[0])
        ]
        if len(same_first) > 1:
            winner = _break_tie(same_first, cand, reading) if break_ties else None
            if winner is not None:
                return winner, [], True
            return None, sorted(ref.bib_id for ref in same_first), False

    scored = [(r, _score_ref(r, cand, reading)) for r in pool]
    scored = [(r, s) for r, s in scored if s[0] > 0]
    if not scored:
        return None, [], False

    scored.sort(key=lambda x: x[1], reverse=True)
    if len(scored) > 1 and (reading.only_one or scored[1][1] == scored[0][1]):
        top_score = scored[0][1]
        top = [r for r, s in scored if reading.only_one or s == top_score]
        winner = _break_tie(top, cand, reading) if break_ties else None
        if winner is not None:
            return winner, [], True
        return None, [r.bib_id for r in top], False
    return scored[0][0], [], False


def _year_pool(
    cand: _Candidate,
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]],
    in_press_refs: list[_RefIdx],
) -> list[_RefIdx]:
    """The references a candidate may cite: those of its exact year (and suffix)."""
    if cand.is_in_press:
        return in_press_refs
    pool: list[_RefIdx] = []
    seen: set[int] = set()
    for key in (
        (cand.year, cand.year_suffix) if cand.year_suffix else None,
        (cand.year, None),
    ):
        if key is None:
            continue
        for r in by_year.get(key, []):
            if r.bib_id not in seen:
                seen.add(r.bib_id)
                pool.append(r)
    return pool


def _spelt_as_listed_elsewhere(
    cand: _Candidate,
    ref: _RefIdx,
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]],
) -> bool:
    """Whether a cited family, as spelt, is an author of another year's reference.

    The citation then names that author with the wrong year, or somebody else:
    a guess at the spelling adds nothing. A further cited family that is an
    author of ``ref`` as spelt (the pair "Crockett and Carlo" against
    "Crocket, Carlo") vouches for it all the same.
    """
    cited = [_exact_keys(family) for family in cand.families]
    if any(_overlap_exact(keys, author) for keys in cited for author in ref.family_key_sets):
        return False
    listed = {
        _compact(_full_key(author))
        for refs in by_year.values()
        for other in refs
        for author in other.family_key_sets
    }
    return any(_compact(_full_key(keys)) in listed for keys in cited)


# How a candidate was placed: "exact" and "tie_break" are the exact reading of
# the names, the others the looser readings (``_LOOSE_READINGS``). The evidence
# and confidence a receipt records for each: the looser the reading, the less
# it vouches for the link.
_READING_EVIDENCE = {
    "exact": ("family_match", "year_match"),
    "tie_break": ("family_match", "year_match", "author_list_tie_break"),
    "prefix": ("family_prefix_match", "year_match"),
    "fuzzy": ("family_fuzzy_match", "year_match"),
}
_READING_CONFIDENCE = {"exact": 0.95, "tie_break": 0.9, "prefix": 0.8, "fuzzy": 0.7}


def _match_reading(
    cand: _Candidate,
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]],
    in_press_refs: list[_RefIdx],
    *,
    rules: bool = False,
) -> tuple[_RefIdx | None, list[int], str | None]:
    """``_match``, and how the reference was placed (a key of ``_READING_EVIDENCE``)."""
    pool = _year_pool(cand, by_year, in_press_refs)
    if not pool:
        return None, [], None

    ref, tied, by_tie_break = _match_pool(pool, cand, _EXACT, break_ties=rules)
    if ref is not None:
        return ref, [], "tie_break" if by_tie_break else "exact"
    if tied or not rules:
        return None, tied, None
    for reading in _LOOSE_READINGS:
        ref, tied, _by_tie_break = _match_pool(pool, cand, reading)
        if ref is not None:
            if reading.name == "fuzzy" and _spelt_as_listed_elsewhere(cand, ref, by_year):
                return None, [], None
            return ref, [], reading.name
        if tied:
            break
    return None, [], None


def _match(
    cand: _Candidate,
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]],
    in_press_refs: list[_RefIdx],
    *,
    rules: bool = False,
) -> tuple[_RefIdx | None, list[int]]:
    """Resolve a candidate to a ref.

    Returns ``(ref_or_None, tied_bib_ids)``:

    - ``(ref, [])`` — unique top scorer.
    - ``(None, [bib_id, ...])`` — 2+ refs tied at the top score; caller can
      route this to a Tier 3 disambiguator using ``tied_bib_ids`` as a
      shortlist.
    - ``(None, [])`` — no candidate ref scored > 0; nothing to disambiguate.

    Names are compared by their match-keys. With *rules* (a run with no LLM),
    references that tie are told apart by their author lists (``_break_tie``),
    and when no reference of the year shares a match-key with the citation the
    looser readings are tried (``_LOOSE_READINGS``); they link a single
    reference or none, and never offer a shortlist, so a citation the exact
    keys could not place is never narrowed to a wrong pair.
    """
    ref, tied, _reading = _match_reading(cand, by_year, in_press_refs, rules=rules)
    return ref, tied


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class AmbiguousCandidate:
    """Candidate citation span where 2+ refs scored equally.

    Surfaced so the caller can route the span to Tier 3 disambiguation
    constrained to ``candidate_bib_ids``.
    """

    text_id: int
    contents: str
    candidate_bib_ids: list[int]


def match_with_candidates(
    body_sents: list[PaperSentence],
    references: list[PaperReference],
    *,
    rules: bool = False,
) -> tuple[list[PaperXref], list[AmbiguousCandidate], list[CitationCandidate]]:
    """Run Tier 2 matching and retain every accepted/rejected source span.

    Caller is responsible for feeding only body sentences (use
    ``_get_body_sentences`` to filter references / abstract / title /
    captions / footnotes).

    *rules*: read the names more loosely, for a run with no LLM to settle
    what the exact reading leaves over. A work listed twice is one work, and
    references that tie are told apart by author count and order
    (``_break_tie``); a family spelt a little differently links when only
    one reference of the year is close to it (``_LOOSE_READINGS``); a
    reference list's leading marker and a surname in capitals are read
    through (``_family_key``, ``_is_initials``). Off, none of it applies and
    the output is that of the exact reading alone.
    """
    if not references:
        return [], [], []

    _refs, by_year, in_press_refs = _build_ref_index(references, rules=rules)

    xrefs: list[PaperXref] = []
    ambiguous: list[AmbiguousCandidate] = []
    candidates: list[CitationCandidate] = []
    seen: set[tuple[int, int]] = set()
    seen_amb: set[tuple[int, str]] = set()

    for sent in body_sents:
        if not sent.text:
            continue
        for cand in _find_candidates(sent.text, sent.text_id, rules=rules):
            ref, tied, reading = _match_reading(cand, by_year, in_press_refs, rules=rules)
            surface_evidence = (
                ("separated_possessive", "work_noun_bridge")
                if cand.surface_kind == "separated_possessive"
                else ()
            )
            if ref is not None:
                candidates.append(
                    CitationCandidate(
                        text_id=cand.text_id,
                        start=cand.start,
                        end=cand.end,
                        raw=cand.raw,
                        style="author-year",
                        bib_ids=(ref.bib_id,),
                        evidence=(*_READING_EVIDENCE[reading or "exact"], *surface_evidence),
                        confidence=_READING_CONFIDENCE[reading or "exact"],
                        accepted=True,
                        rejection_reasons=(),
                    )
                )
                key = (cand.text_id, ref.bib_id)
                if key in seen:
                    continue
                seen.add(key)
                xrefs.append(
                    PaperXref(
                        xref_id=ref.bib_id,
                        xref_type="bib",
                        contents=collapse_ws(cand.raw),
                        text_id=cand.text_id,
                        tier="author-year",
                    )
                )
            elif tied:
                candidates.append(
                    CitationCandidate(
                        text_id=cand.text_id,
                        start=cand.start,
                        end=cand.end,
                        raw=cand.raw,
                        style="author-year",
                        bib_ids=tuple(sorted(tied)),
                        evidence=("family_match", "year_match", *surface_evidence),
                        confidence=0.5,
                        accepted=False,
                        rejection_reasons=("ambiguous_same_surname_year",),
                    )
                )
                amb_key = (cand.text_id, cand.raw)
                if amb_key in seen_amb:
                    continue
                seen_amb.add(amb_key)
                ambiguous.append(
                    AmbiguousCandidate(
                        text_id=cand.text_id,
                        contents=collapse_ws(cand.raw),
                        candidate_bib_ids=tied,
                    )
                )
            else:
                candidates.append(
                    CitationCandidate(
                        text_id=cand.text_id,
                        start=cand.start,
                        end=cand.end,
                        raw=cand.raw,
                        style="author-year",
                        bib_ids=(),
                        evidence=("author_year_surface", *surface_evidence),
                        confidence=0.0,
                        accepted=False,
                        rejection_reasons=("no_reference_match",),
                    )
                )
    return xrefs, ambiguous, candidates


def match_with_ambiguous(
    body_sents: list[PaperSentence],
    references: list[PaperReference],
    *,
    rules: bool = False,
) -> tuple[list[PaperXref], list[AmbiguousCandidate]]:
    """Run Tier 2 matching; return unambiguous xrefs and ambiguous ties."""

    xrefs, ambiguous, _candidates = match_with_candidates(body_sents, references, rules=rules)
    return xrefs, ambiguous


def detect_author_year_xrefs(
    body_sents: list[PaperSentence],
    references: list[PaperReference],
    *,
    rules: bool = False,
) -> list[PaperXref]:
    """Detect author-year citations; return only unambiguous matches.

    Thin wrapper around ``match_with_ambiguous``. Use that function directly
    if you also need the ambiguous-tie list for Tier 3 routing.
    """
    xrefs, _ = match_with_ambiguous(body_sents, references, rules=rules)
    return xrefs


# ---------------------------------------------------------------------------
# Tier 3 without an LLM
# ---------------------------------------------------------------------------


def _match_any_author(pool: list[_RefIdx], cand: _Candidate) -> _RefIdx | None:
    """The one reference of the year that lists every cited family, in any position.

    Last resort for a citation whose first family fits no first author: a
    byline in another order, or a name taken from the end of the list. Reads
    names by their match-keys, as the exact reading does: a name that is only
    near another ("Lee" in a list of three, "Freedman" for "Friedman") is not
    evidence of a position the first author does not hold. Links only a single
    reference, or one that ``_break_tie`` picks from several.
    """
    cited = [_EXACT.cited_keys(family) for family in cand.families]
    if not cited:
        return None
    fits = [
        ref
        for ref in pool
        if ref.family_key_sets
        and all(
            any(_overlap_exact(keys, author) for author in ref.family_key_sets) for keys in cited
        )
    ]
    if len(fits) == 1:
        return fits[0]
    if len(fits) > 1:
        return _break_tie(fits, cand, _EXACT)
    return None


# Words an organisation's initials may skip: "Council for Reef Studies" is
# "CfRS" or "CRS".
_ORG_INITIALS_SKIP = frozenset(
    {"of", "the", "for", "and", "de", "la", "des", "du", "del", "in", "on", "et", "und", "y", "e"}
)


def _match_by_initials(pool: list[_RefIdx], cand: _Candidate) -> _RefIdx | None:
    """An organisation cited by its initials ("BCS"): the one reference of the
    year whose first author spells them out ("Bureau of Coastal Survey").

    The cited name must read as initials: at least two capitals, and at least
    half of its letters. A name such as "Ng" never reaches "Nguyen Giang".
    """
    if len(cand.families) != 1:
        return None
    letters = [char for char in cand.families[0] if char.isalpha()]
    capitals = sum(1 for char in letters if char.isupper())
    if not 2 <= len(letters) <= 8 or capitals < 2 or capitals * 2 < len(letters):
        return None
    cited = _fold("".join(letters))
    hits = []
    for ref in pool:
        words = re.findall(r"[^\W\d_]+", _fold(ref.families[0])) if ref.families else []
        if len(words) < 2:
            continue
        spelled = {
            "".join(word[0] for word in words),
            "".join(word[0] for word in words if word not in _ORG_INITIALS_SKIP),
        }
        if cited in spelled:
            hits.append(ref)
    return hits[0] if len(hits) == 1 else None


# "Harlow (2010, citado en Mendoza et al., 2015)": the work to link is the one the
# paper read, named after the marker.
_SECONDARY_SOURCE_RE = re.compile(
    r"\b(?:citado\s+(?:en|por)|cited\s+(?:in|by)|as\s+cited\s+in|cit[eé]e?s?\s+(?:dans|par)"
    r"|zitiert\s+nach|apud|citat[oa]\s+in)\b",
    re.IGNORECASE,
)


def resolve_citations_by_rules(
    cites: list[tuple[int, str]],
    references: list[PaperReference],
) -> list[tuple[int, str, tuple[int, ...]]]:
    """Resolve citation texts to references with rules only, no LLM.

    Stands in for the Tier-3 LLM when none is available. Each ``(text_id,
    citation_text)`` is read as Tier 2 reads a sentence: a bare "Smith, 2010"
    is read as if parenthesised. Each citation it names is matched with
    ``_match``, then, failing that, to the one reference of the year that lists
    every cited family in any position, or that an organisation's initials
    stand for. A citation that names no reference
    uniquely links nothing. Returns ``(text_id, citation_text, bib_ids)`` per
    input, ``bib_ids`` sorted.
    """
    if not references:
        return [(text_id, text, ()) for text_id, text in cites]
    _refs, by_year, in_press_refs = _build_ref_index(references, rules=True)
    resolved: list[tuple[int, str, tuple[int, ...]]] = []
    for text_id, text in cites:
        secondary = list(_SECONDARY_SOURCE_RE.finditer(text))
        cited_text = text[secondary[-1].end() :] if secondary else text
        candidates = _find_candidates(cited_text, text_id, rules=True) or _find_candidates(
            f"({cited_text})", text_id, rules=True
        )
        linked: set[int] = set()
        for cand in candidates:
            ref, _tied = _match(cand, by_year, in_press_refs, rules=True)
            if ref is None and not _tied:
                pool = _year_pool(cand, by_year, in_press_refs)
                ref = _match_any_author(pool, cand) or _match_by_initials(pool, cand)
            if ref is not None:
                linked.add(ref.bib_id)
        resolved.append((text_id, text, tuple(sorted(linked))))
    return resolved


# ---------------------------------------------------------------------------
# D2: constrained year-less back-reference resolver
# ---------------------------------------------------------------------------

# Capitalised name token(s) optionally joined by "and"/"&", optionally
# followed by "et al.", optionally followed by possessive "'s".
_YEAR_LESS_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-ZÀ-ɏ"
    + _APOS
    + r"\-]+(?:\s+(?:and|&)\s+[A-Z][a-zA-ZÀ-ɏ"
    + _APOS
    + r"\-]+)?(?:\s+et\s+al\.?)?)(?:["
    + _APOS
    + r"]s)?"
)


def match_year_less(
    body_sents,
    references: list[PaperReference],
    cited_bib_ids: set[int],
) -> list[tuple[int, int]]:
    """Resolve bare (year-less) narrative author mentions to refs ALREADY cited
    with a year (``cited_bib_ids``). Returns [(text_id, bib_id)].

    NOT CURRENTLY WIRED into detect_bib_xrefs: on the v1 gold, wiring this dropped
    precision 0.9973→0.9401 (70 new spurious) — single-surname-saturation papers
    (every "Andersson's" fires), leakage into bylines/author-contributions, and the
    gold does not exhaustively annotate year-less back-references. Kept as a building
    block pending: a >=2-family match constraint, a strict body-only section filter,
    and year-less annotations in the gold.
    """
    refs_by_id = {r.bib_id: r for r in references if r.bib_id in cited_bib_ids}
    if not refs_by_id:
        return []
    key_sets = {
        bid: [_family_keys_for(f) for f in extract_families(r.authors or "")]
        for bid, r in refs_by_id.items()
    }
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for sent in body_sents:
        text = getattr(sent, "text", "")
        if not text or re.search(r"\b\d{4}\b", text):
            continue  # year-less only; year-bearing sentences go through Tier 2
        for nm in _YEAR_LESS_NAME_RE.finditer(text):
            fams = _extract_families_from_span(nm.group(1))
            if not fams:
                continue
            cand_keys: set[str] = set()
            for f in fams:
                cand_keys |= _family_keys_for(f)
            matches = [
                bid
                for bid, ksets in key_sets.items()
                if ksets and all(ks & cand_keys for ks in ksets)
            ]
            if len(matches) == 1:  # unique only — never guess on ties
                key = (sent.text_id, matches[0])
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out
