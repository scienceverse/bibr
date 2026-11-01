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

from bibr.models import PaperReference
from bibr.paper_contents import CitationCandidate, PaperSentence, PaperXref
from bibr.utils.text import IN_PRESS_LABEL, collapse_ws

# ---------------------------------------------------------------------------
# Author family extraction
# ---------------------------------------------------------------------------

# Capitalized name token — allows hyphens, apostrophes, Latin extended.
_FAMILY_CORE = r"[A-Z][a-zA-ZÀ-ɏ'\-]+"


def _is_initials(s: str) -> bool:
    """True if s looks like author initials (e.g., 'J.', 'J. A.', 'JA')."""
    s = s.strip().rstrip(".")
    if not s or len(s) > 12:
        return False
    alphas = [c for c in s if c.isalpha()]
    if not alphas:
        return False
    # An unspaced, undotted all-caps run of four letters or more ("NGUYEN") is
    # a surname printed in capitals; initials come as "JA", "J. A." or "J.A.".
    if len(s) >= 4 and s.isalpha() and s.isupper():
        return False
    upper_ratio = sum(1 for c in alphas if c.isupper()) / len(alphas)
    return len(alphas) <= 6 and upper_ratio >= 0.6 and all(c.isalpha() or c in ". " for c in s)


def extract_families(authors_str: str | None) -> list[str]:
    """Best-effort APA-style author parsing — returns family names in order.

    Handles APA "Family, J. A., Family, J., & Family, J." patterns plus
    multi-word names ("Della Sala", "van der Vyver"), hyphenated names
    ("Fei-Fei"), continuation dots ". . ." and "and"/"&" connectors.
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
        while suffix_start > 0 and _is_initials(tokens[suffix_start - 1]):
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
        if _is_initials(cur):
            i += 1
            continue
        families.append(cur)
        if i + 1 < len(parts) and _is_initials(parts[i + 1]):
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


def _family_key(family: str) -> str:
    """Normalize a family name for matching.

    Strips trailing possessive ``’s`` / ``’s`` / ``‘s``
    ("Perruchet’s" -> "perruchet", "Andersson’s" -> "andersson"),
    trailing lone apostrophes, trailing punctuation, leading marker characters
    ("*Berg", "†Berg"), and lowercases.
    """
    s = _LEADING_NON_WORD_RE.sub("", family.strip()).rstrip(",.").lower()
    s = _POSSESSIVE_RE.sub("", s)
    # Citations often omit reference-list diacritics (Garcia vs García). Keep
    # the displayed family untouched, but compare on a decomposed ASCII key.
    return "".join(
        char for char in unicodedata.normalize("NFKD", s) if not unicodedata.combining(char)
    )


def _family_keys_for(family: str) -> set[str]:
    """All match-keys for a family name. Multi-word names also index under
    their last token so a candidate ``Toorn`` can match a ref ``van der Toorn``.
    """
    base = _family_key(family)
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
    # Authors, year and the first 40 characters of the title, folded: two
    # entries with the same key are one work listed twice. None when the entry
    # prints no author or no title, so nothing vouches that it is a repeat.
    duplicate_key: tuple[str, int | None, str] | None = None


def _build_ref_index(
    references: list[PaperReference],
) -> tuple[
    list[_RefIdx],
    dict[tuple[int | None, str | None], list[_RefIdx]],
    list[_RefIdx],
]:
    """Build (refs, year_suffix_index, in_press_refs) from PaperReference list.

    A work listed twice (the same authors, year and title start) is indexed
    once, under its first entry: a citation of it is not ambiguous. Entries
    that print different year suffixes ("2018a", "2018b") stay apart, since
    a citation names one of them by its suffix.
    """
    refs: list[_RefIdx] = []
    in_press: list[_RefIdx] = []
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]] = defaultdict(list)
    listed: set[tuple[tuple[str, int | None, str], str | None]] = set()
    for r in references:
        is_in_press = bool(getattr(r, "is_in_press", False) or r.year == 0)
        year = None if r.year in (0, None) else r.year
        families = tuple(extract_families(r.authors))
        authors, title = _fold(r.authors).strip(), _fold(r.title).strip()[:40]
        idx = _RefIdx(
            bib_id=r.bib_id,
            year=year,
            year_suffix=r.year_suffix,
            families=families,
            is_in_press=is_in_press,
            family_key_sets=tuple(frozenset(_family_keys_for(f)) for f in families),
            duplicate_key=(authors, year, title) if authors and title else None,
        )
        refs.append(idx)
        if idx.duplicate_key is not None:
            slot = (idx.duplicate_key, idx.year_suffix)
            if slot in listed:
                continue
            listed.add(slot)
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


def _find_candidates(text: str, text_id: int) -> list[_Candidate]:
    """Walk a sentence; return all parenthetical + narrative citation candidates."""
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
                            has_et_al="et al" in sub.lower(),
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
                        has_et_al="et al" in sub.lower(),
                        is_in_press=True,
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
                    has_et_al="et al" in prefix.lower(),
                    surface_kind=surface_kind,
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
                has_et_al="et al" in prefix.lower(),
                is_in_press=True,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _overlap_exact(cited: frozenset[str], listed: frozenset[str]) -> bool:
    """A cited family and a listed author share a match-key."""
    return not cited.isdisjoint(listed)


# The prefix match needs at least this many characters on both sides, so that
# "Lee" never reaches "Leeuwen" and "Li" never reaches "Lindqvist".
_PREFIX_MIN_LEN = 4


def _compact(key: str) -> str:
    return key.replace(" ", "").replace("-", "").replace("\u2010", "")


def _overlap_prefix(cited: frozenset[str], listed: frozenset[str]) -> bool:
    """A cited family and a listed author agree once spaces and hyphens are
    dropped, or one starts with the other: "Berg" reaches a "Family Given"
    entry parsed as the family "Berg Anna"."""
    for cited_key in map(_compact, cited):
        for listed_key in map(_compact, listed):
            if cited_key == listed_key:
                return True
            if min(len(cited_key), len(listed_key)) >= _PREFIX_MIN_LEN and (
                cited_key.startswith(listed_key) or listed_key.startswith(cited_key)
            ):
                return True
    return False


_Overlap = Callable[[frozenset[str], frozenset[str]], bool]


def _score_ref(
    ref: _RefIdx, cand: _Candidate, overlaps: _Overlap = _overlap_exact
) -> tuple[int, int, int]:
    """Score how well ``ref`` matches ``cand``. Higher is better.

    Returns ``(matched_families, et_al_bonus, neg_extra_authors)`` for tuple
    sort. The third element prefers refs with fewer extra authors when match
    counts tie.
    """
    cited = [_family_keys_for(f) for f in cand.families]
    ref_key_sets = ref.family_key_sets
    if not ref_key_sets or not any(overlaps(keys, ref_key_sets[0]) for keys in cited):
        return (0, 0, 0)
    matched = sum(1 for author in ref_key_sets if any(overlaps(keys, author) for keys in cited))
    et_al_bonus = 0
    if cand.has_et_al:
        # "et al." supports a multi-author ref, but list length is never
        # disambiguating evidence when first family and year collide.
        et_al_bonus = min(len(ref_key_sets), 5) if len(ref_key_sets) >= 3 else -2
    extra = len(ref_key_sets) - matched
    return (matched, et_al_bonus, -extra)


def _tie_key(ref: _RefIdx, cand: _Candidate, overlaps: _Overlap) -> tuple[bool, int, int]:
    """Rank a reference that ties with others on the same family and year.

    Returns ``(count_ok, order_ok, any_author)``. ``count_ok``: the number of
    listed authors fits how the citation names them: "et al." needs three or
    more, one cited family needs a single-author reference, otherwise the
    counts are equal. ``order_ok``: cited families that match the listed author
    at the same position. ``any_author``: cited families that match any listed
    author.
    """
    cited = [_family_keys_for(family) for family in cand.families]
    listed = ref.family_key_sets
    if cand.has_et_al:
        count_ok = len(listed) >= 3
    elif len(cited) == 1:
        count_ok = len(listed) == 1
    else:
        count_ok = len(listed) == len(cited)
    order_ok = sum(1 for keys, author in zip(cited, listed, strict=False) if overlaps(keys, author))
    any_author = sum(1 for keys in cited if any(overlaps(keys, author) for author in listed))
    return count_ok, order_ok, any_author


def _break_tie(tied: list[_RefIdx], cand: _Candidate, overlaps: _Overlap) -> _RefIdx | None:
    """The one reference of ``tied`` that fits the citation best, if any.

    It must fit the author count and strictly out-rank every other reference
    on ``_tie_key``; two references that fit equally stay tied (a bare "et al."
    does not tell a three-author reference from a six-author one).
    """
    ranked = sorted(
        ((_tie_key(ref, cand, overlaps), ref) for ref in tied), key=lambda p: p[0], reverse=True
    )
    best_key, best = ranked[0]
    if best_key[0] and best_key > ranked[1][0]:
        return best
    return None


def _match_pool(
    pool: list[_RefIdx], cand: _Candidate, overlaps: _Overlap
) -> tuple[_RefIdx | None, list[int]]:
    """Score ``cand`` against the references of its year; see ``_match``."""
    # A bare first-family/year citation cannot distinguish references that
    # share that first family and year by the author list alone: a coauthor or
    # year suffix must disambiguate. The one thing the count of authors does
    # settle is a single-author reference against multi-author ones, because a
    # work by several authors is not cited by its first author alone; an
    # "et al." citation separates three or more authors from one or two, never
    # three from six (``_break_tie``).
    if len(cand.families) == 1 and cand.year_suffix is None:
        candidate_keys = _family_keys_for(cand.families[0])
        same_first = [
            ref
            for ref in pool
            if ref.family_key_sets and overlaps(candidate_keys, ref.family_key_sets[0])
        ]
        if len(same_first) > 1:
            winner = _break_tie(same_first, cand, overlaps)
            if winner is not None:
                return winner, []
            return None, sorted(ref.bib_id for ref in same_first)

    scored = [(r, _score_ref(r, cand, overlaps)) for r in pool]
    scored = [(r, s) for r, s in scored if s[0] > 0]
    if not scored:
        return None, []

    scored.sort(key=lambda x: x[1], reverse=True)
    if len(scored) > 1 and scored[1][1] == scored[0][1]:
        top_score = scored[0][1]
        top = [r for r, s in scored if s == top_score]
        winner = _break_tie(top, cand, overlaps)
        if winner is not None:
            return winner, []
        return None, [r.bib_id for r in top]
    return scored[0][0], []


# Looser ways to read a family name, tried in turn only when no listed author
# shares a match-key with the citation.
_LOOSE_OVERLAPS: tuple[_Overlap, ...] = (_overlap_prefix,)


def _match(
    cand: _Candidate,
    by_year: dict[tuple[int | None, str | None], list[_RefIdx]],
    in_press_refs: list[_RefIdx],
) -> tuple[_RefIdx | None, list[int]]:
    """Resolve a candidate to a ref.

    Returns ``(ref_or_None, tied_bib_ids)``:

    - ``(ref, [])`` — unique top scorer.
    - ``(None, [bib_id, ...])`` — 2+ refs tied at the top score; caller can
      route this to a Tier 3 disambiguator using ``tied_bib_ids`` as a
      shortlist.
    - ``(None, [])`` — no candidate ref scored > 0; nothing to disambiguate.

    Names are compared by their match-keys first. Only when no reference of the
    year shares one with the citation are the looser readings tried
    (``_LOOSE_OVERLAPS``); they link a single reference or none, and never
    offer a shortlist, so a citation the exact keys could not place is never
    narrowed to a wrong pair.
    """
    pool: list[_RefIdx]
    if cand.is_in_press:
        pool = in_press_refs
    else:
        pool = []
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

    if not pool:
        return None, []

    ref, tied = _match_pool(pool, cand, _overlap_exact)
    if ref is not None or tied:
        return ref, tied
    for overlaps in _LOOSE_OVERLAPS:
        ref, tied = _match_pool(pool, cand, overlaps)
        if ref is not None:
            return ref, []
        if tied:
            break
    return None, []


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
) -> tuple[list[PaperXref], list[AmbiguousCandidate], list[CitationCandidate]]:
    """Run Tier 2 matching and retain every accepted/rejected source span.

    Caller is responsible for feeding only body sentences (use
    ``_get_body_sentences`` to filter references / abstract / title /
    captions / footnotes).
    """
    if not references:
        return [], [], []

    _refs, by_year, in_press_refs = _build_ref_index(references)

    xrefs: list[PaperXref] = []
    ambiguous: list[AmbiguousCandidate] = []
    candidates: list[CitationCandidate] = []
    seen: set[tuple[int, int]] = set()
    seen_amb: set[tuple[int, str]] = set()

    for sent in body_sents:
        if not sent.text:
            continue
        for cand in _find_candidates(sent.text, sent.text_id):
            ref, tied = _match(cand, by_year, in_press_refs)
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
                        evidence=("family_match", "year_match", *surface_evidence),
                        confidence=0.95,
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
) -> tuple[list[PaperXref], list[AmbiguousCandidate]]:
    """Run Tier 2 matching; return unambiguous xrefs and ambiguous ties."""

    xrefs, ambiguous, _candidates = match_with_candidates(body_sents, references)
    return xrefs, ambiguous


def detect_author_year_xrefs(
    body_sents: list[PaperSentence],
    references: list[PaperReference],
) -> list[PaperXref]:
    """Detect author-year citations; return only unambiguous matches.

    Thin wrapper around ``match_with_ambiguous``. Use that function directly
    if you also need the ambiguous-tie list for Tier 3 routing.
    """
    xrefs, _ = match_with_ambiguous(body_sents, references)
    return xrefs


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
