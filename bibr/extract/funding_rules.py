"""Structured funders and award numbers from a printed funding statement, with rules only.

A full run hands the funding statement to the LLM, which returns each funder and the award
numbers printed for it. A fast run makes no LLM call, so without this module its funding
statement stays a sentence and ``funding`` is empty. :func:`parse_funding_statement` reads the
same statement locally:

* **Funder names** are runs of capitalised words (with the connectors of organisation names:
  "of", "for", "de", "und", ...) that name an organisation (Foundation, Council, Ministry, ...),
  are an acronym (NSF, CONICET), follow a strong cue ("funded by ..."), or carry a grant number
  in brackets right after them. A name ends before "Grant", "No." or a number, and at a
  possessive. Author initials and author names, places, people after "to", and words such as
  "Grant" or "Program" alone are not funders.
* **Award numbers** are what follows a cue ("grant no.", "Grant Number"), bracketed identifiers,
  two-part identifiers ("R01 MH075750"), and tokens shaped like an identifier anywhere. Years,
  ordinals, call identifiers ("FP7-PEOPLE-...") and citation keys are not.
* **Pairing**: an award goes to the funder right after it when only "from"/"by"/"of" stands
  between ("grant 123 from the NIH"), else to the nearest funder before it in the sentence.
  An acronym and its expansion are one funder.

Sentences that state no funding was received, that say the funders had no role, or that are
about ethics approval or conflicts of interest give nothing.

The rules return what is printed and nothing more: no funder registry lookup, no guessing a
funder for an unnamed grant. They are weaker than the LLM on long multi-agency statements, so
only fast mode (``FUNDING_RULES``) uses them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# --- statement-level filters -----------------------------------------------------------------

_LABEL = re.compile(
    r"^\s*(?:funding(?:\s+(?:information|statement|sources?))?|financial\s+(?:support|disclosure)|"
    r"sources?\s+of\s+funding|grant\s+support|acknowledg(?:e)?ments?)(?:\s*[:.\-–]\s*|\s+(?=[A-Z]))",
    re.I,
)
# A sentence saying nothing was received ("This research received no external funding").
_NO_FUNDING = re.compile(
    r"\b(?:no|not|nor|without|neither)\b[^.]{0,80}?\b(?:fund(?:s|ing|ed)?|financial|grants?|sponsor(?:ship)?|"
    r"support)\b|\bself[- ]funded\b|\bpersonal (?:resources|funds)\b",
    re.I,
)
# ...unless it also says who did fund something.
_POSITIVE_FUNDING = re.compile(
    r"\b(?:supported|funded|financed|sponsored)\s+(?:in part\s+)?by\b|\bfunding from\b|\bgrants? from\b|"
    r"\bfunding (?:was )?provided by\b|\breceived (?:\w+\s+){0,3}(?:funding|support|grants?) from\b",
    re.I,
)
_SKIP_SENTENCE = re.compile(
    r"\bethic|\bapprov(?:al|ed)\b|\bDeclaration of Helsinki\b|\bsolely the responsibility\b|\bofficial views?\b|"
    r"\bconflicts? of interest\b|\bcompeting interests?\b",
    re.I,
)
_FUNDING_WORD = re.compile(r"\b(?:supported|funded|funding|grant)\b", re.I)
_BOILERPLATE = re.compile(
    r"\b(?:funders?|sponsors?|funding (?:bodies|body|agenc(?:y|ies)|sources?))\b[^.]{0,40}\b(?:no|any)\s+"
    r"(?:role|involvement|influence)\b|\bdeclare[sd]?\b[^.]{0,60}\bfinancial support was received\b|"
    r"\bdisclosed receipt of the following\b",
    re.I,
)
_FOLLOWING_FUNDED = re.compile(r"\b(?:supported|funded|funding from|grant)\b[^.]*\b[A-Z]{2,}")
# A line-break hyphen inside an identifier: "CMS- CSST-2021-A01".
_BROKEN_HYPHEN = re.compile(r"(?<=[A-Z0-9])-\s+(?=[A-Z0-9])")
_SENTENCE_END = re.compile(
    r"(?<!\bNo)(?<!\bno)(?<!\bNos)(?<!\bU\.S)(?<!\bInc)(?<!\bDr)(?<!\bSt)(?<!\b[A-Z])\.\s+(?=[A-Z])"
)

# --- funder names ------------------------------------------------------------------------------

# Words that make a run of capitalised words an organisation.
_HEAD = re.compile(
    r"\b(?:foundation|foundations|council|councils|fund|funds|funding|agency|agencies|ministry|ministries|"
    r"institute|institutes|trust|society|academy|commission|office|department|administration|university|"
    r"universit[yi]es|organi[sz]ation|association|centre|center|centers|centres|stiftung|fundação|fundación|"
    r"fundacion|fondation|fondo|fonds|consejo|conselho|conseil|deanship|government|bank|federation|charity|"
    r"alliance|board|authority|laboratory|laboratories|hospital|college|school|corporation|company|inc|ltd|"
    r"gmbh|endowment|secretariat|directorate|forschungsgemeinschaft|kakenhi)\b",
    re.I,
)
# Lower-case words that may sit inside an organisation's name.
_CONNECT = frozenset(
    {"of", "for", "and", "the", "de", "la", "del", "des", "du", "da", "do", "di", "der", "für", "y", "e", "et",
     "&", "in", "on", "à", "zur", "pour", "en", "em", "und", "voor", "van", "von"}
)  # fmt: skip
# A funder name ends before these: "NSF Grant 1931962", "Dutch Research Council Vici Grant ...".
_NAME_STOP = re.compile(
    r"^(?:grants?|awards?|no\.?|nos\.?|numbers?|contract|vici|vidi|veni|fellowships?|scholarships?|senior|"
    r"junior|investigators?|fellows?|professors?|postdoctoral|graduate|phd|ph\.d\.|pi|ref\.?|consolidator|"
    r"starting|advanced|career|synergy|k-award|start-up|seed|exploratory)$",
    re.I,
)
# Phrases that are never a funder on their own.
_NOT_FUNDER = frozenset(
    {"funding", "financial", "support", "grant", "grants", "number", "no", "the", "this", "we", "our", "they",
     "he", "she", "it", "authors", "author", "study", "research", "work", "project", "open", "access", "part",
     "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement", "contract", "award", "awards",
     "fellowship", "program", "programme", "proposal", "dr", "prof", "professor", "mr", "ms", "mrs", "thanks",
     "thank", "conflicts", "conflict", "competing", "interest", "interests", "data", "ethics", "financing", "code",
     "agreement", "nos", "pi", "phd", "md", "msc", "ref", "grant-in-aid", "id", "ga", "core", "covid-19", "hiv"}
)  # fmt: skip
# Words a name never starts with.
_START_SKIP = frozenset(
    {"the", "this", "these", "that", "we", "our", "they", "he", "she", "his", "her", "it", "its", "funding",
     "financial", "part", "in", "a", "an", "all", "both", "each", "no", "nos", "and", "or", "also", "however",
     "data", "development", "collection", "preparation", "acknowledgments", "acknowledgements", "support"}
)  # fmt: skip
_PLACES = frozenset(
    {"argentina", "mexico", "jordan", "china", "japan", "brazil", "germany", "france", "spain", "italy", "canada",
     "usa", "uk", "india", "korea", "norway", "sweden", "finland", "denmark", "netherlands", "the netherlands",
     "switzerland", "austria", "belgium", "poland", "portugal", "greece", "chile", "colombia", "peru",
     "australia", "new zealand", "ireland", "scotland", "england", "wales", "europe", "united states",
     "united kingdom", "russia", "turkey", "iran", "israel", "egypt", "south africa", "nigeria", "kenya", "taiwan",
     "singapore", "malaysia", "indonesia", "thailand", "vietnam", "estonia", "latvia", "lithuania",
     "czech republic", "hungary", "romania", "croatia", "serbia", "slovenia", "slovakia", "bulgaria", "ukraine",
     "pakistan", "bangladesh", "saudi arabia", "qatar", "uae", "princeton"}
)  # fmt: skip
# One-word funders that are neither an acronym nor an organisation word.
_KNOWN_SINGLE = frozenset(
    {"Wellcome", "Inria", "Novartis", "Pfizer", "Google", "Amazon", "Microsoft", "Meta", "IBM", "Intel",
     "Nvidia", "NVIDIA", "Horizon", "Erasmus", "Leverhulme", "Nuffield", "Carlsberg", "Volkswagen", "Mitacs",
     "Fulbright", "Rockefeller", "Gates", "Templeton"}
)  # fmt: skip
# Lower-case words an acronym funder may be followed by ("NSF grant", "ERC research grant"); any
# other ("HPC resources") says the acronym names a thing, not a funder.
_ACRONYM_FOLLOWERS = frozenset(
    {"grant", "grants", "award", "awards", "project", "projects", "and", "or", "under", "through", "via", "for",
     "to", "in", "research", "funding", "fellowship", "program", "programme", "as", "with", "is", "which"}
)  # fmt: skip
# A capitalised or all-caps word, inner hyphens, dots, apostrophes and ampersands allowed (U.S., R&D).
_CAPWORD = re.compile(r"(?:[A-Z][\w’'.&-]*|\d+(?:st|nd|rd|th)\b)")
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
_WORD = re.compile(r"[\w’'.&+-]+|[()\[\]]")
_ACRONYM = re.compile(
    r"^(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])[A-Z][A-Za-z&-]{1,9}$"
)  # NSF, ANPCyT, CONICET
_POSSESSIVE = re.compile(r"[’']s$")
_EXPANSION = re.compile(r"\s*\(([A-Z][^()]{6,90})\)")  # "MRC (Medical Research Council)"
_ACRONYM_AFTER = re.compile(r"\s*\(([A-Z][A-Za-z&.\-]{1,12})\)")  # "Medical Research Council (MRC)"
_JOIN = re.compile(r"\s+(?:and|&)\s+(?:the|The)\s+|\s+(?:and|&)\s+(?=[A-Z])")
_STRONG_CUE = re.compile(
    r"\b(?:supported|funded|financed|sponsored|supplied|provided)\s+(?:in part\s+|partly\s+|partially\s+)?by\s+"
    r"(?:the\s+)?$|\b(?:funding|support|grants?|donations?)\s+from\s+(?:the\s+)?$",
    re.I,
)
_PERSON_CUE = re.compile(r"\b(?:to|awarded to|Dr\.?|Prof\.?)\s*$")
_PERSON = re.compile(r"[A-Z][\w’'\-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][\w’'\-]+){1,2}")
_INITIALS = re.compile(r"(?:[A-Z]\.?){2,4}")
_INITIALS_SUBJECT = re.compile(r"\s+(?:was|is|were|are|has|have|acknowledges?|received|thanks?)\b")
_BARE_GRANT = re.compile(r"(?:Grant|Grants|Award|Awards|Project|Contract)(?:\s+\w+)?")
_ID_IN_BRACKETS = re.compile(
    r"\s*[,]?\s*[(\[]\s*((?:grant|project|award)?\s*(?:no\.?|nos\.?|numbers?|n[o°º◦]\.?)?\s*[:.]?\s*)?"
    r"([^()\[\]]{2,60})[)\]]",
    re.I,
)
_YEARS = re.compile(r"(?:19|20)\d\d(?:\s*[-–]\s*(?:19|20)\d\d)?")

# --- award numbers -----------------------------------------------------------------------------

_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9]*(?:[-–‐/._\\][A-Za-z0-9]+)*"
_ID_TOKEN_RE = re.compile(_ID_PATTERN)
_ID_CUE = re.compile(
    r"\b(?:grants?|awards?|projects?|contracts?|agreements?|proposals?|fellowships?|ref(?:erence)?s?|codes?)?"
    r"\s*(?:no\.?|nos\.?|n[o°º◦]\.?|numbers?|#|id|ids)\s*[:.]?\s*|"
    r"\b(?:grants?|awards?|projects?|contracts?|agreements?)\s*[:#]?\s*(?=[A-Z0-9])",
    re.I,
)
_CUED_ID = re.compile(r"\s*(" + _ID_PATTERN + r"(?:\s(?=[A-Z0-9]{1,4}\s?\d)" + _ID_PATTERN + r")?)")
_ID_LIST_SEP = re.compile(r"\s*(?:,\s*(?:and\s+)?|and\s+|&\s*|/\s*(?=[A-Z]{2,}\d))")
_BRACKETED = re.compile(r"[(\[]([^()\[\]]{2,120})[)\]]")
_BRACKET_TAIL = re.compile(r"\s+(?:to|awarded|for|from)\s.*$")  # "[grant number P12-1017 to MJO]"
_BRACKET_PART = re.compile(r"(?:(?!\s+and\s+)[^,;&])+")
_TWO_PART_ID = re.compile(
    r"\b(?:\d?[A-Z]\d{2}|[A-Z]{2,4})\s(?=[A-Z]{0,3}\d{4,}(?:[-/][\w]+)*\b)([A-Z]{0,3}\d{4,}(?:[-/][\w]+)*)"
)
_TWO_PART_END = re.compile(r"\s*[)\];,.]|\s*$")
_LOOSE_ID = re.compile(r"(?<![\w/\\-])" + _ID_PATTERN)
_URL_BEFORE = re.compile(r"https?://\S*$")
_ID_PREFIX = re.compile(
    r"^(?:grants?|awards?|ref(?:erence)?|no\.?|nos\.?|number)\s*[:.#]?\s*", re.I
)
_ID_LABEL = re.compile(
    r"^(?:[\w.\-]+\s*:\s*)+(?=\S*\d)"
)  # "RaReMem: 101043804", "PI: Fazeli: K99..."
_YEAR = re.compile(r"^(?:19|20)\d\d$")
_CITATION_KEY = re.compile(r"[A-Z][a-z]{2,}\d{2}[a-z]?")
_CALL_ID = re.compile(r"(?:FP7|H2020|HORIZON)-", re.I)
_NOT_ID = frozenset(
    {"COVID-19", "SARS-CoV-2", "FP7", "H2020", "2020", "E3SM", "CO2", "R&D", "B2", "3D", "4TY"}
)

# --- pairing -------------------------------------------------------------------------------------

# Between an award and the funder after it: "grant 123 from the NIH", "(...) awarded by the ...".
_FORWARD_GAP = re.compile(
    r"\s*[\])]?\s*,?\s*(?:(?:awarded|provided|funded|granted)\s+)?(?:from|by|of)\s+(?:the\s+)?",
    re.I,
)

Span = tuple[int, int, str]


def parse_funding_statement(
    statement: str, authors: Sequence[tuple[str, str]] = ()
) -> list[tuple[str, list[str]]]:
    """The funders a funding statement names, each with the award numbers printed for it.

    ``authors`` are the paper's ``(given, family)`` names: their initials and family names are
    not read as funders ("CP was funded by ...", "awarded to Amit Bernstein"). Funders come in
    order of first mention; a funder named twice appears once with the awards of both mentions.
    """
    text = _BROKEN_HYPHEN.sub("-", _LABEL.sub("", statement or "", count=1))
    initials = _author_initials(authors)
    families = {f.strip() for _, f in authors if f and len(f.strip()) >= 3}
    result: list[tuple[str, list[str]]] = []
    index: dict[str, int] = {}
    for start, end in _sentences(text):
        sentence = text[start:end]
        if _SKIP_SENTENCE.search(sentence) and not _FUNDING_WORD.search(sentence):
            continue
        if _BOILERPLATE.search(sentence):
            # "The authors disclosed receipt of the following financial support ...: X was ..."
            if "following" in sentence and ":" in sentence:
                sentence = sentence.split(":", 1)[1]
            elif not (
                "following" in sentence and _FOLLOWING_FUNDED.search(sentence.split(":", 1)[-1])
            ):
                continue
        funders = _funder_spans(sentence, initials, families)
        awards = _award_spans(sentence)
        if _NO_FUNDING.search(sentence) and not awards and not _POSITIVE_FUNDING.search(sentence):
            continue
        funders = [f for f in funders if not any(a[0] <= f[0] and f[1] <= a[1] for a in awards)]
        assigned: dict[int, list[str]] = {k: [] for k in range(len(funders))}
        for award in awards:
            target = _award_target(sentence, award, funders)
            if target is not None:
                assigned[target].append(award[2])
            elif result:  # an award in a sentence of its own goes to the last funder named
                result[-1][1].append(award[2])
        for k, (_, _, name) in enumerate(funders):
            keys = _name_keys(name)
            if not keys:
                continue
            known = next((key for key in keys if key in index), None)
            if known is not None:
                awards_so_far = result[index[known]][1]
                awards_so_far.extend(a for a in assigned[k] if a not in awards_so_far)
            else:
                for key in keys:
                    index.setdefault(key, len(result))
                result.append((name, list(dict.fromkeys(assigned[k]))))
    return result


def _sentences(text: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        bounds.append((start, m.start() + 1))
        start = m.end()
    bounds.append((start, len(text)))
    return bounds


def _author_initials(authors: Sequence[tuple[str, str]]) -> set[str]:
    """Each author's initials, with and without the middle names: "CP", "JGB", "JB"."""
    out: set[str] = set()
    for given, family in authors:
        given_words = [w for w in re.split(r"[\s.\-]+", given or "") if w]
        family_words = [w for w in re.split(r"[\s\-]+", family or "") if w]
        if not family_words:
            continue
        g = "".join(w[0] for w in given_words).upper()
        f = "".join(w[0] for w in family_words).upper()
        out |= {g + f, g + f[:1], g[:1] + f, g[:1] + f[:1]}
    return {x for x in out if len(x) >= 2}


def _has_digit(word: str) -> bool:
    return any(c.isdigit() for c in word) and not _ORDINAL.fullmatch(word)


def _candidate_names(text: str) -> list[Span]:
    """Runs of capitalised words joined by name connectors, with an acronym or expansion after."""
    toks = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]
    spans: list[Span] = []
    i = 0
    while i < len(toks):
        word, start, _ = toks[i]
        if (
            not _CAPWORD.match(word)
            or _has_digit(word)
            or word.lower().strip(".") in _START_SKIP
            or _NAME_STOP.match(word)
        ):
            i += 1
            continue
        j = last = i
        while j + 1 < len(toks):
            nxt = toks[j + 1][0]
            gap = text[toks[j][2] : toks[j + 1][1]]
            if any(c in gap for c in ",;:\n") or _has_digit(nxt) or _NAME_STOP.match(nxt):
                break
            if _POSSESSIVE.search(toks[j][0]):
                break
            if _CAPWORD.match(nxt):
                j = last = j + 1
            elif nxt.lower() in _CONNECT:
                j += 1
            else:
                break
        end = toks[last][2] - (2 if _POSSESSIVE.search(toks[last][0]) else 0)
        name = text[start:end].strip(" .")
        expansion = _EXPANSION.match(text, end)
        if (
            expansion
            and _ACRONYM.match(name.replace(".", ""))
            and _HEAD.search(expansion.group(1))
            and not re.search(r"\d", expansion.group(1))
        ):
            end = expansion.end()
        else:
            acronym = _ACRONYM_AFTER.match(text, end)
            if acronym and _ACRONYM.match(acronym.group(1).replace(".", "")):
                end = acronym.end()
        spans.append((start, end, text[start:end].strip() if end > start else name))
        i = last + 1
        while i < len(toks) and toks[i][1] < end:
            i += 1
    return spans


def _split_joined(text: str, spans: list[Span]) -> list[Span]:
    """Split "X and the Y", and "X and Y" when both sides name an organisation."""
    out: list[Span] = []
    for start, end, _ in spans:
        cut = start
        for m in _JOIN.finditer(text, start, end):
            left, right = text[cut : m.start()], text[m.end() : end]
            if "the" in m.group().lower() or (
                _HEAD.search(left) and _HEAD.search(right) and len(right.split()) > 1
            ):
                out.append((cut, m.start(), text[cut : m.start()].strip(" .")))
                cut = m.end()
        out.append((cut, end, text[cut:end].strip(" .")))
    return out


def _funder_spans(text: str, initials: set[str], families: set[str]) -> list[Span]:
    out: list[Span] = []
    for start, end, name in _split_joined(text, _candidate_names(text)):
        if _is_funder(text, start, end, name, initials, families):
            out.append((start, end, name))
    return out


def _is_funder(
    text: str, start: int, end: int, name: str, initials: set[str], families: set[str]
) -> bool:
    bare = re.sub(r"\s*\([^)]*\)$", "", name).strip()
    words = bare.split()
    low = bare.lower().strip(".")
    if not words or low in _PLACES or low in _NOT_FUNDER:
        return False
    before = text[max(0, start - 40) : start]
    if not _HEAD.search(bare) and (
        any(re.search(rf"\b{re.escape(f)}\b", bare) for f in families)
        or (_PERSON_CUE.search(before) and _PERSON.fullmatch(bare))
    ):
        return False  # a person: an author, or a name after "awarded to"
    parts = [x for x in re.split(r"\s+(?:and|&)\s+|\s*,\s*", bare) if x]
    if len(parts) > 1 and all(_INITIALS.fullmatch(x.strip()) for x in parts):
        return False  # "JH and NM"
    if len(words) == 1:
        word = words[0].strip(".")
        is_acronym = bool(_ACRONYM.match(word))
        if word.upper() in initials and not _HEAD.search(word):
            return False
        if not (is_acronym or word in _KNOWN_SINGLE):
            return False
        if re.fullmatch(r"[A-Z]{2,4}", word) and _INITIALS_SUBJECT.match(text, end):
            return False  # "CP was funded by ..."
        if is_acronym:
            follow = re.match(r"\s+([a-z]+)", text[end:])
            if follow and follow.group(1) not in _ACRONYM_FOLLOWERS:
                return False  # "HPC resources"
    elif not (
        _HEAD.search(bare)
        or any(_ACRONYM.match(x.strip(".")) for x in words)
        or (_STRONG_CUE.search(before) and len(words) <= 5)
        or (_followed_by_id(text, end) and 2 <= len(words) <= 16)
    ):
        return False
    return not _BARE_GRANT.fullmatch(bare)


def _followed_by_id(text: str, end: int) -> bool:
    """A name directly followed by its grant number: "... of Fujian Province (No. 2018J05042)"."""
    m = _ID_IN_BRACKETS.match(text, end)
    return bool(m and re.search(r"\d", m.group(2)) and not _YEARS.fullmatch(m.group(2).strip()))


def _id_shape(token: str, cued: bool) -> bool:
    """Whether ``token`` looks like an award number; a cue ("grant no.") lowers the bar."""
    t = token.strip(".,;:()[]")
    if not t or t in _NOT_ID or t.upper() in _NOT_ID:
        return False
    digits = sum(c.isdigit() for c in t)
    if not digits or _YEAR.match(t) or re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", t):
        return False
    if _CITATION_KEY.fullmatch(t) or _CALL_ID.match(t):
        return False
    if cued:
        return len(t) >= 3
    if t.isdigit():
        return len(t) >= 5
    return len(t) >= 5 and digits >= 3


def _award_spans(text: str) -> list[Span]:
    out: list[Span] = []
    taken = [False] * (len(text) + 1)

    def add(start: int, end: int) -> None:
        value = _ID_LABEL.sub("", _ID_PREFIX.sub("", text[start:end].strip(" .,;:"))).strip(" .,;:")
        if not value or any(taken[start:end]):
            return
        taken[start:end] = [True] * (end - start)
        out.append((start, end, value))

    # 1) after a cue: "grant no. X", "Grant Number 15K08321", "grant numbers 17 K13947, 26780385"
    for m in _ID_CUE.finditer(text):
        pos = m.end()
        while (seg := _CUED_ID.match(text, pos)) is not None:
            token = seg.group(1)
            if not _id_shape(token.replace(" ", ""), True) or token.isalpha():
                break
            add(seg.start(1), seg.end(1))
            sep = _ID_LIST_SEP.match(text, seg.end())
            if sep is None:
                break
            pos = sep.end()
    # 2) bracketed: "(PIP 0662)", "(31472207)", "(No. CFH-2020-2-40911)", "[2016-0-005630031001, ...]"
    for m in _BRACKETED.finditer(text):
        inner = _BRACKET_TAIL.sub("", m.group(1))
        for part in _BRACKET_PART.finditer(inner):
            raw = part.group()
            stripped = _ID_CUE.sub("", raw, count=1) if _ID_CUE.match(raw.strip()) else raw
            candidate = stripped.strip()
            if not candidate or "http" in candidate or "www" in candidate:
                continue
            base = m.start(1) + part.start() + raw.find(candidate)
            words = candidate.split()
            if len(words) > 3 or any(re.fullmatch(r"[a-z]{4,}", w) for w in words):
                # prose in the brackets: keep only identifier-looking tokens
                for t in _ID_TOKEN_RE.finditer(candidate):
                    if _id_shape(t.group(), False):
                        add(base + t.start(), base + t.end())
                continue
            if (
                _id_shape(candidate.replace(" ", ""), True)
                and not re.fullmatch(r"[A-Z][a-z]+", candidate)
                and not (len(words) == 2 and not re.search(r"\d", words[1]))
            ):
                add(base, base + len(candidate))
    # 3) two-part identifiers: NIH activity code + serial ("R01 MH075750"), "GRK 2185/2"
    for t in _TWO_PART_ID.finditer(text):
        if re.fullmatch(r"[A-Z]{2,4}", t.group().split()[0]) and not _TWO_PART_END.match(
            text, t.end()
        ):
            continue
        add(t.start(), t.end())
    # 4) identifier-shaped tokens anywhere, outside URLs
    for t in _LOOSE_ID.finditer(text):
        if _id_shape(t.group(), False) and not _URL_BEFORE.search(
            text[max(0, t.start() - 80) : t.start()]
        ):
            add(t.start(), t.end())
    out.sort()
    return out


def _award_target(sentence: str, award: Span, funders: list[Span]) -> int | None:
    """Index of the funder ``award`` belongs to, or None when the sentence names none."""
    after = [k for k, f in enumerate(funders) if f[0] >= award[1]]
    before = [k for k, f in enumerate(funders) if f[1] <= award[0]]
    if after:
        gap = sentence[award[1] : funders[after[0]][0]]
        if len(gap.split()) <= 4 and _FORWARD_GAP.fullmatch(gap):
            return after[0]
    if before:
        return before[-1]
    return after[0] if after else None


def _name_keys(name: str) -> set[str]:
    """Spellings one funder goes by: its name, without the brackets, and what the brackets hold."""

    def norm(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", x.lower())

    keys = {norm(name), norm(re.sub(r"\s*\([^)]*\)", "", name))}
    for inner in re.findall(r"\(([^)]*)\)", name):
        if _ACRONYM.match(inner.replace(".", "").replace("-", "")) or len(inner.split()) > 1:
            keys.add(norm(inner))
    return {k for k in keys if len(k) >= 2}
