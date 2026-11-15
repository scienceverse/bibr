"""The parts of a printed affiliation string, with rules only.

A full run hands the paper's affiliation strings to the LLM, which returns each one's
institution, department, city and country. A fast run makes no LLM call, so its affiliation
rows carry the printed text alone. :func:`parse_affiliation` splits one string the way it is
usually printed, department first and country last
("Department of Psychology, Utrecht University, Utrecht, The Netherlands"):

* The string is cut at commas and semicolons outside brackets. An email address, a
  correspondence note and a journal's running head read into it come off first, and so do
  a role ("Assistant Professor") and a person ("Mario Zanon is with the ...").
* **Country** is the last segment, or the end of it ("Cambridge MA USA"), when it names a
  country (in English or the country's own language) or is a bracketed country at the end.
* Street addresses, post-office boxes, postcodes and state codes are dropped.
* **Institution** is the last segment that names a university; without one, the last that
  does not start like a department ("Division of ...", "School of ..."). A department and its
  university printed without a comma between them are split.
* **City** is the first place-like segment (a few capitalised words, no digits, no
  organisation word) after the university, or without one the last place-like segment. A
  region after the city ("Muncie, Indiana") is not the city.
* **Department** is what is left before the institution, and the segments after it that are
  not places.

The rules only split what is printed: no country is inferred from a city or a region, and
no registry is consulted. Checked against an LLM's parse of strings they were not tuned on,
country and city agree far more often than institution and department, so the pipeline
(``AFFILIATION_RULES``) fills only the city and the country.
"""

from __future__ import annotations

import re
from typing import TypedDict


class AffiliationParts(TypedDict):
    institution: str | None
    department: str | None
    city: str | None
    country: str | None


_COUNTRIES = frozenset(
    {"afghanistan", "albania", "algeria", "argentina", "armenia", "australia", "austria",
     "azerbaijan", "bahrain", "bangladesh", "belarus", "belgique", "belgium", "belgië", "bolivia",
     "bosnia and herzegovina", "botswana", "brasil", "brazil", "brunei", "bulgaria", "cameroon",
     "can", "canada", "chile", "china", "colombia", "costa rica", "croatia", "cuba", "cyprus",
     "czech republic", "czechia", "danmark", "denmark", "deutschland", "ecuador", "egypt",
     "england", "españa", "estonia", "ethiopia", "finland", "france", "georgia", "germany",
     "ghana", "greece", "guatemala", "hong kong", "hungary", "iceland", "india", "indonesia",
     "iran", "iraq", "ireland", "israel", "italia", "italy", "jamaica", "japan", "jordan",
     "kazakhstan", "kenya", "korea", "kosovo", "kuwait", "la france", "latvia", "lebanon",
     "libya", "lithuania", "luxembourg", "macao", "macau", "malaysia", "malta", "mexico",
     "moldova", "mongolia", "montenegro", "morocco", "méxico", "nederland", "nepal",
     "netherlands", "new zealand", "nigeria", "norge", "north macedonia", "northern ireland",
     "norway", "nz", "oman", "p.r. china", "pakistan", "palestine", "panama", "paraguay",
     "people's republic of china", "peoples r china", "peru", "philippines", "poland", "polska",
     "portugal", "pr china", "qatar", "republic of korea", "romania", "russia",
     "russian federation", "rwanda", "saudi arabia", "schweiz", "scotland", "serbia",
     "singapore", "slovakia", "slovenia", "south africa", "south korea", "spain", "sri lanka",
     "sudan", "suisse", "suomi", "sverige", "sweden", "switzerland", "syria", "taiwan",
     "tanzania", "thailand", "the netherlands", "the united states", "tunisia", "turkey",
     "turkiye", "türkiye", "u.k.", "u.s.", "u.s.a.", "uae", "uganda", "uk", "ukraine",
     "united arab emirates", "united kingdom", "united states", "united states of america",
     "uruguay", "us", "usa", "uzbekistan", "venezuela", "viet nam", "vietnam", "wales", "yemen",
     "zambia", "zimbabwe", "österreich"}
)  # fmt: skip
# States, provinces and territories printed after a city; never the city.
_REGIONS = frozenset(
    {"alabama", "alaska", "alberta", "arizona", "arkansas", "british columbia", "california",
     "colorado", "connecticut", "delaware", "district of columbia", "florida", "georgia",
     "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
     "manitoba", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
     "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
     "new south wales", "new york", "north carolina", "north dakota", "nova scotia", "ohio",
     "oklahoma", "ontario", "oregon", "pennsylvania", "quebec", "queensland", "québec",
     "rhode island", "saskatchewan", "south australia", "south carolina", "south dakota",
     "tasmania", "tennessee", "texas", "utah", "vermont", "victoria", "virginia", "washington",
     "west virginia", "western australia", "wisconsin", "wyoming"}
)  # fmt: skip

# Where "Georgia" closing an address is the country, not the US state.
_GEORGIAN_CITIES = frozenset({"tbilisi", "batumi", "kutaisi", "rustavi", "telavi", "zugdidi"})

_UNI = re.compile(
    r"\b(?:universit\w*|univ\.|universidad\w*|universidade|università|universita|université|"
    r"universite|universiteit|universitat|uniklinik\w*|hochschule|polytechnic|politecnico|"
    r"politécnica|eth|epfl|mit)\b",
    re.I,
)
_ORG = re.compile(
    r"\b(?:college|institut\w*|instituto|istituto|hospital\w*|klinikum|academy|akademie|académie|"
    r"foundation|stiftung|ministry|council|agency|inc|ltd|llc|gmbh|corporation|company|clinic|"
    r"centre|center|centro|laborator\w*|school|schule|organi[sz]ation|society|trust|office|"
    r"bureau|museum|library|government|authority|association|service|services|network|"
    r"consortium|observatory)\b",
    re.I,
)
_DEPT = re.compile(
    r"\b(?:department\w*|dept\.?|division|faculty|facultad|faculdade|fakultät|facoltà|school|"
    r"college|institute\w*|institut\w*|instituto|center|centre|centro|laborator\w*|lab|unit|"
    r"program\w*|group|section|chair|lehrstuhl|klinik|dipartimento|departamento|département|"
    r"departement|vakgroep|graduate|research|cluster|branch|clinic|service|team|core|area|"
    r"cátedra|escuela|escola)\b",
    re.I,
)
_DEPT_START = re.compile(
    r"^(?:the\s+)?(?:(?:\w+\s+){0,3}(?:department|dept\.?|division|faculty|school|college|"
    r"institute|laboratory|lab|unit|program(?:me)?|group|section|chair|centre|center|clinic|"
    r"klinik|dipartimento|departamento|département|graduate school|research group)\b)",
    re.I,
)
_DEPT_PHRASE = re.compile(
    r"\b(?:Department|Dept\.?|Faculty|School|College|Division|Institute|Dipartimento|"
    r"Departamento|Facultad|Faculdade)\s+(?:of|de|di|for|für|d’|d')\b"
)
_DEPT_AFTER_UNI = re.compile(r"\s(?:Department|Dept\.?|Faculty|Division)\s+(?:of|de|di|for|für)\b")
_GROUP_AT = re.compile(
    r"^.*?\b(?:at|in)\s+(?:the\s+)?(?=(?:Department|Dept\.?|Division|Faculty|School|Institute)\b)"
)
_ADDRESS = re.compile(
    r"\b(?:avenue|ave\.?|street|st\.|road|rd\.?|box|strasse|straße|str\.|blvd|boulevard|drive|"
    r"lane|way|platz|calle|rua|via|campus|building|floor|room|suite|po box|p\.o\.|cedex|bp|cp|cc|"
    r"postbus|postfach)\b|av\.|private bag",
    re.I,
)
_STATE = re.compile(r"^[A-Z]{2}$|^[A-Z]\.[A-Z]\.$")
_POSTAL = re.compile(
    r"^(?:[A-Z]{1,2}-?)?\d[\d\- ]{2,}\d$|^[A-Z]\d[A-Z] ?\d[A-Z]\d$|^\d{3,}$"
    r"|^[A-Z]{1,2}\d{1,2}[A-Z]? ?\d[A-Z]{2}$"
)
_POSTAL_IN = re.compile(
    r"(?:^|\s)\d{4}\s?[A-Z]{2}(?=\s|$)|(?:^|\s)(?:[A-Z]{1,2}-)?\d{4,6}(?:-\d{3,4})?(?=\s|$)"
    r"|\s+[A-Z]\d[A-Z] ?\d[A-Z]\d$|\s+[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$"
)
_EMAIL = re.compile(r"\S+@\S+")
_ROLE = re.compile(
    r"^(?:(?:assistant|associate|full|visiting|adjunct|emeritus|senior|junior|research|ug|pg|phd|"
    r"graduate|undergraduate|postgraduate|medical|doctoral|master'?s?)\s+)*(?:professor|lecturer|"
    r"students?|researcher|fellow|scholar|candidate|scientist|physician|resident|intern|"
    r"postdoc\w*)s?$",
    re.I,
)
_PERSON_PREFIX = re.compile(
    r"^(?:(?:[A-Z][\w.\-]+|and)\s+){1,7}(?:is|are)\s+(?:with|at)\s+(?:the\s+)?"
)
_NOT_AFFILIATION = re.compile(r"^(?:corresponding author|e-?mail|tel|phone|fax)\b", re.I)
_PAREN_COUNTRY = re.compile(r"\s*\(([^()]+)\)\s*$")
_COMPANY = re.compile(r"(?:Inc|Ltd|LLC|GmbH|Corp|Co|S\.A|B\.V|AG|plc|S\.L|SpA|S\.p\.A)\.?", re.I)
_UC = re.compile(r"(?:The\s+)?University of California")
# A journal's running head read into the string: "... Psychological Science 2022, Vol. 33(9)".
_RUNNING_HEAD = re.compile(r"\s+(?:[A-Z][\w&]*\s+){1,5}(?:19|20)\d\d,?\s+Vol\..*$|\s*©.*$")
_CUT_TAIL = re.compile(
    r"[.;]?\s*(?:correspondence|corresponding author|e-?mails?|tel\.?|phone|fax)\b.*$", re.I
)
_SENTENCE_STOP = re.compile(r"(?<=[a-zà-ÿ]{3})\.\s+(?=[A-Z])")
_STATE_SUFFIX = re.compile(r"^(.*?)\s+[A-Z]{2}$")  # "Cambridge MA"
_STATE_IN_BRACKETS = re.compile(r"\s*\([A-Z]{2}\)$")  # "Rio de Janeiro (RJ)"
_HOUSE_NUMBER = re.compile(r"^\d{1,4}\s+[A-Z]")
_LEADING_DIGIT = re.compile(r"^(?:[A-Z]{1,2}-)?\d")
_IN_AT_BEFORE_UNI = re.compile(r"(?:\s(?:in|at)\s|[”\"]\s)((?:[A-Z][\w’'.\-]*\s+){0,4})$")


def parse_affiliation(text: str) -> AffiliationParts:
    """The institution, department, city and country printed in one affiliation string.

    A part the string does not print is None.
    """
    out: AffiliationParts = {"institution": None, "department": None, "city": None, "country": None}
    text = _RUNNING_HEAD.sub("", _EMAIL.sub("", text or "").strip(" .,;"))
    text = _PERSON_PREFIX.sub("", _CUT_TAIL.sub("", text)).strip(" .,;")
    text = _SENTENCE_STOP.sub(", ", text)
    if _NOT_AFFILIATION.match(text):
        return out
    m = _PAREN_COUNTRY.search(text)
    if m and _is_country(m.group(1)):
        out["country"] = m.group(1).strip()
        text = text[: m.start()]
    segs = _merge_split_names(
        [x for x in _segments(text) if not (_ROLE.search(x) and not _names_org(x))]
    )
    if segs and "/" in segs[-1] and _is_country(segs[-1].rsplit("/", 1)[1]):
        segs = segs[:-1] + segs[-1].rsplit("/", 1)  # "06420 Ankara/Turkey"
    if not segs:
        return out
    if not out["country"]:
        segs = _take_country(segs, out)
    segs = _drop_addresses(segs)
    if not segs:
        return out
    m = _STATE_SUFFIX.match(segs[-1])
    if m and len(segs) >= 2 and _placeish(m.group(1)):
        segs[-1] = m.group(1)
    if len(segs) >= 2 and " - " in segs[-1] and _placeish(segs[-1].rsplit(" - ", 1)[1]):
        segs = segs[:-1] + segs[-1].rsplit(
            " - ", 1
        )  # "Université ... de Louvain - Louvain-la-Neuve"
    segs = [_STATE_IN_BRACKETS.sub("", x) for x in segs]
    universities = [i for i, x in enumerate(segs) if _UNI.search(x)]
    if universities:
        # Anchored on the university: the places after it are the city, then regions.
        idx = universities[-1]
        places = [x for x in segs[idx + 1 :] if _placeish(x)]
        if places:
            out["city"] = places[0]
        before, after = segs[:idx], [x for x in segs[idx + 1 :] if not _placeish(x)]
    else:
        # No university: the last place is the city ("Public Health England, London").
        while len(segs) >= 3 and segs[-1].lower() in _REGIONS and _placeish(segs[-2]):
            segs = segs[:-1]
        if len(segs) >= 2 and _placeish(segs[-1]):
            out["city"] = segs[-1]
            segs = segs[:-1]
        plain = [i for i, x in enumerate(segs) if not _DEPT_START.match(x)]
        idx = plain[-1] if plain else len(segs) - 1
        before, after = segs[:idx], segs[idx + 1 :]
    institution = segs[idx]
    departments = list(before)
    split = _split_department(institution)
    if split:
        institution, inner = split
        departments.append(inner)
    out["institution"] = institution or None
    departments = [_GROUP_AT.sub("", x) for x in departments + after if x != institution]
    if departments:
        out["department"] = ", ".join(x for x in departments if x)
    return out


def _segments(text: str) -> list[str]:
    """``text`` cut at commas and semicolons outside brackets."""
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    out.append("".join(current).strip())
    return [s for s in out if s]


def _names_org(seg: str) -> bool:
    return bool(_UNI.search(seg) or _ORG.search(seg))


def _merge_split_names(segs: list[str]) -> list[str]:
    """Put back the commas inside a name: "Pharmaceuticals, Inc.", "University of California, Davis"."""
    merged: list[str] = []
    for x in segs:
        if merged and (_COMPANY.fullmatch(x) or (_UC.fullmatch(merged[-1]) and _placeish(x))):
            merged[-1] = f"{merged[-1]}, {x}"
        else:
            merged.append(x)
    return merged


def _is_country(seg: str) -> bool:
    # "Taiwan (Republic of China)" is Taiwan.
    return re.sub(r"\s*\([^()]*\)$", "", seg.lower()).strip(" .") in _COUNTRIES


def _placeish(seg: str) -> bool:
    """A few capitalised words with no digit and no organisation word: a city or a region."""
    if re.search(r"\d", seg) or _names_org(seg) or _DEPT.search(seg):
        return False
    return 1 <= len(seg.split()) <= 4 and seg[:1].isupper()


def _take_country(segs: list[str], out: AffiliationParts) -> list[str]:
    """Move the country, the last segment or its last words ("London UK"), into ``out``.

    A state printed last ("Albuquerque, New Mexico", "Atlanta, Georgia") is no
    country: the address simply stops there. Georgia is the country only after
    one of its cities.
    """
    last = segs[-1]
    region = last.lower().strip(" .")
    if region in _REGIONS and not (
        region == "georgia" and len(segs) >= 2 and segs[-2].strip().lower() in _GEORGIAN_CITIES
    ):
        return segs
    if _is_country(last):
        out["country"] = last.strip(" .")
        return segs[:-1]
    words = last.split()
    for k in (3, 2, 1):
        if len(words) > k and " ".join(words[-k:]).lower().strip(".") in _COUNTRIES:
            rest = " ".join(words[:-k])
            if not _names_org(rest):
                out["country"] = " ".join(words[-k:])
                return [*segs[:-1], rest]
            break
    return segs


def _drop_addresses(segs: list[str]) -> list[str]:
    """Without streets, boxes, postcodes and state codes; "75000 Paris" keeps "Paris"."""
    kept: list[str] = []
    for i, seg in enumerate(segs):
        if _STATE.match(seg) or _POSTAL.match(seg):
            continue
        if _ADDRESS.search(seg) and not _names_org(seg):
            continue
        if re.search(r"\d", seg) and not (_names_org(seg) or _DEPT.search(seg)):
            if _HOUSE_NUMBER.match(seg) and any(_placeish(x) for x in segs[i + 1 :]):
                continue  # a house number and street, with the city still to come
            stripped = _POSTAL_IN.sub(" ", seg).strip()
            stripped = re.sub(r"\s+[A-Z]{2}$", "", stripped).strip()
            if not stripped or re.search(r"\d", stripped) or _STATE.match(stripped):
                continue
            after_city = kept and _placeish(kept[-1]) and len(kept) >= 2
            if not _LEADING_DIGIT.match(seg) and i == len(segs) - 1 and after_city:
                continue  # "Kanagawa 252-0373", "NSW 2560" after the city
            seg = stripped
        kept.append(seg)
    return kept


def _split_department(seg: str) -> tuple[str, str] | None:
    """A department and its university in one segment, as (institution, department).

    "Central China Normal University (CCNU) Department of Psychology",
    "Department of Psychiatry University Hospital (UZBrussel)",
    "College of Medicine in Taif University".
    """
    uni = _UNI.search(seg)
    if not uni:
        return None
    post = _DEPT_AFTER_UNI.search(seg[uni.end() :])
    if post:
        cut = uni.end() + post.start()
        return seg[:cut].strip(" -–,"), seg[cut:].strip(" -–,")
    if not _DEPT_PHRASE.match(seg) or uni.start() == 0:
        return None
    m = _IN_AT_BEFORE_UNI.search(seg[: uni.start()])
    cut = m.start(1) if m else uni.start()
    return seg[cut:].strip(" -–"), re.sub(r"\s+(?:in|at)$", "", seg[:cut].strip(" -–,"))
