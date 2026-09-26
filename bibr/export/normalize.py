"""Normalized companions for printed metadata values.

The export keeps what the paper printed (``metadata.published``,
``metadata.license``, ``author[].role``) and adds a standard form next to it
where one can be derived without guessing: an ISO 8601 date, a license URL and
SPDX identifier, CRediT role URIs, an arXiv identifier. Every function returns
``None`` (or an empty list) when the printed value does not determine the
standard form unambiguously.
"""

from __future__ import annotations

import calendar
import re

# ── dates ─────────────────────────────────────────────────────────────

_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_YEAR = r"(?P<year>(?:1[89]|20)\d{2})"
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_DATE_PATTERNS = [
    re.compile(rf"\b{_YEAR}-(?P<m>[01]?\d)(?:-(?P<d>[0-3]?\d))?(?!\d)"),  # 2026-01-15(T…)
    re.compile(rf"\b{_YEAR}/(?P<m>[01]?\d)/(?P<d>[0-3]?\d)(?!\d)"),  # 2026/01/15
    re.compile(rf"\b{_DAY}\s+{_MONTH},?\s+{_YEAR}\b", re.I),  # 15 January 2026
    re.compile(rf"\b{_MONTH}\s+{_DAY},?\s+{_YEAR}\b", re.I),  # January 15, 2026
    re.compile(rf"\b{_YEAR}\s+{_MONTH}\s+{_DAY}\b", re.I),  # 2026 Jan 15
    re.compile(rf"\b{_YEAR},\s*{_MONTH}\s+{_DAY}\b", re.I),  # 2026, January 15 (APA)
    re.compile(rf"\b{_YEAR},\s*{_MONTH}(?![\w.])", re.I),  # 2026, January (APA)
    re.compile(rf"\b{_MONTH},?\s+{_YEAR}\b", re.I),  # January 2026
    re.compile(rf"^\s*{_YEAR}\s*$"),  # 2026
]
_PUBLISHED_CUE = re.compile(r"publish", re.I)


def _date_from_match(m: re.Match[str]) -> str | None:
    groups = m.groupdict()
    year = int(groups["year"])
    month_text = groups.get("month")
    month = _MONTHS[month_text.lower()] if month_text else groups.get("m")
    day = groups.get("day") or groups.get("d")
    if month is None:
        return f"{year:04d}"
    month = int(month)
    if not 1 <= month <= 12:
        return None
    if day is None:
        return f"{year:04d}-{month:02d}"
    day = int(day)
    if not 1 <= day <= calendar.monthrange(year, month)[1]:
        # An impossible calendar date (an OCR slip or typo like 30 February):
        # the day is undetermined, but the printed month and year still are.
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}-{month:02d}-{day:02d}"


def iso_date(printed: str | None) -> str | None:
    """The publication date in *printed* as ISO 8601 (``YYYY``, ``YYYY-MM`` or
    ``YYYY-MM-DD``); ``None`` when there is no date or several without a
    "published" cue to choose between them."""
    if not printed or not printed.strip():
        return None
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(printed):
            if any(m.start() < end and m.end() > start for start, end in taken):
                continue
            value = _date_from_match(m)
            if value is not None:
                found.append((m.start(), value))
                taken.append(m.span())
    if not found:
        return None
    found.sort()
    if len({value for _, value in found}) == 1:
        return found[0][1]
    cue = _PUBLISHED_CUE.search(printed)
    if cue is None:
        return None
    after = [value for start, value in found if start > cue.start()]
    return after[0] if after else None


# ── licenses ──────────────────────────────────────────────────────────

_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
_CC_URL = re.compile(
    r"creativecommons\.org/(?:licenses/(?P<kind>by(?:-nc)?(?:-sa|-nd)?)/(?P<version>\d\.\d)"
    r"|publicdomain/zero/(?P<zero>1\.0))",
    re.I,
)
_CC_SHORT = re.compile(
    r"\bCC[\s-]?(?P<kind>BY(?:[\s-]NC)?(?:[\s-](?:SA|ND))?)\b(?:[\s-]*(?P<version>\d\.\d))?",
    re.I,
)
_CC_LONG = re.compile(
    r"Creative\s+Commons\s+Attribution"
    r"(?P<nc>[\s-]+Non-?Commercial)?"
    r"(?:(?P<sa>[\s-]+Share-?Alike)|(?P<nd>[\s-]+No-?Deriv(?:atives|s|ative)?))?"
    r"(?:\s+\(CC[\s-]BY[^)]*\))?"
    r"(?:\s+License)?(?:\s+(?P<version>\d\.\d))?",
    re.I,
)
_CC0 = re.compile(r"\bCC\s?0\b|Creative\s+Commons\s+Zero|public\s+domain\s+dedication", re.I)


def _cc(kind: str, version: str) -> tuple[str, str]:
    """(SPDX identifier, canonical URL) of a Creative Commons license."""
    parts = kind.lower().replace(" ", "-").split("-")
    slug = "-".join(parts)
    spdx = "CC-" + "-".join(p.upper() for p in parts) + f"-{version}"
    return spdx, f"https://creativecommons.org/licenses/{slug}/{version}/"


def license_ids(printed: str | None) -> tuple[str | None, str | None]:
    """``(license_url, license_spdx)`` for a printed license statement.

    The URL is the one printed, else the canonical Creative Commons URL when
    the license is identified. The SPDX identifier is given only for a Creative
    Commons license whose version is known, or CC0.
    """
    if not printed or not printed.strip():
        return None, None
    printed_url = next((u.rstrip(".,;") for u in _URL.findall(printed)), None)
    spdx: str | None = None
    canonical: str | None = None
    if m := _CC_URL.search(printed):
        if m.group("zero"):
            spdx, canonical = "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"
        else:
            spdx, canonical = _cc(m.group("kind"), m.group("version"))
    elif _CC0.search(printed):
        spdx, canonical = "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"
    elif (m := _CC_SHORT.search(printed)) and m.group("version"):
        spdx, canonical = _cc(re.sub(r"[\s-]+", "-", m.group("kind")), m.group("version"))
    elif (m := _CC_LONG.search(printed)) and m.group("version"):
        kind = "by" + ("-nc" if m.group("nc") else "")
        kind += "-sa" if m.group("sa") else "-nd" if m.group("nd") else ""
        spdx, canonical = _cc(kind, m.group("version"))
    return printed_url or canonical, spdx


# ── CRediT roles ──────────────────────────────────────────────────────

CREDIT_BASE_URI = "https://credit.niso.org/contributor-roles/"

# CRediT term slug -> lowercase phrases that name it in contribution statements.
# The term names themselves come first; the rest are the common free-text
# spellings. Deliberately short: an unmatched role stays unmapped.
_CREDIT_PHRASES: dict[str, tuple[str, ...]] = {
    "conceptualization": ("conceptualization", "conceptualisation", "conceived"),
    "data-curation": ("data curation", "curated the data"),
    "formal-analysis": ("formal analysis", "analyzed the data", "analysed the data"),
    "funding-acquisition": ("funding acquisition", "acquired funding", "obtained funding"),
    "investigation": ("investigation", "collected the data", "performed the experiments"),
    "methodology": ("methodology",),
    "project-administration": ("project administration",),
    "resources": ("resources",),
    "software": ("software",),
    "supervision": ("supervision", "supervised"),
    "validation": ("validation",),
    "visualization": ("visualization", "visualisation"),
    "writing-original-draft": (
        "writing - original draft",
        "writing – original draft",
        "writing—original draft",
        "original draft",
        "wrote the manuscript",
        "wrote the first draft",
        "drafted the manuscript",
    ),
    "writing-review-editing": (
        "writing - review",
        "writing – review",
        "writing—review",
        "review & editing",
        "review and editing",
        "revised the manuscript",
    ),
}


def credit_roles(roles: list[str]) -> list[str]:
    """CRediT term URIs for free-text contribution *roles*, in CRediT order."""
    text = " | ".join(role.lower() for role in roles)
    return [
        CREDIT_BASE_URI + slug + "/"
        for slug, phrases in _CREDIT_PHRASES.items()
        if any(phrase in text for phrase in phrases)
    ]


# ── identifiers ───────────────────────────────────────────────────────

_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(?P<id>\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
# The vertical stamp arXiv prints on page 1: "arXiv:2101.12345v2 [cs.CL] 3 Feb 2021".
_ARXIV_STAMP = re.compile(r"\barXiv:(?P<id>\d{4}\.\d{4,5})(?:v\d+)?\s*\[[A-Za-z.-]+\]")


def arxiv_id(doi: str | None, first_page_texts: list[str]) -> str | None:
    """The paper's own arXiv identifier, from an arXiv DOI or the page-1 stamp."""
    if doi and (m := _ARXIV_DOI.match(doi.strip())):
        return m.group("id")
    for text in first_page_texts:
        if m := _ARXIV_STAMP.search(text):
            return m.group("id")
    return None
