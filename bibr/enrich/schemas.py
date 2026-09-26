"""Pydantic schemas for external service responses.

Provides typed models for the subset of Crossref API fields that bibr
consumes, catching upstream API changes at the parsing boundary.
"""

from __future__ import annotations

import html
import re

from pydantic import BaseModel

_ROR_ID = re.compile(r"^(?:https?://)?ror\.org/(?P<id>0[a-z0-9]{6}\d{2})/?$", re.I)
_FUNDER_DOI = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?(?P<doi>10\.13039/\S+)$", re.I)
# The license of the article itself, best first. "tdm" licenses grant text
# mining, not reuse of the work, so they never stand for the article's license.
_LICENSE_VERSIONS = ("vor", "unspecified", "am")
# An inline element of a deposited title: JATS/HTML face markup (<i>, <sub>,
# <scp>) and MathML (<mml:msub>). Its text is part of the title; the tag is not.
# Attributes must be name="value", as XML requires, so a title's own "<y and z>"
# is text, not a tag.
_INLINE_TAG = re.compile(
    r"(</?[A-Za-z][\w.:-]*(?:\s+[\w.:-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'))*\s*/?>)"
)
# A line break inside a title separates the words on either side of it.
_LINE_BREAK_TAG = re.compile(r"<(?:br|break)\s*/?>", re.IGNORECASE)
_SCRIPT_TAG = re.compile(r"</?su[bp]\b", re.IGNORECASE)
# An element symbol that continues a formula after a subscript ("N<sub>2</sub>O").
_FORMULA_SYMBOL = re.compile(r"[A-Z][a-z]?(?![^\W\d_])")
# Punctuation that attaches to the word before it, so it follows a closing tag
# without a space ("<i>R</i> + newline + ': An'"), and punctuation that attaches
# to the word after it, so it precedes an opening tag without one ("(" + newline
# + "<i>Festuca</i>"). Dashes and slashes attach on both sides. A comma or
# period before an element and a "(" after one keep their space ("Stink Bug,
# <i>Halyomorpha halys</i> (Stål)"), and so does a straight double quote, which
# may open or close.
_JOINING_MARKS = "-‐‑‒–—―/"
_ATTACHES_BEFORE = frozenset(",.;:!?)]}’”»'" + _JOINING_MARKS)
_ATTACHES_AFTER = frozenset("([{‘“«'" + _JOINING_MARKS)


def plain_text(value: object) -> str | None:
    """A Crossref title as plain text: inline tags dropped (their text kept),
    character entities decoded, whitespace collapsed.

    Crossref returns titles as deposited, e.g. ``Effects of CO<sub>2</sub>``
    or ``Genes &amp; Development``. Scored as-is the tags cost enough fuzzy
    similarity to lose the correct record, and an accepted match exported them
    into ``bib_match`` and, through consolidation, into ``bib``.

    The line break a pretty-printed deposit puts around an element becomes a
    space only where the title had one: not inside the element, not before a
    sub- or superscript ("CO" + newline + "<sub>2</sub>"), not between an
    element and punctuation that attaches to it, and not before an element
    symbol that continues a formula ("C<sub>2</sub>" + newline + "H<sub>6</sub>").
    """
    if not isinstance(value, str):
        return None
    pieces = _INLINE_TAG.split(_LINE_BREAK_TAG.sub(" ", value))
    out: list[str] = []
    for index in range(0, len(pieces), 2):
        text = html.unescape(pieces[index])
        opened_by = pieces[index - 1] if index else ""
        closed_by = pieces[index + 1] if index + 1 < len(pieces) else ""
        body = text.lstrip()
        if opened_by and "\n" in text[: len(text) - len(body)]:
            joined = (
                not opened_by.startswith("</")
                or body[:1] in _ATTACHES_BEFORE
                or bool(_SCRIPT_TAG.match(opened_by) and _FORMULA_SYMBOL.match(body))
            )
            text = body if joined else " " + body
        body = text.rstrip()
        if closed_by and "\n" in text[len(body) :]:
            joined = (
                closed_by.startswith("</")
                or bool(_SCRIPT_TAG.match(closed_by))
                or body[-1:] in _ATTACHES_AFTER
            )
            text = body if joined else body + " "
        out.append(text)
    return " ".join("".join(out).split()) or None


def canonical_ror(value: object) -> str | None:
    """``https://ror.org/<id>`` for a ROR ID in URI or bare form, else ``None``."""
    if not isinstance(value, str):
        return None
    m = _ROR_ID.match(value.strip())
    return f"https://ror.org/{m.group('id').lower()}" if m else None


def _funder_doi(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    m = _FUNDER_DOI.match(value.strip())
    return m.group("doi") if m else None


def _ror_in(ids: object) -> str | None:
    """The first ROR ID in a Crossref ``id`` list."""
    for entry in ids if isinstance(ids, list) else []:
        if (
            isinstance(entry, dict)
            and str(entry.get("id-type", "")).upper() == "ROR"
            and (ror := canonical_ror(entry.get("id")))
        ):
            return ror
    return None


class CrossrefOrganization(BaseModel):
    """An affiliation of a Crossref author."""

    name: str | None = None
    ror: str | None = None


class CrossrefFunder(BaseModel):
    """A funder of a Crossref work."""

    name: str | None = None
    funder_doi: str | None = None
    ror: str | None = None
    award_ids: list[str] = []


class CrossrefAuthor(BaseModel):
    """Author as returned by Crossref API."""

    given: str = ""
    family: str = ""
    orcid: str | None = None
    sequence: str | None = None
    affiliations: list[CrossrefOrganization] = []


def _organizations(raw: object) -> list[CrossrefOrganization]:
    out = []
    for aff in raw if isinstance(raw, list) else []:
        if not isinstance(aff, dict):
            continue
        name = (aff.get("name") or "").strip() or None
        ror = _ror_in(aff.get("id"))
        if name or ror:
            out.append(CrossrefOrganization(name=name, ror=ror))
    return out


def _funders(raw: object) -> list[CrossrefFunder]:
    out = []
    for funder in raw if isinstance(raw, list) else []:
        if not isinstance(funder, dict):
            continue
        ids = funder.get("id")
        doi = _funder_doi(funder.get("DOI"))
        for entry in ids if isinstance(ids, list) else []:
            if doi is None and isinstance(entry, dict) and entry.get("id-type") == "DOI":
                doi = _funder_doi(entry.get("id"))
        name = (funder.get("name") or "").strip() or None
        awards = [a.strip() for a in funder.get("award") or [] if isinstance(a, str) and a.strip()]
        ror = _ror_in(ids)
        if name or doi or ror:
            out.append(CrossrefFunder(name=name, funder_doi=doi, ror=ror, award_ids=awards))
    return out


def _license_url(raw: object) -> str | None:
    """The URL of the work's own license: version of record first."""
    licenses = [lic for lic in raw if isinstance(lic, dict)] if isinstance(raw, list) else []
    for version in _LICENSE_VERSIONS:
        for lic in licenses:
            if lic.get("content-version") == version and isinstance(lic.get("URL"), str):
                return lic["URL"].strip() or None
    return None


class CrossrefWorkItem(BaseModel):
    """Parsed subset of a Crossref work item.

    Use ``CrossrefWorkItem.from_raw(dict)`` to parse a raw API response.
    """

    doi: str | None = None
    title: str | None = None
    container_title: str | None = None
    volume: str | None = None
    issue: str | None = None
    page: str | None = None
    publisher: str | None = None
    work_type: str | None = None
    url: str | None = None
    authors: list[CrossrefAuthor] = []
    editors: list[CrossrefAuthor] = []
    year: int | None = None
    date: str | None = None
    api_score: float | None = None
    license_url: str | None = None
    funders: list[CrossrefFunder] = []

    @classmethod
    def from_raw(cls, raw: dict) -> CrossrefWorkItem:
        """Parse a raw Crossref API work item dict into a typed model."""
        titles = raw.get("title", [])
        title = plain_text(titles[0]) if titles else None

        containers = raw.get("container-title", [])
        container_title = plain_text(containers[0]) if containers else None

        authors = [
            CrossrefAuthor(
                given=a.get("given", ""),
                family=a.get("family", ""),
                orcid=a.get("ORCID"),
                sequence=a.get("sequence"),
                affiliations=_organizations(a.get("affiliation")),
            )
            for a in raw.get("author", [])
        ]

        editors = [
            CrossrefAuthor(
                given=e.get("given", ""),
                family=e.get("family", ""),
            )
            for e in raw.get("editor", [])
        ]

        pub_year = None
        pub_date = None
        issued = raw.get("issued", {})
        date_parts = issued.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            if parts and parts[0]:
                pub_year = int(parts[0])
                # Crossref emits ``[year, 0]`` (and occasionally ``[year, m, 0]``)
                # for year-only / year-and-month-only records. Treat those
                # zero placeholders as "missing" rather than formatting them
                # as the invalid month/day ``"00"``.
                month = parts[1] if len(parts) >= 2 else None
                day = parts[2] if len(parts) >= 3 else None
                if month and day:
                    pub_date = f"{parts[0]:04d}-{month:02d}-{day:02d}"
                elif month:
                    pub_date = f"{parts[0]:04d}-{month:02d}"

        return cls(
            doi=raw.get("DOI"),
            title=title,
            container_title=container_title,
            volume=raw.get("volume"),
            issue=raw.get("issue"),
            page=raw.get("page"),
            publisher=raw.get("publisher"),
            work_type=raw.get("type"),
            url=raw.get("URL"),
            authors=authors,
            editors=editors,
            year=pub_year,
            date=pub_date,
            api_score=raw.get("score"),
            license_url=_license_url(raw.get("license")),
            funders=_funders(raw.get("funder")),
        )
