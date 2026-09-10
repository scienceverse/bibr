"""Pydantic schemas for external service responses.

Provides typed models for the subset of Crossref API fields that bibr
consumes, catching upstream API changes at the parsing boundary.
"""

from __future__ import annotations

from pydantic import BaseModel


class CrossrefAuthor(BaseModel):
    """Author as returned by Crossref API."""

    given: str = ""
    family: str = ""
    orcid: str | None = None
    sequence: str | None = None


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

    @classmethod
    def from_raw(cls, raw: dict) -> CrossrefWorkItem:
        """Parse a raw Crossref API work item dict into a typed model."""
        titles = raw.get("title", [])
        title = titles[0] if titles else None

        containers = raw.get("container-title", [])
        container_title = containers[0] if containers else None

        authors = [
            CrossrefAuthor(
                given=a.get("given", ""),
                family=a.get("family", ""),
                orcid=a.get("ORCID"),
                sequence=a.get("sequence"),
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
        )
