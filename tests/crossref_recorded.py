"""A Crossref client that answers from responses recorded in
``tests/fixtures/crossref/`` (trimmed to the fields bibr reads).

Recorded October 2026 from api.crossref.org and doi.org/ra:

- ``work_carney``: 10.1177/0956797610383437, Carney, Cuddy & Yap (2010),
  "Power Posing" with its subtitle deposited separately.
- ``work_wakefield``: 10.1016/s0140-6736(97)11096-0, Wakefield et al. (1998),
  titled "RETRACTED: ..." and updated by a retraction and a correction.
- ``work_osc``: 10.1126/science.aac4716, Open Science Collaboration (2015),
  whose one author is an organization.
- ``work_cohen_book``: 10.4324/9780203771587, Cohen's "Statistical Power
  Analysis for the Behavioral Sciences", a book issued 2013 (first printed 1988).
- ``search_carney`` / ``search_miss``: /works?query.bibliographic= answers.
- ``doi_ra``: doi.org/ra for a Crossref DOI, a DataCite DOI, a DOI that
  does not exist and the Wakefield DOI.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import unquote

import httpx

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "crossref"

CARNEY_DOI = "10.1177/0956797610383437"
WAKEFIELD_DOI = "10.1016/S0140-6736(97)11096-0"
OSC_DOI = "10.1126/science.aac4716"
COHEN_DOI = "10.4324/9780203771587"
DATACITE_DOI = "10.5281/zenodo.3242074"
UNREGISTERED_DOI = "10.1177/0956797610383499"

WORKS = {
    CARNEY_DOI: "work_carney",
    WAKEFIELD_DOI: "work_wakefield",
    OSC_DOI: "work_osc",
    COHEN_DOI: "work_cohen_book",
}


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _not_found(path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.crossref.org" + path)
    response = httpx.Response(404, text="Resource not found.", request=request)
    return httpx.HTTPStatusError("Not Found", request=request, response=response)


class RecordedCrossref:
    """Answers ``works`` from the recorded records (404 for any other DOI) and
    ``search`` from ``searches``: the first recorded answer whose key occurs
    in the query, else the recorded miss. DOIs in ``hang`` never answer."""

    def __init__(self, searches: dict[str, str] | None = None, hang: set[str] = frozenset()):
        self.searches = {"power posing": "search_carney", **(searches or {})}
        self.hang = {doi.casefold() for doi in hang}
        self.calls: list[tuple[str, str]] = []
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def enrich_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(8)
        return self._semaphore

    async def prefetch_works_by_doi(self, dois):  # noqa: ARG002
        return 0

    async def works(self, ids: str, **_kw) -> dict:
        self.calls.append(("works", ids))
        if ids.casefold() in self.hang:
            await asyncio.Event().wait()
        for doi, name in WORKS.items():
            if doi.casefold() == ids.casefold():
                return load(name)
        raise _not_found(f"/works/{ids}")

    async def search(self, query: str, limit: int = 3, **_kw) -> dict:
        self.calls.append(("search", query))
        for key, name in self.searches.items():
            if key in query.casefold():
                return load(name)
        return load("search_miss")


class RecordedDoiOrg:
    """doi.org/ra from the recorded answer: an httpx-like client for
    :func:`bibr.enrich.lookup.registration_agencies`."""

    def __init__(self):
        self.urls: list[str] = []

    async def get(self, url: str):
        self.urls.append(url)
        asked = {unquote(doi).casefold() for doi in url.rsplit("/ra/", 1)[1].split(",")}
        rows = [row for row in load("doi_ra") if row["DOI"].casefold() in asked]
        return httpx.Response(200, json=rows, request=httpx.Request("GET", url))

    async def aclose(self):
        return None
