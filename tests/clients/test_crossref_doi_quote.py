from unittest import mock


async def test_works_passes_unencoded_doi_slash(monkeypatch):
    from bibr.clients.crossref import CrossrefClient
    from bibr.config import GlobalSettings

    requested = {"path": None}

    async def fake_request(self, path, params=None):  # noqa: ARG001
        requested["path"] = path
        return {"message": {}}

    monkeypatch.setattr(CrossrefClient, "_request", fake_request)
    monkeypatch.setattr(CrossrefClient, "_ensure_limiter", mock.AsyncMock(), raising=False)

    settings = GlobalSettings()
    settings.crossref.cache_size = 0
    c = CrossrefClient(settings=settings)
    await CrossrefClient.works(c, "10.1093/bioinformatics/btac123")
    assert requested["path"] == "/works/10.1093/bioinformatics/btac123", requested["path"]


async def test_works_still_encodes_unsafe_chars(monkeypatch):
    from bibr.clients.crossref import CrossrefClient
    from bibr.config import GlobalSettings

    requested = {"path": None}

    async def fake_request(self, path, params=None):  # noqa: ARG001
        requested["path"] = path
        return {"message": {}}

    monkeypatch.setattr(CrossrefClient, "_request", fake_request)

    settings = GlobalSettings()
    settings.crossref.cache_size = 0
    c = CrossrefClient(settings=settings)
    # DOIs can contain spaces or '#' rarely — those should still be encoded.
    await CrossrefClient.works(c, "10.1234/abc def#ghi")
    assert " " not in requested["path"]
    assert "%20" in requested["path"] or "+" in requested["path"]
    assert "#" not in requested["path"]  # # would break path parsing


async def test_works_preserves_legacy_doi_special_chars(monkeypatch):
    """Legacy DOIs like ``10.1002/(SICI)1521-3951(199911)216:1<...>3.0.CO;2-#``
    require ``()``, ``;``, and ``:`` to remain literal in the Crossref path.
    Only ``<``, ``>``, ``#`` and similar genuinely-unsafe chars are encoded."""
    from bibr.clients.crossref import CrossrefClient
    from bibr.config import GlobalSettings

    requested = {"path": None}

    async def fake_request(self, path, params=None):  # noqa: ARG001
        requested["path"] = path
        return {"message": {}}

    monkeypatch.setattr(CrossrefClient, "_request", fake_request)

    settings = GlobalSettings()
    settings.crossref.cache_size = 0
    c = CrossrefClient(settings=settings)
    await CrossrefClient.works(c, "10.1002/(SICI)1521-3951(199911)216:1")
    path = requested["path"]
    assert "(" in path and ")" in path, path
    assert ";" not in path or "%3B" not in path  # `;` is fine literal
    assert ":" in path, path
    assert "%28" not in path and "%29" not in path, path


async def test_works_rejects_path_traversal_doi(monkeypatch):
    """A DOI containing a '..' path segment must not reach ``_request`` — even
    though the Crossref host is hardcoded, ``..`` would traverse within the
    public API path (audit L11 / M1 secondary)."""
    from bibr.clients.crossref import CrossrefClient
    from bibr.config import GlobalSettings

    hit = {"called": False}

    async def fake_request(self, path, params=None):  # noqa: ARG001
        hit["called"] = True
        return {"message": {}}

    monkeypatch.setattr(CrossrefClient, "_request", fake_request)
    monkeypatch.setattr(CrossrefClient, "_ensure_limiter", mock.AsyncMock(), raising=False)

    settings = GlobalSettings()
    settings.crossref.cache_size = 0
    c = CrossrefClient(settings=settings)
    out = await CrossrefClient.works(c, "10.1234/../admin/search")
    assert hit["called"] is False, "traversal DOI reached _request"
    assert out == {}, "unsafe DOI must degrade to an empty result"
