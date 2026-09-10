import pytest


@pytest.fixture(autouse=True)
def _isolate_resolver_env(monkeypatch, tmp_path):
    # These tests assert defaults; shield them from the developer's real .env
    # (read from CWD at settings construction) and exported BIBR_RESOLVER_* vars.
    monkeypatch.chdir(tmp_path)
    for var in (
        "BIBR_RESOLVER_URL",
        "BIBR_RESOLVER_ENRICH",
        "BIBR_RESOLVER_TIMEOUT",
        "BIBR_RESOLVER_LIMIT",
        "BIBR_RESOLVER_AUTHORITATIVE",
        "BIBR_RESOLVER_SOURCES",
        "BIBR_RESOLVER_FALLBACK_SOURCES",
        "BIBR_RESOLVER_FALLBACK_SEARCH_CONCURRENCY",
        "BIBR_RESOLVER_FALLBACK_TIMEOUT",
        "CROSSREF_ENRICH_CONCURRENCY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolver_options_defaults():
    from bibr.config import ResolverOptions

    o = ResolverOptions()
    assert o.url is None
    assert o.enrich is True
    assert o.timeout == 10.0
    assert o.limit == 20
    # Default off: the resolver is a fast accelerator over CrossRef, so a resolver
    # miss still falls through to CrossRef unless the operator asserts the resolver
    # is backed by the same corpus.
    assert o.authoritative is False


def test_resolver_options_env(monkeypatch):
    monkeypatch.setenv("BIBR_RESOLVER_URL", "http://gpu-box:2010")
    monkeypatch.setenv("BIBR_RESOLVER_LIMIT", "5")
    from bibr.config import ResolverOptions

    o = ResolverOptions()
    assert o.url == "http://gpu-box:2010"
    assert o.limit == 5


def test_resolver_sources_default_is_crossref_only():
    from bibr.config import ResolverOptions

    # CrossRef only: OpenAlex lives on the resolver's slow-disk shards (8 indices,
    # ~128 Quickwit splits per query vs 86 for CrossRef, ~8x the latency per search),
    # so it must not sit on the primary path every reference takes. Reach it through
    # fallback_sources, which only queries the refs CrossRef missed.
    assert ResolverOptions().sources == ["crossref"]


def test_resolver_sources_env_comma_split(monkeypatch):
    # Env is a comma-separated string (Compose/.env friendly), normalized to a
    # lowercased list.
    monkeypatch.setenv("BIBR_RESOLVER_SOURCES", "CrossRef, OpenAlex")
    from bibr.config import ResolverOptions

    assert ResolverOptions().sources == ["crossref", "openalex"]


def test_resolver_sources_env_empty_disables_selection(monkeypatch):
    # An empty value falls back to the resolver's own default tier (no sources key).
    monkeypatch.setenv("BIBR_RESOLVER_SOURCES", "")
    from bibr.config import ResolverOptions

    assert ResolverOptions().sources == []


def test_resolver_fallback_defaults_are_disabled_and_bounded():
    from bibr.config import ResolverOptions

    options = ResolverOptions()
    assert options.fallback_sources == []
    assert options.fallback_search_concurrency == 4
    assert options.fallback_timeout == 30.0


def test_resolver_fallback_sources_env_comma_split(monkeypatch):
    monkeypatch.setenv("BIBR_RESOLVER_FALLBACK_SOURCES", " OpenAlex, DATACITE ")
    from bibr.config import ResolverOptions

    assert ResolverOptions().fallback_sources == ["openalex", "datacite"]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("BIBR_RESOLVER_FALLBACK_SEARCH_CONCURRENCY", "0"),
        ("BIBR_RESOLVER_FALLBACK_TIMEOUT", "0"),
    ),
)
def test_resolver_fallback_bounds_must_be_positive(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    from bibr.config import ResolverOptions

    with pytest.raises(ValueError):
        ResolverOptions()


def test_resolver_authoritative_env(monkeypatch):
    monkeypatch.setenv("BIBR_RESOLVER_AUTHORITATIVE", "true")
    from bibr.config import ResolverOptions

    assert ResolverOptions().authoritative is True


def test_global_settings_registers_resolver():
    from bibr.config import GlobalSettings

    assert GlobalSettings().resolver.url is None


def test_resolver_enabled_raises_enrich_concurrency(monkeypatch):
    # With the resolver on, most lookups are fast/local — raise above the
    # CrossRef-only polite default so the local resolver path isn't serialized.
    monkeypatch.setenv("BIBR_RESOLVER_URL", "http://gpu-box:2010")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.resolver.url == "http://gpu-box:2010"
    assert s.crossref.enrich_concurrency == 16


def test_resolver_disabled_keeps_polite_enrich_concurrency():
    # CrossRef-only users keep the polite pipeline-fill default (the shared rate
    # limiter, not concurrency, enforces politeness).
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.resolver.url is None
    assert s.crossref.enrich_concurrency == 12


def test_explicit_enrich_concurrency_respected_with_resolver(monkeypatch):
    # An explicit override wins over the auto-raise.
    monkeypatch.setenv("BIBR_RESOLVER_URL", "http://gpu-box:2010")
    monkeypatch.setenv("CROSSREF_ENRICH_CONCURRENCY", "5")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.crossref.enrich_concurrency == 5
