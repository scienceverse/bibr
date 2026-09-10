"""_normalize_export_url: strip PDF line-wrap artifacts (embedded whitespace,
sentence-final period) from exported hrefs so metacheck's downstream tempfix
patch (gsub("\\s", "", href) + trailing-dot strip in R/import-read.R) can be
deleted."""

import pytest

from bibr.export.json_export import _normalize_export_url

PROBES = [
    ("https://example.org/a\r\nb", "https://example.org/ab"),
    ("https://example.org/a b", "https://example.org/ab"),
    ("https://example.org/path.", "https://example.org/path"),
    ("https://example.org/path", "https://example.org/path"),
    ("https://doi.org/10.1177/0956797613520608", "https://doi.org/10.1177/0956797613520608"),
    # display-truncated ellipsis: a single-pass strip left one dot behind
    # ("…/path.." instead of "…/path") — all trailing dots must go.
    ("https://example.org/path...", "https://example.org/path"),
]


@pytest.mark.parametrize(("raw", "expected"), PROBES)
def test_line_wrap_artifacts_are_removed(raw, expected):
    assert _normalize_export_url(raw) == expected


def test_a_trailing_slash_is_preserved():
    assert _normalize_export_url("https://example.org/") == "https://example.org/"


@pytest.mark.parametrize("raw", [raw for raw, _ in PROBES])
def test_normalization_is_idempotent(raw):
    once = _normalize_export_url(raw)
    assert _normalize_export_url(once) == once


# ── the normalization must reach every exported URL surface ────────────


def _paper_with_reference_url(url):
    from bibr.models import PaperMetadata, PaperReference
    from tests.test_export_units import _minimal_paper

    reference = PaperReference(
        bib_id=1,
        title="A title",
        first_page=None,
        volume=None,
        authors="Smith, J.",
        year=2020,
        container=None,
        url=url,
    )
    metadata = PaperMetadata(doi="10.1234/test", title="T", references=[reference])
    return _minimal_paper(metadata=metadata)


def test_bib_url_is_normalized_like_url_href():
    """``bib[].url`` is parsed from line-joined reference text and carries the
    same wrap artifacts as ``url[].href``. Both must be repaired here, or a
    consumer that drops its own whitespace/trailing-dot patch on the strength
    of this release sees ``bib[].url`` regress."""
    from bibr.export.json_export import export_paper_to_json

    payload = export_paper_to_json(_paper_with_reference_url("https://example.org/a\r\n b/paper."))

    assert payload["bib"][0]["url"] == "https://example.org/ab/paper"


def test_bib_url_stays_none_when_absent():
    from bibr.export.json_export import export_paper_to_json

    assert export_paper_to_json(_paper_with_reference_url(None))["bib"][0]["url"] is None
