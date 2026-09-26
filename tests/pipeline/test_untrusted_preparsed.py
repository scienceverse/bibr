"""Generic HTML page meta is an untrusted preparsed record.

A page with only a ``<title>`` and ``<html lang>`` (no Highwire, Dublin Core
or OPF front matter) used to come back with no preparsed record at all, so a
no-LLM export lost the language main exported. The record now stays, marked
untrusted: a no-LLM run keeps it, an LLM run extracts the front matter from
the printed article and fills only the fields it left empty.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.input.html_native import HtmlParser
from bibr.models import PaperAuthor, PaperMetadata

_GENERIC = (
    b"""<html lang="en"><head><title>Sleep and memory | Journal | Press</title>"""
    b"""<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">"""
    b"""<meta name="keywords" content="sleep, memory">"""
    b"""<meta name="author" content="Jane Smith, John Doe"></head>"""
    b"""<body><article><h1>Sleep and memory</h1><h2>Abstract</h2>"""
    b"""<p>We tested 80 people.</p></article></body></html>"""
)


def _generic_contents():
    contents = HtmlParser(_GENERIC).parse()
    assert contents.preparsed_metadata_trusted is False
    return contents


@pytest.mark.asyncio
async def test_no_llm_keeps_the_untrusted_record_and_its_language():
    from bibr.pipeline.stages import post_parse

    contents = _generic_contents()
    with patch("bibr.extract.extractor.MetadataExtractor") as extractor_cls:
        result = await post_parse._extract_metadata_and_equations(
            contents, file_hash="h", no_llm=True, llm_client=None
        )
    extractor_cls.assert_not_called()
    assert result is contents.preparsed_metadata
    assert result.language == "en"
    assert result.license == "https://creativecommons.org/licenses/by/4.0/"


@pytest.mark.asyncio
async def test_llm_run_extracts_front_matter_for_an_untrusted_record():
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import post_parse

    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False
    contents = _generic_contents()
    extracted = PaperMetadata(
        doi="",
        title="Sleep and memory",
        abstract="We tested 80 people.",
        authors=[PaperAuthor(author_id=1, given="Ann", family="Lee", affiliation="")],
    )
    extractor = MagicMock()
    extractor.extract_all_metadata = AsyncMock(return_value=extracted)
    extractor.validation_issues = []
    with patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor) as cls:
        result = await post_parse._extract_metadata_and_equations(
            contents, file_hash="h", no_llm=False, llm_client=MagicMock(), settings=settings
        )
    cls.assert_called_once()
    assert result is extracted

    post_parse._fill_from_untrusted_preparsed(contents, result, llm_active=True)
    # Fill-empty fields come from the page meta ...
    assert result.language == "en"
    assert result.license == "https://creativecommons.org/licenses/by/4.0/"
    assert result.keywords == ["sleep", "memory"]
    # ... the SEO-prone identity fields never do.
    assert result.title == "Sleep and memory"
    assert [(a.given, a.family) for a in result.authors] == [("Ann", "Lee")]


def test_fill_leaves_extracted_values_and_trusted_records_alone():
    from bibr.pipeline.stages import post_parse

    contents = _generic_contents()
    extracted = PaperMetadata(doi="", title="T", language="de")
    post_parse._fill_from_untrusted_preparsed(contents, extracted, llm_active=True)
    assert extracted.language == "de"

    # No-LLM runs use the record itself; nothing to fill.
    other = PaperMetadata(doi="", title="T")
    post_parse._fill_from_untrusted_preparsed(contents, other, llm_active=False)
    assert other.language is None

    trusted = HtmlParser(
        b"""<html lang="en"><head><meta name="citation_title" content="P">"""
        b"""<meta name="citation_author" content="Jane Smith"></head>"""
        b"""<body><article><h2>Intro</h2><p>Text.</p></article></body></html>"""
    ).parse()
    assert trusted.preparsed_metadata_trusted is True
    fresh = PaperMetadata(doi="", title="T")
    post_parse._fill_from_untrusted_preparsed(trusted, fresh, llm_active=True)
    assert fresh.language is None


@pytest.mark.parametrize(("llm_active", "required"), [(True, True), (False, False)])
def test_front_matter_target_follows_the_trust_flag(llm_active, required):
    from bibr.pipeline.stages import post_parse

    contents = _generic_contents()
    seen = {}

    def fake_resolve(contents, *, expected_identity, target_required, settings):
        seen["target_required"] = target_required
        return None, []

    with patch("bibr.extract.front_matter.resolve_front_matter", fake_resolve):
        post_parse._attach_front_matter_resolution(contents, None, metadata_llm_active=llm_active)
    assert seen["target_required"] is required
