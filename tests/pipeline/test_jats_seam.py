"""JATS post-parse seam + native reference-segmentation branch."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from bibr.models import PaperMetadata, PaperReference
from bibr.paper_contents import PaperContents


def _contents(**kw) -> PaperContents:
    return PaperContents(
        sentences=[],
        sections=[],
        tables=[],
        links=[],
        sections_text={},
        **kw,
    )


def _ref(bib_id: int) -> PaperReference:
    return PaperReference(
        bib_id=bib_id,
        title=f"Ref {bib_id}",
        first_page=None,
        volume=None,
        authors=None,
        year=2000,
        container=None,
    )


# ---------------------------------------------------------------------------
# post_parse: preparsed metadata bypasses core LLM extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preparsed_native_refs_bypass_metadata_extractor(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import post_parse

    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False
    meta = PaperMetadata(doi="10.1/x", title="Preparsed Title")
    refs = [_ref(1), _ref(2)]
    contents = _contents(preparsed_metadata=meta, native_references=refs)

    with patch("bibr.extract.extractor.MetadataExtractor") as ME:
        result = await post_parse._extract_metadata_and_equations(
            contents,
            file_hash="h",
            no_llm=False,
            llm_client=MagicMock(),
            settings=settings,
        )

    ME.assert_not_called()  # core LLM extraction skipped
    assert result is meta
    assert result.references == refs
    assert contents.equations == []


@pytest.mark.asyncio
async def test_preparsed_survives_no_llm():
    from bibr.pipeline.stages import post_parse

    meta = PaperMetadata(doi="10.1/x", title="Preparsed Title")
    refs = [_ref(1)]
    contents = _contents(preparsed_metadata=meta, native_references=refs)

    with patch("bibr.extract.extractor.MetadataExtractor") as ME:
        result = await post_parse._extract_metadata_and_equations(
            contents, file_hash="h", no_llm=True, llm_client=None
        )

    ME.assert_not_called()
    assert result is meta
    assert result.references == refs  # LLM-free structured refs still applied
    assert contents.equations == []


@pytest.mark.asyncio
async def test_no_preparsed_no_llm_returns_empty():
    from bibr.pipeline.stages import post_parse

    contents = _contents()
    result = await post_parse._extract_metadata_and_equations(
        contents, file_hash="h", no_llm=True, llm_client=None
    )
    assert result.doi == "" and result.title == ""
    assert contents.equations == []


@pytest.mark.asyncio
async def test_preparsed_ref_strings_run_extractor(monkeypatch):
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import post_parse

    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False
    meta = PaperMetadata(doi="10.1/x", title="T")
    contents = _contents(
        preparsed_metadata=meta,
        native_ref_strings=["Alpha (2000). One.", "Beta (2001). Two."],
    )
    parsed = [_ref(1), _ref(2)]

    extractor = MagicMock()
    extractor._collect_reference_rows.return_value = pd.DataFrame({"text": ["a", "b"]})
    extractor._extract_references = AsyncMock(return_value=parsed)

    with patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor):
        result = await post_parse._extract_metadata_and_equations(
            contents,
            file_hash="h",
            no_llm=False,
            llm_client=MagicMock(),
            settings=settings,
        )

    assert result is meta
    assert result.references == parsed
    extractor._extract_references.assert_awaited_once()


# ---------------------------------------------------------------------------
# ref_extractor: native segmentation branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_seg_branch_returns_strings_verbatim():
    from bibr.extract.ref_extractor import ReferenceExtractor

    strings = ["First ref (2000).", "Second ref (2001)."]
    contents = _contents(native_ref_strings=strings)
    ext = ReferenceExtractor(contents, llm_client=MagicMock())

    # Even with a configured strategy, the native strings win and no cascade runs.
    result = await ext._segment_references("ignored ref text", "geom")
    assert result == strings


@pytest.mark.asyncio
async def test_refs_off_skips_native_references(monkeypatch):
    """--refs off means no refs in the output even when JATS pre-parsed them."""
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import post_parse

    settings = snapshot_settings()
    settings.EQUATION_EXTRACTION = False
    meta = PaperMetadata(doi="10.1/x", title="Preparsed Title")
    contents = _contents(preparsed_metadata=meta, native_references=[_ref(1)])

    result = await post_parse._extract_metadata_and_equations(
        contents,
        file_hash="h",
        no_llm=False,
        llm_client=MagicMock(),
        ref_parse_strategy="off",
        settings=settings,
    )
    assert result is meta
    assert result.references == []


@pytest.mark.asyncio
async def test_refs_off_skips_native_references_under_no_llm():
    from bibr.pipeline.stages import post_parse

    meta = PaperMetadata(doi="10.1/x", title="Preparsed Title")
    contents = _contents(preparsed_metadata=meta, native_references=[_ref(1)])

    result = await post_parse._extract_metadata_and_equations(
        contents,
        file_hash="h",
        no_llm=True,
        llm_client=None,
        ref_parse_strategy="off",
    )
    assert result is meta
    assert result.references == []
