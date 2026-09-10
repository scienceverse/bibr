"""Per-paper LLM token-usage attribution and export.

Covers: contextvar-scoped per-file buckets in LLMClient (race-free across
concurrent asyncio tasks), post_parse attaching ``paper.llm_usage_labels``,
and the ``extraction.usage`` block in the JSON export.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from bibr.config import Settings
from tests.export.conftest import extraction_block as _extraction_block


@pytest.fixture()
def _enable_tracking(monkeypatch):
    """Enable LLM usage tracking for the duration of a test."""
    monkeypatch.setattr(Settings.llm, "track_usage", True)


def _mock_limiter():
    limiter = mock.AsyncMock()
    limiter.acquire = mock.AsyncMock()
    return limiter


def _completion(input_tokens, output_tokens, cached=0):
    """Anthropic-shaped fake completion (SimpleNamespace so absent attrs raise)."""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=cached,
    )
    return SimpleNamespace(usage=usage)


# ── per-file usage buckets in LLMClient ────────────────────────────────


@pytest.mark.usefixtures("_enable_tracking")
class TestPerFileUsage:
    async def test_per_file_buckets_isolated_across_tasks(self):
        """Two concurrent tasks with different file hashes must not bleed."""
        from bibr.clients.llm import LLMClient, usage_file_context

        client = LLMClient()

        async def record(file_hash, completions):
            with usage_file_context(file_hash):
                for c in completions:
                    # Yield so the sibling task interleaves mid-context.
                    await asyncio.sleep(0)
                    client._record_usage(c)

        await asyncio.gather(
            record("hash-a", [_completion(100, 10, cached=64), _completion(50, 5)]),
            record("hash-b", [_completion(7, 3, cached=2), _completion(7, 3)]),
        )

        model = Settings.llm.model
        assert client.usage_for_file("hash-a") == {
            model: {
                "input_tokens": 150,
                "output_tokens": 15,
                "total_tokens": 165,
                "cached_input_tokens": 64,
            }
        }
        assert client.usage_for_file("hash-b") == {
            model: {
                "input_tokens": 14,
                "output_tokens": 6,
                "total_tokens": 20,
                "cached_input_tokens": 2,
            }
        }
        # Global aggregate is the sum of both files.
        assert client.usage[model]["total_tokens"] == 185
        assert client.usage[model]["cached_input_tokens"] == 66

    async def test_usage_for_file_returns_deep_copy(self):
        from bibr.clients.llm import LLMClient, usage_file_context

        client = LLMClient()
        with usage_file_context("hash-a"):
            client._record_usage(_completion(10, 5))

        model = Settings.llm.model
        snapshot = client.usage_for_file("hash-a")
        snapshot[model]["input_tokens"] = 999
        assert client.usage_for_file("hash-a")[model]["input_tokens"] == 10

    async def test_no_file_context_records_only_global(self):
        from bibr.clients.llm import LLMClient

        client = LLMClient()
        client._record_usage(_completion(10, 5))

        model = Settings.llm.model
        assert client.usage[model]["total_tokens"] == 15
        assert client._usage_by_file == {}
        assert client.usage_for_file("anything") == {}

    async def test_extract_call_records_per_file(self):
        """The production call path attributes usage to the active file context."""
        from bibr.clients.llm import LLMClient, usage_file_context
        from bibr.schemas import TitleKeywordsLLM

        fake_response = TitleKeywordsLLM(title="Test Paper", keywords=["test"])

        client = LLMClient()
        client._limiter = _mock_limiter()
        fake_instructor = mock.AsyncMock()
        fake_instructor.create_with_completion = mock.AsyncMock(
            return_value=(fake_response, _completion(100, 50, cached=20))
        )
        client._client = fake_instructor

        with usage_file_context("hash-x"):
            await client.extract_title_keywords("Some paper text")

        model = Settings.llm.model
        counts = client.usage_for_file("hash-x")[model]
        assert counts["input_tokens"] == 100
        assert counts["output_tokens"] == 50
        assert counts["cached_input_tokens"] == 20

    async def test_usage_pop_file_evicts_bucket(self):
        from bibr.clients.llm import LLMClient, usage_file_context

        client = LLMClient()
        with usage_file_context("hash-a"):
            client._record_usage(_completion(10, 5))

        model = Settings.llm.model
        popped = client.usage_pop_file("hash-a")
        assert popped[model]["total_tokens"] == 15
        # Evicted: the map stays bounded and a re-read yields nothing.
        assert client._usage_by_file == {}
        assert client.usage_pop_file("hash-a") == {}
        # Global aggregate untouched by the pop.
        assert client.usage[model]["total_tokens"] == 15

    async def test_new_usage_context_key_unique_per_invocation(self):
        from bibr.clients.llm import LLMClient, new_usage_context_key, usage_file_context

        k1 = new_usage_context_key("samehash")
        k2 = new_usage_context_key("samehash")
        assert k1 != k2 and k1.startswith("samehash#") and k2.startswith("samehash#")
        assert new_usage_context_key("") is None
        assert new_usage_context_key(None) is None

        # Reprocessing the same content hash never accumulates across runs.
        client = LLMClient()
        with usage_file_context(k1):
            client._record_usage(_completion(10, 5))
        first = client.usage_pop_file(k1)
        with usage_file_context(k2):
            client._record_usage(_completion(10, 5))
        second = client.usage_pop_file(k2)
        model = Settings.llm.model
        assert first[model]["total_tokens"] == 15
        assert second[model]["total_tokens"] == 15


# ── post_parse attaches paper.llm_usage ────────────────────────────────


def _minimal_contents():
    from bibr.paper_contents import PaperContents, PaperSection

    return PaperContents(
        sentences=[],
        sections=[PaperSection(section_id=0, header="Root", level=0, parent_section_id=None)],
        tables=[],
        links=[],
        sections_text={0: ""},
    )


class TestPostParseLlmUsage:
    async def test_post_parse_attaches_llm_usage(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch

        from bibr.models import PaperMetadata
        from bibr.pipeline.stages.post_parse import post_parse

        monkeypatch.setattr(Settings, "EQUATION_EXTRACTION", False)

        file_usage = {"model-x": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
        file_usage_labels = {
            ("extract_title_keywords", "google", "model-x"): {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            }
        }
        llm_client = MagicMock()
        llm_client._track_usage = True
        llm_client.usage_pop_file = MagicMock(return_value=file_usage)
        llm_client.usage_labels_pop_file = MagicMock(return_value=file_usage_labels)

        extractor = MagicMock()
        extractor.extract_all_metadata = AsyncMock(return_value=PaperMetadata(doi="", title="T"))

        with (
            patch("bibr.extract.extractor.MetadataExtractor", return_value=extractor),
            patch(
                "bibr.structure.implicit_sections.detect_implicit_sections",
                AsyncMock(return_value=None),
            ),
            patch("bibr.structure.citation_linker.detect_bib_xrefs", AsyncMock(return_value=[])),
        ):
            paper = await post_parse(
                contents=_minimal_contents(),
                file_name="x.pdf",
                file_hash="deadbeef",
                llm_client=llm_client,
            )

        # The attribution key is unique per invocation: "<file_hash>#<n>",
        # popped (not read) so reprocessing never accumulates.
        llm_client.usage_pop_file.assert_called_once()
        (key,) = llm_client.usage_pop_file.call_args.args
        assert key.startswith("deadbeef#")
        # ``llm_usage_labels`` is the only usage surface Paper still carries;
        # the by-model-only ``llm_usage`` field went with the root export key.
        assert paper.llm_usage_labels == file_usage_labels

    async def test_post_parse_no_llm_has_empty_usage(self):
        from bibr.pipeline.stages.post_parse import post_parse

        paper = await post_parse(
            contents=_minimal_contents(),
            file_name="x.pdf",
            file_hash="deadbeef",
            no_llm=True,
        )

        assert paper.llm_usage_labels == {}


# ── JSON export ────────────────────────────────────────────────────────


def _minimal_paper(**overrides):
    from bibr.input.file import InputFile, InputFormat
    from bibr.models import PaperMetadata
    from bibr.paper import Paper

    input_file = InputFile(
        path=Path("/tmp/test.pdf"),
        file_hash="abc123",
        input_format=InputFormat(
            file_extension=".pdf", detected_mime_type="application/pdf", file_type="pdf"
        ),
    )
    defaults = {
        "input_file": input_file,
        "metadata": PaperMetadata(doi="10.1234/test", title="Test Paper"),
        "contents": _minimal_contents(),
    }
    defaults.update(overrides)
    return Paper(**defaults)


class TestExportLlmUsage:
    """v11 replaced the per-model root ``llm_usage`` with ``extraction.usage``,
    whose ``breakdown`` rows carry the (label, provider, model) dimensions."""

    def test_export_emits_usage_under_extraction(self):
        from bibr.export.json_export import export_paper_to_json, validate_export
        from bibr.export.usage import build_usage_export

        paper = _minimal_paper(
            llm_usage_labels={
                ("extract_authors", "google", "gemini-x"): {
                    "calls": 2,
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 20,
                    "total_tokens": 120,
                }
            }
        )
        paper.extraction = _extraction_block(
            usage=build_usage_export(paper.llm_usage_labels),
        )
        result = export_paper_to_json(paper)

        assert result["extraction"]["usage"]["totals"]["total_tokens"] == 120
        assert result["extraction"]["usage"]["breakdown"][0]["model"] == "gemini-x"
        assert "llm_usage" not in result
        assert validate_export(result) == []

    def test_export_omits_usage_when_empty(self):
        from bibr.export.json_export import export_paper_to_json, validate_export

        paper = _minimal_paper()
        paper.extraction = _extraction_block()
        result = export_paper_to_json(paper)

        assert "usage" not in result["extraction"]
        assert validate_export(result) == []
