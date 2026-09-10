"""Client-layer audit fixes.

L1 (a byte-identical "recovery" re-roll), L4 (case-duplicated Crossref cache
keys), L7 (a resolver body whose ``candidates`` is null), L13 (batch resume
never checked its manifest).

M7 lives in ``tests/extract/test_audit_ref_fixes.py`` and L5/L14 in
``tests/test_paper_ocr_metadata.py`` / ``tests/ocr/test_image_utils.py``,
alongside the code each one covers.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibr.clients.batch import AnthropicBatchAdapter, BatchRequest
from bibr.models import PaperMetadata


class TestJsonModeRerollIsSkippedWhenItCannotDiffer:
    """L1: for every provider but openai the re-roll is byte-identical."""

    def _client(self, provider):
        from bibr.clients.llm import LLMClient
        from bibr.config import GlobalSettings

        return LLMClient(settings=GlobalSettings(llm={"provider": provider}))

    def test_openai_can_actually_change_mode(self):
        assert self._client("openai").json_mode_reroll_is_distinct()

    @pytest.mark.parametrize("provider", ["google", "anthropic", "groq", "ollama"])
    def test_other_providers_cannot(self, provider):
        assert not self._client(provider).json_mode_reroll_is_distinct()


class TestCrossrefCacheKey:
    """L4: DOIs are case-insensitive; the key was not."""

    async def test_case_variants_share_one_cache_entry(self):
        from bibr.clients.crossref import CrossrefClient

        client = CrossrefClient(mailto="x@y.z")
        seen = []

        async def fake_cached(key, factory):
            seen.append(key)
            return {}

        with patch.object(client, "_cached", side_effect=fake_cached):
            await client.works("10.1037/ABC123")
            await client.works("10.1037/abc123")

        assert len(set(seen)) == 1


class TestResolverNullCandidates:
    """L7: ``{"candidates": null}`` made search() return None, not a list."""

    async def _search(self, payload):
        from bibr.clients.resolver import ResolverClient

        client = ResolverClient(base_url="http://resolver.invalid")
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value=payload)
        client._client = MagicMock()
        client._client.post = AsyncMock(return_value=response)
        return await client.search("Some title", 2020, 5)

    async def test_null_candidates_becomes_an_empty_list(self):
        assert await self._search({"candidates": None}) == []

    async def test_a_non_dict_body_becomes_an_empty_list(self):
        assert await self._search(["unexpected"]) == []

    async def test_real_candidates_pass_through(self):
        assert await self._search({"candidates": [{"doi": "10.1/x"}]}) == [{"doi": "10.1/x"}]


class TestBatchManifestVerification:
    """L13: resuming against a foreign state file returned partial results."""

    def _request(self, custom_id, text):
        return BatchRequest(custom_id=custom_id, schema=PaperMetadata, system="sys", user_text=text)

    def _state(self, tmp_path, adapter, requests):
        path = tmp_path / "state.json"
        import hashlib

        path.write_text(
            json.dumps(
                {
                    "provider": "anthropic",
                    "model": adapter.model,
                    "batch_id": "batch_abc",
                    "manifest": {
                        r.custom_id: hashlib.sha256(r.user_text.encode()).hexdigest()[:16]
                        for r in requests
                    },
                }
            )
        )
        return path

    def test_a_matching_manifest_resumes(self, tmp_path):
        adapter = AnthropicBatchAdapter(client=MagicMock())
        requests = [self._request("a", "paper one")]
        path = self._state(tmp_path, adapter, requests)

        adapter._verify_manifest(json.loads(path.read_text()), requests, path)

    def test_a_foreign_state_file_is_refused(self, tmp_path):
        adapter = AnthropicBatchAdapter(client=MagicMock())
        path = self._state(tmp_path, adapter, [self._request("a", "paper one")])
        other = [self._request("a", "a completely different paper")]

        with pytest.raises(ValueError, match="different requests"):
            adapter._verify_manifest(json.loads(path.read_text()), other, path)

    def test_an_unknown_custom_id_is_refused(self, tmp_path):
        adapter = AnthropicBatchAdapter(client=MagicMock())
        path = self._state(tmp_path, adapter, [self._request("a", "paper one")])
        other = [self._request("b", "paper one")]

        with pytest.raises(ValueError, match="different requests"):
            adapter._verify_manifest(json.loads(path.read_text()), other, path)

    def test_resuming_with_a_subset_is_allowed(self, tmp_path):
        """Callers legitimately resume after writing some results durably."""
        adapter = AnthropicBatchAdapter(client=MagicMock())
        requests = [self._request("a", "paper one"), self._request("b", "paper two")]
        path = self._state(tmp_path, adapter, requests)

        adapter._verify_manifest(json.loads(path.read_text()), requests[:1], path)

    def test_a_pre_manifest_state_file_is_tolerated(self, tmp_path):
        adapter = AnthropicBatchAdapter(client=MagicMock())
        path = tmp_path / "state.json"

        adapter._verify_manifest({"batch_id": "b"}, [self._request("a", "x")], path)
