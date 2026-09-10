"""StructuredBackend seam — the injectable transport under LLMClient.

LLMClient keeps the limiter/breaker/retry/usage machinery; the innermost
structured-output call is a swappable backend so a non-Instructor engine
(e.g. NuExtract 3's template dialect) can slot in without forking the client.
"""

import pytest

from bibr.clients.llm import InstructorBackend, LLMClient
from bibr.clients.prompts import part, prompt_text
from bibr.clients.structured import StructuredBackend
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM


def _settings(**llm_overrides) -> GlobalSettings:
    settings = GlobalSettings()
    for key, value in llm_overrides.items():
        setattr(settings.llm, key, value)
    return settings


class _FakeBackend:
    """Records create() calls; returns a canned validated model."""

    def __init__(self, result, failures=0, exc=None):
        self.result = result
        self.calls: list[dict] = []
        self._failures = failures
        self._exc = exc

    async def create(
        self,
        *,
        response_model,
        system,
        messages,
        want_completion,
        reasoning_effort=None,
        max_tokens=None,
        client_override=None,
    ):
        self.calls.append(
            {
                "response_model": response_model,
                "system": system,
                "messages": messages,
                "want_completion": want_completion,
                "max_tokens": max_tokens,
            }
        )
        if self._failures > 0:
            self._failures -= 1
            raise self._exc
        return self.result, None


class TestBackendSubstitution:
    async def test_public_method_routes_through_injected_backend(self):
        canned = TitleKeywordsLLM(title="T", abstract=None, keywords=[])
        fake = _FakeBackend(canned)
        client = LLMClient(backend=fake)

        result = await client.extract_title_keywords("Some front page text")

        assert result is canned
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["response_model"] is TitleKeywordsLLM
        assert call["system"] == "You are a scientific paper metadata extractor."
        assert "Some front page text" in prompt_text(call["messages"][0]["content"])
        # No instructor client was ever built for a substituted backend.
        assert client._client is None

    async def test_transient_backend_failure_still_retried_by_client(self, monkeypatch):
        class _Transient(Exception):
            status_code = 503

        canned = TitleKeywordsLLM(title="T", abstract=None, keywords=[])
        fake = _FakeBackend(canned, failures=2, exc=_Transient("boom"))
        client = LLMClient(backend=fake)
        monkeypatch.setattr(LLMClient, "_RETRY_BASE_DELAY", 0.001)

        result = await client.extract_title_keywords("text")

        assert result is canned
        assert len(fake.calls) == 3  # 2 transient failures + 1 success

    async def test_fatal_backend_failure_not_retried(self):
        class _Fatal(Exception):
            status_code = 400

        fake = _FakeBackend(None, failures=5, exc=_Fatal("bad request"))
        client = LLMClient(backend=fake)

        from bibr.exceptions import UpstreamServiceError

        with pytest.raises(UpstreamServiceError):
            await client.extract_title_keywords("text")

        assert len(fake.calls) == 1


class TestDefaultBackend:
    def test_default_backend_is_instructor(self):
        client = LLMClient()
        assert isinstance(client._backend, InstructorBackend)

    @pytest.mark.parametrize(
        "model",
        [
            "numind/NuExtract3",
            "numind/NuExtract3-FP8",
            "numind/NuExtract3-mlx-8bits",
            "numind/NuExtract3-GGUF:Q4_K_M",
        ],
    )
    def test_nuextract_auto_backend_is_instructor_until_qualified(self, model):
        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            model=model,
            structured_backend="auto",
        )

        assert isinstance(LLMClient(settings=settings)._backend, InstructorBackend)

    def test_auto_backend_keeps_instructor_for_non_mlx_models(self):
        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            model="google/gemma-4-e4b-it",
        )

        client = LLMClient(settings=settings)

        assert isinstance(client._backend, InstructorBackend)

    @pytest.mark.parametrize(
        "model",
        [
            "numind/NuExtract3",
            "numind/NuExtract3-FP8",
            "numind/NuExtract3-mlx-8bits",
            "numind/NuExtract3-GGUF:Q4_K_M",
        ],
    )
    def test_nuextract_explicit_native_backend_remains_available(self, model):
        from bibr.clients.nuextract import NuExtractNativeBackend

        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            model=model,
            structured_backend="nuextract-native",
        )

        assert isinstance(LLMClient(settings=settings)._backend, NuExtractNativeBackend)

    def test_gguf_nuextract_without_base_url_keeps_instructor(self):
        settings = _settings(
            provider="openai",
            base_url=None,
            model="numind/NuExtract3-GGUF:Q4_K_M",
        )

        client = LLMClient(settings=settings)

        assert isinstance(client._backend, InstructorBackend)

    def test_explicit_instructor_override_beats_gguf_auto_routing(self):
        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            model="numind/NuExtract3-GGUF:Q4_K_M",
            structured_backend="instructor",
        )

        client = LLMClient(settings=settings)

        assert isinstance(client._backend, InstructorBackend)

    def test_backends_satisfy_protocol(self):
        from bibr.clients.nuextract import NuExtractNativeBackend

        assert isinstance(InstructorBackend(LLMClient()), StructuredBackend)
        assert isinstance(NuExtractNativeBackend(settings=_settings()), StructuredBackend)
        assert isinstance(_FakeBackend(None), StructuredBackend)


class TestInstructorBackend:
    async def test_prepends_system_and_returns_completion_when_wanted(self, monkeypatch):
        client = LLMClient()
        seen = {}

        class _FakeInstructor:
            async def create_with_completion(self, *, response_model, messages, max_retries, **kw):
                seen["messages"] = messages
                return "MODEL", "COMPLETION"

            async def create(self, *, response_model, messages, max_retries, **kw):
                seen["messages"] = messages
                return "MODEL", None

        monkeypatch.setattr(client, "_get_client", lambda: _FakeInstructor())
        result, completion = await client._backend.create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[{"role": "user", "content": "hi"}],
            want_completion=True,
        )

        assert (result, completion) == ("MODEL", "COMPLETION")
        assert seen["messages"][0] == {"role": "system", "content": "SYS"}
        assert seen["messages"][1] == {"role": "user", "content": "hi"}

    async def test_recovery_backend_suppresses_inherited_reasoning_and_hidden_reasks(self):
        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            reasoning_effort="minimal",
            validation_attempts=3,
        )
        client = LLMClient(settings=settings)
        seen = {}

        class _FakeInstructor:
            async def create(self, **kwargs):
                seen.update(kwargs)
                return "MODEL"

        backend = client._make_recovery_instructor_backend()
        result, completion = await backend.create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[{"role": "user", "content": "hi"}],
            want_completion=False,
            reasoning_effort=None,
            client_override=_FakeInstructor(),
        )

        assert (result, completion) == ("MODEL", None)
        assert "reasoning_effort" not in seen
        assert seen["max_retries"].stop.max_attempt_number == 1

    async def test_ordinary_backend_retains_reasoning_and_validation_configuration(self):
        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            reasoning_effort="minimal",
            validation_attempts=3,
        )
        client = LLMClient(settings=settings)
        seen = {}

        class _FakeInstructor:
            async def create(self, **kwargs):
                seen.update(kwargs)
                return "MODEL"

        result, completion = await client._backend.create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[{"role": "user", "content": "hi"}],
            want_completion=False,
            reasoning_effort="high",
            client_override=_FakeInstructor(),
        )

        assert (result, completion) == ("MODEL", None)
        assert seen["reasoning_effort"] == "high"
        assert seen["max_retries"].stop.max_attempt_number == 3


class TestNuExtractNativeBackend:
    def test_unannotated_prompt_falls_back_to_flattened_document(self):
        from bibr.clients.nuextract import _project_native_prompt

        projected, instructions = _project_native_prompt(
            [
                {
                    "role": "user",
                    "content": [part("Document prefix", cache=True), part("\nRemainder")],
                }
            ]
        )

        assert projected == [{"role": "user", "content": "Document prefix\nRemainder"}]
        assert instructions is None

    def test_template_uses_semantic_nuextract_leaf_types(self):
        from bibr.clients.nuextract import template_for_model
        from bibr.schemas import AuthorsLLM, PaperClassificationLLM

        title = template_for_model(TitleKeywordsLLM)
        authors = template_for_model(AuthorsLLM)
        classification = template_for_model(PaperClassificationLLM)

        assert title["title"] == "verbatim-string"
        assert title["keywords"] == ["verbatim-string"]
        assert authors["authors"][0]["given"] == "string"
        assert authors["authors"][0]["email"] == "email-address"
        assert authors["authors"][0]["corresponding"] == "boolean"
        assert classification["paper_type"] == [
            "empirical",
            "review",
            "meta-analysis",
            "case-study",
            "commentary",
            "corrigendum",
            "erratum",
            "retraction",
        ]

    def test_template_fails_closed_on_unsupported_schema_branches(self):
        from pydantic import BaseModel

        from bibr.clients.nuextract import template_for_model

        class MixedSchema(BaseModel):
            good: str
            unsupported: str | int

        with pytest.raises(ValueError, match="unsupported|dropped"):
            template_for_model(MixedSchema)

    def test_every_registered_prompt_has_a_supported_native_template(self, caplog):
        from bibr.clients.nuextract import template_for_model
        from bibr.clients.prompts import PROMPTS

        with caplog.at_level("WARNING", logger="bibr.clients.nuextract"):
            templates = {
                name: template_for_model(spec.response_model) for name, spec in PROMPTS.items()
            }

        assert all(isinstance(template, dict) and template for template in templates.values())
        assert "dropped unsupported schema branches" not in caplog.text

    async def test_sends_native_template_without_response_format(self, monkeypatch):
        from types import SimpleNamespace

        from bibr.clients.nuextract import NuExtractNativeBackend

        settings = _settings(
            provider="openai",
            base_url="http://127.0.0.1:8767/v1",
            api_key="not-needed",
            model="numind/NuExtract3-mlx-8bits",
            max_tokens=123,
        )
        seen = {}

        class _FakeCompletions:
            async def create(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content=(
                                    '{"title":"T","abstract":null,"keywords":[],'
                                    '"journal":null,"volume":null,"issue":null,'
                                    '"first_page":null,"last_page":null,"issn":null,'
                                    '"publisher":null,"published":null,"license":null}'
                                )
                            ),
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                seen["client_kwargs"] = kwargs

            chat = _FakeChat()

        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)

        result, completion = await NuExtractNativeBackend(settings=settings).create(
            response_model=TitleKeywordsLLM,
            system="SYS",
            messages=[
                {
                    "role": "user",
                    "content": [
                        part(
                            "DOCUMENT",
                            cache=True,
                            nuextract_role="document",
                        ),
                        part(
                            "TASK INSTRUCTIONS",
                            nuextract_role="instructions",
                        ),
                    ],
                }
            ],
            want_completion=True,
        )

        assert isinstance(result, TitleKeywordsLLM)
        assert completion is not None
        assert seen["client_kwargs"]["base_url"] == "http://127.0.0.1:8767/v1"
        assert seen["model"] == "numind/NuExtract3-mlx-8bits"
        assert seen["temperature"] == 0.0
        assert seen["max_tokens"] == 123
        assert "response_format" not in seen
        assert seen["messages"] == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "DOCUMENT"},
        ]
        template_kwargs = seen["extra_body"]["chat_template_kwargs"]
        assert template_kwargs["enable_thinking"] is False
        assert template_kwargs["instructions"] == "TASK INSTRUCTIONS"
        assert "DOCUMENT" not in template_kwargs["instructions"]
        assert '"title": "verbatim-string"' in template_kwargs["template"]
