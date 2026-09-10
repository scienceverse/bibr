"""Pinned NuExtract Jinja and production request-construction contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bibr.clients.nuextract import (
    NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256,
    NUEXTRACT3_FP8_EXPECTED_REVISION,
)
from bibr.clients.prompts import part
from bibr.config import GlobalSettings
from bibr.schemas import TitleKeywordsLLM

DATA_DIR = Path(__file__).parent / "data"
JINJA_PATH = DATA_DIR / "nuextract3_fp8_d88964_chat_template.jinja"
GOLDEN_PATH = DATA_DIR / "nuextract3_fp8_d88964_rendered.golden.txt"


def _settings(**llm_overrides) -> GlobalSettings:
    settings = GlobalSettings()
    for key, value in llm_overrides.items():
        setattr(settings.llm, key, value)
    return settings


def render_pinned_template(**kwargs) -> str:
    # transformers ships in the optional ``ml`` extra; a core install can still
    # exercise the request-builder contracts below, just not the render.
    pytest.importorskip("transformers")
    from transformers.utils.chat_template_utils import _compile_jinja_template

    return _compile_jinja_template(JINJA_PATH.read_text(encoding="utf-8")).render(**kwargs)


def _minimal_render_kwargs() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Document"},
        ],
        "template": '{"title":"verbatim-string"}',
        "instructions": "Extract title",
        "enable_thinking": False,
        "add_generation_prompt": True,
    }


def test_pinned_fp8_jinja_rendering_contract():
    source = JINJA_PATH.read_text(encoding="utf-8")
    assert NUEXTRACT3_FP8_EXPECTED_REVISION == "d88964bad5ba47333cb721b351e19045ee6a6fc0"
    assert hashlib.sha256(source.encode()).hexdigest() == NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256

    rendered = render_pinned_template(**_minimal_render_kwargs())

    assert rendered == GOLDEN_PATH.read_text(encoding="utf-8")
    assert "【task】structured" in rendered
    assert rendered.index("【template_start】") < rendered.index("【instructions_start】")
    assert rendered.index("【instructions_start】") < rendered.index("【document_start】")
    assert "【document_end】" in rendered
    assert rendered.endswith("<think>\n\n</think>\n\n")


def test_truthy_template_selects_structured_mode_without_explicit_mode():
    kwargs = _minimal_render_kwargs()

    omitted = render_pinned_template(**kwargs)
    explicit = render_pinned_template(**kwargs, mode="structured")

    assert omitted == explicit


def test_native_request_builder_projects_roles_and_only_supported_template_kwargs():
    from bibr.clients.nuextract import build_native_request_kwargs

    settings = _settings(
        provider="openai",
        base_url="http://127.0.0.1:8767/v1",
        model="numind/NuExtract3-FP8",
        temperature=0.2,
    )

    kwargs = build_native_request_kwargs(
        settings=settings,
        response_model=TitleKeywordsLLM,
        system="System",
        messages=[
            {
                "role": "user",
                "content": [
                    part("Document", cache=True, nuextract_role="document"),
                    part("Extract title", nuextract_role="instructions"),
                ],
            }
        ],
        max_tokens=321,
    )

    assert kwargs["model"] == "numind/NuExtract3-FP8"
    assert kwargs["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Document"},
    ]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 321
    assert "response_format" not in kwargs
    assert "reasoning_effort" not in kwargs
    assert "reasoning" not in kwargs
    template_kwargs = kwargs["extra_body"]["chat_template_kwargs"]
    assert set(template_kwargs) == {"template", "instructions", "enable_thinking"}
    assert isinstance(template_kwargs["template"], str)
    assert json.loads(template_kwargs["template"])["title"] == "verbatim-string"
    assert template_kwargs["instructions"] == "Extract title"
    assert template_kwargs["enable_thinking"] is False
    assert "mode" not in template_kwargs


def test_native_request_builder_rejects_assistant_input():
    from bibr.clients.nuextract import build_native_request_kwargs

    with pytest.raises(ValueError, match="assistant"):
        build_native_request_kwargs(
            settings=_settings(model="numind/NuExtract3-FP8"),
            response_model=TitleKeywordsLLM,
            system="System",
            messages=[{"role": "assistant", "content": "Prior answer"}],
            max_tokens=None,
        )


def test_native_prompt_projection_separates_distinct_instruction_parts():
    from bibr.clients.nuextract import _project_native_prompt

    _messages, instructions = _project_native_prompt(
        [
            {
                "role": "user",
                "content": [
                    part("First instruction.", nuextract_role="instructions"),
                    part("Document", nuextract_role="document"),
                    part("Second instruction.", nuextract_role="instructions"),
                ],
            }
        ]
    )

    assert instructions == "First instruction.\nSecond instruction."


async def test_native_backend_delegates_payload_construction(monkeypatch):
    import bibr.clients.nuextract as nuextract

    expected_kwargs = {
        "model": "numind/NuExtract3-FP8",
        "messages": [{"role": "system", "content": "System"}],
        "temperature": 0.2,
        "extra_body": {"chat_template_kwargs": {}},
    }
    builder = MagicMock(return_value=expected_kwargs)
    monkeypatch.setattr(nuextract, "build_native_request_kwargs", builder)
    seen = {}

    class FakeCompletions:
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
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )

    backend = nuextract.NuExtractNativeBackend(
        settings=_settings(model="numind/NuExtract3-FP8", temperature=0.2)
    )
    monkeypatch.setattr(
        backend,
        "_get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    await backend.create(
        response_model=TitleKeywordsLLM,
        system="System",
        messages=[{"role": "user", "content": "Document"}],
        want_completion=False,
        reasoning_effort="high",
        max_tokens=123,
    )

    builder.assert_called_once_with(
        settings=backend._settings,
        response_model=TitleKeywordsLLM,
        system="System",
        messages=[{"role": "user", "content": "Document"}],
        max_tokens=123,
    )
    assert seen == expected_kwargs
