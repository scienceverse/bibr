"""The ref-llm-sft@1 contract for a dedicated fine-tuned reference server.

One printed entry per request, using the training prompt with thinking off.
The server must expose the fine-tuned model (or its LoRA alias), not the base
Qwen checkpoint. This client never loads GPU libraries or changes the metadata
LLM. Failed entries retain their input position for the caller's NER recovery.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

import httpx

from bibr.config import GlobalSettings
from bibr.exceptions import ConfigurationError

FIELDS = (
    "authors",
    "year",
    "year_suffix",
    "title",
    "container",
    "volume",
    "issue",
    "first_page",
    "last_page",
    "editors",
    "publisher",
    "edition",
    "doi",
    "url",
)
SYSTEM_PROMPT = (
    "You parse one bibliographic reference into JSON. Copy every value verbatim "
    "from the reference as printed, keep its language and spelling, and use null "
    "for anything not printed. authors and editors are the full name lists as "
    "written. year is an integer; a letter right after it (2005a) goes in "
    "year_suffix. title is the cited work; container is the journal, book or "
    "proceedings it appeared in. Output only the JSON object."
)


def json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            key: {"type": ["integer" if key == "year" else "string", "null"]} for key in FIELDS
        },
        "required": list(FIELDS),
        "additionalProperties": False,
    }


def _fold(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold().replace("–", "-").replace("—", "-")


def validate_record(raw: str, reference: str) -> dict[str, Any] | None:
    """Reject incomplete shapes, invented fields and impossible page ranges.

    Verbatim checks establish source support, not field accuracy or calibrated
    confidence. The full model record stays together; no fieldwise voting.
    """
    try:
        record = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict) or set(record) != set(FIELDS):
        return None
    source = _fold(reference)
    for key, value in record.items():
        if value is None:
            continue
        if key == "year":
            if type(value) is not int or not 1500 <= value <= 2100:
                return None
            if not re.search(rf"(?<!\d){value}(?!\d)", reference):
                return None
        elif not isinstance(value, str) or not value.strip() or _fold(value) not in source:
            return None
    if not record["title"] and not record["authors"]:
        return None
    if record["doi"] and not re.fullmatch(r"10\.\d{4,9}/\S+", record["doi"]):
        return None
    first, last = record["first_page"], record["last_page"]
    if (
        first
        and last
        and first.isdigit()
        and last.isdigit()
        and len(last) >= len(first)
        and int(last) < int(first)
    ):
        return None
    return record


async def parse_batch(
    references: list[str],
    settings: GlobalSettings,
    *,
    usage_recorder: Callable[[dict], None] | None = None,
) -> list[dict[str, Any] | None]:
    """Bounded, aligned greedy requests; cancellation closes the HTTP client."""
    if not references:
        return []
    if not settings.REF_LLM_BASE_URL or not settings.REF_LLM_MODEL:
        raise ConfigurationError(
            "ref-llm requires REF_LLM_BASE_URL and REF_LLM_MODEL pointing to the "
            "fine-tuned reference model or LoRA alias"
        )
    url = settings.REF_LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {}
    if settings.REF_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.REF_LLM_API_KEY}"
    aligned: list[dict[str, Any] | None] = [None] * len(references)
    gate = asyncio.Semaphore(settings.REF_LLM_MAX_CONCURRENCY)
    async with httpx.AsyncClient(
        headers=headers,
        timeout=settings.REF_LLM_TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as client:

        async def parse_one(index: int, reference: str) -> None:
            # Decline rather than silently truncate the printed source.
            if len(reference) > settings.REF_LLM_MAX_INPUT_CHARS or not reference.strip():
                return
            async with gate:
                try:
                    async with asyncio.timeout(settings.REF_LLM_TIMEOUT_SECONDS):
                        response = await client.post(
                            url,
                            json={
                                "model": settings.REF_LLM_MODEL,
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {
                                        "role": "user",
                                        "content": re.sub(r"\s+", " ", reference).strip(),
                                    },
                                ],
                                "temperature": 0.0,
                                "max_tokens": settings.REF_LLM_MAX_TOKENS,
                                "chat_template_kwargs": {"enable_thinking": False},
                                "response_format": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "reference",
                                        "strict": True,
                                        "schema": json_schema(),
                                    },
                                },
                            },
                        )
                        response.raise_for_status()
                        payload = response.json()
                        if not isinstance(payload, dict):
                            return
                        if usage_recorder is not None and isinstance(payload.get("usage"), dict):
                            usage_recorder(payload["usage"])
                        choices = payload.get("choices")
                        if (
                            not isinstance(choices, list)
                            or len(choices) != 1
                            or not isinstance(choices[0], dict)
                            or choices[0].get("finish_reason") != "stop"
                            or not isinstance(choices[0].get("message"), dict)
                        ):
                            return
                        aligned[index] = validate_record(
                            choices[0]["message"]["content"],
                            reference,
                        )
                except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError):
                    # The caller reports fallback counts without echoing server
                    # response bodies, reference text, endpoint URLs or secrets.
                    return

        try:
            async with asyncio.timeout(settings.REF_LLM_TOTAL_TIMEOUT_SECONDS):
                await asyncio.gather(*(parse_one(i, ref) for i, ref in enumerate(references)))
        except TimeoutError:
            # Retain completed slots; asyncio cancelled and awaited unfinished
            # requests before leaving the shared client context.
            pass
    return aligned
