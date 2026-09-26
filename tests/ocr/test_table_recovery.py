"""Paddle incomplete-table recovery shared by the local and serve transports.

The local ``PaddleHttpOcrClient`` (which also serves the managed vLLM,
MLX-VLM and strict Rapid-MLX clients) must retry a truncated table exactly
like the serve backend does — and neither may retry a deterministic
malformed-but-complete table.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from PIL import Image

from bibr.local.ocr import PaddleHttpOcrClient
from bibr.ocr.backend import OcrText
from bibr.ocr.profiles import (
    GLM_PROFILE,
    PADDLE_PROFILE,
    PADDLE_TABLE_RECOVERY_MAX_TOKENS,
    resolve_ocr_profile,
)
from bibr.ocr.table_recovery import recover_paddle_table

TRUNCATED = "<fcel>A<fcel>B<nl><fcel>C"
COMPLETE = "<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>"
# Closed grid (every row terminated, rectangular) whose spans are broken:
# deterministic output a retry would just reproduce.
MALFORMED_COMPLETE = "<fcel>A<fcel>B<nl><ucel><xcel><nl>"


def _text(content, finish_reason=None):
    return OcrText(content, finish_reason=finish_reason)


async def _never_retry():
    raise AssertionError("must not retry")


class TestRecoverPaddleTable:
    @pytest.mark.asyncio
    async def test_retries_length_truncation_and_returns_recovery(self):
        retry = AsyncMock(return_value=_text(COMPLETE, "stop"))
        out = await recover_paddle_table(
            profile=PADDLE_PROFILE,
            prompt=PADDLE_PROFILE.prompts["table"],
            result=_text(TRUNCATED, "length"),
            retry=retry,
        )
        assert out == COMPLETE
        retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_ragged_and_unterminated_tables_without_finish_reason(self):
        """Structural truncation signals retry only when the provider reports
        no finish reason. A ``stop``-terminated ragged grid at temperature 0
        reproduces identically (see below)."""
        for raw in ("<fcel>A<fcel>B<nl><fcel>C<nl>", "<fcel>A<fcel>B"):
            retry = AsyncMock(return_value=_text(COMPLETE, "stop"))
            out = await recover_paddle_table(
                profile=PADDLE_PROFILE,
                prompt=PADDLE_PROFILE.prompts["table"],
                result=_text(raw, None),
                retry=retry,
            )
            assert out == COMPLETE
            retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_retry_stop_terminated_ragged_or_unterminated_tables(self):
        """A generation that ended on EOS was not cut short: a retry at a
        higher budget is the same greedy decode and only burns a generation."""
        for raw in (
            "<fcel>A<fcel>B<nl><fcel>C<nl>",
            "<fcel>A<fcel>B",
            "<fcel>a<fcel>b<nl><fcel>c<nl><fcel>d<fcel>e<nl>",
            "<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>Note: n=12",
        ):
            retry = AsyncMock(return_value=_text(COMPLETE, "stop"))
            out = await recover_paddle_table(
                profile=PADDLE_PROFILE,
                prompt=PADDLE_PROFILE.prompts["table"],
                result=_text(raw, "stop"),
                retry=retry,
            )
            assert out == raw
            retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_retry_complete_table(self):
        out = await recover_paddle_table(
            profile=PADDLE_PROFILE,
            prompt=PADDLE_PROFILE.prompts["table"],
            result=_text(COMPLETE, "stop"),
            retry=_never_retry,
        )
        assert out == COMPLETE

    @pytest.mark.asyncio
    async def test_does_not_retry_malformed_but_complete_table(self):
        """Guard: deterministic structure errors must not cost a retry."""
        out = await recover_paddle_table(
            profile=PADDLE_PROFILE,
            prompt=PADDLE_PROFILE.prompts["table"],
            result=_text(MALFORMED_COMPLETE, "stop"),
            retry=_never_retry,
        )
        assert out == MALFORMED_COMPLETE

    @pytest.mark.asyncio
    async def test_does_not_retry_empty_or_non_grid_output(self):
        for raw in ("", "   ", "plain text"):
            out = await recover_paddle_table(
                profile=PADDLE_PROFILE,
                prompt=PADDLE_PROFILE.prompts["table"],
                result=_text(raw, "stop"),
                retry=_never_retry,
            )
            assert out == raw

    @pytest.mark.asyncio
    async def test_does_not_retry_non_table_or_non_paddle(self):
        out = await recover_paddle_table(
            profile=PADDLE_PROFILE,
            prompt=PADDLE_PROFILE.prompts["text"],
            result=_text("half a sentence", "length"),
            retry=_never_retry,
        )
        assert out == "half a sentence"
        out = await recover_paddle_table(
            profile=GLM_PROFILE,
            prompt=GLM_PROFILE.prompts["table"],
            result=_text("half a table", "length"),
            retry=_never_retry,
        )
        assert out == "half a table"

    @pytest.mark.asyncio
    async def test_retries_when_override_is_below_recovery_budget(self):
        """A smaller override (e.g. 2048) still leaves room for a larger retry."""
        profile = resolve_ocr_profile(
            explicit="paddle",
            backend="paddle-http",
            model="paddle-ocr-vl-1.6",
            max_tokens=2048,
        )
        assert profile.request.max_tokens_for("table") == 2048
        retry = AsyncMock(return_value=_text(COMPLETE, "stop"))
        out = await recover_paddle_table(
            profile=profile,
            prompt=profile.prompts["table"],
            result=_text(TRUNCATED, "length"),
            retry=retry,
        )
        assert out == COMPLETE
        retry.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cap", [8192, 16384])
    async def test_does_not_retry_when_first_table_budget_meets_recovery_budget(self, cap):
        """An OCR_GENERATION_MAX_TOKENS override at or above the recovery
        budget must not trigger a retry: re-generating at an equal-or-smaller
        budget can only reproduce a shorter prefix of the same decode and
        would replace the longer first output."""
        profile = resolve_ocr_profile(
            explicit="paddle",
            backend="paddle-http",
            model="paddle-ocr-vl-1.6",
            max_tokens=cap,
        )
        assert profile.request.max_tokens_for("table") == cap
        retry = AsyncMock(return_value=_text(COMPLETE, "stop"))
        out = await recover_paddle_table(
            profile=profile,
            prompt=profile.prompts["table"],
            result=_text(TRUNCATED, "length"),
            retry=retry,
        )
        assert out == TRUNCATED
        retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_retry_keeps_initial_output(self):
        async def boom():
            raise RuntimeError("server hiccup")

        out = await recover_paddle_table(
            profile=PADDLE_PROFILE,
            prompt=PADDLE_PROFILE.prompts["table"],
            result=_text(TRUNCATED, "length"),
            retry=boom,
        )
        assert out == TRUNCATED

    @pytest.mark.asyncio
    async def test_empty_retry_keeps_initial_output(self):
        retry = AsyncMock(return_value=_text("", "stop"))
        out = await recover_paddle_table(
            profile=PADDLE_PROFILE,
            prompt=PADDLE_PROFILE.prompts["table"],
            result=_text(TRUNCATED, "length"),
            retry=retry,
        )
        assert out == TRUNCATED


def _local_client(payloads, responses, profile=PADDLE_PROFILE):
    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        content, finish_reason = responses[len(payloads) - 1]
        body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
        return httpx.Response(200, content=json.dumps(body).encode())

    client = PaddleHttpOcrClient(base_url="http://localhost:1", profile=profile)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.asyncio
async def test_local_client_retries_truncated_table_at_8192():
    """Local runs get the same recovery serve has, through the shared helper."""
    payloads: list[dict] = []
    client = _local_client(payloads, [(TRUNCATED, "length"), (COMPLETE, "stop")])

    with patch("bibr.local.ocr.encode_region_for_ocr", return_value="aW1n"):
        out = await client.recognize(Image.new("RGB", (8, 8)), PADDLE_PROFILE.prompts["table"])

    assert out == COMPLETE
    assert [payload["max_tokens"] for payload in payloads] == [
        4096,
        PADDLE_TABLE_RECOVERY_MAX_TOKENS,
    ]
    assert PADDLE_TABLE_RECOVERY_MAX_TOKENS == 8192


@pytest.mark.asyncio
async def test_local_client_does_not_retry_stop_terminated_ragged_table():
    """A stop-terminated ragged grid reproduces identically — no 8192 retry."""
    payloads: list[dict] = []
    ragged_stop = "<fcel>A<fcel>B<nl><fcel>C<nl>"
    client = _local_client(payloads, [(ragged_stop, "stop"), (COMPLETE, "stop")])

    with patch("bibr.local.ocr.encode_region_for_ocr", return_value="aW1n"):
        out = await client.recognize(Image.new("RGB", (8, 8)), PADDLE_PROFILE.prompts["table"])

    assert out == ragged_stop
    assert [payload["max_tokens"] for payload in payloads] == [4096]


@pytest.mark.asyncio
async def test_local_client_retries_truncation_without_finish_reason():
    """Structural truncation without a reported finish reason still retries."""
    payloads: list[dict] = []
    ragged = "<fcel>A<fcel>B<nl><fcel>C<nl>"
    client = _local_client(payloads, [(ragged, None), (COMPLETE, "stop")])

    with patch("bibr.local.ocr.encode_region_for_ocr", return_value="aW1n"):
        out = await client.recognize(Image.new("RGB", (8, 8)), PADDLE_PROFILE.prompts["table"])

    assert out == COMPLETE
    assert [payload["max_tokens"] for payload in payloads] == [
        4096,
        PADDLE_TABLE_RECOVERY_MAX_TOKENS,
    ]


@pytest.mark.asyncio
async def test_local_client_does_not_retry_malformed_complete_table():
    payloads: list[dict] = []
    client = _local_client(payloads, [(MALFORMED_COMPLETE, "stop")])

    with patch("bibr.local.ocr.encode_region_for_ocr", return_value="aW1n"):
        out = await client.recognize(Image.new("RGB", (8, 8)), PADDLE_PROFILE.prompts["table"])

    assert out == MALFORMED_COMPLETE
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_local_client_does_not_retry_complete_table():
    payloads: list[dict] = []
    client = _local_client(payloads, [(COMPLETE, "stop")])

    with patch("bibr.local.ocr.encode_region_for_ocr", return_value="aW1n"):
        out = await client.recognize(Image.new("RGB", (8, 8)), PADDLE_PROFILE.prompts["table"])

    assert out == COMPLETE
    assert len(payloads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [8192, 16384])
async def test_local_client_skips_recovery_when_first_budget_meets_recovery_budget(
    cap,
):
    """Pin: with an OCR_GENERATION_MAX_TOKENS override at or above the
    recovery budget, a length-truncated table keeps its (longer) first output
    and costs exactly one request — no 8192 retry may replace or repeat it."""
    profile = resolve_ocr_profile(
        explicit="paddle",
        backend="paddle-http",
        model="paddle-ocr-vl-1.6",
        max_tokens=cap,
    )
    payloads: list[dict] = []
    client = _local_client(payloads, [(TRUNCATED, "length")], profile=profile)

    with patch("bibr.local.ocr.encode_region_for_ocr", return_value="aW1n"):
        out = await client.recognize(Image.new("RGB", (8, 8)), profile.prompts["table"])

    assert out == TRUNCATED
    assert [payload["max_tokens"] for payload in payloads] == [cap]
