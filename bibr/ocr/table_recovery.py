"""Paddle incomplete-table recovery shared by every OCR transport.

A Paddle table generation cut short by its token budget exports a ragged or
unterminated OTSL grid. One retry of the same region at
``PADDLE_TABLE_RECOVERY_MAX_TOKENS`` usually completes it — but only when
that budget is larger than the first request's table cap (the resolved
profile's table budget, including an ``OCR_GENERATION_MAX_TOKENS``
override). Retrying at an equal or smaller budget can only reproduce a
shorter prefix of the same decode and replace the longer output the first
request kept, so in that case the first result is returned unchanged and no
second request is sent. This policy used to
live only in the serve HTTP backend, so local runs (the default
``paddle-vllm`` chain, ``paddle-http``, the managed MLX clients) exported
truncated tables with only a warning.

Both the local HTTP client (``PaddleHttpOcrClient``) and the serve backend
(``BibrServeOcrBackend``) implement the retry through
:func:`recover_paddle_table`: the transports differ (circuit breaker,
semaphores, payload shape), but the decision of *when* to retry and the
fallback of *what* to keep are one implementation. Only a generation the
provider cut short (``finish_reason == \"length\"``, or structural truncation
signals when the provider reports no finish reason) is retried — a
stop-terminated ragged or unterminated grid at temperature 0 reproduces
deterministically, as does a closed grid whose spans are malformed, an empty
result, or plain non-grid text (see ``otsl_looks_truncated``).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from bibr.ocr.otsl import check_otsl_completeness, otsl_looks_truncated
from bibr.ocr.profiles import PADDLE_TABLE_RECOVERY_MAX_TOKENS

if TYPE_CHECKING:
    from bibr.ocr.profiles import OcrProfile

logger = logging.getLogger(__name__)

#: Result type preserved through recovery (``str`` or the ``OcrText`` subclass).
T = TypeVar("T", bound=str)


async def recover_paddle_table(
    *,
    profile: OcrProfile,
    prompt: str,
    result: T,
    retry: Callable[[], Awaitable[T]],
) -> T:
    """Retry a truncated Paddle table once; otherwise return *result* as-is.

    Args:
        profile: The request profile that produced *result*.
        prompt: The original region prompt (used to resolve the OCR task).
        result: The first recognition result (``str`` or ``OcrText``).
        retry: Zero-argument async callable re-running the same region at
            ``PADDLE_TABLE_RECOVERY_MAX_TOKENS``.

    Returns:
        The recovered result when the retry yields non-empty output, else the
        original result. A failed retry never loses the first output. No retry
        is sent when the first request's table budget already meets or exceeds
        ``PADDLE_TABLE_RECOVERY_MAX_TOKENS`` — the retry could only return a
        shorter prefix of the same decode.
    """
    if profile.name != "paddle" or profile.task_for_prompt(prompt) != "table":
        return result
    completeness = check_otsl_completeness(result)
    finish_reason = getattr(result, "finish_reason", None)
    if not otsl_looks_truncated(completeness, finish_reason):
        return result
    first_budget = profile.request.max_tokens_for("table")
    if first_budget >= PADDLE_TABLE_RECOVERY_MAX_TOKENS:
        logger.debug(
            "Skipping Paddle table recovery: first request already used %d tokens "
            "(recovery budget %d; finish_reason=%s)",
            first_budget,
            PADDLE_TABLE_RECOVERY_MAX_TOKENS,
            finish_reason,
        )
        return result
    logger.warning(
        "Paddle table output incomplete; retrying once with output limit %d "
        "(finish_reason=%s, reasons=%s)",
        PADDLE_TABLE_RECOVERY_MAX_TOKENS,
        finish_reason,
        completeness.reasons,
    )
    try:
        recovered = await retry()
    except Exception:
        logger.warning(
            "Paddle table recovery failed; retaining initial output",
            exc_info=True,
        )
        return result
    if not recovered:
        logger.warning("Paddle table recovery returned empty output; retaining initial output")
        return result
    return recovered
