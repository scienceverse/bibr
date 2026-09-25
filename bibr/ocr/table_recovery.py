"""Paddle incomplete-table recovery shared by every OCR transport.

A Paddle table generation cut short by its token budget exports a ragged or
unterminated OTSL grid. One retry of the same region at
``PADDLE_TABLE_RECOVERY_MAX_TOKENS`` usually completes it. This policy used to
live only in the serve HTTP backend, so local runs (the default
``paddle-vllm`` chain, ``paddle-http``, the managed MLX clients) exported
truncated tables with only a warning.

Both the local HTTP client (``PaddleHttpOcrClient``) and the serve backend
(``BibrServeOcrBackend``) implement the retry through
:func:`recover_paddle_table`: the transports differ (circuit breaker,
semaphores, payload shape), but the decision of *when* to retry and the
fallback of *what* to keep are one implementation. Only truncation is
retried — a closed grid whose spans are malformed, an empty result, or plain
non-grid text reproduces deterministically (see ``otsl_looks_truncated``).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from bibr.ocr.otsl import check_otsl_completeness, otsl_looks_truncated
from bibr.ocr.profiles import PADDLE_TABLE_RECOVERY_MAX_TOKENS

if TYPE_CHECKING:
    from bibr.ocr.profiles import OcrProfile

logger = logging.getLogger(__name__)


async def recover_paddle_table(
    *,
    profile: OcrProfile,
    prompt: str,
    result,
    retry: Callable[[], Awaitable],
):
    """Retry a truncated Paddle table once; otherwise return *result* as-is.

    Args:
        profile: The request profile that produced *result*.
        prompt: The original region prompt (used to resolve the OCR task).
        result: The first recognition result (``str`` or ``OcrText``).
        retry: Zero-argument async callable re-running the same region at
            ``PADDLE_TABLE_RECOVERY_MAX_TOKENS``.

    Returns:
        The recovered result when the retry yields non-empty output, else the
        original result. A failed retry never loses the first output.
    """
    if profile.name != "paddle" or profile.task_for_prompt(prompt) != "table":
        return result
    completeness = check_otsl_completeness(result)
    finish_reason = getattr(result, "finish_reason", None)
    if completeness.complete and finish_reason != "length":
        return result
    if not otsl_looks_truncated(completeness, finish_reason):
        return result
    logger.warning(
        "Paddle table output incomplete; retrying once at %d tokens (finish_reason=%s, reasons=%s)",
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
