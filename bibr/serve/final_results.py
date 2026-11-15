"""Whether an export is a final answer for its input: what serve may cache.

Kept apart from ``bibr.serve.deployments.pipeline`` so it imports nothing from
the ``serve`` extra: the fast-mode and text-layer tests of a core install ask
it of their exports too.
"""

from __future__ import annotations

# A blocking issue that is a decision about the input, not a failure: the same
# input and code make it again, so the result is final.
FINAL_BLOCKING_CODES = frozenset({"VAL_METADATA_MULTI_ITEM"})


def is_final_result(payload: dict) -> bool:
    """Whether an export is a final answer for its input, safe to cache.

    Not when a failure a retry could avoid shaped it: a blocking validation
    issue other than a front-matter abstention (a failed title call,
    incomplete references), incomplete enrichment, or a warning in
    ``NOT_FINAL_CODES``. Caching such a result would replay one timeout or
    outage to every request for the TTL. A ``failed`` field state alone does
    not count: a deterministic failure such as ``REF_SEG_FAILED`` fails the
    same way on every run.
    """
    from bibr.processing_warnings import NOT_FINAL_CODES
    from bibr.validation import payload_validation

    validation = payload_validation(payload) or {}
    if any(
        isinstance(issue, dict)
        and issue.get("blocking")
        and issue.get("code") not in FINAL_BLOCKING_CODES
        for issue in validation.get("issues") or ()
    ):
        return False
    extraction = payload.get("extraction")
    if not isinstance(extraction, dict):
        return True
    enrichment = extraction.get("enrichment")
    if isinstance(enrichment, dict) and enrichment.get("complete") is False:
        return False
    return not any(
        isinstance(warning, dict) and warning.get("code") in NOT_FINAL_CODES
        for warning in extraction.get("warnings") or ()
    )
