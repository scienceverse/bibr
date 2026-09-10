"""Pure aggregator for the ``qualification_provenance`` export surface.

An external qualification runner reads ``prediction["qualification_provenance"]``
from bibr's result JSON to verify the deployed NuExtract native protocol. This
module folds the per-label usage counters and per-label protocol hashes that
``LLMClient`` already tracks into the single provenance object the gate expects.

No raw document or completion text enters this surface — only sha256 hashes
(computed upstream in the backends), counts, and identity strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fixed scan order for the first-present native-invalid category. Mirrors the
# ``native_invalid_<cat>`` counter names recorded by ``LLMClient``.
_NATIVE_INVALID_CATEGORIES = (
    "empty",
    "non_json",
    "truncated",
    "trailing_content",
    "non_object",
    "schema_invalid",
)


@dataclass(frozen=True)
class DeploymentIdentity:
    """Identity of the deployment that produced an extraction."""

    bibr_sha: str | None
    platform_sha: str | None
    model_id: str | None
    model_revision: str | None
    jinja_sha256: str | None
    structured_backend: str | None  # "nuextract-native" | "instructor"
    temperature: float | None
    thinking_mode: bool | None


def _sum_counter(usage_by_label: dict[str, dict[str, int]], name: str) -> int:
    return sum(int(bucket.get(name, 0) or 0) for bucket in usage_by_label.values())


def build_qualification_provenance(
    *,
    usage_by_label: dict[str, dict[str, int]],
    protocol_hashes_by_label: dict[str, dict[str, str]],
    identity: DeploymentIdentity,
) -> dict[str, Any] | None:
    """Fold usage + protocol hashes + identity into the provenance object.

    Returns ``None`` only when there were zero LLM tasks (both maps empty).
    """
    if not usage_by_label and not protocol_hashes_by_label:
        return None

    is_native = identity.structured_backend == "nuextract-native"

    native_invalid_total = _sum_counter(usage_by_label, "native_invalid_outputs")
    raw_native_valid = (native_invalid_total == 0) if is_native else None

    raw_native_category: str | None = None
    if is_native and native_invalid_total > 0:
        for category in _NATIVE_INVALID_CATEGORIES:
            if _sum_counter(usage_by_label, f"native_invalid_{category}") > 0:
                raw_native_category = category
                break

    fallback_attempts = _sum_counter(usage_by_label, "protocol_fallbacks")
    fallback_recovered = _sum_counter(usage_by_label, "protocol_fallbacks_recovered")
    fallback_attempted = fallback_attempts > 0
    if fallback_attempts == 0:
        fallback_outcome = "not_attempted"
    elif fallback_recovered >= fallback_attempts:
        fallback_outcome = "recovered"
    else:
        fallback_outcome = "failed"

    return {
        "bibr_sha": identity.bibr_sha,
        "platform_sha": identity.platform_sha,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "jinja_sha256": identity.jinja_sha256,
        "structured_backend": identity.structured_backend,
        "temperature": identity.temperature,
        "thinking_mode": identity.thinking_mode,
        "protocol_hashes": dict(protocol_hashes_by_label) if protocol_hashes_by_label else {},
        "raw_native_valid": raw_native_valid,
        "raw_native_category": raw_native_category,
        "fallback_attempted": fallback_attempted,
        "fallback_outcome": fallback_outcome,
        "physical_request_count": _sum_counter(usage_by_label, "attempts"),
        "logical_request_count": _sum_counter(usage_by_label, "logical_calls"),
    }
