"""Aggregate per-label LLM usage buckets into the v11 ``extraction.usage`` block.

``totals`` is a pure aggregation of ``breakdown`` — denormalized deliberately,
because re-summing is the 90% read path. A test enforces the invariant.

Input is keyed by the ``(label, provider, model)`` triple, not by label alone
— a label (e.g. ``parse_refs``) can run under more than one engine within a
single file, and each engine's counters must land in its own row rather than
blending into a bucket stamped with whichever provider created it first. This
mirrors ``LLMClient.usage_labels_pop_file()`` / ``_labels_by_file``, which
partition the same way.
"""

from __future__ import annotations

_COUNTERS = ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")


def build_usage_export(
    labels: dict[tuple[str, str | None, str | None], dict] | None,
) -> dict | None:
    """Return ``{"totals": ..., "breakdown": [...]}``; ``None`` when nothing tracked.

    ``labels`` keys are ``(label, provider, model)`` triples; values are the
    numeric usage buckets (extra keys beyond ``_COUNTERS`` are dropped).
    """
    if not labels:
        return None

    # Sort key only — ``None`` provider/model (permitted by the type, not
    # produced today) must not hit `str < NoneType` in a raw tuple sort.
    # The sentinel never leaks into the emitted row: ``provider``/``model``
    # there still come from the unmodified key, so ``None`` serializes as
    # ``null``, not ``""``.
    def _sort_key(key: tuple[str, str | None, str | None]) -> tuple[str, str, str]:
        label, provider, model = key
        return (label, provider or "", model or "")

    breakdown = []
    for key in sorted(labels, key=_sort_key):
        label, provider, model = key
        bucket = labels[key] or {}
        row = {
            "label": label,
            "provider": provider,
            "model": model,
        }
        row.update({name: int(bucket.get(name, 0) or 0) for name in _COUNTERS})
        breakdown.append(row)

    totals = {name: sum(row[name] for row in breakdown) for name in _COUNTERS}
    return {"totals": totals, "breakdown": breakdown}
