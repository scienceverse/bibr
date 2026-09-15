"""Account for source-backed bibliography loss without fabricating parsed entries."""

from collections import Counter

from bibr.paper_contents import ReferenceSourceLoss, ReferenceYieldLosses
from bibr.utils.text import collapse_ws


def reference_losses(
    source_rows: list[tuple[str, int | None]],
    raw_segments: list[str],
    retained_before_split: list[str],
    retained_segments: list[str],
    parse_unresolved: dict[int, str],
    *,
    parse_alignment_available: bool,
    selected_section_ids: tuple[int, ...] = (),
) -> ReferenceYieldLosses:
    """Offsets address the normalized reference source used by the segmenter.

    Unlocatable model-rewritten strings have no proven source span and are not
    copied into this evidence ledger. Unknown alignment is explicitly reported.
    """
    normalized = [collapse_ws(text) for text, _ in source_rows]
    source = "\n".join(normalized)
    row_spans = []
    cursor = 0
    for text, (_, text_id) in zip(normalized, source_rows, strict=True):
        row_spans.append((cursor, cursor + len(text), text_id))
        cursor += len(text) + 1
    losses: list[ReferenceSourceLoss] = []

    def add(stage: str, reason: str, start: int, stop: int) -> None:
        if not source[start:stop].strip():
            return
        losses.append(
            ReferenceSourceLoss(
                stage=stage,
                reason=reason,
                source_text=source[start:stop],
                source_span=(start, stop),
                source_text_ids=tuple(
                    dict.fromkeys(
                        text_id
                        for left, right, text_id in row_spans
                        if text_id is not None and left < stop and right > start
                    )
                ),
            )
        )

    # Locate each occurrence in source order, preserving equal citations at
    # distinct offsets. A rewritten segment makes gap accounting uncertain.
    raw_spans: list[tuple[int, int] | None] = []
    cursor = 0
    for segment in raw_segments:
        start = source.find(segment, cursor) if segment else -1
        if start < 0:
            raw_spans.append(None)
            continue
        stop = start + len(segment)
        raw_spans.append((start, stop))
        cursor = stop
    if all(span is not None for span in raw_spans):
        cursor = 0
        for span in raw_spans:
            assert span is not None  # noqa: S101 — established by the alignment guard
            start, stop = span
            add("segmentation", "unassigned_source_span", cursor, start)
            cursor = stop
        add("segmentation", "unassigned_source_span", cursor, len(source))

    keep = Counter(retained_before_split)
    filtered = 0
    for segment, span in zip(raw_segments, raw_spans, strict=True):
        if keep[segment]:
            keep[segment] -= 1
            continue
        filtered += 1
        if span is not None:
            add("filtering", "non_reference_segment", *span)

    cursor = 0
    unlocated_unresolved_count = 0
    source_alignment_available = all(span is not None for span in raw_spans)
    for index, segment in enumerate(retained_segments):
        start = source.find(segment, cursor) if segment else -1
        if start < 0:
            source_alignment_available = False
            unlocated_unresolved_count += int(index in parse_unresolved)
            continue
        stop = start + len(segment)
        if index in parse_unresolved:
            add("parsing", parse_unresolved[index], start, stop)
        cursor = stop
    return ReferenceYieldLosses(
        selected_source_row_count=sum(bool(text) for text in normalized),
        segmented_count=len(raw_segments),
        retained_segment_count=len(retained_segments),
        filtered_segment_count=filtered,
        parse_alignment_available=parse_alignment_available,
        source_alignment_available=source_alignment_available,
        unlocated_unresolved_count=unlocated_unresolved_count,
        selected_section_ids=selected_section_ids,
        unresolved=tuple(losses),
    )
