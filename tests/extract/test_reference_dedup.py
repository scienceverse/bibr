"""Exact, provenance-aware reference deduplication.

Compact fixtures are transcribed from the mounted 2026-07-21 audit.  In
particular, ord227 contains a replayed 93-reference list, while ord70 contains
three genuinely distinct EU-regulation citations that must remain separate.
"""

from bibr.extract import ref_extractor

_prepare_segmented_references = getattr(ref_extractor, "_prepare_segmented_references", None)


def _spans(text: str, segments: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for segment in segments:
        start = text.index(segment, cursor)
        end = start + len(segment)
        spans.append((start, end))
        cursor = end
    return spans


def test_ord227_exact_list_replay_is_suppressed_without_title_matching():
    assert callable(_prepare_segmented_references)
    refs = [
        "1. Mitzkat A, Berger S, Reeves S, Mahler C. More terminological clarity. 2016.",
        "2. Walkenhorst U, Mahler C, Aistleithner R. Position statement GMA. 2015.",
        "3. Kent F, Keating JL. Interprofessional education in primary health care. 2015.",
    ]
    text = "\n".join(refs)
    source_spans = _spans(text, refs)

    kept, kept_spans, duplicate_rate, reasons = _prepare_segmented_references(
        text, [*source_spans, *source_spans]
    )

    assert kept == refs
    assert kept_spans == tuple(source_spans)
    assert duplicate_rate == 0.5
    assert reasons == ("duplicate_source_span",)


def test_ord70_similar_regulation_occurrences_at_distinct_offsets_survive():
    assert callable(_prepare_segmented_references)
    refs = [
        "2. REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT. Official Journal L 119/1.",
        "6. REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT. Art. 5. Official Journal L 119/1.",
        "7. REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT. Art. 4. Official Journal L 119/1.",
    ]
    text = "\n".join(refs)

    kept, kept_spans, duplicate_rate, reasons = _prepare_segmented_references(
        text, _spans(text, refs)
    )

    assert kept == refs
    assert kept_spans == tuple(_spans(text, refs))
    assert duplicate_rate == 0.0
    assert reasons == ()


def test_equal_surface_at_two_offsets_is_not_enough_to_deduplicate():
    assert callable(_prepare_segmented_references)
    ref = "World Health Organization. (2018). Global report. WHO Press."
    text = f"{ref}\n{ref}"

    kept, kept_spans, duplicate_rate, reasons = _prepare_segmented_references(
        text, _spans(text, [ref, ref])
    )

    assert kept == [ref, ref]
    assert len(kept_spans) == 2
    assert duplicate_rate == 0.0
    assert reasons == ()


def test_equal_list_run_at_distinct_offsets_is_not_enough_to_deduplicate():
    assert callable(_prepare_segmented_references)
    refs = [
        "1. Smith J. First exact source reference. 2020.",
        "2. Doe A. Second exact source reference. 2021.",
        "3. Roe B. Third exact source reference. 2022.",
    ]
    repeated = [*refs, *refs]
    text = "\n".join(repeated)

    kept, kept_spans, duplicate_rate, reasons = _prepare_segmented_references(
        text, _spans(text, repeated)
    )

    assert kept == repeated
    assert len(kept_spans) == 6
    assert duplicate_rate == 0.0
    assert reasons == ()


def test_same_source_span_is_deduplicated_after_unicode_and_marker_normalization():
    assert callable(_prepare_segmented_references)
    ref = "[1]\u00a0Smith, J. (2020). A study. Journal of Tests, 1, 1–3."

    kept, kept_spans, duplicate_rate, reasons = _prepare_segmented_references(
        ref, [(0, len(ref)), (0, len(ref))]
    )

    assert kept == [ref]
    assert kept_spans == ((0, len(ref)),)
    assert duplicate_rate == 0.5
    assert "duplicate_source_span" in reasons
