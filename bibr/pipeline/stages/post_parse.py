"""PostParseStage — concurrent extractor invocation across files."""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import time
import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING

from bibr.exceptions import LlmCallError, ProcessingError
from bibr.field_states import FieldScope, set_field_source
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.utils.text import NAME_CHAR_CLS

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.extract.front_matter import FrontMatterResolution
    from bibr.pipeline.classifier_resources import ClassifierResources
    from bibr.pipeline.context import PipelineContext
    from bibr.pipeline.identity import ExpectedIdentity
    from bibr.validation import ValidationIssue

logger = logging.getLogger(__name__)

# In-text author-year citation marker: a capitalised surname, optional
# co-authors / "et al.", then a 4-digit year. Used only to size the body's
# distinct citation count for the reference-undercount safety net (residual #3);
# numeric "[1]" citation styles yield zero here and are sized from the citation
# linker's receipt instead (``_count_numbered_citations``).
_INTEXT_CITE_RE = re.compile(
    r"([A-Z]" + NAME_CHAR_CLS + r"+)"
    r"(?:[,\s]+(?:et\s+al\.?|&|and|[A-Z]" + NAME_CHAR_CLS + r"+))*"
    r"[,\s]*\(?\s*((?:1[6-9]|20)\d{2})[a-z]?\)?"
)

# Only judge when the body cites enough distinct works, and only warn on a
# gross deficit — conservative so a dropped lead reference (off-by-one, which
# leaves no output trace) never trips it and short papers are never flagged.
_MIN_BODY_CITES_FOR_WARNING = 15
_REF_DEFICIT_RATIO = 0.5

# A numbered bibliography is cited almost densely from 1 upward. Cited numbers
# only count up to the highest n that has at least this share of 1..n cited, so
# stray bracketed integers far above the list (values, intervals) cannot
# inflate the count.
_NUMBERED_CITE_MIN_DENSITY = 0.5

_RESULTS_CHILD_CUE_RE = re.compile(
    r"\b(?:results?|findings?|outcomes?|views?|perspectives?|experiences?|themes?|engagement)\b",
    re.IGNORECASE,
)


def _reconcile_result_subsection_types(sections) -> None:
    """Repair weak METHOD predictions for result-oriented child headings.

    The trained classifier can over-weight procedural vocabulary in a body
    snippet even when layout hierarchy and the heading identify a subsection
    of Results (for example, ``Professional view on engagement``). Only weak
    model predictions are changed; aliases, LLM decisions, and strongly scored
    method subsections remain untouched.
    """
    from bibr.paper_contents import CanonicalSection

    by_id = {section.section_id: section for section in sections}
    for section in sections:
        parent = by_id.get(section.parent_section_id)
        if (
            parent is not None
            and parent.section_type == CanonicalSection.RESULTS
            and section.section_type == CanonicalSection.METHODS
            and section.classification_source == "model"
            and section.classification_score < 0.8
            and _RESULTS_CHILD_CUE_RE.search(section.header)
        ):
            section.section_type = CanonicalSection.RESULTS
            section.classification_score = max(section.classification_score, 0.8)
            section.classification_source = "parent_context"


def _inherit_unknown_child_section_types(sections) -> None:
    """Fill UNKNOWN child section types from their body-section parent.

    This is a conservative fallback for subsections that are structurally
    inside Methods/Results/etc. but whose own heading is too specific for the
    classifier. Explicit child classifications are preserved.
    """
    from bibr.paper_contents import CanonicalSection
    from bibr.structure.section_tree import INTERLUDE_TYPES

    inheritable = {
        CanonicalSection.INTRODUCTION,
        CanonicalSection.METHODS,
        CanonicalSection.RESULTS,
        CanonicalSection.DISCUSSION,
    }
    by_id = {section.section_id: section for section in sections}
    pos_by_id = {section.section_id: idx for idx, section in enumerate(sections)}
    for section in sections:
        if section.section_type not in (CanonicalSection.UNKNOWN, None):
            continue
        parent = by_id.get(section.parent_section_id)
        if parent is None or parent.section_type not in inheritable:
            continue
        parent_pos = pos_by_id.get(parent.section_id)
        child_pos = pos_by_id.get(section.section_id)
        if parent_pos is None or child_pos is None or parent_pos >= child_pos:
            continue
        if any(
            intervening.section_type in INTERLUDE_TYPES
            for intervening in sections[parent_pos + 1 : child_pos]
        ):
            continue
        section.section_type = parent.section_type
        parent_score = float(parent.classification_score or 0.7)
        section.classification_score = max(
            float(section.classification_score or 0.0),
            min(parent_score, 0.75),
        )
        section.classification_source = "parent_context"


def _count_distinct_intext_citations(text: str) -> int:
    """Count distinct (surname, year) author-year citations in *text*."""
    if not text:
        return 0
    return len({(m.group(1).lower(), m.group(2)) for m in _INTEXT_CITE_RE.finditer(text)})


def _count_numbered_citations(numbers: set[int]) -> int:
    """Count the distinct cited reference numbers inside the dense run from 1.

    With the numbers sorted, the i-th smallest one ``n`` has exactly ``i``
    cited numbers at or below it; the run extends to the largest ``n`` where
    that is at least ``_NUMBERED_CITE_MIN_DENSITY`` of ``1..n``.
    """
    count = 0
    for i, number in enumerate(sorted(numbers), start=1):
        if i >= _NUMBERED_CITE_MIN_DENSITY * number:
            count = i
    return count


def _low_reference_count_warning(
    body_text: str, n_refs: int, *, cited_numbers: set[int] | None = None
) -> ProcessingWarning | None:
    """Return a warning when extracted references are grossly fewer than the
    body's distinct in-text citations (likely OCR reference-region omission),
    else None. General gross-drop net — not a single-dropped-reference detector.

    Author-year citations are counted from *body_text*; numeric ones from
    *cited_numbers*, the printed reference numbers the body's markers cite
    (``citation_linker.cited_reference_numbers``). The larger count decides.
    """
    n_author_year = _count_distinct_intext_citations(body_text)
    n_numbered = _count_numbered_citations(cited_numbers or set())
    n_cited = max(n_author_year, n_numbered)
    if n_cited >= _MIN_BODY_CITES_FOR_WARNING and n_refs < _REF_DEFICIT_RATIO * n_cited:
        kind = "numbered in-text citations" if n_numbered > n_author_year else "in-text citations"
        return ProcessingWarning(
            WarningCode.REF_UNDER_EXTRACTION_SUSPECTED,
            f"{n_refs} references parsed vs {n_cited} distinct {kind} in the body — some "
            f"reference entries may have been dropped (OCR region omission).",
        )
    return None


def _attach_text_quality(paper, contents, settings: GlobalSettings | None = None) -> None:
    """Compute and attach the report-only parse-quality score (never alters
    extracted data).

    Text units are the layout regions preserved on ``contents.region_summaries``
    — they carry the per-region label (mapped via ``PDFParser.LABEL_TREATMENT``)
    and source page the donor's per-cell rating needs. ``RegionSummary.content``
    is the parser's 200-char region preview, which is ample for the pervasive
    garbage patterns rated (mojibake, token loops, symbol soup, spaced letters).
    Empty scoreable regions are kept (not filtered here): the scorer treats an
    empty prose region as an OCR-coverage failure (0.0), so a page whose text
    regions came back blank is reflected in the score. DOCX-native input
    populates no region summaries, so its score stays ``None`` (there is no OCR
    garbage to rate). Sets ``paper.text_quality`` and, below
    ``Settings.pipeline.text_quality_warn_threshold``, appends a
    ``LOW_TEXT_QUALITY`` processing warning.
    """
    from bibr.config import snapshot_settings
    from bibr.structure.text_quality import paper_text_quality

    effective = settings if settings is not None else snapshot_settings()

    # Empty regions are intentionally NOT filtered out here: an empty scoreable
    # region (content/heading/footnote/section-hint) is an OCR-coverage failure
    # the scorer rates 0.0. Non-scoreable empties are ignored inside
    # paper_text_quality via the treatment filter, so this stays targeted.
    regions = (
        (rs.page, rs.label, rs.content)
        for rs in (getattr(contents, "region_summaries", None) or [])
    )
    report = paper_text_quality(regions)
    if report is None:
        return
    paper.text_quality = report.score
    threshold = effective.pipeline.text_quality_warn_threshold
    if report.score < threshold:
        warning = ProcessingWarning(
            WarningCode.LOW_TEXT_QUALITY,
            f"text-quality score {report.score:.2f} is below {threshold:g}",
        )
        logger.warning("Low text quality: %s", warning.message)
        paper.processing_warnings.append(warning)


async def _classify_sections(
    contents,
    layout_hints,
    no_llm: bool,
    llm_client,
    *,
    classifier_resources: ClassifierResources | None = None,
    settings: GlobalSettings | None = None,
) -> None:
    """Section classification (lookup + LLM fallback), title tagging,
    layout-hint overrides, and scoped hierarchy assignment. Mutates
    ``contents``. Study-scope ids are transient locals — never serialized."""
    from bibr.paper_contents import CanonicalSection
    from bibr.structure.section_classifier import (
        classify_headers_batch,
        classify_headers_batch_async,
    )
    from bibr.structure.section_tree import (
        assign_hierarchy_from_top_level,
        assign_provisional_scopes,
        close_scopes,
        detect_study_markers,
    )

    # Study-marker detection runs on raw headers BEFORE classification; the
    # regex is the sole authority on scope boundaries.
    markers = detect_study_markers(contents.sections)
    provisional_scopes = assign_provisional_scopes(contents.sections, markers)

    # The detected paper title is excluded from the LLM-driven header
    # classification (it's a unique string the LLM has no chance to
    # canonicalize) and assigned section_type=TITLE directly below.
    classifiable = [
        s
        for s in contents.sections
        if s.level > 0 and not (contents.detected_title and s.header == contents.detected_title)
    ]
    if classifiable:
        headers = [s.header for s in classifiable]
        if no_llm:
            # Sync lookup-only path returns 2-tuples (no model, no LLM). The
            # source is derived from the score: 1.0 = exact alias table hit,
            # 0.95 = word-boundary substring hit, 0.0 = miss (UNKNOWN).
            lookup_classifications = classify_headers_batch(headers)
            classifications: list[tuple[CanonicalSection, float, bool | None, str | None]] = [
                (
                    canon,
                    score,
                    None,
                    ("exact_alias" if score >= 1.0 else "substring_alias") if score > 0 else None,
                )
                for canon, score in lookup_classifications
            ]
        else:
            body_snippets = [contents.sections_text.get(s.section_id, "") for s in classifiable]
            scope_context: list[str | None] | None = None
            if markers:
                # Label each header with the raw text of the marker that
                # opened its provisional scope (None outside any scope).
                label_by_scope: dict[int, str] = {}
                for sec in contents.sections:
                    info = markers.get(sec.section_id)
                    if info is not None:
                        label_by_scope.setdefault(
                            provisional_scopes.get(sec.section_id, 0), info.header
                        )
                scope_context = [
                    label_by_scope.get(provisional_scopes.get(s.section_id, 0))
                    for s in classifiable
                ]
            classifier_kwargs = {}
            if classifier_resources is not None:
                classifier_kwargs["classifier_resources"] = classifier_resources
            if settings is not None:
                classifier_kwargs["settings"] = settings
            classifications = await classify_headers_batch_async(
                headers,
                body_snippets=body_snippets,
                llm_client=llm_client,
                scope_context=scope_context,
                degradation_warnings=contents.processing_warnings,
                **classifier_kwargs,
            )
        for section, (canon, score, is_top, source) in zip(
            classifiable, classifications, strict=True
        ):
            section.section_type = canon
            section.classification_score = score
            section.classification_source = source
            # The trained classifier emits a per-section is_top_level
            # prediction. Route it onto the section so the hierarchy
            # builder (assign_hierarchy_from_top_level) can consume it.
            # ``None`` for alias-lookup hits / LLM-path / no_llm — leaves
            # the type-based positional rule in charge.
            if is_top is not None:
                section.is_top_level_predicted = is_top

        # Compound markers ("Study 2: Methods"): the post-marker remainder's
        # alias type is deterministic — it overrides the classifier.
        for section in classifiable:
            info = markers.get(section.section_id)
            if info is not None and info.remainder_type is not None:
                section.section_type = info.remainder_type
                section.classification_score = 0.95
                section.classification_source = "exact_alias"

    # Tag the title section directly — it was excluded from classification
    # because its header text is unique to the paper.
    if contents.detected_title:
        for section in contents.sections:
            if section.level > 0 and section.header == contents.detected_title:
                section.section_type = CanonicalSection.TITLE
                section.classification_score = 1.0
                section.classification_source = "title"
                break

    _reconcile_result_subsection_types(contents.sections)

    if layout_hints:
        contents.layout_hints = layout_hints

    # Scope closing needs the final section types: General Discussion and
    # back matter revert themselves and all followers to scope 0.
    scope_ids = close_scopes(contents.sections, provisional_scopes, markers)

    # Section hierarchy: assign levels + parents using the alias-driven
    # `is_top_level_predicted` signal (when set) plus a positional rule
    # that folds UNKNOWN sections into the most-recent IMRaD anchor —
    # both per study scope, so Study 2's Methods doesn't fold under
    # Study 1's. Numbered headings keep the level/parent inferred from
    # their numbering prefix in `pdf_parser._handle_heading`.
    #
    # Runs for DOCX too: it intentionally canonicalizes the native Word
    # hierarchy that `docx_native` builds, flattening unnumbered headings to
    # the same IMRaD 1-2 shape as the PDF path (output uniformity by design —
    # see specs/2026-06-15-docx-hierarchy-flattening-issue.md).
    assign_hierarchy_from_top_level(contents.sections, scope_ids=scope_ids, marker_ids=set(markers))
    _inherit_unknown_child_section_types(contents.sections)


async def _normalize_section_structure(
    contents,
    no_llm: bool,
    llm_client,
    file_hash: str,
    *,
    settings: GlobalSettings,
) -> None:
    """Detect implicit Abstract/Introduction boundaries (LLM, gated on
    availability) then enforce IMRaD ordering and section sanity.

    Implicit detection classifies front-matter text to recover an Abstract or
    Introduction the layout model missed, falling back to the positional
    page-1 heuristic when the LLM is disabled or fails. The two ``enforce_*``
    passes run unconditionally (pure, no LLM). Mutates ``contents``.
    """
    from bibr.paper import enforce_imrad_order, enforce_section_sanity

    if not no_llm:
        from bibr.structure.implicit_sections import detect_implicit_sections

        await detect_implicit_sections(
            contents,
            file_hash=file_hash,
            llm_client=llm_client,
            settings=settings,
        )

    enforce_imrad_order(contents.sections)
    enforce_section_sanity(contents.sections)


def _attach_front_matter_resolution(
    contents,
    expected_identity: ExpectedIdentity | None = None,
    *,
    metadata_llm_active: bool = False,
    settings: GlobalSettings | None = None,
):
    """Build record ownership before normalization mutates section boundaries."""

    from bibr.extract.front_matter import resolve_front_matter

    target_required = bool(
        expected_identity is not None
        and contents.preparsed_metadata is None
        and (
            expected_identity.doi_required
            or expected_identity.expected_doi is not None
            or expected_identity.expected_doi_sha256 is not None
            or expected_identity.expected_title is not None
            or expected_identity.target_block_hint is not None
        )
    ) or bool(metadata_llm_active and contents.preparsed_metadata is None)
    resolution, issues = resolve_front_matter(
        contents,
        expected_identity=expected_identity,
        target_required=target_required,
        settings=settings,
    )
    contents.front_matter_resolution = resolution
    return issues


def _notify_references(listener: Callable[[list], None] | None, references: list) -> None:
    """Hand parsed references to the optional listener; its failure never propagates."""
    if listener is None or not references:
        return
    try:
        listener(references)
    except Exception:  # noqa: BLE001 — a listener must never break extraction
        logger.warning("on_references_ready callback failed", exc_info=True)


async def _resolve_preparsed_references(
    contents,
    paper_metadata,
    file_hash: str,
    llm_client,
    ref_seg_strategy: str | None,
    ref_parse_strategy: str | None,
    *,
    settings: GlobalSettings,
    on_references_ready: Callable[[list], None] | None = None,
):
    """Fill references onto a natively-preparsed (JATS) ``PaperMetadata``.

    Structured element-citations arrive already parsed on
    ``contents.native_references`` and are assigned directly (no extractor).
    Otherwise the normal ``ReferenceExtractor`` runs — its ``native``
    segmentation branch consumes ``contents.native_ref_strings`` so LLM/geom
    segmentation is skipped, then parses via the configured strategy. Mutates
    and returns *paper_metadata*.

    ``ref_seg_strategy`` / ``ref_parse_strategy`` are the already-resolved
    strategy names (``post_parse`` resolves once at the top); they are passed
    verbatim to the extractor, which normalizes them idempotently.
    """
    # ``--refs off`` means no references in the output (and no downstream
    # Crossref enrichment), even when the JATS ref-list came pre-parsed for
    # free — checked before the native assignment so the semantics match the
    # PDF/DOCX path.
    if ref_parse_strategy == "off":
        from bibr.validation import clear_reference_state

        clear_reference_state(paper_metadata)
        return paper_metadata

    if contents.native_references is not None:
        paper_metadata.references = contents.native_references
        _notify_references(on_references_ready, paper_metadata.references)
        return paper_metadata

    from bibr.extract.extractor import MetadataExtractor

    extractor = MetadataExtractor(
        contents,
        file_hash=file_hash,
        llm_client=llm_client,
        ref_seg_strategy=ref_seg_strategy,
        ref_parse_strategy=ref_parse_strategy,
        settings=settings,
    )
    try:
        ref_df = extractor._collect_reference_rows()
    except ValueError as e:
        logger.warning(f"Reference section not found: {e}")
        contents.processing_warnings.append(
            ProcessingWarning(
                WarningCode.REF_SECTION_NOT_FOUND, f"{e}; the reference list is empty"
            )
        )
        return paper_metadata
    except ProcessingError:
        raise
    except Exception as e:  # noqa: BLE001 — core metadata already exists
        from bibr.validation import mark_references_incomplete

        mark_references_incomplete(paper_metadata, e)
        logger.error(
            "Native reference row collection failed after core metadata succeeded: %s",
            paper_metadata._references_incomplete_diagnostic,
        )
        return paper_metadata
    if ref_df is not None and not ref_df.empty:
        try:
            paper_metadata.references = await extractor._extract_references(ref_df)
            _notify_references(on_references_ready, paper_metadata.references)
        except asyncio.CancelledError:
            raise
        except ProcessingError:
            raise
        except Exception as e:  # noqa: BLE001 — refs are best-effort, never fatal
            from bibr.validation import mark_references_incomplete

            mark_references_incomplete(paper_metadata, e)
            logger.error(
                "Native reference extraction failed after core metadata succeeded: %s",
                paper_metadata._references_incomplete_diagnostic,
            )
    return paper_metadata


async def _extract_metadata_and_equations(
    contents,
    file_hash: str,
    no_llm: bool,
    llm_client,
    extract_equations: bool = True,
    ref_seg_strategy: str | None = None,
    ref_parse_strategy: str | None = None,
    *,
    settings: GlobalSettings | None = None,
    classifier_resources: ClassifierResources | None = None,
    front_matter_resolution: FrontMatterResolution | None = None,
    validation_issue_sink: list[ValidationIssue] | None = None,
    on_references_ready: Callable[[list], None] | None = None,
):
    """Phase 1: metadata + equation extraction in parallel.

    Returns the extracted ``PaperMetadata``; equation results are written to
    ``contents.equations``.

    When the input was parsed natively (JATS XML), the front matter is already
    on ``contents.preparsed_metadata`` — that becomes the base and the core LLM
    extraction (title/authors/classification) is skipped. References still run
    (directly from structured element-citations, or via the extractor's
    ``native`` segmentation branch). This holds under ``no_llm`` too: the
    preparsed metadata is entirely LLM-free.

    ``ref_seg_strategy`` / ``ref_parse_strategy`` are the already-resolved
    strategy names when called from ``post_parse`` (resolved once at the top);
    direct callers may pass raw names or ``None`` — only the ``== "off"`` test
    matters here, and both ``None`` and ``"ner"`` read as "not off".
    """
    from bibr.config import snapshot_settings
    from bibr.extract.extractor import MetadataExtractor
    from bibr.models import PaperMetadata

    effective_settings = settings if settings is not None else snapshot_settings()

    preparsed = contents.preparsed_metadata
    extractor = None

    if no_llm:
        contents.equations = []
        if preparsed is not None:
            if ref_parse_strategy == "off":
                from bibr.validation import clear_reference_state

                clear_reference_state(preparsed)
            elif contents.native_references is not None:
                preparsed.references = contents.native_references
            return preparsed
        return PaperMetadata(doi="", title="")

    if preparsed is not None:
        meta_coro = _resolve_preparsed_references(
            contents,
            preparsed,
            file_hash,
            llm_client,
            ref_seg_strategy,
            ref_parse_strategy,
            settings=effective_settings,
            on_references_ready=on_references_ready,
        )
    else:
        extractor = MetadataExtractor(
            contents,
            file_hash=file_hash,
            llm_client=llm_client,
            ref_seg_strategy=ref_seg_strategy,
            ref_parse_strategy=ref_parse_strategy,
            settings=effective_settings,
            classifier_resources=classifier_resources,
            front_matter_resolution=front_matter_resolution,
        )
        # Pass the listener only when one is set so extractor doubles that take no
        # kwargs (and every non-prefetching path) see the unchanged call.
        extract_kwargs = {"on_references_ready": on_references_ready} if on_references_ready else {}
        meta_coro = extractor.extract_all_metadata(**extract_kwargs)

    eq_coro = None
    regex_equations: list = []
    if extract_equations and effective_settings.EQUATION_EXTRACTION:
        from bibr.extract.equation_extractor import EquationExtractor

        eq_extractor = EquationExtractor()
        # The regex pass is synchronous and always completes; only the LLM
        # fan-out can exceed the budget. Running it inside the wait_for meant a
        # timeout destroyed equations that had already been extracted, and the
        # export shipped "eq": [] with nothing in the log. Threaded because
        # this runs on the shared serve event loop.
        regex_equations = await asyncio.to_thread(
            eq_extractor.extract_from_sentences, contents.sentences, contents.sections
        )
        eq_coro = asyncio.wait_for(
            eq_extractor.extract_with_llm_fallback(
                contents.sentences,
                contents.sections,
                llm_client,
                min_regex_stats=effective_settings.EQUATION_LLM_FALLBACK_MIN_REGEX_STATS,
                regex_equations=regex_equations,
            ),
            timeout=float(effective_settings.EQUATION_EXTRACTION_TIMEOUT_SECONDS),
        )

    if eq_coro is None:
        metadata = await meta_coro
        if validation_issue_sink is not None and extractor is not None:
            validation_issue_sink.extend(extractor.validation_issues)
        return metadata

    results = await asyncio.gather(meta_coro, eq_coro, return_exceptions=True)
    for result in results:
        if isinstance(result, ProcessingError):
            raise result
    if isinstance(results[0], BaseException):
        raise results[0]
    if isinstance(results[1], BaseException):
        timed_out = isinstance(results[1], (TimeoutError, asyncio.TimeoutError))
        if timed_out:
            logger.warning(
                "Equation LLM fallback timed out after %ss; keeping %d regex equation(s)",
                effective_settings.EQUATION_EXTRACTION_TIMEOUT_SECONDS,
                len(regex_equations),
            )
            contents.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.EQUATION_LLM_FALLBACK_TIMEOUT,
                    "timed out after "
                    f"{effective_settings.EQUATION_EXTRACTION_TIMEOUT_SECONDS}s; kept regex-only "
                    "equation extraction",
                )
            )
        else:
            logger.warning("Equation extraction failed: %s", results[1])
            contents.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.EQUATION_LLM_FALLBACK_FAILED,
                    f"{type(results[1]).__name__}; kept regex-only equation extraction",
                )
            )
        # The regex pass ran to completion before the fan-out started — ship it.
        contents.equations = regex_equations
    else:
        contents.equations = results[1]
    failed_batches = eq_extractor.llm_batch_failures
    if failed_batches:
        # Each failed batch kept only its regex equations; say how many and why.
        contents.processing_warnings.append(
            ProcessingWarning(
                WarningCode.EQUATION_LLM_FALLBACK_FAILED,
                f"{len(failed_batches)} of {eq_extractor.llm_batch_count} LLM batch(es) failed "
                f"({', '.join(sorted(set(failed_batches)))}); their sentences kept regex-only "
                "equation extraction",
            )
        )
    if validation_issue_sink is not None and extractor is not None:
        validation_issue_sink.extend(extractor.validation_issues)
    return results[0]


def _note_extraction_sources(contents, paper_metadata, parse_strategy: str | None) -> None:
    """Note the source of what metadata extraction produced, for ``extraction.fields``.

    Front matter the input declares (JATS, HTML meta tags) is ``native``; the
    core extractor notes its own sources. The reference list comes from the
    input's structured citations or the configured parser.
    """
    if contents.preparsed_metadata is not None:
        for field in ("title", "abstract", "keywords", "published", "journal", "author"):
            set_field_source(paper_metadata, field, "native")
        set_field_source(paper_metadata, "funding_statement", "native")
    set_field_source(
        paper_metadata,
        "bib",
        "native" if contents.native_references is not None else str(parse_strategy or "llm"),
    )


def _body_text_excluding_references(contents) -> str:
    """Concatenate body sentence text, excluding the reference-list section
    (whose author-year entries would otherwise dominate the citation count)."""
    from bibr.paper_contents import CanonicalSection

    ref_section_ids = {
        s.section_id for s in contents.sections if s.section_type == CanonicalSection.REFERENCES
    }
    return " ".join(s.text for s in contents.sentences if s.section_id not in ref_section_ids)


def _abstract_suspicion_reasons(contents, abstract_text, selection) -> tuple[str, ...]:
    from bibr.paper_contents import CanonicalSection
    from bibr.validation import abstract_suspicion_reasons

    section_types = {section.section_id: section.section_type for section in contents.sections}
    reference_ids = {
        section_id
        for section_id, section_type in section_types.items()
        if section_type in {CanonicalSection.REFERENCES, CanonicalSection.ENDNOTE}
    }
    selected_ids = frozenset(selection.text_ids)
    prose_sentences = [
        sentence
        for sentence in contents.sentences
        if sentence.section_id not in reference_ids and not sentence.is_display_formula
    ]
    return abstract_suspicion_reasons(
        abstract_text,
        source_text=selection.text,
        outside_texts=(
            sentence.text for sentence in prose_sentences if sentence.text_id not in selected_ids
        ),
        non_reference_prose=" ".join(sentence.text for sentence in prose_sentences),
    )


def _finalize_abstract_and_keywords(
    contents,
    paper_metadata,
    *,
    resolution=None,
    validation_issue_sink: list[ValidationIssue] | None = None,
) -> None:
    """Extraction policy, finalized once per paper (mutates ``paper_metadata``):

    - abstract: prefer the LLM-extracted string (clean, deduplicated); fall
      back to joining ABSTRACT-typed section sentences only when the LLM
      produced nothing usable. The LLM string wins because layout regions
      (running headers, copyright lines, affiliation blocks) routinely flow
      into the abstract section and would corrupt a blind join.
    - keywords: recover from KEYWORD-typed sections when LLM extraction missed
      them. Without LLM the keywords section often spills into the intro, so
      only the first sentence is used and anything that doesn't look like a
      keyword list (long or sentence-like entries) is rejected.
    - commentary guard (residual #1): the OCR layout model mislabels a
      commentary's opening body as an ABSTRACT region; both the LLM string and
      the section fallback can carry that body text. Suppress it here — using
      the FINAL keywords, so a genuine commentary with a keywords block is
      preserved.

    This lives in post-parse, not the export layer: ``json_export`` serializes
    metadata verbatim and must not re-derive it.
    """
    from bibr.paper_contents import CanonicalSection
    from bibr.structure.implicit_sections import select_abstract_span

    selection = select_abstract_span(contents, resolution) if resolution is not None else None

    abstract_text = (paper_metadata.abstract or "").strip()
    # A layout hint or positional inference can name body prose "Abstract".
    # Preserve a completed extraction's explicit null unless the document has
    # a printed abstract heading in the selected span. Missing fields and
    # no-LLM runs retain the existing fallback behavior.
    explicit_absence = paper_metadata._abstract_explicitly_absent
    selected_section_ids = (
        {
            sentence.section_id
            for sentence in contents.sentences
            if sentence.text_id in selection.text_ids
        }
        if selection is not None
        else {section.section_id for section in contents.sections}
    )
    printed_abstract = any(
        section.section_id in selected_section_ids
        and section.section_type == CanonicalSection.ABSTRACT
        and section.header.strip()
        and not section.header_is_synthetic
        for section in contents.sections
    )
    if not abstract_text and (not explicit_absence or printed_abstract):
        if selection is not None:
            abstract_text = selection.text
        else:
            abstract_section_ids = set()
            for s in contents.sections:
                if s.section_type == CanonicalSection.ABSTRACT:
                    abstract_section_ids.add(s.section_id)
                elif s.section_type in (CanonicalSection.UNKNOWN, None) and s.header:
                    from bibr.utils.text import normalize_text

                    if normalize_text(s.header) == "abstract":
                        abstract_section_ids.add(s.section_id)
            if abstract_section_ids:
                abstract_text = " ".join(
                    sent.text
                    for sent in contents.sentences
                    if sent.section_id in abstract_section_ids and not sent.is_display_formula
                )

    keywords = paper_metadata.keywords
    if not keywords:
        kw_section_ids = {
            s.section_id for s in contents.sections if s.section_type == CanonicalSection.KEYWORDS
        }
        if kw_section_ids:
            kw_sentences = [
                sent.text for sent in contents.sentences if sent.section_id in kw_section_ids
            ]
            if kw_sentences:
                raw = kw_sentences[0].rstrip(" .;")
                candidates = [k.strip() for k in raw.split(",") if k.strip()]
                if 1 <= len(candidates) <= 15 and all(
                    len(k) <= 80 and "." not in k for k in candidates
                ):
                    keywords = candidates

    if abstract_text and selection is not None and validation_issue_sink is not None:
        reasons = _abstract_suspicion_reasons(contents, abstract_text, selection)
        if reasons:
            from bibr.validation import IssueSeverity, ValidationIssue

            validation_issue_sink.append(
                ValidationIssue(
                    code="VAL_ABSTRACT_SUSPECT",
                    severity=IssueSeverity.WARNING,
                    message=f"abstract suspicion: {', '.join(reasons)}",
                    origin_stage="extract",
                    evidence_ids=selection.evidence_ids,
                    blocking=False,
                )
            )

    paper_metadata.abstract = abstract_text.strip()
    paper_metadata.keywords = keywords


# Layout doc_title vs LLM title reconciliation (masthead guard). A journal /
# publisher name shorter than this is too generic to treat a match as a
# masthead signal.
_MASTHEAD_MIN_NAME_CHARS = 6
# detected_title fuzzy ratio at/above which it "is" the journal/publisher name.
_MASTHEAD_NAME_RATIO = 0.90
# detected vs LLM title ratio at/above which they agree (keep the verbatim
# layout title even on a masthead match — avoids overriding when the LLM merely
# paraphrased a genuine title).
_TITLE_AGREE_RATIO = 0.85
# Use a narrow masthead grammar: a URL or ISSN is stronger page-furniture evidence than a volume
# label or bare DOI.
_MASTHEAD_MARKER_RE = re.compile(r"https?://|www\.|ISSN\s*\d{4}", re.IGNORECASE)


def _normalize_for_match(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .:;,-")


def _title_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_for_match(a), _normalize_for_match(b)).ratio()


def _name_matches(detected_norm: str, name: str | None) -> bool:
    if not name:
        return False
    n = _normalize_for_match(name)
    if len(n) < _MASTHEAD_MIN_NAME_CHARS:
        return False
    return n in detected_norm or difflib.SequenceMatcher(None, detected_norm, n).ratio() >= (
        _MASTHEAD_NAME_RATIO
    )


def _detected_title_is_masthead(detected: str, journal: str | None, publisher: str | None) -> bool:
    """True when the layout-detected title is really a masthead/banner.

    Layout detection can label the journal or publisher banner as a document title. Matching an independently extracted journal/publisher name or finding a banner-only URL/ISSN marker provides specific evidence that the text is page furniture.
    """
    d = _normalize_for_match(detected)
    if _name_matches(d, journal) or _name_matches(d, publisher):
        return True
    return bool(_MASTHEAD_MARKER_RE.search(detected))


# Reject a composite heading only when every content token belongs to the heading vocabulary,
# using accent-insensitive matching.
_BODY_HEADING_WORDS = frozenset(
    {
        # presentation / introduction
        "apresentacao",
        "presentacion",
        "presentazione",
        "presentation",
        "introducao",
        "introduccion",
        "introduzione",
        "introduction",
        # method / materials
        "metodologia",
        "metodologias",
        "metodologie",
        "methodologie",
        "metodo",
        "metodos",
        "metodi",
        "methode",
        "methodes",
        "materiais",
        "materiales",
        "materiali",
        "materiel",
        "materiels",
        "procedimentos",
        "procedimientos",
        # results / analysis
        "resultado",
        "resultados",
        "resultat",
        "resultats",
        "risultati",
        "risultato",
        "analise",
        "analises",
        "analisis",
        "analisi",
        "analyse",
        "analyses",
        "dados",
        "datos",
        "dati",
        "donnees",
        # discussion / conclusion
        "discussao",
        "discussoes",
        "discusion",
        "discusiones",
        "discussione",
        "discussioni",
        "discussion",
        "conclusao",
        "conclusoes",
        "conclusion",
        "conclusiones",
        "conclusione",
        "conclusioni",
        "conclusions",
        "consideracoes",
        "consideraciones",
        "considerazioni",
        "considerations",
        "sintese",
        "sintesi",
        "synthese",
        "limitacoes",
        "limitaciones",
        "limitazioni",
        "recomendacoes",
        "recomendaciones",
        "raccomandazioni",
        "recommandations",
        "final",
        "finais",
        "finales",
        "finali",
        # framing / literature
        "revisao",
        "revision",
        "revisione",
        "revue",
        "literatura",
        "letteratura",
        "litterature",
        "fundamentacao",
        "fundamentacion",
        "fundamentos",
        "teorica",
        "teorico",
        "teoricos",
        "theorique",
        "objetivo",
        "objetivos",
        "obiettivi",
        "obiettivo",
        "objectif",
        "objectifs",
        "hipotese",
        "hipoteses",
        "hipotesis",
        "ipotesi",
        "hypothese",
        "hypotheses",
        # front/back matter
        "resumo",
        "resumen",
        "riassunto",
        "resume",
        "palavras",
        "palabras",
        "parole",
        "chave",
        "clave",
        "chiave",
        "cle",
        "cles",
        "agradecimentos",
        "agradecimientos",
        "ringraziamenti",
        "remerciements",
        "referencias",
        "riferimenti",
        "bibliografia",
        "bibliografias",
    }
)
# Function words and articles that carry no heading/title signal on their own.
_HEADING_FILLER_WORDS = frozenset(
    {
        "a",
        "ai",
        "al",
        "alla",
        "and",
        "as",
        "com",
        "con",
        "da",
        "das",
        "de",
        "degli",
        "dei",
        "del",
        "della",
        "delle",
        "dello",
        "des",
        "di",
        "do",
        "dos",
        "du",
        "e",
        "ed",
        "el",
        "em",
        "en",
        "et",
        "gli",
        "i",
        "il",
        "in",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "of",
        "os",
        "para",
        "per",
        "pour",
        "the",
        "u",
        "um",
        "uma",
        "un",
        "una",
        "unas",
        "uno",
        "unos",
        "y",
    }
)
_HEADING_TOKEN_RE = re.compile(r"[^\W\d_]+")
# A printed heading is short; beyond this the row is prose, not a heading.
_MAX_HEADING_TOKENS = 8


def _strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", value) if not unicodedata.combining(char)
    )


def _is_ordinary_body_heading(normalized: str) -> bool:
    """Whether a normalized row is a bare body heading rather than a title.

    Complements ``_ORDINARY_HEADING_TEXT``'s exact-membership test: numbering
    and function words are dropped, and the row is a heading only when every
    remaining token is heading vocabulary.
    """

    tokens = _HEADING_TOKEN_RE.findall(_strip_accents(normalized))
    if not tokens or len(tokens) > _MAX_HEADING_TOKENS:
        return False
    content = [token for token in tokens if token not in _HEADING_FILLER_WORDS]
    return bool(content) and all(token in _BODY_HEADING_WORDS for token in content)


def _title_text_is_unsafe(text: str, normalized: str, region_label: str, paper_metadata) -> bool:
    """Shared text-level filters for any printed row proposed as the title.

    Journal furniture, generic article labels, bare section headings and
    mastheads all fail closed: asserting a wrong title is worse than asserting
    none.
    """

    from bibr.extract.front_matter import (
        _ORDINARY_HEADING_TEXT,
        _looks_like_masthead,
        is_exact_front_matter_furniture,
    )
    from bibr.utils.metadata import is_exact_generic_article_label

    return bool(
        not text
        or is_exact_generic_article_label(text)
        or is_exact_front_matter_furniture(text)
        or normalized in _ORDINARY_HEADING_TEXT
        or _is_ordinary_body_heading(normalized)
        or _looks_like_masthead(text, region_label)
        or _detected_title_is_masthead(text, paper_metadata.journal, paper_metadata.publisher)
    )


def _selected_front_matter_block(contents):
    """The resolved front-matter block, or None when there is no ownership."""

    resolution = getattr(contents, "front_matter_resolution", None)
    if resolution is None or resolution.selected_block_id is None:
        return None
    return next(
        (block for block in resolution.blocks if block.block_id == resolution.selected_block_id),
        None,
    )


def _safe_title_candidate(candidate, paper_metadata) -> bool:
    """Whether one selected-record candidate may stand in as the article title."""

    text = candidate.raw_text.strip()
    return not (
        "title" not in candidate.roles
        or not candidate.roles.isdisjoint({"abstract", "affiliation", "byline", "doi"})
        or _title_text_is_unsafe(
            text,
            candidate.normalized_text,
            (candidate.region_label or "").casefold(),
            paper_metadata,
        )
    )


def _resolve_selected_title(
    contents,
    paper_metadata,
    *,
    validation_issue_sink: list[ValidationIssue],
) -> bool:
    """Recover a null LLM title from exactly one safe selected-record candidate.

    Reads only the selected front-matter block's title candidates — never the
    global detected title or unknown-section headers. Ambiguity (two distinct
    candidates after normalization), mastheads, journal furniture, generic
    article labels, bare section headings, and composite title+byline/abstract/
    affiliation/DOI rows all fail closed and leave the title null.
    """

    from bibr.validation import IssueSeverity, ValidationIssue

    selected_block = _selected_front_matter_block(contents)
    if selected_block is None:
        return False
    resolution = contents.front_matter_resolution

    by_id = {candidate.candidate_id: candidate for candidate in resolution.candidates}
    safe = [
        candidate
        for candidate_id in selected_block.title_candidate_ids
        if (candidate := by_id.get(candidate_id)) is not None
        and _safe_title_candidate(candidate, paper_metadata)
    ]

    distinct: dict[str, object] = {}
    for candidate in safe:
        distinct.setdefault(_normalize_for_match(candidate.raw_text), candidate)
    if len(distinct) != 1:
        return False
    candidate = next(iter(distinct.values()))
    paper_metadata.title = candidate.raw_text
    validation_issue_sink.append(
        ValidationIssue(
            code="VAL_TITLE_RECOVERED",
            severity=IssueSeverity.WARNING,
            message="Recovered null model title from the selected front-matter record",
            origin_stage="extract",
            evidence_ids=(candidate.candidate_id,),
            blocking=False,
        )
    )
    logger.info(
        "Selected-title fallback recovered the null model title from candidate %s",
        candidate.candidate_id,
    )
    return True


def _resolve_detected_title_fallback(
    contents,
    paper_metadata,
    *,
    validation_issue_sink: list[ValidationIssue],
) -> bool:
    """Last-resort: accept the layout ``detected_title`` for a still-null title.

    Under ownership scope, a null model title and no safe selected candidate can leave title=None even when layout has the printed title. Reuse the selected-title filters without enabling the broader unknown-header scan. This path only fills an absent title.
    """

    from bibr.extract.front_matter import _normalize_text
    from bibr.validation import IssueSeverity, ValidationIssue

    if paper_metadata.title:
        return False
    detected = (getattr(contents, "detected_title", None) or "").strip()
    if _title_text_is_unsafe(detected, _normalize_text(detected), "doc_title", paper_metadata):
        return False

    paper_metadata.title = detected
    validation_issue_sink.append(
        ValidationIssue(
            code="VAL_TITLE_RECOVERED",
            severity=IssueSeverity.WARNING,
            message="Recovered null model title from the layout-detected title",
            origin_stage="extract",
            blocking=False,
        )
    )
    logger.info("Detected-title fallback recovered the null model title: %r", detected)
    return True


def _prefer_byline_adjacent_title(
    contents,
    paper_metadata,
    *,
    validation_issue_sink: list[ValidationIssue],
    settings,
) -> bool:
    """Prefer the printed title row that sits directly above the byline.

    Multilingual front matter may print the original title above the byline and a translation above another abstract. This path can replace an asserted title, so it is gated by PIPELINE_TITLE_PREFER_BYLINE_ADJACENT and defaults off pending broader validation.
    """

    from bibr.validation import IssueSeverity, ValidationIssue

    if not getattr(getattr(settings, "pipeline", None), "title_prefer_byline_adjacent", False):
        return False
    llm_title = (paper_metadata.title or "").strip()
    if not llm_title:
        return False
    selected_block = _selected_front_matter_block(contents)
    if selected_block is None:
        return False

    owned = set(selected_block.candidate_ids) | set(selected_block.title_candidate_ids)
    in_block = [
        candidate
        for candidate in contents.front_matter_resolution.candidates
        if candidate.candidate_id in owned
    ]
    byline_order = min(
        (candidate.reading_order for candidate in in_block if "byline" in candidate.roles),
        default=None,
    )
    if byline_order is None:
        return False
    above = [
        candidate
        for candidate in in_block
        if candidate.reading_order < byline_order
        and _safe_title_candidate(candidate, paper_metadata)
    ]
    if not above:
        return False
    candidate = max(above, key=lambda item: item.reading_order)

    wanted = _normalize_for_match(llm_title)
    if (
        wanted in _normalize_for_match(candidate.raw_text)
        or _title_ratio(candidate.raw_text, llm_title) >= _TITLE_AGREE_RATIO
    ):
        return False

    paper_metadata.title = candidate.raw_text
    validation_issue_sink.append(
        ValidationIssue(
            code="VAL_TITLE_BYLINE_ADJACENT",
            severity=IssueSeverity.WARNING,
            message="Preferred the printed title row directly above the byline over the model title",
            origin_stage="extract",
            evidence_ids=(candidate.candidate_id,),
            blocking=False,
        )
    )
    logger.info(
        "Byline-adjacency title preference replaced %r with candidate %s",
        llm_title,
        candidate.candidate_id,
    )
    return True


def _resolve_title(contents, paper_metadata) -> None:
    """Prefer the layout-detected title; otherwise fall back to the first
    unclassified non-canonical header.

    When the layout title is specifically identified as a journal or publisher banner and disagrees with the language model, prefer the language-model title. Disagreement alone is insufficient: the masthead evidence must be specific.
    """
    from bibr.paper_contents import CanonicalSection

    detected = contents.detected_title
    llm_title = paper_metadata.title
    if detected:
        from bibr.utils.metadata import is_exact_generic_article_label

        resolution = getattr(contents, "front_matter_resolution", None)
        grounded_llm_title = False
        if (
            llm_title
            and not is_exact_generic_article_label(llm_title)
            and resolution is not None
            and resolution.selected_block_id is not None
        ):
            selected = {
                candidate_id
                for block in resolution.blocks
                if block.block_id == resolution.selected_block_id
                for candidate_id in block.title_candidate_ids
            }
            wanted = _normalize_for_match(llm_title)
            grounded_llm_title = any(
                candidate.candidate_id in selected
                and "title" in candidate.roles
                and candidate.roles.isdisjoint({"abstract", "byline", "affiliation", "doi"})
                and wanted
                and wanted in _normalize_for_match(candidate.raw_text)
                for candidate in resolution.candidates
            )
        if is_exact_generic_article_label(detected) and grounded_llm_title:
            paper_metadata.title = llm_title
        elif (
            llm_title
            and _detected_title_is_masthead(
                detected, paper_metadata.journal, paper_metadata.publisher
            )
            and _title_ratio(detected, llm_title) < _TITLE_AGREE_RATIO
        ):
            # Layout labeled the journal or publisher banner as the document title; the language
            # model has the article title from the full front matter.
            logger.info(
                "Masthead-title guard: layout doc_title %r matches journal/publisher; "
                "using LLM title %r instead",
                detected,
                llm_title,
            )
            paper_metadata.title = llm_title
        else:
            paper_metadata.title = detected
    if not paper_metadata.title:
        _canonical_headers = {s.value for s in CanonicalSection if s != CanonicalSection.UNKNOWN}
        for sec in contents.sections:
            if sec.level > 0 and sec.section_type == CanonicalSection.UNKNOWN:
                header_lower = sec.header.lower().strip()
                if header_lower and header_lower not in _canonical_headers:
                    paper_metadata.title = sec.header
                    break


async def _link_citations(
    contents,
    paper_metadata,
    file_hash: str,
    llm_client,
    *,
    validation_issue_sink: list[ValidationIssue] | None = None,
) -> None:
    """Inline citation → bib linking.

    Detects inline citations and links them to bibliography entries, extending
    ``contents.xrefs``. Read-only on sentences/sections, so it runs
    concurrently with research-integrity extraction. Superscript cleanup is
    intentionally *not* here — it mutates body-sentence text and must run after
    abstract finalization, so ``post_parse`` calls
    ``strip_citation_superscripts`` as a separate late step.
    """
    from bibr.export.validation import xref_low_coverage_issue
    from bibr.structure.citation_linker import detect_bib_xrefs

    receipt_sink = []
    bib_xrefs = await detect_bib_xrefs(
        sentences=contents.sentences,
        sections=contents.sections,
        references=paper_metadata.references,
        llm_client=llm_client,
        file_hash=file_hash,
        receipt_sink=receipt_sink,
    )
    contents.xrefs.extend(bib_xrefs)
    contents.citation_receipt = receipt_sink[0] if receipt_sink else None
    if contents.citation_receipt is not None:
        failed = [
            reason.removeprefix("llm_failed:")
            for candidate in contents.citation_receipt.candidates
            for reason in candidate.rejection_reasons
            if reason.startswith("llm_failed:")
        ]
        if failed:
            contents.processing_warnings.append(
                ProcessingWarning(
                    WarningCode.CITATION_LLM_FAILED,
                    f"{failed[0]}: {len(failed)} ambiguous in-text citation(s) left unlinked",
                )
            )
    if validation_issue_sink is not None:
        issue = xref_low_coverage_issue(
            {reference.bib_id for reference in paper_metadata.references},
            {xref.xref_id for xref in bib_xrefs if xref.xref_type == "bib"},
            bib_count=len(paper_metadata.references),
            origin_stage="post_parse",
        )
        if issue is not None:
            validation_issue_sink.append(issue)


def _usage_totals_by_label(
    labels: dict[tuple[str, str, str], dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Sum ``(label, provider, model)``-keyed usage buckets down to label-only.

    ``LLMClient.usage_labels_pop_file`` partitions by the full triple (a label
    can run under more than one provider/model within a file) so the per-triple
    counts never blend a stale provider stamp with a new engine's token counts.
    The export carries those triples verbatim (``extraction.usage.breakdown``),
    but two consumers are label-keyed by contract and must not gain an engine
    dimension: ``SafeLlmDiagnostics`` (which requires unique labels) and
    ``qualification_provenance`` (read by an external deployment gate). Both are
    fed from here.
    """
    collapsed: dict[str, dict[str, int]] = {}
    for (label, _provider, _model), counts in labels.items():
        bucket = collapsed.setdefault(label, {})
        for name, value in counts.items():
            bucket[name] = bucket.get(name, 0) + value
    return collapsed


def _build_paper(contents, paper_metadata, file_name: str, file_hash: str, paper_id: str | None):
    """Assemble the final ``Paper`` object from extracted parts."""
    from bibr.input.file import InputFile, InputFormat
    from bibr.paper import Paper, ProcessingStatus

    ext_lower = os.path.splitext(file_name)[1].lower()
    file_type = {
        ".pdf": "PDF",
        ".docx": "DOCX",
        ".xml": "XML",
        ".html": "HTML",
        ".htm": "HTML",
        ".epub": "EPUB",
    }.get(ext_lower, "UNKNOWN")

    input_file_obj = InputFile(path=file_name)
    input_file_obj.file_hash = file_hash
    input_file_obj.input_format = InputFormat(
        file_extension=ext_lower.lstrip("."),
        detected_mime_type="",
        file_type=file_type,
    )

    return Paper(
        input_file=input_file_obj,
        contents=contents,
        metadata=paper_metadata,
        processing_status=ProcessingStatus(parsed=True),
        paper_id=paper_id,
    )


def _build_qualification_provenance(
    llm_client,
    settings: GlobalSettings,
    *,
    usage_by_label: dict,
    protocol_hashes: dict,
) -> dict | None:
    """Assemble the deployment-qualification provenance surface for a paper."""
    from bibr.clients.nuextract import (
        NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256,
        NUEXTRACT3_FP8_EXPECTED_REVISION,
        NUEXTRACT3_FP8_MODEL_ID,
    )
    from bibr.export.qualification_provenance import (
        DeploymentIdentity,
        build_qualification_provenance,
    )

    structured_backend = getattr(llm_client, "resolved_structured_backend", None)
    model_id = settings.llm.model
    # The expected pins belong to the FP8 repo. Stamping them on any other
    # NuExtract 3 deployment (bf16, GGUF, MLX, whichever backend) would record a
    # commit and template that deployment never loaded, so it reports only
    # what LLM_MODEL_REVISION / LLM_JINJA_SHA256 declare.
    is_nuextract_fp8 = (model_id or "").strip().lower() == NUEXTRACT3_FP8_MODEL_ID.lower()
    identity = DeploymentIdentity(
        bibr_sha=settings.pipeline.bibr_sha or settings.BIBR_BUILD_SHA,
        platform_sha=settings.pipeline.platform_sha,
        model_id=model_id,
        model_revision=settings.llm.model_revision
        or (NUEXTRACT3_FP8_EXPECTED_REVISION if is_nuextract_fp8 else None),
        jinja_sha256=settings.llm.jinja_sha256
        or (NUEXTRACT3_FP8_EXPECTED_JINJA_SHA256 if is_nuextract_fp8 else None),
        structured_backend=structured_backend,
        # Temperature is a declared qualification axis only for the native
        # protocol arms. For non-native backends it is not part of the
        # protocol identity the gate compares, so report None (the arm's
        # requested temperature is likewise null) rather than the ambient
        # sampling temperature, which would false-flag an identity mismatch.
        temperature=(
            settings.llm.temperature if structured_backend == "nuextract-native" else None
        ),
        thinking_mode=False,
    )
    return build_qualification_provenance(
        usage_by_label=usage_by_label,
        protocol_hashes_by_label=protocol_hashes,
        identity=identity,
    )


async def post_parse(
    contents,
    file_name: str,
    file_hash: str,
    paper_id: str | None = None,
    ocr_metadata: dict | None = None,
    layout_hints: list | None = None,
    no_llm: bool = False,
    extract_equations: bool = True,
    *,
    llm_client=None,
    ref_seg_strategy: str | None = None,
    ref_parse_strategy: str | None = None,
    settings: GlobalSettings | None = None,
    classifier_resources: ClassifierResources | None = None,
    expected_identity: ExpectedIdentity | None = None,
    enrichment_prefetch: bool = False,
):
    """Post-parse pipeline: classification, extraction, linking.

    Shared between LitServe and local pipelines.

    When ``enrichment_prefetch`` is set (the run will enrich references), the
    enrich stage's up-front network work starts as a task the moment the
    references are parsed — while citation linking and structured-integrity
    LLM calls are still in flight — and rides the returned ``Paper`` as
    ``paper.enrichment_prefetch`` for ``CrossrefEnricher`` to consume. It is
    cancelled if post-parse fails after it started. Nothing is started for
    ``no_llm``, ``refs=off`` or an empty reference list.

    When ``no_llm=True``, all LLM-driven steps are skipped: section
    classification falls back to alias-table lookup only, implicit section
    detection / metadata extraction / equation extraction / citation linking
    are all bypassed. The returned ``Paper`` has an empty ``PaperMetadata``
    (title populated from OCR-detected title when available). Intended for
    ML-training-data preprocessing where metadata will be re-labeled later.
    """
    from bibr.clients.llm import LLMClient, new_usage_context_key, usage_file_context
    from bibr.config import snapshot_settings
    from bibr.extract.ref_extractor import _resolve_ref_strategies
    from bibr.paper import _merge_ocr_metadata

    # Single LLMClient for the entire post-parse pipeline — shared across
    # section classification, implicit section detection, and metadata extraction
    # so that the agent cache, rate limiter, and circuit breaker state are reused.
    effective_settings = settings if settings is not None else snapshot_settings()
    owns_client = False
    if no_llm:
        llm_client = None
    elif llm_client is None:
        llm_client = LLMClient(settings=effective_settings)
        owns_client = True

    # Resolve the reference seg/parse strategies ONCE per file. The resolved
    # pair flows to the extractor (which re-normalizes idempotently) and to the
    # under-extraction gate below, so no phase re-resolves per file.
    seg_strategy, parse_strategy = _resolve_ref_strategies(
        ref_seg_strategy,
        ref_parse_strategy,
        settings=effective_settings,
    )

    file_usage: dict[str, dict[str, int]] = {}
    file_usage_labels: dict[tuple[str, str, str], dict[str, int]] = {}
    file_protocol_hashes: dict[str, dict[str, str]] = {}
    file_llm_trace: list[dict] = []
    front_matter_issues = ()
    metadata_issues: list[ValidationIssue] = []

    prefetch_handle = None
    on_references_ready = None
    if enrichment_prefetch and not no_llm and parse_strategy != "off":
        from bibr.pipeline.enrich_prefetch import start_enrichment_prefetch

        def on_references_ready(references: list) -> None:
            nonlocal prefetch_handle
            if prefetch_handle is None:
                prefetch_handle = start_enrichment_prefetch(references, settings=effective_settings)

    extraction_completed = False
    # The client is shared across concurrently-processed files; the contextvar
    # scopes usage attribution to this file's task tree (no snapshot diffing).
    # The key is unique per invocation (not the bare content hash) so
    # reprocessing a file never accumulates across runs and concurrent
    # duplicate files don't share a bucket; popped below to stay bounded.
    usage_key = new_usage_context_key(file_hash)
    terminal_processing_error: ProcessingError | None = None
    with usage_file_context(usage_key):
        try:
            # --- Sections: classify then normalize (implicit + enforce) ---
            await _classify_sections(
                contents,
                layout_hints,
                no_llm,
                llm_client,
                classifier_resources=classifier_resources,
                settings=effective_settings,
            )
            front_matter_issues = _attach_front_matter_resolution(
                contents,
                expected_identity,
                metadata_llm_active=not no_llm,
                settings=effective_settings,
            )
            await _normalize_section_structure(
                contents,
                no_llm,
                llm_client,
                file_hash,
                settings=effective_settings,
            )

            # --- Metadata + Equations in parallel ---
            paper_metadata = await _extract_metadata_and_equations(
                contents,
                file_hash,
                no_llm,
                llm_client,
                extract_equations=extract_equations,
                ref_seg_strategy=seg_strategy,
                ref_parse_strategy=parse_strategy,
                settings=effective_settings,
                classifier_resources=classifier_resources,
                front_matter_resolution=contents.front_matter_resolution,
                validation_issue_sink=metadata_issues,
                on_references_ready=on_references_ready,
            )
            _note_extraction_sources(contents, paper_metadata, parse_strategy)
            metadata_ownership_scoped = bool(
                contents.preparsed_metadata is None
                and contents.front_matter_resolution is not None
                and (not no_llm or any(issue.blocking for issue in front_matter_issues))
            )
            metadata_abstained = bool(
                metadata_ownership_scoped
                and contents.front_matter_resolution.selected_block_id is None
            )
            from bibr.utils.metadata import is_exact_generic_article_label

            # Ownership-scoped null-title safety net: one safe selected-record
            # candidate may replace a null LLM title, then — still only when the
            # title is otherwise absent — the layout detected title, under the
            # same filters. `_resolve_title`'s unknown-header scan stays skipped
            # under ownership scope. With a title in hand the only (default-off)
            # policy is byline adjacency for multilingual front matter.
            if metadata_ownership_scoped and not metadata_abstained and not paper_metadata.title:
                if _resolve_selected_title(
                    contents,
                    paper_metadata,
                    validation_issue_sink=metadata_issues,
                ):
                    set_field_source(paper_metadata, "title", "front_matter_candidate")
                elif _resolve_detected_title_fallback(
                    contents,
                    paper_metadata,
                    validation_issue_sink=metadata_issues,
                ):
                    set_field_source(paper_metadata, "title", "layout_title")
            elif (
                metadata_ownership_scoped
                and not metadata_abstained
                and _prefer_byline_adjacent_title(
                    contents,
                    paper_metadata,
                    validation_issue_sink=metadata_issues,
                    settings=effective_settings,
                )
            ):
                set_field_source(paper_metadata, "title", "byline_adjacent")
            if not metadata_ownership_scoped or is_exact_generic_article_label(
                contents.detected_title
            ):
                title_before = paper_metadata.title
                _resolve_title(contents, paper_metadata)
                if paper_metadata.title != title_before:
                    set_field_source(
                        paper_metadata,
                        "title",
                        "layout_title"
                        if paper_metadata.title == contents.detected_title
                        else "section_header",
                    )

            # OCR/doc-info fallback can supply authors when primary extraction
            # abstains. Merge it before freezing the author-name snapshot used
            # to ground named funding declarations.
            if ocr_metadata and not metadata_ownership_scoped:
                _merge_ocr_metadata(paper_metadata, ocr_metadata)

            # Build statement candidates while section labels and source IDs
            # are stable, but delay materialization until after late cleaning.
            # This keeps exported scalar text and structured-funding evidence
            # byte-aligned with the final sentence text.
            from bibr.extract.integrity_statements import (
                apply_integrity_resolution,
                resolve_integrity_statements,
            )
            from bibr.extract.research_integrity import extract_structured_integrity

            # Synchronous regex sweep over every sentence — median 28-37 ms,
            # max 66 ms on 479 real PMC exports. Threaded for the same reason
            # as the citation tiers: serve runs one async worker, so this on
            # the loop delays every co-resident request.
            integrity_resolution = await asyncio.to_thread(
                resolve_integrity_statements,
                contents,
                mode=effective_settings.pipeline.integrity_statement_mode,
                author_names=tuple(
                    (author.given, author.family) for author in paper_metadata.authors
                ),
            )

            # Citation linking must see the raw superscript/LaTeX patterns.
            # Structured integrity is intentionally sequenced later because it
            # must render selected IDs after final text cleaning.
            if not no_llm:
                await _link_citations(
                    contents,
                    paper_metadata,
                    file_hash,
                    llm_client,
                    validation_issue_sink=metadata_issues,
                )

            # Extraction policy (abstract fallback, keyword recovery,
            # commentary guard) finalizes metadata here — the export layer
            # serializes it verbatim.
            if not metadata_abstained:
                abstract_before = paper_metadata.abstract
                keywords_before = paper_metadata.keywords
                _finalize_abstract_and_keywords(
                    contents,
                    paper_metadata,
                    resolution=(
                        contents.front_matter_resolution if metadata_ownership_scoped else None
                    ),
                    validation_issue_sink=metadata_issues,
                )
                if paper_metadata.abstract and not (abstract_before or "").strip():
                    set_field_source(paper_metadata, "abstract", "abstract_section")
                if paper_metadata.keywords and not keywords_before:
                    set_field_source(paper_metadata, "keywords", "keywords_section")

            # Superscript cleanup runs last: it needs the ``^{N}`` markers
            # preserved through citation detection above, and must follow
            # abstract finalization (which reads unstripped body text) — hence
            # it is split out of ``_link_citations``.
            if not no_llm:
                from bibr.structure.citation_linker import strip_citation_superscripts

                strip_citation_superscripts(
                    contents.sentences,
                    contents.sections,
                    contents.citation_receipt,
                )

            # Late text cleaning now precedes integrity rendering. Candidate
            # acceptance and provenance were frozen above; only their selected
            # text IDs are rendered from the cleaned sentence objects here.
            await asyncio.to_thread(contents.finalize_text)
            apply_integrity_resolution(contents, paper_metadata, integrity_resolution)
            metadata_issues.extend(integrity_resolution.issues)

            if not no_llm:
                await extract_structured_integrity(
                    contents,
                    paper_metadata,
                    llm_client,
                    file_hash,
                    integrity_resolution=integrity_resolution,
                )
                set_field_source(paper_metadata, "funding", "llm")
            extraction_completed = True

        except ProcessingError as exc:
            terminal_processing_error = exc
            raise
        finally:
            # Post-parse failed (or was cancelled) after the reference task
            # kicked off the enrichment prefetch: nothing will consume it, so
            # cancel rather than let it run to completion on its own.
            if not extraction_completed and prefetch_handle is not None:
                prefetch_handle.cancel()
            # Capture-and-evict in the finally so the bucket is removed even
            # when a phase raises — shared (serve/local) clients are never
            # closed per file, so a leaked bucket would accumulate forever.
            if (
                usage_key is not None
                and llm_client is not None
                and hasattr(llm_client, "usage_pop_file")
            ):
                popped = llm_client.usage_pop_file(usage_key)
                if getattr(llm_client, "_track_usage", False):
                    file_usage = popped
            if (
                usage_key is not None
                and llm_client is not None
                and hasattr(llm_client, "usage_labels_pop_file")
            ):
                popped_labels = llm_client.usage_labels_pop_file(usage_key)
                if getattr(llm_client, "_track_usage", False):
                    file_usage_labels = popped_labels
            if (
                usage_key is not None
                and llm_client is not None
                and hasattr(llm_client, "protocol_hashes_pop_file")
            ):
                popped_hashes = llm_client.protocol_hashes_pop_file(usage_key)
                if getattr(llm_client, "_track_usage", False):
                    file_protocol_hashes = popped_hashes
            if (
                usage_key is not None
                and llm_client is not None
                and hasattr(llm_client, "traces_pop_file")
            ):
                popped_trace = llm_client.traces_pop_file(usage_key)
                if getattr(llm_client, "_track_usage", False):
                    file_llm_trace = popped_trace
            if (
                terminal_processing_error is not None
                and terminal_processing_error.safe_diagnostics is not None
            ):
                terminal_processing_error.safe_diagnostics = (
                    terminal_processing_error.safe_diagnostics.with_file_usage(
                        file_usage,
                        _usage_totals_by_label(file_usage_labels),
                    )
                )
            # Clean up LLM client resources (rate limiter / Redis connections)
            # Only close if we created the client ourselves (owns_client=True).
            if owns_client and llm_client is not None:
                try:
                    await llm_client.close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if terminal_processing_error is None:
                        raise
                    logger.warning("LLM client cleanup failed after terminal processing error")

    # Populate Section.children from parent_section_id pointers, after all
    # section list mutations (implicit detection, re-parenting, IMRaD ordering).
    from bibr.structure.section_tree import build_section_tree

    build_section_tree(contents.sections)

    paper = _build_paper(contents, paper_metadata, file_name, file_hash, paper_id)
    paper.field_scope = FieldScope(
        no_llm=no_llm,
        native_metadata=contents.preparsed_metadata is not None,
        references_off=parse_strategy == "off",
    )
    paper.enrichment_prefetch = prefetch_handle
    paper.validation_issues.extend(front_matter_issues)
    paper.validation_issues.extend(metadata_issues)
    paper.llm_usage_labels = file_usage_labels
    paper.llm_trace = file_llm_trace
    if llm_client is not None and getattr(llm_client, "_track_usage", False):
        paper.qualification_provenance = _build_qualification_provenance(
            llm_client,
            effective_settings,
            # Label-keyed by contract — the external gate's shape must not gain
            # an engine dimension.
            usage_by_label=_usage_totals_by_label(file_usage_labels),
            protocol_hashes=file_protocol_hashes,
        )

    # Content-level warnings (e.g. reference-segmentation CRF fallback) come
    # first; post-parse-level warnings are appended after them.
    paper.processing_warnings.extend(contents.processing_warnings)

    # Surface suspected reference under-extraction (residual #3): a body that
    # cites far more distinct works than were parsed signals dropped reference
    # regions (OCR omission). Warning-only — never alters extracted data.
    # Skipped when reference extraction is off by design (--refs off): zero
    # parsed references would spuriously trip the net on every cited paper.
    if not no_llm and parse_strategy != "off":
        from bibr.structure.citation_linker import cited_reference_numbers

        warning = _low_reference_count_warning(
            _body_text_excluding_references(contents),
            len(paper_metadata.references),
            cited_numbers=cited_reference_numbers(contents.citation_receipt),
        )
        if warning:
            logger.warning("Reference under-extraction suspected: %s", warning.message)
            paper.processing_warnings.append(warning)

    # Report-only parse-quality score (Docling port): rates each region's text
    # for extraction garbage and attaches a paper-level scalar + threshold
    # warning. Never alters extracted data.
    if effective_settings.pipeline.text_quality_report:
        _attach_text_quality(paper, contents, effective_settings)

    return paper


class PostParseStage:
    name = "extract"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("contents", "file_hash", "native_metadata")
    produces = ("paper",)

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        alive = ctx.alive()
        sem = asyncio.Semaphore(ctx.settings.pipeline.max_concurrent_post_parse)
        # Start enrichment's network prefetch under the LLM tail only when this
        # run will actually enrich (post_parse itself skips it for refs=off /
        # no_llm / no references).
        enrichment_prefetch = bool(ctx.config.enrichment_enabled(ctx.settings))

        async def _gated(fs):
            async with sem:
                fs_t0 = time.monotonic()
                file_hash = fs.file_hash or ""
                result = await post_parse(
                    contents=fs.contents,
                    file_name=fs.path.name,
                    file_hash=file_hash,
                    paper_id=fs.paper_id,
                    ocr_metadata=fs.native_metadata,
                    layout_hints=fs.contents.layout_hints or None if fs.contents else None,
                    no_llm=ctx.config.no_llm,
                    extract_equations=ctx.config.equations,
                    llm_client=None if ctx.config.no_llm else ctx.resources.llm_client,
                    ref_seg_strategy=ctx.config.ref_seg_strategy,
                    ref_parse_strategy=ctx.config.ref_parse_strategy,
                    settings=ctx.settings,
                    classifier_resources=ctx.resources.classifiers,
                    expected_identity=fs.expected_identity,
                    enrichment_prefetch=enrichment_prefetch,
                )
                fs.stage_times[self.name] = time.monotonic() - fs_t0
                return result

        results = await asyncio.gather(*(_gated(fs) for fs in alive), return_exceptions=True)
        for fs, result in zip(alive, results, strict=True):
            if isinstance(result, BaseException):
                typed_processing = isinstance(result, ProcessingError)
                protocol_failure = typed_processing and result.error_code == "llm_invalid_output"
                if typed_processing and result.error_code:
                    error_code = result.error_code
                elif isinstance(result, LlmCallError):
                    # Say how the LLM failed (llm_timeout, llm_truncated, ...)
                    # instead of the generic extraction code.
                    error_code = result.error_code
                else:
                    error_code = "extraction_failed"
                if typed_processing:
                    result.failed_stage = result.failed_stage or self.name
                if protocol_failure:
                    # ``gather(return_exceptions=True)`` returns the exception
                    # with its full coroutine traceback. Do not retain paper or
                    # completion frame locals on the long-lived FileState.
                    result.__traceback__ = None
                    result.__context__ = None
                    cause = result.__cause__
                    if cause is not None:
                        cause.__traceback__ = None
                fs.set_error(
                    f"Post-parse failed: {result}",
                    code=error_code,
                    stage=self.name,
                    exc=result,
                )
                # exc_info=result preserves the original traceback through
                # asyncio.gather's return_exceptions=True path. Without this
                # the warning becomes a one-line message and intermittent
                # bugs (e.g. enum-coercion flakes from LLM responses) lose
                # all debugging context.
                if protocol_failure:
                    logger.warning(
                        "Post-parse failed for %s: %s",
                        fs.path.name,
                        result,
                    )
                else:
                    logger.warning(
                        "Post-parse failed for %s: %s",
                        fs.path.name,
                        result,
                        exc_info=result,
                    )
            else:
                # The export's ``source.sha256`` is the full digest the validate
                # stage computed; ``file_hash`` keeps its 16-character prefix for
                # the caches keyed on it.
                result.input_file.sha256 = fs.content_sha256
                fs.paper = result

        logger.debug("Post-parse stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
