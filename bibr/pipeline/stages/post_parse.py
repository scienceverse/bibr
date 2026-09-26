"""PostParseStage — concurrent extractor invocation across files."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from bibr.exceptions import LlmCallError, ProcessingError
from bibr.field_states import FieldScope
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
    memory_mode: str | None = None,
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
        memory_mode=memory_mode,
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
    memory_mode: str | None = None,
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
            memory_mode=memory_mode,
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
            memory_mode=memory_mode,
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


def _decide_record_fields(paper_metadata, *, native: bool, doc_info_authors=None) -> None:
    """Decide the fields the core extractor decides, when it did not run.

    An input that declares its front matter (JATS, HTML) supplies its own
    authors, publication date, journal and publisher; a run without an LLM has
    none, and the PDF doc-info may fill the authors (*doc_info_authors*, given
    only when the doc-info may fill).
    """
    from bibr.extract.field_decisions import (
        Classification,
        FieldDecision,
        apply_decision,
        decide_authors,
        decide_value,
        field_decisions_of,
        incumbent_candidate,
    )

    ledger = field_decisions_of(paper_metadata)
    if ledger is not None and "author" in ledger:
        return
    source = "native" if native else None
    authors = [incumbent_candidate(paper_metadata, "author", source=source)]
    if doc_info_authors is not None:
        authors.append(doc_info_authors)
    apply_decision(paper_metadata, decide_authors(authors))
    for name in ("published", "journal", "publisher"):
        apply_decision(
            paper_metadata,
            decide_value(name, incumbent_candidate(paper_metadata, name, source=source)),
        )
    apply_decision(
        paper_metadata,
        FieldDecision(
            "paper_type",
            Classification(
                paper_metadata.paper_type,
                paper_metadata.paper_type_confidence,
                paper_metadata.oecd_l1,
                paper_metadata.oecd_l2,
                paper_metadata.oecd_confidence,
            ),
            None,
            "not_classified",
        ),
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
    native: bool = False,
    doc_info_keywords=None,
    abstained: bool = False,
) -> None:
    """Decide the abstract and keywords, once per paper (writes ``paper_metadata``):

    - abstract: prefer the LLM-extracted string (clean, deduplicated); fall
      back to joining ABSTRACT-typed section sentences only when the LLM
      produced nothing usable. The LLM string wins because layout regions
      (running headers, copyright lines, affiliation blocks) routinely flow
      into the abstract section and would corrupt a blind join.
    - keywords: the extracted list, else the PDF doc-info's (*doc_info_keywords*,
      given only when the doc-info may fill), else the KEYWORD-typed sections.
      Without LLM the keywords section often spills into the intro, so only
      the first sentence is used and anything that doesn't look like a keyword
      list (long or sentence-like entries) is rejected.
    - a correction notice's extracted abstract and keywords are never used
      (the core extractor vetoes them); nothing is chosen when front-matter
      selection *abstained*.

    *native* says the input declared the incumbent values (JATS, HTML). This
    lives in post-parse, not the export layer: ``json_export`` serializes
    metadata verbatim and must not re-derive it.
    """
    from bibr.extract.field_decisions import (
        FieldCandidate,
        apply_decision,
        decide_abstract,
        decide_keywords,
        incumbent_candidate,
    )
    from bibr.paper_contents import CanonicalSection
    from bibr.structure.implicit_sections import select_abstract_span

    source = "native" if native else None
    abstract_incumbent = incumbent_candidate(paper_metadata, "abstract", source=source)
    keywords_incumbent = incumbent_candidate(paper_metadata, "keywords", source=source)
    if abstained:
        for decision in (
            decide_abstract(
                abstract_incumbent,
                fallback=None,
                explicitly_absent=False,
                printed_abstract=False,
                abstained=True,
            ),
            decide_keywords(keywords_incumbent, doc_info=None, section=None, abstained=True),
        ):
            apply_decision(paper_metadata, decision)
        return

    selection = select_abstract_span(contents, resolution) if resolution is not None else None

    extracted_abstract = (
        (abstract_incumbent.value or "").strip() if abstract_incumbent.veto is None else ""
    )
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
    fallback = None
    if selection is not None:
        fallback = FieldCandidate(
            "abstract",
            "abstract_section",
            selection.text,
            evidence_ids=tuple(selection.evidence_ids),
        )
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
            fallback = FieldCandidate(
                "abstract",
                "abstract_section",
                " ".join(
                    sent.text
                    for sent in contents.sentences
                    if sent.section_id in abstract_section_ids and not sent.is_display_formula
                ),
            )
    # The text the suspicion check reads: the extracted string, or the
    # fallback's text before the final strip.
    abstract_text = extracted_abstract
    if not abstract_text and (not explicit_absence or printed_abstract) and fallback is not None:
        abstract_text = fallback.value

    section_keywords = None
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
                section_keywords = FieldCandidate("keywords", "keywords_section", candidates)

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

    for decision in (
        decide_abstract(
            abstract_incumbent,
            fallback=fallback,
            explicitly_absent=explicit_absence,
            printed_abstract=printed_abstract,
            abstained=False,
        ),
        decide_keywords(
            keywords_incumbent,
            doc_info=doc_info_keywords,
            section=section_keywords,
            abstained=False,
        ),
    ):
        apply_decision(paper_metadata, decision)


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
    memory_mode: str | None = None,
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
    from bibr.extract.field_decisions import (
        apply_decision,
        decide_title,
        field_decisions_of,
        incumbent_candidate,
    )
    from bibr.extract.ref_extractor import _resolve_ref_strategies
    from bibr.paper import _merge_ocr_metadata, doc_info_candidates

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
                memory_mode=memory_mode,
            )
            metadata_ownership_scoped = bool(
                contents.preparsed_metadata is None
                and contents.front_matter_resolution is not None
                and (not no_llm or any(issue.blocking for issue in front_matter_issues))
            )
            metadata_abstained = bool(
                metadata_ownership_scoped
                and contents.front_matter_resolution.selected_block_id is None
            )
            native_metadata = contents.preparsed_metadata is not None
            # The OCR/doc-info metadata may fill empty fields only when no
            # front-matter record owns them.
            doc_info = (
                doc_info_candidates(ocr_metadata)
                if ocr_metadata and not metadata_ownership_scoped
                else {}
            )
            # The core extractor decides the authors and the fields only it
            # produces; decide them here when it did not run. The doc-info
            # authors are in hand before the author-name snapshot below, which
            # grounds named funding declarations, is taken.
            _decide_record_fields(
                paper_metadata, native=native_metadata, doc_info_authors=doc_info.get("author")
            )

            # The title: the extracted one, the ownership-scoped null-title
            # safety net (one safe selected-record title row, then the layout
            # title under the same filters), byline adjacency for multilingual
            # front matter (default off), the layout title and unknown-header
            # scan outside ownership scope, then the doc-info.
            title_decision = decide_title(
                incumbent_candidate(
                    paper_metadata, "title", source="native" if native_metadata else None
                ),
                resolution=contents.front_matter_resolution,
                detected_title=contents.detected_title,
                sections=contents.sections,
                journal=paper_metadata.journal,
                publisher=paper_metadata.publisher,
                scoped=metadata_ownership_scoped,
                abstained=metadata_abstained,
                prefer_byline_adjacent=bool(
                    getattr(
                        getattr(effective_settings, "pipeline", None),
                        "title_prefer_byline_adjacent",
                        False,
                    )
                ),
                doc_info=doc_info.get("title"),
            )
            apply_decision(paper_metadata, title_decision)
            metadata_issues.extend(title_decision.issues)

            # The OCR/doc-info DOI fills an empty DOI outside ownership scope.
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

            # The abstract and keywords are decided here (extracted values,
            # section and doc-info fallbacks) — the export layer serializes
            # them verbatim.
            _finalize_abstract_and_keywords(
                contents,
                paper_metadata,
                resolution=(
                    contents.front_matter_resolution if metadata_ownership_scoped else None
                ),
                validation_issue_sink=metadata_issues,
                native=native_metadata,
                doc_info_keywords=doc_info.get("keywords"),
                abstained=metadata_abstained,
            )

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
        # The reference list comes from the input's structured citations or
        # the configured parser.
        references_source=(
            "native" if contents.native_references is not None else str(parse_strategy or "llm")
        ),
    )
    paper.field_decisions = field_decisions_of(paper_metadata)
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
                    memory_mode=ctx.config.memory_mode,
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

        if ctx.config.memory_mode == "aggressive":
            # The NER reference parser (~1 GB) is a process-wide singleton
            # outside the layout/segmenter lifecycle: release it now so the
            # next chunk's OCR/LLM phases get the whole machine, mirroring
            # the layout/segmenter unloads in aggressive mode.
            ctx.resources.unload_ner_parser()

        logger.debug("Post-parse stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
