"""Metadata Extractor for scientific papers — orchestration facade.

``MetadataExtractor`` composes the three collaborators that used to live in
this one module and keeps the public entry point (``extract_all_metadata``)
plus back-compat access to their internals:

* :class:`bibr.extract.ref_locator.RefLocator` — WHERE the front matter and
  reference list live in ``sentences_df``.
* :class:`bibr.extract.core_metadata.CoreMetadataExtractor` — title/authors/
  DOI/abstract/keywords via the LLM, with the post-LLM guards.
* :class:`bibr.extract.ref_extractor.ReferenceExtractor` — reference
  segmentation (geom→LLM→CRF cascade) and parsing (LLM or NER, via
  ``REF_PARSE_STRATEGIES``).

Training capture lives in :mod:`bibr.extract.training_capture`. The
re-exports below keep ``from bibr.extract.extractor import X`` working for
every name this module historically defined; new code should import from the
owning module directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from bibr.clients.llm import LLMClient
from bibr.clients.llm_protocol import LlmClient
from bibr.exceptions import BibrError, ProcessingError
from bibr.extract.core_metadata import (  # noqa: F401 — re-exported for back-compat
    _CORRECTION_NOTICE_TITLE_RE,
    EMPTY_AUTHORS_WARNING_PREFIX,
    CoreMetadataExtractor,
)
from bibr.extract.ref_extractor import (  # noqa: F401 — re-exported for back-compat
    GEOM_CASCADE_WARNING_PREFIX,
    MARKER_SPLIT_RECOVERY_PREFIX,
    MERGE_SPLIT_WARNING_PREFIX,
    REF_EXTRACTION_ERROR_PREFIX,
    REF_PARSE_STRATEGIES,
    REF_SEG_HARD_FAILURE_PREFIX,
    SEG_FALLBACK_WARNING_PREFIX,
    IncompleteOutputException,
    ReferenceExtractor,
    _backfill_issue,
    _build_ref_text,
    _chunk,
    _clean_bib_field,
    _expand_compact_last_page,
    _finalize_reference_fields,
    _get_geom_segmenter,
    _get_ner_parser,
    _get_ner_segmenter,
    _infer_bibtype,
    _is_degenerate_ref_failure,
    _is_in_press,
    _map_bib_text_ids,
    _marker_split_refs,
    _normalize_vol_issue,
    _parse_refs_via_llm,
    _parse_refs_via_ner,
    _parse_year,
    _rescue_authors_from_segment,
    _rescue_doi_from_segment,
    _resolve_ref_strategies,
    _resolve_repeated_authors,
    _sequence_references,
    _split_vol_issue,
    _strip_enum_markers,
)
from bibr.extract.ref_locator import (  # noqa: F401 — re-exported for back-compat
    _ENTRY_NUMBERING_RE,
    _REF_HEADER_RE,
    RefLocator,
    _looks_like_complete_entry_start,
)
from bibr.paper import PaperMetadata, PaperReference
from bibr.paper_contents import PaperContents
from bibr.validation import ValidationIssue

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.extract.front_matter import FrontMatterResolution
    from bibr.pipeline.classifier_resources import ClassifierResources

logger = logging.getLogger(__name__)


async def _cancel_and_await(*tasks: asyncio.Task) -> None:
    """Cancel unfinished sibling tasks and always retrieve their results."""

    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _await_core_and_reference_tasks(
    core_task: asyncio.Task[None],
    ref_task: asyncio.Task[list[PaperReference]],
) -> list[PaperReference] | Exception:
    """Await concurrent extraction while making cancellation fail promptly."""

    try:
        done, _ = await asyncio.wait((core_task, ref_task), return_when=asyncio.FIRST_COMPLETED)
        if ref_task in done:
            if ref_task.cancelled():
                await _cancel_and_await(core_task)
                raise asyncio.CancelledError
            ref_error = ref_task.exception()
            if isinstance(ref_error, ProcessingError):
                await _cancel_and_await(core_task)
                raise ref_error
            if ref_error is not None and not isinstance(ref_error, Exception):
                await _cancel_and_await(core_task)
                raise ref_error

        try:
            await core_task
        except BaseException:
            await _cancel_and_await(ref_task)
            raise

        try:
            return await ref_task
        except asyncio.CancelledError:
            raise
        except ProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 — caller marks references incomplete
            return exc
    except asyncio.CancelledError:
        await _cancel_and_await(core_task, ref_task)
        raise


class MetadataExtractor:
    """
    Extracts metadata from scientific papers.

    Uses section classification and LLM-based extraction to identify
    title, authors, DOI, keywords, and references from paper content.

    The heavy lifting is delegated to :class:`RefLocator`,
    :class:`CoreMetadataExtractor` and :class:`ReferenceExtractor`. A few
    ``_``-prefixed methods below remain as thin delegates because external
    callers (``bibr/pipeline/stages/post_parse.py``) still reach them on the
    facade rather than on the owning collaborator.
    """

    # Priority order to find the "split" point in the paper
    CUTOFF_PRIORITY = RefLocator.CUTOFF_PRIORITY

    def __init__(
        self,
        contents: PaperContents,
        file_hash: str = "unknown",
        llm_client: LlmClient | None = None,
        ref_seg_strategy: str | None = None,
        ref_parse_strategy: str | None = None,
        settings: GlobalSettings | None = None,
        classifier_resources: ClassifierResources | None = None,
        front_matter_resolution: FrontMatterResolution | None = None,
    ):
        """
        Initialize the metadata extractor.

        Args:
            contents: The parsed paper contents.
            file_hash: SHA256 hash of the source file for caching.
            llm_client: Optional LLM client instance. If not provided,
                a default client will be created.
            ref_seg_strategy: Per-run segmentation strategy override
                (``None`` = resolve from Settings).
            ref_parse_strategy: Per-run parse strategy override
                (``None`` = resolve from Settings).
        """
        from bibr.config import snapshot_settings

        self.contents = contents
        self.sentences_df = contents.sentences_df
        self._settings = settings if settings is not None else snapshot_settings()
        self._llm_client = llm_client or LLMClient(settings=self._settings)
        self.metadata: PaperMetadata | None = None
        self.file_hash = file_hash
        self._ref_seg_strategy = ref_seg_strategy
        self._ref_parse_strategy = ref_parse_strategy
        self.validation_issues: list[ValidationIssue] = []
        self.locator = RefLocator(contents, settings=self._settings)
        self.refs = ReferenceExtractor(
            contents,
            file_hash=file_hash,
            llm_client=self.llm_client,
            seg_strategy=ref_seg_strategy,
            parse_strategy=ref_parse_strategy,
            settings=self._settings,
        )
        core_kwargs = {}
        if settings is not None:
            core_kwargs["settings"] = self._settings
        if classifier_resources is not None:
            core_kwargs["classifier_resources"] = classifier_resources
        self.core = CoreMetadataExtractor(
            contents,
            file_hash=file_hash,
            llm_client=self.llm_client,
            locator=self.locator,
            front_matter_resolution=front_matter_resolution,
            **core_kwargs,
        )
        # Back-compatible facade handle. Core owns construction because only it
        # knows whether the harvester must be scoped to selected text IDs.
        self._email_harvester = self.core._email_harvester

    @property
    def llm_client(self) -> LlmClient:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, value: LlmClient) -> None:
        # Post-construction reassignment (a long-standing test/eval pattern)
        # must reach the collaborators, which captured the client at __init__.
        self._llm_client = value
        if hasattr(self, "refs"):
            self.refs.llm_client = value
        if hasattr(self, "core"):
            self.core.llm_client = value

    async def extract_all_metadata(self) -> PaperMetadata:
        """
        Extract all metadata from the paper.

        Runs core metadata extraction concurrently with LLM-based reference
        extraction.

        Returns:
            PaperMetadata object containing extracted information.
        """
        total_t0 = time.monotonic()
        self.validation_issues.clear()
        ref_issues = getattr(self.refs, "validation_issues", None)
        if isinstance(ref_issues, list):
            ref_issues.clear()

        if self.sentences_df.empty or "text" not in self.sentences_df.columns:
            logger.warning("No sentences in document (OCR may have failed). Skipping extraction.")
            self.metadata = PaperMetadata(doi="", title="", keywords=[], authors=[])
            return self.metadata

        # Collect reference rows — skipped entirely in no-references mode
        # (--refs off): no segmentation, no parsing, references stay [].
        _, parse_strategy = _resolve_ref_strategies(
            self._ref_seg_strategy,
            self._ref_parse_strategy,
            settings=self._settings,
        )
        ref_df = None
        ref_collection_error: Exception | None = None
        if parse_strategy == "off":
            logger.info("Reference extraction disabled (refs=off)")
        else:
            try:
                ref_df = self._collect_reference_rows()
            except ValueError as e:
                logger.warning(f"Reference section not found: {e}")
            except Exception as e:  # noqa: BLE001 — preserve core, mark incomplete below
                ref_collection_error = e

        if ref_collection_error is not None:
            t0 = time.monotonic()
            await self.extract_core_metadata()
            core_time = time.monotonic() - t0
            refs_time = 0.0

            from bibr.validation import mark_references_incomplete

            assert self.metadata is not None  # noqa: S101 — core extraction succeeded
            mark_references_incomplete(self.metadata, ref_collection_error)
            logger.error(
                "Reference row collection failed after core metadata succeeded: %s",
                self.metadata._references_incomplete_diagnostic,
            )
        elif ref_df is not None and not ref_df.empty:
            # Run core metadata + LLM references concurrently
            t0 = time.monotonic()
            core_task = asyncio.create_task(self.extract_core_metadata())
            ref_task = asyncio.create_task(self._extract_references(ref_df))
            ref_result = await _await_core_and_reference_tasks(core_task, ref_task)
            core_time = time.monotonic() - t0
            refs_time = core_time  # concurrent, so same wall clock

            if isinstance(ref_result, ProcessingError):
                raise ref_result
            if isinstance(ref_result, Exception):
                from bibr.validation import mark_references_incomplete

                assert self.metadata is not None  # noqa: S101 — core task succeeded
                mark_references_incomplete(self.metadata, ref_result)
                # Keep main's stable processing warning for unexpected local
                # runtime failures while using the durable incomplete marker for
                # every recoverable reference-extraction exception.
                if not isinstance(ref_result, BibrError):
                    self._record_ref_extraction_failure(ref_result)
                logger.error(
                    "Reference extraction failed after core metadata succeeded: %s",
                    self.metadata._references_incomplete_diagnostic,
                )
            elif self.metadata is not None:
                self.metadata.references = ref_result
            else:
                logger.error(
                    f"Core metadata missing after extraction — "
                    f"dropping {len(ref_result)} extracted references"
                )
        else:
            # No reference section — just extract core metadata
            t0 = time.monotonic()
            await self.extract_core_metadata()
            core_time = time.monotonic() - t0
            refs_time = 0.0

        total_time = time.monotonic() - total_t0
        for issue in getattr(self.refs, "validation_issues", ()):
            if issue not in self.validation_issues:
                self.validation_issues.append(issue)
        logger.info(
            f"Metadata extraction timing: core_metadata={core_time:.2f}s, "
            f"references={refs_time:.2f}s, total={total_time:.2f}s"
        )

        assert self.metadata is not None  # noqa: S101 — extract_core_metadata sets it
        return self.metadata

    async def extract_core_metadata(self) -> None:
        """Extract core metadata (title/authors/DOI/…) and store it on
        ``self.metadata``."""
        self.metadata = await self.core.extract()
        self.validation_issues = list(self.core.validation_issues)

    # ------------------------------------------------------------------
    # Load-bearing delegates — reached directly on the facade by
    # bibr/pipeline/stages/post_parse.py, so these stay even though the
    # rest of the back-compat surface was removed.
    # ------------------------------------------------------------------

    def _collect_reference_rows(self):
        return self.locator.collect_reference_rows()

    async def _extract_references(self, ref_df) -> list[PaperReference]:
        return await self.refs.extract(ref_df)

    def _record_ref_extraction_failure(self, exc: BaseException) -> None:
        """Surface a swallowed reference-extraction exception on the export.

        ``extract_all_metadata`` keeps core metadata when reference extraction
        raises a non-BibrError (resilience: partial results beat none). But a
        total reference wipeout — the CUDA OOM the local NER parser hit on a
        VRAM-starved serve is the observed case — must not be invisible: records
        a stable, greppable HIGH-severity warning naming the exception type so
        monitoring/eval catches it instead of only the downstream
        VAL_REF_COUNT_MISMATCH symptom.
        """
        detail = " ".join(str(exc).split())[:200]
        warning = f"{REF_EXTRACTION_ERROR_PREFIX}: {type(exc).__name__}: {detail}"
        # Real PaperContents carries a list; be defensive against test doubles /
        # a cleared attribute so the warning is never lost.
        warnings = getattr(self.contents, "processing_warnings", None)
        if not isinstance(warnings, list):
            warnings = []
            self.contents.processing_warnings = warnings
        warnings.append(warning)

    _should_suppress_commentary_abstract = staticmethod(
        CoreMetadataExtractor._should_suppress_commentary_abstract
    )
