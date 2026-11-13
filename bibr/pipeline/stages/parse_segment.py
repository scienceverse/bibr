"""ParseSegmentStage — PDFParser + wtpsplit sentence segmentation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from bibr.exceptions import ModelOverrideError
from bibr.processing_warnings import ProcessingWarning, WarningCode

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def _parse_pdf(ocr_regions, outline=None, *, settings=None, first_page_index=0):
    """Build PDFParser and run parse() — sync, CPU-heavy.

    ``outline`` (PDF bookmarks) is threaded through only when present, so the
    default (outline-off) construction stays ``PDFParser(ocr_regions)``.
    ``first_page_index`` tells the parser which absolute page the slice starts
    at, so front-matter heuristics keep firing under ``--pages``/``start_page``.
    """
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(
        ocr_regions,
        outline=outline,
        settings=settings,
        first_page_index=first_page_index,
    )
    contents = parser.parse()
    return parser, contents


async def _document_layer(fs, settings, *, start_page, end_page):
    """The file's document layer with its regions as blocks; never raise."""
    try:
        from bibr.document.rebuild import ensure_document_layer

        return await asyncio.to_thread(
            ensure_document_layer, fs, settings, start_page=start_page, end_page=end_page
        )
    except Exception:  # noqa: BLE001 - the internal layer must never fail parsing
        logger.warning("Document layer failed; continuing without it", exc_info=True)
        return None


def paper_key(fs) -> str:
    """The name a paper's model override failures are counted under."""
    return str(getattr(fs, "file_hash", None) or getattr(fs, "paper_id", None) or fs.path)


def _note_override_failure(settings, env: str, exc: BaseException, warnings, paper) -> None:
    """Record an overridden model's error on the paper; several in a row stop the run."""
    if settings is None:
        return
    from bibr.utils.model_overrides import note_inference_failure

    warning = note_inference_failure(settings, env, exc, paper=paper)
    if warning is not None and warnings is not None:
        warnings.append(warning)


def _note_override_ran(settings, env: str, paper) -> None:
    if settings is not None:
        from bibr.utils.model_overrides import note_inference_success

        note_inference_success(settings, env, paper=paper)


async def _predict_front_roles(
    rm, ocr_regions, *, first_page_index: int, settings=None, warnings=None, paper=None
):
    """Run the optional front-role classifier off the event loop.

    Never raises, except for an overridden classifier that cannot be loaded, or
    that raised on several papers in a row. An overridden classifier's error on
    this paper adds a warning to ``warnings`` (``settings`` are the run's).
    Loading it is off the loop too: the first paper reads the bundle (a
    download, on a first run).
    """
    from bibr.extract.front_role import FrontRolePredictions

    ensure = getattr(rm, "ensure_front_role", None)
    if ensure is None:
        return None
    try:
        classifier = await asyncio.to_thread(ensure)
        if classifier is None:
            return None
        predictions = await asyncio.to_thread(
            classifier.predict_pages, ocr_regions, first_page_index=first_page_index
        )
    except ModelOverrideError:
        raise  # an overridden model that cannot load (bibr.utils.model_overrides)
    except Exception as exc:  # noqa: BLE001 - an optional prior must never fail parsing
        logger.warning("Front-role classification failed; continuing without it", exc_info=True)
        _note_override_failure(settings, "ML_FRONT_ROLE_MODEL_ID", exc, warnings, paper)
        return None
    _note_override_ran(settings, "ML_FRONT_ROLE_MODEL_ID", paper)
    return predictions if isinstance(predictions, FrontRolePredictions) else None


async def _tag_front_matter(
    rm, ocr_regions, *, first_page_index: int, settings=None, warnings=None, paper=None
):
    """Run the optional front-matter tagger off the event loop.

    Never raises, except for an overridden tagger that cannot be loaded, or that
    raised on several papers in a row; an overridden tagger's error on this
    paper adds a warning to ``warnings``.

    ``settings`` are the run's: a fast run turns the tagger on for itself, so
    the resource manager's own settings would leave it off. Loading the tagger
    is off the loop too: the first paper reads the bundle (a download, on a
    first run) and opens the session.
    """
    ensure = getattr(rm, "ensure_front_tagger", None)
    if ensure is None:
        return None
    try:
        tagger = await asyncio.to_thread(ensure, settings)
        if tagger is None:
            return None
        # The run decides: a fast run reads the front matter with the local rules,
        # a full run leaves those readings to the LLM.
        rules = bool(getattr(settings, "FRONT_TAGGER_RULES", False))
        from bibr.extract.front_role import build_front_regions
        from bibr.front_tagger.runtime import tag_front_matter

        def run():
            regions = build_front_regions(ocr_regions, first_page_index=first_page_index)
            return tag_front_matter(tagger, regions, first_page_index=first_page_index, rules=rules)

        tagged = await asyncio.to_thread(run)
    except ModelOverrideError:
        raise  # an overridden model that cannot load (bibr.utils.model_overrides)
    except Exception as exc:  # noqa: BLE001 - an optional reading must never fail parsing
        logger.warning("Front-matter tagging failed; continuing without it", exc_info=True)
        _note_override_failure(settings, "ML_FRONT_TAGGER_MODEL_ID", exc, warnings, paper)
        return None
    _note_override_ran(settings, "ML_FRONT_TAGGER_MODEL_ID", paper)
    return tagged


def _front_page_was_read(fs, first_page_index: int) -> bool:
    """Whether OCR read the paper's first page (a scan or a broken text layer).

    The front-matter tagger learned from born-digital pages and misses most
    abstracts on scans, so a run that has an LLM leaves those to it; a fast run
    tags them anyway and notes it. A page with no class (no inspection) counts
    as born-digital, as everywhere else.
    """
    try:
        from bibr.scan.page_kind import PageKind, ensure_page_kinds

        kind = ensure_page_kinds(fs).get(first_page_index)
    except Exception:  # noqa: BLE001 - a missing class is not a scan
        return False
    return kind in (PageKind.SCAN, PageKind.BROKEN_TEXT_LAYER)


def _split_scanned_articles(fs) -> None:
    """Drop neighbouring articles' regions from the file's scanned pages; never raise."""
    try:
        from bibr.scan.articles import describe, split_articles
        from bibr.scan.page_kind import ensure_page_kinds

        split = split_articles(fs.ocr_regions, ensure_page_kinds(fs))
    except Exception:  # noqa: BLE001 - the split is an addition, never a dependency
        logger.warning(
            "Article split failed for %s; parsing every region", fs.path.name, exc_info=True
        )
        return
    if not split.changed:
        return
    fs.ocr_regions = split.pages
    message = describe(split)
    logger.info("%s: %s", fs.path.name, message)
    fs.warnings.append(ProcessingWarning(WarningCode.SCAN_ARTICLE_SPLIT, message))


class ParseSegmentStage:
    name = "parse"
    # FileState fields consumed / populated (see validate_stage_contracts).
    requires = ("ocr_regions",)
    produces = ("contents",)

    async def run(self, ctx: PipelineContext) -> None:
        ctx.progress.stage_start(self.name)
        t0 = time.monotonic()
        rm = ctx.resources
        try:
            rm.ensure_segmenter()
        except Exception as exc:  # noqa: BLE001 - every file needs it; record per file
            for fs in ctx.alive():
                fs.set_error(
                    f"Sentence segmenter failed to load: {exc}",
                    code="parse_failed",
                    stage=self.name,
                    exc=exc,
                )
            logger.warning("Sentence segmenter failed to load", exc_info=True)
            ctx.progress.stage_end(self.name)
            return

        async def _process_one(fs):
            try:
                fs_t0 = time.monotonic()
                native_parser = getattr(fs, "_native_parser", None)
                if native_parser is not None:
                    parser = native_parser
                    contents = fs.contents
                else:
                    if ctx.settings.pipeline.scan_article_split is True and fs.ocr_regions:
                        _split_scanned_articles(fs)
                    # Offload sync CPU-heavy parse to a thread so concurrent
                    # files can interleave on the event loop and the thread
                    # pool can fan out across cores. The PDF outline (when
                    # harvested) rides along as an authoritative heading-level
                    # signal; when absent the call stays single-arg so the
                    # default path is unchanged.
                    outline = getattr(fs, "pdf_outline", None)
                    parse_args = (fs.ocr_regions,) if outline is None else (fs.ocr_regions, outline)
                    parser, contents = await asyncio.to_thread(
                        _parse_pdf,
                        *parse_args,
                        settings=ctx.settings,
                        first_page_index=ctx.config.start_page or 0,
                    )
                    fs.contents = contents

                # Hand captured reference-line geometry to the extract stage.
                if contents is not None:
                    contents.ref_line_geometry = getattr(fs, "ref_line_geometry", None)
                    contents.ref_page_lines = getattr(fs, "ref_page_lines", None)
                    contents.pdf_uri_links = getattr(fs, "pdf_uri_links", None)

                # The document layer, with this run's regions as blocks
                # (rebuilt from the PDF when an OCR-bundle hit skipped
                # NativeTextStage). Internal only; it never fails the paper.
                # "is True" keeps it off under Mock settings in tests.
                if (
                    contents is not None
                    and native_parser is None
                    and ctx.settings.pipeline.document_layer is True
                ):
                    contents.document = await _document_layer(
                        fs,
                        ctx.settings,
                        start_page=ctx.config.start_page,
                        end_page=ctx.config.end_page,
                    )

                # Score every OCR region's front-matter role while the raw
                # regions are still resident (they are freed after this
                # stage). Evidence only: front_matter.py decides what to do
                # with it, and a missing model leaves the heuristics alone.
                if contents is not None and native_parser is None and fs.ocr_regions:
                    predictions = await _predict_front_roles(
                        rm,
                        fs.ocr_regions,
                        first_page_index=ctx.config.start_page or 0,
                        settings=ctx.settings,
                        warnings=fs.warnings,
                        paper=paper_key(fs),
                    )
                    if predictions is not None:
                        contents.front_role_predictions = predictions
                    first_page = ctx.config.start_page or 0
                    scanned = _front_page_was_read(fs, first_page)
                    # With the tagger alone there is no LLM to leave a scan to.
                    tagger_alone = ctx.settings.ml.front_tagger_mode == "only"
                    if scanned and not tagger_alone:
                        logger.debug(
                            "%s: front page went through OCR; the front-matter tagger skips it",
                            fs.path.name,
                        )
                    else:
                        tagged = await _tag_front_matter(
                            rm,
                            fs.ocr_regions,
                            first_page_index=first_page,
                            settings=ctx.settings,
                            warnings=fs.warnings,
                            paper=paper_key(fs),
                        )
                        if tagged is not None and scanned:
                            tagged = replace(
                                tagged, problems=[*tagged.problems, "scanned_front_page"]
                            )
                        contents.front_tagger_result = tagged

                assembler = parser.assembler
                if len(assembler):
                    texts = assembler.segmentable_texts
                    if texts:
                        # segment_batch serializes access to the shared,
                        # non-thread-safe segmenter internally.
                        all_segments = await rm.segmenter.segment_batch(texts)
                    else:
                        all_segments = []
                    await asyncio.to_thread(parser.apply_segmentation, contents, all_segments)
                await asyncio.to_thread(parser.create_content_sections, contents)

                fs.stage_times["parse_segment"] = time.monotonic() - fs_t0
            except Exception as e:  # noqa: BLE001
                fs.set_error(
                    f"Parse/segment failed: {e}",
                    code="parse_failed",
                    stage=self.name,
                    exc=e,
                )
                logger.warning("Parse failed for %s", fs.path.name, exc_info=True)

        try:
            await asyncio.gather(*(_process_one(fs) for fs in ctx.alive()))
        finally:
            if ctx.config.memory_mode == "aggressive":
                rm.unload_segmenter()

        logger.debug("Parse+segment stage: %.1fs", time.monotonic() - t0)
        ctx.progress.stage_end(self.name)
