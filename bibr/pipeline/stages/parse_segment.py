"""ParseSegmentStage — PDFParser + wtpsplit sentence segmentation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from bibr.exceptions import ModelOverrideError
from bibr.processing_warnings import ProcessingWarning, WarningCode
from bibr.structure.table_links import add_table_links, repair_links_from_targets
from bibr.utils.link_targets import LinkTargets

if TYPE_CHECKING:
    from bibr.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def _parse_pdf(ocr_regions, outline=None, *, settings=None, first_page_index=0, link_targets=None):
    """Build PDFParser and run parse() — sync, CPU-heavy.

    ``outline`` (PDF bookmarks) is threaded through only when present, so the
    default (outline-off) construction stays ``PDFParser(ocr_regions)``.
    ``first_page_index`` tells the parser which absolute page the slice starts
    at, so front-matter heuristics keep firing under ``--pages``/``start_page``.
    ``link_targets`` (the targets of the PDF's link annotations) is passed only
    when the PDF has any.
    """
    from bibr.structure.pdf_parser import PDFParser

    parser = PDFParser(
        ocr_regions,
        outline=outline,
        settings=settings,
        first_page_index=first_page_index,
        **({"link_targets": link_targets} if link_targets else {}),
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
    rm,
    ocr_regions,
    *,
    first_page_index: int,
    settings=None,
    warnings=None,
    paper=None,
    front_regions=None,
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
        kwargs = {"first_page_index": first_page_index}
        if front_regions is not None:
            kwargs["regions"] = front_regions
        try:
            predictions = await asyncio.to_thread(classifier.predict_pages, ocr_regions, **kwargs)
        except TypeError:
            if "regions" not in kwargs:
                raise
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
    rm,
    ocr_regions,
    *,
    first_page_index: int,
    settings=None,
    warnings=None,
    paper=None,
    front_regions=None,
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
        min_confidence = getattr(getattr(settings, "ml", None), "front_tagger_min_confidence", None)
        from bibr.extract.front_role import build_front_regions
        from bibr.front_tagger.runtime import tag_front_matter

        def run():
            regions = (
                front_regions
                if front_regions is not None
                else build_front_regions(ocr_regions, first_page_index=first_page_index)
            )
            return tag_front_matter(
                tagger,
                regions,
                first_page_index=first_page_index,
                rules=rules,
                min_confidence=min_confidence,
            )

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
                first_page = ctx.config.start_page or 0
                doc_layer_coro = None
                roles_coro = None
                tagger_coro = None
                role_warnings: list[ProcessingWarning] = []
                tagger_warnings: list[ProcessingWarning] = []
                scanned = False

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
                    # The PDF's link targets ride along only when it has any.
                    links = LinkTargets.from_uri_dicts(getattr(fs, "pdf_uri_links", None))
                    parse_coro = asyncio.to_thread(
                        _parse_pdf,
                        *parse_args,
                        settings=ctx.settings,
                        first_page_index=first_page,
                        **({"link_targets": links} if links else {}),
                    )

                    # Document layer, front-role classifier, and front-matter
                    # tagger depend only on fs.ocr_regions (not on _parse_pdf),
                    # so run them concurrently with _parse_pdf and share the
                    # built FrontRegion list between role and tagger models.
                    if ctx.settings.pipeline.document_layer is True:
                        doc_layer_coro = _document_layer(
                            fs,
                            ctx.settings,
                            start_page=ctx.config.start_page,
                            end_page=ctx.config.end_page,
                        )

                    if fs.ocr_regions:
                        from bibr.extract.front_role import build_front_regions

                        front_regions = build_front_regions(
                            fs.ocr_regions, first_page_index=first_page
                        )
                        pkey = paper_key(fs)
                        roles_coro = _predict_front_roles(
                            rm,
                            fs.ocr_regions,
                            first_page_index=first_page,
                            settings=ctx.settings,
                            warnings=role_warnings,
                            paper=pkey,
                            front_regions=front_regions,
                        )
                        scanned = _front_page_was_read(fs, first_page)
                        # With the tagger alone there is no LLM to leave a scan to.
                        tagger_alone = ctx.settings.ml.front_tagger_mode == "only"
                        if scanned and not tagger_alone:
                            logger.debug(
                                "%s: front page went through OCR; the front-matter tagger skips it",
                                fs.path.name,
                            )
                        else:
                            tagger_coro = _tag_front_matter(
                                rm,
                                fs.ocr_regions,
                                first_page_index=first_page,
                                settings=ctx.settings,
                                warnings=tagger_warnings,
                                paper=pkey,
                                front_regions=front_regions,
                            )

                    async def _noop_none():
                        return None

                    (parser, contents), doc_layer, predictions, tagged = await asyncio.gather(
                        parse_coro,
                        doc_layer_coro if doc_layer_coro is not None else _noop_none(),
                        roles_coro if roles_coro is not None else _noop_none(),
                        tagger_coro if tagger_coro is not None else _noop_none(),
                    )
                    fs.contents = contents
                    if role_warnings:
                        fs.warnings.extend(role_warnings)
                    if tagger_warnings:
                        fs.warnings.extend(tagger_warnings)

                # Hand captured reference-line geometry to the extract stage.
                if contents is not None:
                    contents.ref_line_geometry = getattr(fs, "ref_line_geometry", None)
                    contents.ref_page_lines = getattr(fs, "ref_page_lines", None)
                    contents.pdf_uri_links = getattr(fs, "pdf_uri_links", None)
                    if native_parser is None:
                        if doc_layer_coro is not None:
                            contents.document = doc_layer
                        if predictions is not None:
                            contents.front_role_predictions = predictions
                        if tagger_coro is not None:
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

                if contents is not None:
                    # Links printed in table cells, which are in no sentence.
                    # HTML, DOCX and JATS input reads only a URL printed in
                    # full as a link from text, here as in its sentences.
                    # Both run after segmentation: sentence links exist only
                    # from there on.
                    add_table_links(contents, bare=native_parser is None)
                    repair_links_from_targets(contents)

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
