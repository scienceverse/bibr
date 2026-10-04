"""The real chain with the layer on, through ``LocalPipeline`` and a ``start_page``.

NativeTextStage builds the layer inside ``inspect_pdf``, ParseSegmentStage
attaches the post-OCR regions as blocks and hands it on as
``PaperContents.document``, PostParse reads it, and the balanced memory mode
frees it after PostParse. Only layout, segmentation, OCR and the LLM are
faked, as in ``tests/test_pipeline_smoke.py``.
"""

from __future__ import annotations

from bibr.config import Settings
from bibr.document import rebuild as rebuild_mod
from bibr.document import views
from tests.document import _pdfs
from tests.test_pipeline_smoke import _LINES, FakeLayout, FakeSegmenter, _fake_llm_client


class _QuietOcr:
    loaded = True

    async def recognize(self, image, prompt):
        return ""


def _two_pages() -> bytes:
    body = b"".join(_pdfs.text(line, x, y, size=size) for x, y, size, line in _LINES)
    cover = _pdfs.text("A cover page outside the processed range", 72.0, 700.0)
    return _pdfs.build_pdf([_pdfs.PageSpec(cover), _pdfs.PageSpec(body)])


async def test_the_local_pipeline_hands_the_layer_of_its_page_range_to_post_parse(
    tmp_path, monkeypatch
):
    import bibr.pipeline.stages.post_parse as post_parse_mod
    from bibr.local.pipeline import LocalPipeline
    from bibr.pipeline.context import PipelineContext
    from bibr.pipeline.resources import ResourceManager

    monkeypatch.setattr(Settings.pipeline, "document_layer", True)
    # No OCR bundle hit can skip NativeTextStage.
    monkeypatch.setattr(Settings.cache, "ocr", False)
    monkeypatch.setattr(Settings.ocr, "native_text_min_chars", 4)
    monkeypatch.setattr(Settings.ml, "section_classifier_model_id", None)
    seen: dict = {}

    def no_rebuild(*_args, **_kwargs):
        raise AssertionError("NativeTextStage builds the layer inline")

    real_attach = rebuild_mod.attach_blocks

    def attach_blocks(layer, ocr_regions):
        seen["regions"] = ocr_regions
        real_attach(layer, ocr_regions)

    real_post_parse = post_parse_mod.post_parse

    async def post_parse(*, contents, **kwargs):
        seen["layer"] = contents.document
        seen["summaries"] = list(contents.region_summaries)
        return await real_post_parse(contents=contents, **kwargs)

    real_free = PipelineContext.free_after_stage

    def free_after_stage(self, stage_name):
        real_free(self, stage_name)
        if stage_name == "extract":
            seen["freed"] = [
                (fs.doc_layer, fs.doc_layer_attempted, fs.contents.document)
                for fs in self.file_states
            ]

    monkeypatch.setattr(rebuild_mod, "rebuild_document_layer", no_rebuild)
    monkeypatch.setattr(rebuild_mod, "attach_blocks", attach_blocks)
    monkeypatch.setattr(post_parse_mod, "post_parse", post_parse)
    monkeypatch.setattr(PipelineContext, "free_after_stage", free_after_stage)
    pdf_path = tmp_path / "two_pages.pdf"
    pdf_path.write_bytes(_two_pages())

    pipeline = LocalPipeline(crossref=False, memory_mode="balanced", start_page=1)
    rm = ResourceManager(
        memory_mode=pipeline.memory_mode,
        ocr_backend=pipeline.ocr_backend,
        ocr_url=pipeline.ocr_url,
        ocr_model=pipeline.ocr_model,
        device=pipeline.device,
        settings=pipeline.settings,
        layout=FakeLayout(),
        segmenter=FakeSegmenter(),
    )
    pipeline._resources = rm
    rm._ocr = _QuietOcr()
    rm._llm_client = _fake_llm_client()
    try:
        result = await pipeline.process_file(pdf_path)
    finally:
        await pipeline.aclose()

    assert result["metadata"]["title"]
    layer = seen["layer"]
    assert layer is not None
    assert [page.index for page in layer.pages] == [1]
    assert layer.component_errors == {}
    page = layer.page(1)
    # The OCR stage pads the page before start_page; a region's index is its position.
    regions = seen["regions"]
    assert regions[0] == []
    assert [region.index for region in regions[1]] == list(range(len(regions[1])))
    assert [block.block_id for block in page.blocks] == [
        f"p1.r{position}" for position in range(len(regions[1]))
    ]
    summaries = seen["summaries"]
    assert summaries
    assert {summary.page for summary in summaries} == {2}
    for summary in summaries:
        block = views.block_for_region(layer, page_no=summary.page, region_index=summary.index)
        assert block is page.blocks[summary.index]
    native = [block for block in page.blocks if block.chosen == "native"]
    assert native
    for block in native:
        assert views.block_text(layer, block.block_id) == block.text["native"]
    # Balanced memory mode frees the layer after PostParse; it was built once.
    assert seen["freed"] == [(None, True, None)]
