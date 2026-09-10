from unittest import mock


async def test_remote_ocr_shares_one_region_semaphore(monkeypatch):
    """Remote OCR runs files concurrently and bounds total in-flight regions
    via a single cross-file semaphore. Without that, N parallel files would
    multiply the effective region concurrency by N."""
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import ocr as ocr_mod

    semaphores_seen: list[int] = []

    async def fake_ocr_page_regions(*_a, **_kw):
        # The semaphore is the 6th positional arg (page_img, regions, idx, name, fn, sem).
        sem = _a[5]
        semaphores_seen.append(id(sem))
        return []

    monkeypatch.setattr(ocr_mod, "ocr_page_regions", fake_ocr_page_regions)

    files = []
    for fi in range(2):
        fs = mock.MagicMock()
        fs.path = mock.MagicMock(name=f"f{fi}.pdf")
        fs.path.name = f"f{fi}.pdf"
        fs.contents = None
        fs.ocr_regions = None
        fs.page_indices = [0, 1]
        fs.page_images = [mock.MagicMock(), mock.MagicMock()]
        fs.layout_results = [[], []]
        fs.free_pre_ocr = mock.MagicMock()
        fs.set_error = mock.MagicMock()
        files.append(fs)

    ctx = mock.MagicMock()
    ctx.alive.return_value = files
    ctx.config = mock.MagicMock(include_figures=False)
    ctx.settings = snapshot_settings()

    stage = ocr_mod.OcrStage()
    await stage._run_remote(ctx, mock.AsyncMock())

    # Every region — across files and pages — uses the same semaphore.
    assert len(set(semaphores_seen)) == 1, semaphores_seen


async def test_remote_ocr_does_not_call_free_pre_ocr_directly(monkeypatch):
    """_run_remote must NOT call fs.free_pre_ocr() — freeing is centralized
    in PipelineContext.free_after_stage, called by the pipeline orchestrator
    after the OCR stage completes (see bibr/pipeline/pipeline.py)."""
    from bibr.config import snapshot_settings
    from bibr.pipeline.stages import ocr as ocr_mod

    async def fake_ocr_page_regions(*_a, **_kw):
        return []

    monkeypatch.setattr(ocr_mod, "ocr_page_regions", fake_ocr_page_regions)
    monkeypatch.setattr(
        ocr_mod,
        "_postprocess_ocr_regions",
        lambda x, *_args, **_kwargs: x,
    )

    files = []
    for fi in range(3):
        fs = mock.MagicMock()
        fs.path = mock.MagicMock()
        fs.path.name = f"f{fi}.pdf"
        fs.contents = None
        fs.ocr_regions = None
        fs.page_indices = [0]
        fs.page_images = [mock.MagicMock()]
        fs.layout_results = [[]]
        fs.set_error = mock.MagicMock()
        files.append(fs)

    ctx = mock.MagicMock()
    ctx.alive.return_value = files
    ctx.config = mock.MagicMock(include_figures=False)
    ctx.settings = snapshot_settings()

    stage = ocr_mod.OcrStage()
    await stage._run_remote(ctx, mock.AsyncMock())
    for fs in files:
        fs.free_pre_ocr.assert_not_called()
