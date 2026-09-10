import asyncio
from unittest import mock


async def test_post_parse_stage_caps_concurrency(monkeypatch):
    """Concurrency through PostParseStage must not exceed the configured cap."""
    from bibr.config import GlobalSettings
    from bibr.pipeline.stages.post_parse import PostParseStage

    settings = GlobalSettings()
    settings.pipeline.max_concurrent_post_parse = 2

    inflight = 0
    peak = 0

    async def fake_post_parse(**_kw):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.sleep(0.05)
        finally:
            inflight -= 1
        return mock.MagicMock()

    monkeypatch.setattr(
        "bibr.pipeline.stages.post_parse.post_parse",
        fake_post_parse,
    )

    files = []
    for i in range(8):
        fs = mock.MagicMock()
        fs.path = mock.MagicMock(name=f"f{i}.pdf")
        fs.path.name = f"f{i}.pdf"
        fs.contents = mock.MagicMock(layout_hints=None)
        fs.stage_times = {}
        fs.error = None
        fs.paper_id = None
        files.append(fs)

    ctx = mock.MagicMock()
    ctx.alive.return_value = files
    ctx.config = mock.MagicMock(no_llm=True)
    ctx.progress = mock.MagicMock()
    ctx.settings = settings

    stage = PostParseStage()
    await stage.run(ctx)
    assert peak <= 2, f"peak concurrency {peak} exceeded cap of 2"
