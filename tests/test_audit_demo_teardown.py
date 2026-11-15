"""``bibr demo`` closes its pipeline once Gradio stops, so managed servers do not outlive it."""

import sys

import pytest

gr = pytest.importorskip("gradio")  # demo is gated behind gradio
if not hasattr(gr, "Blocks"):
    pytest.skip("gradio not fully installed", allow_module_level=True)

import bibr.config
import bibr.demo.server as server


@pytest.fixture
def closed(monkeypatch):
    closed: list[str] = []

    class _Pipeline:
        def __init__(self, **_kw):
            pass

        async def aclose(self) -> None:
            closed.append("pipeline")

    monkeypatch.setattr("bibr.local.pipeline.LocalPipeline", _Pipeline)
    section = bibr.config.Settings.cache
    original = section.ocr
    was_explicit = "ocr" in section.model_fields_set
    yield closed
    section.ocr = original
    if not was_explicit:
        section.model_fields_set.discard("ocr")


@pytest.mark.parametrize("stopped_by", [None, KeyboardInterrupt])
def test_bibr_demo_closes_its_pipeline_when_gradio_stops(monkeypatch, closed, stopped_by):
    def launch(self, **_kw):
        # Gradio returns from launch() after Ctrl-C; a Ctrl-C while it starts raises.
        if stopped_by is not None:
            raise stopped_by

    monkeypatch.setattr(gr.Blocks, "launch", launch)
    monkeypatch.setattr(sys, "argv", ["bibr demo", "--ocr", "glm-mlx"])
    monkeypatch.delenv("DEMO_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("DEMO_MAX_FILE_SIZE_MB", raising=False)

    if stopped_by is None:
        server.main()
    else:
        with pytest.raises(stopped_by):
            server.main()

    assert closed == ["pipeline"]
