"""Pipeline lifecycle: per-request teardown vs explicit ``aclose``.

The serve worker reuses one ``ServePipeline`` instance across requests, so
``process_chunk`` must NOT shut down long-lived resources (vllm-mlx LLM
server, OCR engine) in a per-request ``finally``. The owner is responsible
for calling :meth:`Pipeline.aclose` exactly once at end of life.
"""

from __future__ import annotations

from pathlib import Path

from bibr.config import GlobalSettings
from bibr.pipeline.context import RunConfig
from bibr.pipeline.pipeline import Pipeline
from bibr.pipeline.progress import NullProgress
from bibr.pipeline.resources import ResourceManager
from bibr.pipeline.state import FileState


class _NoopStage:
    name = "noop"

    async def run(self, ctx):  # noqa: ARG002
        return


def _make_pipeline_with_fake_llm():
    settings = GlobalSettings()
    rm = ResourceManager(settings=settings)
    shutdown_calls = {"n": 0}

    class FakeServer:
        def shutdown(self):
            shutdown_calls["n"] += 1

    rm._llm_server = FakeServer()

    pl = Pipeline(stages=[_NoopStage()], resources=rm, config=RunConfig(), settings=settings)
    return pl, shutdown_calls


async def test_process_chunk_does_not_shutdown_llm_server():
    pl, shutdown_calls = _make_pipeline_with_fake_llm()
    fs1 = FileState(path=Path("x.pdf"))
    fs2 = FileState(path=Path("y.pdf"))

    await pl.process_chunk([fs1], progress=NullProgress(), config=None)
    await pl.process_chunk([fs2], progress=NullProgress(), config=None)
    assert shutdown_calls["n"] == 0, "per-request shutdown must not happen"


async def test_aclose_shuts_down_llm_server_once():
    pl, shutdown_calls = _make_pipeline_with_fake_llm()
    fs = FileState(path=Path("x.pdf"))

    await pl.process_chunk([fs], progress=NullProgress(), config=None)
    assert shutdown_calls["n"] == 0

    await pl.aclose()
    assert shutdown_calls["n"] == 1, "explicit aclose must shut down once"


async def test_aclose_shuts_down_ocr():
    settings = GlobalSettings()
    rm = ResourceManager(settings=settings)
    ocr_shutdowns = {"n": 0}

    class FakeOcr:
        loaded = True

        def shutdown(self):
            ocr_shutdowns["n"] += 1

    rm._ocr = FakeOcr()

    pl = Pipeline(stages=[], resources=rm, config=RunConfig(), settings=settings)

    await pl.aclose()
    assert ocr_shutdowns["n"] == 1


async def test_aclose_shuts_down_classifiers_before_llm_server():
    events = []
    settings = GlobalSettings()
    rm = ResourceManager(settings=settings)

    class FakeClassifiers:
        async def close(self):
            events.append("classifiers")

    class FakeServer:
        def shutdown(self):
            events.append("llm")

    rm._classifiers = FakeClassifiers()
    rm._llm_server = FakeServer()
    pl = Pipeline(stages=[], resources=rm, config=RunConfig(), settings=settings)

    await pl.aclose()
    assert events[:2] == ["classifiers", "llm"]


def test_release_llm_server_helper_is_safe_when_no_server():
    """The finalize callback must not raise when no LLM server was started."""
    import pytest

    pytest.importorskip("litserve")

    from bibr.serve.deployments.pipeline import _release_llm_server

    rm = ResourceManager()
    # No _llm_server attached — shutdown_llm_server early-returns.
    _release_llm_server(rm)  # must not raise


def test_release_llm_server_helper_swallows_errors():
    """Finalize callbacks must never propagate exceptions."""
    import pytest

    pytest.importorskip("litserve")

    from bibr.serve.deployments.pipeline import _release_llm_server

    class BoomResources:
        def shutdown_llm_server(self):
            raise RuntimeError("boom")

    # Must not raise — the helper swallows finalizer errors.
    _release_llm_server(BoomResources())


async def test_ocr_load_runs_off_event_loop():
    """The blocking constructor must not be invoked on the event loop thread."""
    import threading

    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager()
    main_thread = threading.get_ident()
    seen_threads: list[int] = []

    def slow_create():
        seen_threads.append(threading.get_ident())

        class FakeOcr:
            loaded = True

        return FakeOcr()

    rm._create_ocr_client = slow_create
    await rm.await_ocr()
    assert len(seen_threads) == 1
    assert seen_threads[0] != main_thread, (
        "OCR constructor must run on a worker thread, not the event loop"
    )


def test_finalizer_calls_release_on_gc():
    """weakref.finalize on BibrPipelineAPI must release the LLM server on GC."""
    import gc
    import weakref

    import pytest

    pytest.importorskip("litserve")

    from bibr.serve.deployments.pipeline import BibrPipelineAPI, _release_llm_server

    shutdown_calls = {"n": 0}

    class FakeServer:
        def shutdown(self):
            shutdown_calls["n"] += 1

    rm = ResourceManager()
    rm._llm_server = FakeServer()

    api = BibrPipelineAPI.__new__(BibrPipelineAPI)
    # Mirror the wiring `setup()` performs, without booting the full LitAPI.
    api._finalizer = weakref.finalize(api, _release_llm_server, rm)

    del api
    gc.collect()

    assert shutdown_calls["n"] == 1, "finalize callback must release the LLM server"


async def test_process_file_chains_original_exception():
    """ProcessingError raised by process_file must carry the originating
    exception via __cause__ so debugging tools and structured loggers can
    inspect the real failure, not just the stage's error string."""
    import pytest

    from bibr.exceptions import ProcessingError

    class FailingStage:
        name = "boom"

        async def run(self, ctx):
            for fs in ctx.alive():
                try:
                    raise RuntimeError("real cause")
                except RuntimeError as e:
                    fs.set_error("boom failed", code="boom", stage=self.name, exc=e)

    settings = GlobalSettings()
    rm = ResourceManager(settings=settings)
    pl = Pipeline(stages=[FailingStage()], resources=rm, config=RunConfig(), settings=settings)

    with pytest.raises(ProcessingError) as exc_info:
        await pl.process_file("dummy.pdf")

    assert exc_info.value.__cause__ is not None, "expected chained __cause__"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "real cause" in str(exc_info.value.__cause__)
