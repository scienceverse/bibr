"""Tests for ResourceManager.llm_client property and close_llm_client."""

import asyncio
import time

import pytest


def test_resource_manager_returns_singleton_llm_client():
    from bibr.pipeline.resources import ResourceManager

    rm = ResourceManager()
    a = rm.llm_client
    b = rm.llm_client
    assert a is b


async def test_resource_manager_close_llm_client(monkeypatch):
    from bibr.pipeline.resources import ResourceManager

    closed = []

    class FakeClient:
        async def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "bibr.pipeline.resources._make_llm_client",
        lambda: FakeClient(),
    )
    rm = ResourceManager()
    _ = rm.llm_client
    await rm.close_llm_client()
    assert closed == [True]
    # Idempotent: calling again must not raise / not double-close
    await rm.close_llm_client()
    assert closed == [True]


async def test_start_llm_server_is_idempotent(monkeypatch):
    """start_llm_server must no-op when a server is already up.

    Managed servers bind a fixed port and are reused across chunks
    (process_chunk never tears them down). A second construction would spawn
    another subprocess that fails to bind the port, hard-failing later chunks.
    """
    import bibr.local.vllm_llm as vllm_llm
    from bibr.pipeline.resources import ResourceManager

    constructed = []

    class FakeServer:
        def __init__(self):
            constructed.append(True)

        def configure_llm_client(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(vllm_llm, "VllmLlmServer", FakeServer)

    rm = ResourceManager()
    await rm.start_llm_server(backend="vllm")
    await rm.start_llm_server(backend="vllm")  # second call must be a no-op

    assert len(constructed) == 1
    assert rm._llm_server is not None


async def test_concurrent_start_llm_server_constructs_one_server(monkeypatch):
    import bibr.local.vllm_llm as vllm_llm
    from bibr.pipeline.resources import ResourceManager

    constructed = []

    class FakeServer:
        def __init__(self):
            time.sleep(0.05)
            constructed.append(self)

        def configure_llm_client(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(vllm_llm, "VllmLlmServer", FakeServer)

    rm = ResourceManager()
    await asyncio.gather(
        rm.start_llm_server(backend="vllm"),
        rm.start_llm_server(backend="vllm"),
    )

    assert len(constructed) == 1
    assert rm._llm_server is constructed[0]


async def test_failed_llm_configuration_shuts_down_unpublished_server(monkeypatch):
    import bibr.local.vllm_llm as vllm_llm
    from bibr.pipeline.resources import ResourceManager

    class FakeServer:
        def __init__(self):
            self.shutdown_calls = 0

        def configure_llm_client(self):
            raise RuntimeError("bad client configuration")

        def shutdown(self):
            self.shutdown_calls += 1

    server = FakeServer()
    monkeypatch.setattr(vllm_llm, "VllmLlmServer", lambda: server)

    rm = ResourceManager()
    with pytest.raises(RuntimeError, match="bad client configuration"):
        await rm.start_llm_server(backend="vllm")

    assert server.shutdown_calls == 1
    assert rm._llm_server is None
