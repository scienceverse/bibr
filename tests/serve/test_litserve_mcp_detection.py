"""LitServe's own MCP connector must stay off, whatever is installed.

LitServe 0.2.17 sets ``litserve.server._MCP_AVAILABLE`` from ``mcp`` being
importable, but ``litserve/mcp.py`` only binds ``MCPServer`` when the
third-party ``fastmcp`` package is present — with bibr's ``mcp`` extra alone,
``LitServer.run()`` raised ``NameError: name 'MCPServer' is not defined``.
bibr mounts its own ``/mcp`` endpoint and never wants LitServe's.
"""

from __future__ import annotations

import pytest


def test_build_server_disables_litserve_mcp_detection(monkeypatch):
    litserve_server = pytest.importorskip("litserve.server")

    # Simulate the failing install: LitServe thinks MCP is available.
    monkeypatch.setattr(litserve_server, "_MCP_AVAILABLE", True)

    from bibr.serve.app import build_server

    build_server()

    assert litserve_server._MCP_AVAILABLE is False
