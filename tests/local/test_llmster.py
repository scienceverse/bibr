from unittest.mock import MagicMock

import pytest

from bibr.exceptions import UpstreamServiceError


def test_missing_lms_cli_has_manual_install_command(monkeypatch):
    from bibr.local import llmster

    monkeypatch.setattr(llmster, "find_lms", lambda: None)

    with pytest.raises(UpstreamServiceError, match="lmstudio.ai/install.sh"):
        llmster.LlmsterLlmServer(model="org/model", identifier="bibr-model")


def test_starts_only_missing_resources_and_unloads_owned_model(monkeypatch):
    from bibr.local import llmster

    calls: list[list[str]] = []
    state = {"daemon": False, "server": False, "loaded": False}

    def run(args: list[str], *, json_output: bool = False):
        calls.append(args)
        if args[:2] == ["daemon", "status"]:
            return {"status": "running" if state["daemon"] else "not-running"}
        if args[:2] == ["daemon", "up"]:
            state["daemon"] = True
            return {"status": "running"}
        if args[:2] == ["server", "status"]:
            return {"running": state["server"], "port": 1234}
        if args[:2] == ["server", "start"]:
            state["server"] = True
            return None
        if args == ["ls", "--json"]:
            return [{"modelKey": "org/model"}]
        if args == ["ps", "--json"]:
            return [{"identifier": "bibr-model"}] if state["loaded"] else []
        if args[0] == "load":
            state["loaded"] = True
            return None
        if args == ["unload", "bibr-model"]:
            state["loaded"] = False
            return None
        if args[:2] == ["server", "stop"]:
            state["server"] = False
            return None
        if args[:2] == ["daemon", "down"]:
            state["daemon"] = False
            return None
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")
    server = llmster.LlmsterLlmServer(
        model="org/model", identifier="bibr-model", port=1234, context_length=8192, runner=run
    )

    assert ["daemon", "up", "--json"] in calls
    assert ["server", "start", "--port", "1234"] in calls
    assert [
        "load",
        "org/model",
        "--identifier",
        "bibr-model",
        "--context-length",
        "8192",
    ] in calls

    server.shutdown()

    assert ["unload", "bibr-model"] in calls
    assert ["server", "stop"] in calls
    assert ["daemon", "down"] in calls


def test_shutdown_preserves_preexisting_daemon_server_and_model(monkeypatch):
    from bibr.local import llmster

    calls: list[list[str]] = []

    def run(args: list[str], *, json_output: bool = False):
        calls.append(args)
        if args[:2] == ["daemon", "status"]:
            return {"status": "running"}
        if args[:2] == ["server", "status"]:
            return {"running": True, "port": 1234}
        if args == ["ls", "--json"]:
            return {"models": [{"modelKey": "org/model"}]}
        if args == ["ps", "--json"]:
            return {"models": [{"identifier": "bibr-model"}]}
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")
    server = llmster.LlmsterLlmServer(
        model="org/model", identifier="bibr-model", port=1234, runner=run
    )
    calls.clear()

    server.shutdown()

    assert calls == []


def test_missing_local_model_never_downloads(monkeypatch):
    from bibr.local import llmster

    calls: list[list[str]] = []

    def run(args: list[str], *, json_output: bool = False):
        calls.append(args)
        if args[:2] == ["daemon", "status"]:
            return {"status": "running"}
        if args[:2] == ["server", "status"]:
            return {"running": True, "port": 1234}
        if args == ["ls", "--json"]:
            return []
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")

    with pytest.raises(UpstreamServiceError, match=r"lms get org/model"):
        llmster.LlmsterLlmServer(model="org/model", identifier="bibr-model", runner=run)

    assert not any(command and command[0] == "get" for command in calls)


def test_configure_llm_client_uses_openai_endpoint(monkeypatch):
    from bibr.config import GlobalSettings
    from bibr.local import llmster

    server = llmster.LlmsterLlmServer.__new__(llmster.LlmsterLlmServer)
    server._identifier = "bibr-model"
    server._port = 4321
    server._settings = GlobalSettings()
    server._settings.llm.provider = "google"
    server._settings.llm.base_url = None
    server._settings.llm.api_key = "old"
    server._settings.llm.model = "old"

    server.configure_llm_client()

    assert server._settings.llm.provider == "openai"
    assert server._settings.llm.base_url == "http://127.0.0.1:4321/v1"
    assert server._settings.llm.api_key == "lm-studio"
    assert server._settings.llm.model == "bibr-model"


def test_subprocess_error_keeps_command_and_stderr(monkeypatch):
    from bibr.local import llmster

    completed = MagicMock(returncode=2, stdout="", stderr="daemon unavailable")
    monkeypatch.setattr(llmster.subprocess, "run", MagicMock(return_value=completed))

    with pytest.raises(UpstreamServiceError, match="daemon unavailable"):
        llmster.run_lms_command("/usr/local/bin/lms", ["daemon", "status", "--json"])


# ---------------------------------------------------------------------------
# local-runtimes sweep: load-timeout mapping, stale-identifier reuse,
# rate-limit raise (findings 3, 5, 12)
# ---------------------------------------------------------------------------


def _ready_runner(*, loaded_items, port=1234):
    """Fake `lms` runner with daemon/server up and our model downloaded."""

    def run(args: list[str], *, json_output: bool = False):
        if args[:2] == ["daemon", "status"]:
            return {"status": "running"}
        if args[:2] == ["server", "status"]:
            return {"running": True, "port": port}
        if args == ["ls", "--json"]:
            return [{"modelKey": "org/model"}]
        if args == ["ps", "--json"]:
            return loaded_items
        raise AssertionError(f"unexpected command: {args}")

    return run


def _models_server(served_ids):
    """Ephemeral loopback server answering GET /v1/models with *served_ids*."""
    import http.server
    import json as _json
    import socketserver
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            body = _json.dumps({"data": [{"id": i} for i in served_ids]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_load_timeout_surfaces_actionable_error(monkeypatch):
    """A hung `lms load` maps to UpstreamServiceError naming the model (3)."""
    import subprocess

    from bibr.local import llmster

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=120)

    monkeypatch.setattr(llmster.subprocess, "run", hang)

    with pytest.raises(UpstreamServiceError, match="org/model.*LLMSTER_CONTEXT_LENGTH"):
        llmster.run_lms_command(
            "/usr/local/bin/lms", ["load", "org/model", "--identifier", "bibr-model"]
        )


def test_reused_identifier_pointing_at_other_model_raises(monkeypatch):
    """`ps` evidence that the identifier serves another model refuses reuse (5)."""
    from bibr.local import llmster

    monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")
    run = _ready_runner(loaded_items=[{"identifier": "bibr-model", "modelKey": "other/model"}])

    with pytest.raises(UpstreamServiceError, match="lms unload bibr-model"):
        llmster.LlmsterLlmServer(model="org/model", identifier="bibr-model", runner=run)


def test_reused_identifier_not_served_raises(monkeypatch):
    """Server alive but not serving our identifier: stale, refuse reuse (5)."""
    from bibr.local import llmster

    httpd = _models_server(["someone-elses-model"])
    try:
        port = httpd.server_address[1]
        monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")
        run = _ready_runner(loaded_items=[{"identifier": "bibr-model"}], port=port)

        with pytest.raises(UpstreamServiceError, match="not serving it"):
            llmster.LlmsterLlmServer(
                model="org/model", identifier="bibr-model", port=port, runner=run
            )
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_reused_identifier_served_ok(monkeypatch):
    """Server actually serving our identifier: reuse proceeds without a load (5 guard)."""
    from bibr.local import llmster

    httpd = _models_server(["bibr-model"])
    try:
        port = httpd.server_address[1]
        monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")
        calls: list[list[str]] = []
        base = _ready_runner(loaded_items=[{"identifier": "bibr-model"}], port=port)

        def run(args: list[str], *, json_output: bool = False):
            calls.append(args)
            return base(args, json_output=json_output)

        server = llmster.LlmsterLlmServer(
            model="org/model", identifier="bibr-model", port=port, runner=run
        )
        assert not any(c and c[0] == "load" for c in calls)
        server.shutdown()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_reused_identifier_only_in_ps_trusts_cli(monkeypatch):
    """Identifier-only `ps` entry + unreachable server keeps the old reuse (5 guard)."""
    from bibr.local import llmster

    monkeypatch.setattr(llmster, "find_lms", lambda: "/usr/local/bin/lms")
    calls: list[list[str]] = []
    base = _ready_runner(loaded_items=[{"identifier": "bibr-model"}])

    def run(args: list[str], *, json_output: bool = False):
        calls.append(args)
        return base(args, json_output=json_output)

    # Nothing listens on 1234 here → liveness probe degrades to trusting `ps`.
    llmster.LlmsterLlmServer(model="org/model", identifier="bibr-model", port=1234, runner=run)
    assert not any(c and c[0] == "load" for c in calls)


def test_configure_llm_client_raises_rate_limit_rpm(monkeypatch):
    """Unset rpm auto-raises to the managed-local value; explicit is kept (12)."""
    from bibr.config import GlobalSettings
    from bibr.local import llmster

    server = llmster.LlmsterLlmServer.__new__(llmster.LlmsterLlmServer)
    server._identifier = "bibr-model"
    server._port = 4321
    server._settings = GlobalSettings()
    server._settings.llm.model_fields_set.discard("rate_limit_rpm")
    server.configure_llm_client()
    assert server._settings.llm.rate_limit_rpm == 600

    server._settings = GlobalSettings()
    server._settings.llm.rate_limit_rpm = 30
    assert "rate_limit_rpm" in server._settings.llm.model_fields_set
    server.configure_llm_client()
    assert server._settings.llm.rate_limit_rpm == 30
