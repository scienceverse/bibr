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
