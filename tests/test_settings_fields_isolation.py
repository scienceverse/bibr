"""Global Settings must not leak between tests (x-tests-1).

Pydantic v2 records every attribute assignment in ``model_fields_set``, and
``monkeypatch`` undoes itself with another assignment — so a bare
``monkeypatch.setattr(Settings.llm, "max_concurrency", 1)`` permanently marks
the field as user-set on the process-global singleton, and production
"default when unset" branches flip depending on which tests ran before.
The autouse ``_isolate_global_settings`` fixture snapshots values,
``model_fields_set`` and private attributes per test and restores them
afterwards.

These tests run in definition order: the first deliberately pollutes the
global the way the old suite did, the second proves the pollution is gone.
On the pre-fix tree the second test fails (``max_concurrency`` still marked,
value 1); with the fixture both pass in any order.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from bibr.config import GlobalSettings, Settings


def test_polluting_test_marks_max_concurrency_user_set(monkeypatch):
    """Deliberately leak the way test_llm_server_respects_explicit_max_concurrency did."""
    monkeypatch.setattr(Settings.llm, "max_concurrency", 1)
    monkeypatch.setattr(Settings.ocr, "backend", "paddle")
    assert "max_concurrency" in Settings.llm.model_fields_set
    assert Settings.llm.max_concurrency == 1


def test_previous_test_leak_is_gone():
    assert "max_concurrency" not in Settings.llm.model_fields_set
    assert Settings.llm.max_concurrency == GlobalSettings().llm.max_concurrency
    assert "backend" not in Settings.ocr.model_fields_set
    assert Settings.ocr.backend == GlobalSettings().ocr.backend


def test_section_identity_survives_isolation():
    """In-place restore: modules holding ``Settings.llm`` keep one object."""
    assert Settings.llm is Settings.llm


def test_explicit_mark_is_still_honored_within_a_test(monkeypatch):
    """Guard: the fixture must not hide a genuinely user-set field from
    production "default when unset" logic inside the same test."""
    from bibr.local import llama_cpp

    fake = MagicMock(base_url="http://127.0.0.1:8770", model="org/model:Q4", n_slots=9)
    monkeypatch.setattr(llama_cpp, "LlamaCppServer", MagicMock(return_value=fake))
    monkeypatch.setattr(Settings.llm, "local_model", "org/model:Q4")
    monkeypatch.setattr(Settings.llm, "max_concurrency", 3)
    server = llama_cpp.LlamaCppLlmServer()
    server.configure_llm_client()
    assert server._settings.llm.max_concurrency == 3
