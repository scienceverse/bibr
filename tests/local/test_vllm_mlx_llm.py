import contextlib
from unittest.mock import MagicMock


@contextlib.contextmanager
def _unset_field(settings_section, field_name):
    """Temporarily remove ``field_name`` from ``model_fields_set`` so a
    default-computing code path treats it as not explicitly set, then restore
    the original explicit/unset state on exit."""
    was_explicit = field_name in settings_section.model_fields_set
    settings_section.model_fields_set.discard(field_name)
    try:
        yield
    finally:
        if was_explicit:
            settings_section.model_fields_set.add(field_name)


@contextlib.contextmanager
def _set_field_explicit(settings_section, field_name):
    was_explicit = field_name in settings_section.model_fields_set
    settings_section.model_fields_set.add(field_name)
    try:
        yield
    finally:
        if not was_explicit:
            settings_section.model_fields_set.discard(field_name)


def test_llm_server_uses_serial_engine_by_default(monkeypatch):
    """Continuous batching is opt-in: vllm-mlx 0.2.6's BatchGenerator crash-loops
    on any model whose cache includes a plain ArraysCache layer (e.g. the
    default local model, numind/NuExtract3-mlx-nvfp4) because it aliases
    ArraysCache as MambaCache and calls it with the old MambaCache signature.
    """
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    server = MagicMock()
    constructor = MagicMock(return_value=server)
    monkeypatch.setattr(ocr, "VllmMlxServer", constructor)
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_port", 8767)
    monkeypatch.setattr(Settings.llm, "vllm_mlx_extra_args", "")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", False)

    wrapped = mod.VllmMlxLlmServer()

    constructor.assert_called_once_with(
        model="org/text-model",
        port=8767,
        continuous_batching=False,
        multimodal=False,
        extra_args=[],
        settings=wrapped._settings,
    )
    assert wrapped._server is server


def test_llm_server_continuous_batching_opt_in(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    server = MagicMock()
    constructor = MagicMock(return_value=server)
    monkeypatch.setattr(ocr, "VllmMlxServer", constructor)
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_port", 8767)
    monkeypatch.setattr(Settings.llm, "vllm_mlx_extra_args", "")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", True)

    mod.VllmMlxLlmServer()

    assert constructor.call_args.kwargs["continuous_batching"] is True


def test_nuextract_llm_server_uses_llm_role_settings(monkeypatch):
    """The text LLM server must read its own ``vllm_mlx_*`` settings, not the
    (permanently disabled — see ``raise_vllm_mlx_ocr_disabled``) OCR role's.
    """
    from bibr.config import Settings
    from bibr.local import llm as llm_mod
    from bibr.local import ocr as ocr_mod

    calls = []

    class FakeServer:
        base_url = "http://localhost:8767"

        @property
        def loaded(self):
            return True

    def fake_server(**kwargs):
        calls.append(kwargs)
        return FakeServer()

    monkeypatch.setattr(ocr_mod, "VllmMlxServer", fake_server)
    # Unset LLM_LOCAL_MODEL resolves to the registry MLX default for this backend.
    monkeypatch.setattr(Settings.llm, "local_model", None)
    monkeypatch.setattr(Settings.llm, "vllm_mlx_port", 8767)
    monkeypatch.setattr(Settings.llm, "vllm_mlx_extra_args", "")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", True)

    llm_mod.VllmMlxLlmServer()

    assert calls[0]["model"] == "numind/NuExtract3-mlx-8bits"
    assert calls[0]["port"] == 8767
    assert calls[0]["multimodal"] is False


def test_configure_llm_client_raises_default_timeout(monkeypatch):
    """MLX SimpleEngine on Apple Silicon runs single-digit tok/s under concurrent
    load; the 30s cloud-API default timeout was killing in-flight local calls
    (references, authors) before the model finished generating.
    """
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "timeout_seconds", 30)

    with _unset_field(Settings.llm, "timeout_seconds"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.timeout_seconds == 300


def test_configure_llm_client_respects_explicit_timeout(monkeypatch):
    """A user-set LLM_TIMEOUT_SECONDS must not be silently overridden."""
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "timeout_seconds", 45)

    with _set_field_explicit(Settings.llm, "timeout_seconds"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.timeout_seconds == 45


def test_configure_llm_client_caps_concurrency_for_simple_engine(monkeypatch):
    """bibr fans out multiple concurrent LLM calls (core metadata + reference
    batches). SimpleEngine generates one request at a time internally, so a
    queued call's wall-clock can exceed the server's own 300s request timeout
    — whose abort path crashes the Metal context mid-generation
    (`Completed handler provided after commit call`), taking down every
    other in-flight call with a connection error. Capping client-side
    concurrency to 1 avoids ever queueing past the timeout.
    """
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", False)
    monkeypatch.setattr(Settings.llm, "max_concurrency", 0)

    with _unset_field(Settings.llm, "max_concurrency"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 1


def test_configure_llm_client_respects_explicit_concurrency(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", False)
    monkeypatch.setattr(Settings.llm, "max_concurrency", 4)

    with _set_field_explicit(Settings.llm, "max_concurrency"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 4


def test_configure_llm_client_caps_default_concurrency_with_batching(monkeypatch):
    """BatchedEngine still shares one Apple-Silicon model and Metal context.

    vllm-mlx 0.4 can stall when bibr fans out several structured JSON-schema
    calls with large output budgets, so the constrained (json_schema logits)
    path stays serialized even with batching. Operators can still opt into
    higher concurrency with LLM_MAX_CONCURRENCY once a runtime/model
    combination is validated.
    """
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "instructor_mode", "")
    monkeypatch.setattr(Settings.llm, "structured_backend", "auto")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", True)
    monkeypatch.setattr(Settings.llm, "max_concurrency", 0)

    with _unset_field(Settings.llm, "max_concurrency"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 1


def test_configure_llm_client_serializes_even_for_nuextract_batched(monkeypatch):
    """Measured 2026-07-09 on M4/16GB: 3 concurrent bibr-shaped calls ran 0.71x
    the speed of serial (prefill-heavy calls serialize on the GPU; extra
    in-flight KV adds memory pressure), so even the plain-generation NuExtract
    path stays at concurrency 1 unless the operator raises it explicitly."""
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "numind/NuExtract3-mlx-8bits")
    monkeypatch.setattr(Settings.llm, "instructor_mode", "")
    monkeypatch.setattr(Settings.llm, "structured_backend", "auto")
    monkeypatch.setattr(Settings.llm, "vllm_mlx_continuous_batching", True)
    monkeypatch.setattr(Settings.llm, "max_concurrency", 0)

    with _unset_field(Settings.llm, "max_concurrency"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.max_concurrency == 1


def test_configure_llm_client_sets_safe_default_token_cap(monkeypatch):
    """Managed vllm-mlx should not inherit the server's huge default cap.

    The OpenAI provider omits max_completion_tokens for generic self-hosted
    base_url endpoints unless LLM_MAX_TOKENS is explicit. For bibr-managed
    vllm-mlx, we know the endpoint and can set a safe cap so concurrent or
    degenerate generations do not reserve/run toward 32k+ output tokens.
    """
    from bibr.clients import providers
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "max_tokens", 65536)

    with _unset_field(Settings.llm, "max_tokens"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()

        assert server._settings.llm.max_tokens == 8192
        assert "max_tokens" in server._settings.llm.model_fields_set
        kwargs = providers.get("openai", settings=server._settings).call_kwargs(
            reasoning_effort=None
        )
        assert kwargs["max_tokens"] == 8192
        assert "max_completion_tokens" not in kwargs


def test_configure_llm_client_respects_explicit_token_cap(monkeypatch):
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "max_tokens", 2048)

    with _set_field_explicit(Settings.llm, "max_tokens"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.max_tokens == 2048


def test_configure_llm_client_raises_default_rate_limit(monkeypatch):
    """LLM_RATE_LIMIT_RPM's 60 default guards a cloud provider's quota. This
    server is ours — there is no external budget — and calls here are already
    serialized by max_concurrency=1, so the default only inserts dead time.
    """
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr
    from bibr.local.http_runtime import MANAGED_LOCAL_LLM_RATE_LIMIT_RPM

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "rate_limit_rpm", 60)

    with _unset_field(Settings.llm, "rate_limit_rpm"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.rate_limit_rpm == MANAGED_LOCAL_LLM_RATE_LIMIT_RPM


def test_configure_llm_client_respects_explicit_rate_limit(monkeypatch):
    """A user-set LLM_RATE_LIMIT_RPM must not be silently overridden."""
    from bibr.config import Settings
    from bibr.local import llm as mod
    from bibr.local import ocr

    monkeypatch.setattr(
        ocr, "VllmMlxServer", MagicMock(return_value=MagicMock(base_url="http://x"))
    )
    monkeypatch.setattr(Settings.llm, "local_model", "org/text-model")
    monkeypatch.setattr(Settings.llm, "rate_limit_rpm", 25)

    with _set_field_explicit(Settings.llm, "rate_limit_rpm"):
        server = mod.VllmMlxLlmServer()
        server.configure_llm_client()
        assert server._settings.llm.rate_limit_rpm == 25
