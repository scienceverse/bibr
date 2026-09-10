"""HTTP OCR backends must request the alias the server advertises.

``OCR_LOCAL_MODEL`` (``THUDM/GLM-OCR``) is the HuggingFace repo a *local*
runtime loads weights from; an externally-managed GLM-OCR server advertises
``glm-ocr`` on ``/v1/models``. With no ``OCR_MODEL`` set, glm-http asked for
the repo id, so ``wait_for_server`` polled for a model id the server never
lists and burned its whole ``PIPELINE_DEPLOYMENT_READY_TIMEOUT`` before
failing — the wizard's private-server tier and the library ``ocr_url=`` entry
point both land here.
"""

import pytest

from bibr.config import GlobalSettings
from bibr.ocr.profiles import GLM_SERVED_MODEL_ALIAS, resolve_ocr_runtime_identity
from bibr.ocr.registry import resolve_backend_candidates
from bibr.pipeline.context import RunConfig


@pytest.fixture
def settings():
    """Settings with nothing user-set, i.e. no ``OCR_MODEL``."""
    return GlobalSettings(ocr={})


@pytest.mark.parametrize(
    "cfg",
    [
        RunConfig(ocr_backend="glm-http"),
        # An ``ocr_url`` override rewrites any local GLM runtime to glm-http.
        RunConfig(ocr_url="http://ocr.internal:8002"),
        RunConfig(ocr_backend="glm-rapid-mlx", ocr_url="http://ocr.internal:8002"),
    ],
    ids=["explicit", "url-only", "url-overrides-local"],
)
def test_glm_http_identity_uses_the_served_alias(cfg, settings):
    identity = resolve_ocr_runtime_identity(cfg, settings)
    assert identity.backend == "glm-http"
    assert identity.model == GLM_SERVED_MODEL_ALIAS
    assert identity.profile == "glm"


def test_glm_http_startup_candidate_uses_the_served_alias(settings):
    candidates = resolve_backend_candidates("glm-http", settings)
    assert [(c.backend, c.model) for c in candidates] == [("glm-http", GLM_SERVED_MODEL_ALIAS)]


def test_explicit_ocr_model_still_wins(settings):
    settings.ocr.model = "my-org/custom-ocr"
    identity = resolve_ocr_runtime_identity(RunConfig(ocr_backend="glm-http"), settings)
    assert identity.model == "my-org/custom-ocr"
    candidates = resolve_backend_candidates("glm-http", settings)
    assert candidates[0].model == "my-org/custom-ocr"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("glm-rapid-mlx", "mlx-community/GLM-OCR-8bit"),
        ("glm-llama", "ggml-org/GLM-OCR-GGUF:Q8_0"),
        ("paddle-http", "paddle-ocr-vl-1.6"),
    ],
)
def test_local_and_paddle_backends_are_unchanged(backend, expected, settings):
    identity = resolve_ocr_runtime_identity(RunConfig(ocr_backend=backend), settings)
    assert identity.model == expected
    assert resolve_backend_candidates(backend, settings)[0].model == expected


def test_transport_default_matches_the_resolved_identity():
    """The client's own fallback and the resolver must not drift apart."""
    from bibr.local.ocr_transport import BaseHttpOcrClient

    assert BaseHttpOcrClient._DEFAULT_MODEL == GLM_SERVED_MODEL_ALIAS


def test_dry_run_preview_matches_the_resolved_identity(settings):
    """``--dry-run`` printed ``glm-ocr`` while the run asked for the repo id."""
    from bibr.local.cli.dry_run import _OCR_HTTP_DEFAULT_SERVED_NAME

    identity = resolve_ocr_runtime_identity(RunConfig(ocr_backend="glm-http"), settings)
    assert _OCR_HTTP_DEFAULT_SERVED_NAME["glm-http"] == identity.model
