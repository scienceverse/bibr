import os

import pytest

# Load no dotenv file at all. Real env vars outrank dotenv in pydantic-settings,
# so the pins below were already safe from a developer's ``.env`` — but every
# setting they do *not* pin (OCR_BACKEND, LLM_MODEL, SERVE_*, ...) was read
# straight out of ./.env or ~/.bibr/.env, so a local .env silently changed what
# the suite asserted (and individual tests had started working around it by
# chdir-ing to a tmp_path and faking Path.home). One switch covers every
# settings model, since _BibrSettings.__init__ resolves the chain centrally.
# Tests that exercise the chain itself delenv this; tests that want a specific
# file still pass ``_env_file=``.
os.environ["BIBR_ENV_FILE"] = ""

# Set default environment variables for tests
os.environ["REDIS_URL"] = "memory://"
os.environ["REDIS_PASSWORD"] = "test-redis-password"
os.environ["GOOGLE_API_KEY"] = "test-google-key"
os.environ["LLM_PROVIDER"] = "google"
os.environ["LLM_BACKEND"] = "cloud"
os.environ["LLM_INSTRUCTOR_MODE"] = ""
os.environ["ENVIRONMENT"] = "development"
os.environ["LLM_TRACK_USAGE"] = "false"
# Keep unit tests hermetic from a developer's real resolver .env. Individual
# resolver tests opt in explicitly via monkeypatch.
os.environ["BIBR_RESOLVER_URL"] = ""
os.environ["BIBR_RESOLVER_ENRICH"] = "false"
os.environ["BIBR_RESOLVER_AUTHORITATIVE"] = "false"
# Production default is "ner" (local parser); pinned to "llm" here so the test
# suite doesn't pull the ModernBERT-CRF checkpoint. See
# test_config.test_ref_strategy_defaults_are_local for the real default.
os.environ["REF_PARSE_STRATEGY"] = "llm"
# Production default is "geom" (local geometry segmenter); pinned to "llm" here
# so the suite doesn't fetch the remote geom HF artifact. Tests that assert
# the real default delenv this first; geom tests set it explicitly.
os.environ["REF_SEG_STRATEGY"] = "llm"
# Production default is the published paper classifier; disabled here so extract
# tests don't pull the HF snapshot. Classifier-path tests opt in via monkeypatch;
# test_config.test_paper_classifier_defaults_to_published_model asserts the real
# default after delenv.
os.environ["ML_PAPER_CLASSIFIER_MODEL_ID"] = ""
# Keep section-classifier tests offline for the same reason. Tests for the
# trained path opt in explicitly; config-default tests clear this override.
os.environ["ML_SECTION_CLASSIFIER_MODEL_ID"] = ""
# Production default is the published front-role bundle; disabled here so
# front-matter tests don't pull the HF snapshot. Tests for the model path build
# their own FrontRolePredictions or point at a local bundle.
os.environ["ML_FRONT_ROLE_MODEL_ID"] = ""
# Production default is "auto" (probe the Hub for a model's ONNX bundle, else
# torch). Pinned to "torch" here so constructing a detector/classifier in a
# test never makes a network round-trip; ONNX-runtime tests set ML_RUNTIME (or
# a settings override) explicitly and point at local bundles.
os.environ["ML_RUNTIME"] = "torch"
# Never inherit a developer's local resolver service from ``.env``. Tests that
# exercise resolver routing pass an explicit client and opt into authoritative
# mode with monkeypatch; all other tests should remain offline and hermetic.
os.environ["BIBR_RESOLVER_URL"] = ""
os.environ["BIBR_RESOLVER_ENRICH"] = "false"
os.environ["BIBR_RESOLVER_AUTHORITATIVE"] = "false"


# ---------------------------------------------------------------------------
# Guard: never let a test signal process group 0 or 1.
#
# The managed-server shutdown paths call ``os.killpg(proc.pid, ...)``. A test
# that fakes the process with a bare ``MagicMock`` gets ``pid`` back as a mock,
# and ``os.killpg`` coerces it through ``__index__`` — which MagicMock answers
# with 1. The result is ``killpg(1, SIGTERM)``: a SIGTERM to init's process
# group. On a developer machine that is EPERM and invisible; on a CI runner it
# terminates the runner agent mid-suite ("The runner has received a shutdown
# signal"), which is how this went unnoticed for weeks.
#
# pgid 0 (our own group) is equally fatal, so both are refused loudly. A test
# exercising a shutdown path must either patch ``os.killpg`` in the module under
# test or give the fake process a plausible integer pid.
# ---------------------------------------------------------------------------
# Windows has no process-group signalling API. Keep it absent there so
# tests exercise the real platform surface; Unix-backend tests mock it explicitly.
_real_killpg = getattr(os, "killpg", None)


def _guarded_killpg(pgid, sig):
    try:
        target = pgid.__index__()
    except (AttributeError, TypeError):
        target = pgid
    if isinstance(target, int) and target <= 1:
        raise RuntimeError(
            f"os.killpg({target!r}, {sig!r}) blocked: signalling process group "
            f"{target} would hit init (1) or this test session (0). The process "
            "under test is almost certainly a mock whose .pid is not a real pid — "
            "patch os.killpg in the module under test, or set a plausible int pid."
        )
    return _real_killpg(pgid, sig)


if _real_killpg is not None:
    os.killpg = _guarded_killpg


@pytest.fixture(autouse=True)
def _killpg_guard():
    """Reinstate the guard if a test replaced ``os.killpg`` without restoring it."""
    yield
    if _real_killpg is not None and getattr(os, "killpg", None) is not _guarded_killpg:
        os.killpg = _guarded_killpg


@pytest.fixture(autouse=True)
def _pin_cuda_probe_for_platform_tests(monkeypatch):
    """The automatic Linux OCR chain now asks the hardware before listing
    ``paddle-vllm``. Tests that pin ``sys.platform`` to linux were written
    against the old unconditional chain, so pin a roomy GPU here; tests that
    exercise the CPU-only / small-GPU paths patch the probe themselves."""
    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: 24.0)
