"""Tests for model-specific OCR request profiles."""

import inspect
from typing import get_type_hints

import pytest

from bibr.ocr.profiles import (
    GLM_PROFILE,
    PADDLE_PROFILE,
    PADDLE_TABLE_RECOVERY_MAX_TOKENS,
    OcrProfile,
    OcrProfileName,
    OcrRequestSettings,
    OcrTask,
    resolve_ocr_profile,
    resolve_ocr_runtime_identity,
)
from bibr.pipeline.context import RunConfig


def test_profile_shared_interface_uses_named_task_and_profile_types():
    profile_hints = get_type_hints(OcrProfile)
    prompt_hints = get_type_hints(OcrProfile.prompt_for)
    resolver_hints = get_type_hints(resolve_ocr_profile)

    assert profile_hints["name"] == OcrProfileName
    assert profile_hints["prompts"] == dict[OcrTask, str]
    assert prompt_hints["task"] == OcrTask
    assert resolver_hints["explicit"] == str | None
    assert "profile" not in inspect.signature(resolve_ocr_profile).parameters


def test_paddle_profile_prompts_and_request_settings():
    assert PADDLE_PROFILE.prompt_for("text") == "OCR:"
    assert PADDLE_PROFILE.prompt_for("table") == "Table Recognition:"
    assert PADDLE_PROFILE.prompt_for("formula") == "Formula Recognition:"
    assert PADDLE_PROFILE.request == OcrRequestSettings(
        temperature=0.0,
        max_tokens=1024,
        task_max_tokens=(("table", 4096),),
    )
    assert PADDLE_PROFILE.request.max_tokens_for("text") == 1024
    assert PADDLE_PROFILE.request.max_tokens_for("formula") == 1024
    assert PADDLE_PROFILE.request.max_tokens_for("table") == 4096
    assert PADDLE_PROFILE.task_for_prompt("Table Recognition:") == "table"


def test_paddle_table_recovery_budget_is_8192():
    assert PADDLE_TABLE_RECOVERY_MAX_TOKENS == 8192


def test_glm_profile_prompts_and_request_settings():
    assert GLM_PROFILE.prompt_for("text") == "Text Recognition:"
    assert GLM_PROFILE.request == OcrRequestSettings(
        temperature=0.01,
        max_tokens=16384,
        top_k=1,
        repetition_penalty=1.1,
    )


def test_legacy_task_prompts_is_the_glm_profile_mapping():
    from bibr.ocr.layout import TASK_PROMPTS

    assert TASK_PROMPTS is GLM_PROFILE.prompts


@pytest.mark.parametrize(
    ("backend", "model"),
    [
        ("paddle-ocr-vl-1.6", ""),
        ("paddle", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ],
)
def test_resolve_ocr_profile_infers_paddle_from_backend_or_model(backend, model):
    assert resolve_ocr_profile(explicit=None, backend=backend, model=model).name == "paddle"


@pytest.mark.parametrize(
    ("backend", "model"),
    [
        ("glm-ocr", ""),
        ("glm", "zai-org/GLM-OCR"),
    ],
)
def test_resolve_ocr_profile_infers_glm_from_backend_or_model(backend, model):
    assert resolve_ocr_profile(explicit=None, backend=backend, model=model).name == "glm"


def test_resolve_ocr_profile_honors_explicit_override_and_generation_overrides():
    resolved_profile = resolve_ocr_profile(
        explicit="paddle",
        backend="private-ocr",
        model="private/model",
        max_tokens=2048,
        temperature=0.2,
    )

    assert resolved_profile.name == "paddle"
    assert resolved_profile.request == OcrRequestSettings(temperature=0.2, max_tokens=2048)
    assert resolved_profile.request.max_tokens_for("table") == 2048


def test_resolve_ocr_profile_requires_explicit_override_for_unknown_custom_alias():
    with pytest.raises(ValueError, match="OCR_PROFILE"):
        resolve_ocr_profile(explicit=None, backend="private-ocr", model="private/model")


def test_paddle_http_url_runtime_identity_keeps_paddle_backend_and_alias():
    from bibr.config import GlobalSettings

    identity = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend="paddle-http", ocr_url="http://host:8000"),
        GlobalSettings(),
    )

    assert identity.backend == "paddle-http"
    assert identity.model == "paddle-ocr-vl-1.6"
    assert identity.profile == "paddle"


def test_paddle_vllm_runtime_identity_matches_its_served_client_alias():
    from bibr.config import GlobalSettings

    identity = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend="paddle-vllm"),
        GlobalSettings(),
    )

    assert identity.backend == "paddle-vllm"
    assert identity.model == "paddle-ocr-vl-1.6"
    assert identity.profile == "paddle"


def test_paddle_vllm_identity_keeps_served_alias_when_source_model_is_overridden():
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    identity = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend="paddle-vllm", ocr_model="some/source-model"),
        settings,
    )

    assert identity.model == settings.ocr.paddle_served_model
    assert identity.profile == "paddle"


def test_cloud_runtime_identity_honors_explicit_paddle_profile():
    from bibr.config import GlobalSettings

    identity = resolve_ocr_runtime_identity(
        RunConfig(ocr_backend="openai", ocr_profile="paddle"),
        GlobalSettings(),
    )

    assert identity.profile == "paddle"


def test_paddle_geometry_matches_the_served_model_preprocessor_config():
    """PaddleOCR-VL's own preprocessor_config.json is the authority here.

    temporal_patch_size=1, max_pixels=1003520, min_pixels=112896, and
    patch_size=14 x merge_size=2 = the 28 px alignment factor. Drifting from
    these silently changes resolution instead of failing.
    """
    geometry = PADDLE_PROFILE.image

    assert geometry.t_patch_size == 1
    assert geometry.max_pixels == 1_003_520
    assert geometry.min_pixels == 112_896
    assert geometry.patch_expand_factor == 1


def test_glm_and_paddle_do_not_share_geometry():
    """Regression: until 2026-08-03 GLM's constants encoded every backend.

    GLM's t_patch_size=2 is a temporal (video frame-pairing) factor that
    ``smart_resize`` folds into the pixel-budget test, so applying it to Paddle
    halved the budget that model actually accepts.
    """
    assert GLM_PROFILE.image != PADDLE_PROFILE.image
    assert GLM_PROFILE.image.t_patch_size == 2
    assert PADDLE_PROFILE.image.t_patch_size == 1


def test_paddle_geometry_yields_the_full_budget():
    """The temporal factor must not shrink Paddle's usable pixel budget."""
    pytest.importorskip("cv2")  # smart_resize lives behind the ml extra

    from bibr.ocr.image_processing import smart_resize

    geometry = PADDLE_PROFILE.image
    # A crop far above the budget, so the cap branch decides the output size.
    h_bar, w_bar = smart_resize(
        t=geometry.t_patch_size,
        h=3000,
        w=3000,
        t_factor=geometry.t_patch_size,
        h_factor=28,
        w_factor=28,
        min_pixels=geometry.min_pixels,
        max_pixels=geometry.max_pixels,
    )

    # Paddle's own smart_resize caps on h*w alone, with no temporal term.
    assert h_bar * w_bar <= geometry.max_pixels
    assert h_bar * w_bar > geometry.max_pixels * 0.9

    glm = GLM_PROFILE.image
    glm_h, glm_w = smart_resize(
        t=glm.t_patch_size,
        h=3000,
        w=3000,
        t_factor=glm.t_patch_size,
        h_factor=28,
        w_factor=28,
        min_pixels=glm.min_pixels,
        max_pixels=glm.max_pixels,
    )
    # The GLM constants land near half the budget on the same input.
    assert glm_h * glm_w < h_bar * w_bar * 0.6


def test_geometry_cache_fingerprint_distinguishes_every_field():
    from dataclasses import replace

    base = PADDLE_PROFILE.image
    seen = {base.cache_fingerprint()}
    for field, value in (
        ("t_patch_size", 2),
        ("max_pixels", 501_760),
        ("min_pixels", 12_544),
        ("patch_expand_factor", 2),
    ):
        fingerprint = replace(base, **{field: value}).cache_fingerprint()
        assert fingerprint not in seen, f"{field} does not change the fingerprint"
        seen.add(fingerprint)


def test_resolve_ocr_profile_preserves_geometry():
    """Request-setting overrides must not reset the model's geometry."""
    resolved = resolve_ocr_profile(
        explicit="paddle",
        backend="serve-http",
        model="paddle-ocr-vl-1.6",
        max_tokens=256,
        temperature=0.5,
    )

    assert resolved.image == PADDLE_PROFILE.image
    assert resolved.request.max_tokens == 256
