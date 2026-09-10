import os
from pathlib import Path
from unittest import mock

import pytest

from bibr.config import Settings


def test_retired_llm_and_ml_settings_are_not_public():
    from bibr.config import LlmOptions, MlOptions

    assert {
        "batch_provider",
        "vllm_mlx_cache_mb",
    }.isdisjoint(LlmOptions.model_fields)
    assert {
        "enabled",
        "section",
        "ref_seg",
        "ref_parse",
        "section_accept_threshold",
        "section_flag_threshold",
        "section_repo_id",
        "ref_seg_repo_id",
        "ref_parse_repo_id",
    }.isdisjoint(MlOptions.model_fields)


def test_retired_ocr_settings_are_not_public():
    from bibr.config import OcrOptions

    assert {
        "api_host",
        "api_port",
        "config_path",
        "api_path",
        "api_mode",
        "enable_layout",
        "local_device",
        "vllm_mlx_model",
        "vllm_mlx_port",
        "vllm_mlx_cache_mb",
        "vllm_mlx_extra_args",
        "vllm_mlx_continuous_batching",
    }.isdisjoint(OcrOptions.model_fields)


def test_settings_default():
    # If LLM_MODEL is not set in env (or if we mock it to be unset), it should be default
    with mock.patch.dict(os.environ, {}, clear=True):
        # We need to reload the module or just check the class attribute if it was dynamic
        # Since Settings is a class with static attributes evaluated at import time,
        # testing defaults after import is tricky without reload.
        # But we can test that it has SOME value.
        assert Settings.llm.model is not None


def test_jobs_queue_defaults_are_bounded(monkeypatch):
    monkeypatch.delenv("JOBS_MAX_RUNNING", raising=False)
    from bibr.config import GlobalSettings

    jobs = GlobalSettings().jobs
    assert jobs.max_running == 2
    assert not hasattr(jobs, "max_upload_admission_retries")


def test_pipeline_default_upload_limit_is_50_mib(monkeypatch):
    monkeypatch.delenv("PIPELINE_MAX_FILE_SIZE", raising=False)
    from bibr.config import GlobalSettings

    assert GlobalSettings().pipeline.max_file_size == 50 * 1024 * 1024


def test_pipeline_litserve_hardening_defaults(monkeypatch):
    monkeypatch.delenv("PIPELINE_MULTIPART_OVERHEAD_BYTES", raising=False)
    monkeypatch.delenv("PIPELINE_RESTART_WORKERS", raising=False)
    from bibr.config import GlobalSettings

    pipeline = GlobalSettings(_env_file=None).pipeline

    assert pipeline.multipart_overhead_bytes == 1024 * 1024
    assert pipeline.restart_workers is False


def test_pipeline_multipart_overhead_rejects_negative(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("PIPELINE_MULTIPART_OVERHEAD_BYTES", "-1")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_litserve_dependency_is_bounded_below_03():
    from importlib.metadata import requires

    from packaging.requirements import Requirement

    litserve_requirement = next(
        Requirement(requirement)
        for requirement in requires("bibr") or []
        if Requirement(requirement).name == "litserve"
    )

    assert litserve_requirement.specifier == Requirement("litserve>=0.2.16,<0.3").specifier


def test_build_sha_accepts_full_lowercase_commit_and_affects_cache(monkeypatch):
    from bibr.config import GlobalSettings, compute_behavior_fingerprint

    first_sha = "a" * 40
    second_sha = "b" * 40
    monkeypatch.setenv("BIBR_BUILD_SHA", first_sha)
    first = GlobalSettings()
    monkeypatch.setenv("BIBR_BUILD_SHA", second_sha)
    second = GlobalSettings()

    assert first_sha == first.BIBR_BUILD_SHA
    assert second_sha == second.BIBR_BUILD_SHA
    assert compute_behavior_fingerprint(first) != compute_behavior_fingerprint(second)


def test_build_sha_rejects_short_or_noncanonical_value(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("BIBR_BUILD_SHA", "abc123")

    with pytest.raises(ValidationError):
        GlobalSettings()


def test_manual_override():
    # Test that we can read the value
    assert isinstance(Settings.llm.model, str)
    assert len(Settings.llm.model) > 0


def test_llm_task_output_cap_defaults_and_env_override(monkeypatch):
    from bibr.config import GlobalSettings

    monkeypatch.setenv("LLM_TITLE_MAX_TOKENS", "2048")
    llm = GlobalSettings().llm
    assert llm.title_max_tokens == 2048
    assert llm.authors_max_tokens == 8192
    assert llm.paper_classification_max_tokens == 1024
    assert llm.paper_type_max_tokens == 512
    assert llm.section_max_tokens == 4096
    assert llm.integrity_max_tokens == 4096
    assert llm.equation_max_tokens == 4096
    assert llm.citation_max_tokens == 8192


def test_llm_task_output_caps_accept_zero_and_reject_negative():
    from pydantic import ValidationError

    from bibr.config import LlmOptions

    assert LlmOptions(title_max_tokens=0).title_max_tokens == 0
    assert LlmOptions(title_max_tokens=None).title_max_tokens is None
    with pytest.raises(ValidationError):
        LlmOptions(title_max_tokens=-1)


def test_wtpsplit_threshold_defaults_to_none(monkeypatch):
    monkeypatch.delenv("WTPSPLIT_THRESHOLD", raising=False)
    from bibr.config import GlobalSettings

    assert GlobalSettings().WTPSPLIT_THRESHOLD is None


def test_wtpsplit_threshold_loads_from_environment(monkeypatch):
    monkeypatch.setenv("WTPSPLIT_THRESHOLD", "0.42")
    from bibr.config import GlobalSettings

    assert GlobalSettings().WTPSPLIT_THRESHOLD == 0.42


def test_wtpsplit_windowing_defaults_to_runtime_defaults(monkeypatch):
    monkeypatch.delenv("WTPSPLIT_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("WTPSPLIT_STRIDE", raising=False)
    from bibr.config import GlobalSettings

    settings = GlobalSettings()

    assert settings.WTPSPLIT_BLOCK_SIZE is None
    assert settings.WTPSPLIT_STRIDE is None


def test_wtpsplit_windowing_loads_as_a_pair(monkeypatch):
    monkeypatch.setenv("WTPSPLIT_BLOCK_SIZE", "256")
    monkeypatch.setenv("WTPSPLIT_STRIDE", "128")
    from bibr.config import GlobalSettings

    settings = GlobalSettings()

    assert settings.WTPSPLIT_BLOCK_SIZE == 256
    assert settings.WTPSPLIT_STRIDE == 128


def test_wtpsplit_windowing_rejects_partial_configuration(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("WTPSPLIT_BLOCK_SIZE", "256")
    monkeypatch.delenv("WTPSPLIT_STRIDE", raising=False)

    with pytest.raises(ValidationError, match="WTPSPLIT_BLOCK_SIZE.*WTPSPLIT_STRIDE"):
        GlobalSettings()


@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan"])
def test_wtpsplit_threshold_rejects_values_outside_unit_interval(monkeypatch, value):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("WTPSPLIT_THRESHOLD", value)
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_native_text_enabled_defaults_true():
    assert Settings.ocr.native_text_enabled is True


def test_native_text_min_chars_default():
    assert Settings.ocr.native_text_min_chars == 20


def test_native_text_min_printable_ratio_default():
    assert Settings.ocr.native_text_min_printable_ratio == pytest.approx(0.85)


def test_native_text_min_printable_ratio_env_override(monkeypatch):
    monkeypatch.setenv("OCR_NATIVE_TEXT_MIN_PRINTABLE_RATIO", "0.5")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr.native_text_min_printable_ratio == pytest.approx(0.5)


def test_no_worker_count_setting_is_exposed():
    # Worker count is pinned in build_server(), not configurable: one worker
    # already serves concurrent requests via the LitServe async loop + the
    # layout/segmenter GpuBatcher, and a second would duplicate the models,
    # split the batcher, and parallelize only GIL-bound Python.
    assert not hasattr(Settings.pipeline, "workers_per_device")


def test_pipeline_resource_limits_are_safe_by_default():
    assert Settings.pipeline.max_pages == 200
    assert Settings.pipeline.max_active_uploads == 8


def test_pipeline_resource_limits_reject_zero(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("PIPELINE_MAX_PAGES", "0")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_layout_batch_size_rejects_zero(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("LAYOUT_BATCH_SIZE", "0")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_flat_env_var_loading(monkeypatch):
    """`OCR_BACKEND=glm-llama` (section prefix + field) sets Settings.ocr.backend."""
    monkeypatch.setenv("OCR_BACKEND", "glm-llama")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr.backend == "glm-llama"


def test_ocr_options_default_to_paddle_profile_settings():
    from bibr.config import GlobalSettings

    ocr = GlobalSettings().ocr

    assert ocr.backend == "paddle"
    assert ocr.profile is None
    assert ocr.generation_max_tokens is None
    assert ocr.generation_temperature is None
    assert ocr.paddle_model == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert ocr.paddle_revision == "66317acc4c9fc17bd154591ce650735cd2855f3e"
    assert ocr.paddle_served_model == "paddle-ocr-vl-1.6"
    assert ocr.paddle_vllm_port == 8774
    assert ocr.paddle_vllm_startup_timeout == 900
    assert ocr.paddle_vllm_extra_args == ""
    assert ocr.paddle_mlx_model == "olragon/PaddleOCR-VL-1.6-8bit"
    assert ocr.paddle_mlx_port == 8775
    assert ocr.paddle_mlx_startup_timeout == 600
    assert ocr.paddle_mlx_extra_args == ""
    assert ocr.paddle_rapid_mlx_model == "olragon/PaddleOCR-VL-1.6-8bit"


def test_unknown_ocr_alias_requires_explicit_profile(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("OCR_BACKEND", "private-ocr")
    monkeypatch.setenv("OCR_MODEL", "private/model")
    monkeypatch.delenv("OCR_PROFILE", raising=False)

    with pytest.raises(ValidationError, match="OCR_PROFILE"):
        GlobalSettings()


@pytest.mark.parametrize("profile", ["paddle", "glm"])
def test_explicit_ocr_profile_allows_unknown_ocr_alias(monkeypatch, profile):
    from bibr.config import GlobalSettings

    monkeypatch.setenv("OCR_BACKEND", "private-ocr")
    monkeypatch.setenv("OCR_MODEL", "private/model")
    monkeypatch.setenv("OCR_PROFILE", profile)

    assert GlobalSettings().ocr.profile == profile


@pytest.mark.parametrize(
    "backend",
    [
        "paddle",
        "paddle-vllm",
        "paddle-rapid-mlx",
        "paddle-mlx-vlm",
        "paddle-http",
        "glm-llama",
        "glm-mlx",
        "glm-rapid-mlx",
        "glm-llama",
        "glm-http",
        "gemini",
        "openai",
        "anthropic",
    ],
)
def test_known_ocr_and_cloud_vision_backends_do_not_require_explicit_profile(monkeypatch, backend):
    from bibr.config import GlobalSettings

    monkeypatch.setenv("OCR_BACKEND", backend)
    monkeypatch.delenv("OCR_PROFILE", raising=False)

    assert GlobalSettings().ocr.backend == backend


def test_invalid_ref_seg_strategy_rejected(monkeypatch):
    """A typo'd strategy must fail loudly, not silently behave as 'llm'."""
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("REF_SEG_STRATEGY", "rfl")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_invalid_ref_parse_strategy_rejected(monkeypatch):
    from pydantic import ValidationError

    from bibr.config import GlobalSettings

    monkeypatch.setenv("REF_PARSE_STRATEGY", "grobid")
    with pytest.raises(ValidationError):
        GlobalSettings()


def test_ref_split_merged_refs_defaults_on(monkeypatch):
    monkeypatch.delenv("REF_SPLIT_MERGED_REFS", raising=False)
    from bibr.config import GlobalSettings

    assert GlobalSettings().REF_SPLIT_MERGED_REFS is True


def test_ref_strategy_defaults_are_local(monkeypatch):
    """Unset, parsing defaults to the local ModernBERT-CRF ("ner") and
    segmentation to the local geometry GBM ("geom").

    conftest pins REF_PARSE_STRATEGY=llm and REF_SEG_STRATEGY=llm for the
    suite, so delete them to see the real field defaults. The retired
    REF_EXTRACTION_STRATEGY alias must be gone entirely.
    """
    from bibr.config import GlobalSettings

    monkeypatch.delenv("REF_PARSE_STRATEGY", raising=False)
    monkeypatch.delenv("REF_SEG_STRATEGY", raising=False)
    s = GlobalSettings()
    assert not hasattr(s, "REF_EXTRACTION_STRATEGY")
    assert s.REF_PARSE_STRATEGY == "ner"
    assert s.REF_SEG_STRATEGY == "geom"  # local geometry segmenter is the default


def test_section_classifier_model_default_points_to_latest_hf_repo(monkeypatch):
    monkeypatch.delenv("ML_SECTION_CLASSIFIER_MODEL_ID", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ml.section_classifier_model_id == "scienceverse/bibr-section-classifier"


def test_front_role_defaults_to_the_published_bundle(monkeypatch):
    """The front-role tagger is on by default: it beat the heuristics on every
    column of the 192-paper gold replay (title 0.849 -> 0.901, byline 0.260 ->
    0.698) with nothing broken. Set the id to null to fall back to the lexical
    heuristics alone."""
    monkeypatch.delenv("ML_FRONT_ROLE_MODEL_ID", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ml.front_role_model_id == "scienceverse/bibr-front-role-v1"
    assert s.ml.front_role_revision == "7f01b57e1999d93cb5f17895ed10fdfa27cf6e0b"
    assert s.ml.front_role_enabled is True
    assert s.ml.front_role_min_confidence == 0.5
    assert s.ml.front_role_masthead_confidence == 0.8
    assert s.ml.front_role_record_root_confidence == 0.9


def test_paper_classifier_defaults_to_published_model(monkeypatch):
    """paper_classifier_model_id defaults to the published multitask classifier
    so the trained OECD/paper_type model is live by default (set to null to fall
    back to the LLM classification path)."""
    monkeypatch.delenv("ML_PAPER_CLASSIFIER_MODEL_ID", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ml.paper_classifier_model_id == "scienceverse/bibr-paper-classifier"
    assert s.ml.paper_classifier_revision == "6046171b3198a255acb1f07f81a586a32f399ac4"
    assert s.ml.paper_classifier_min_confidence == 0.5
    assert s.ml.paper_classifier_l2_min_confidence == 0.5
    assert s.ml.paper_classifier_llm_escalation is True
    assert s.ml.paper_classifier_device is None


def test_paper_classifier_model_id_overridable_via_env(monkeypatch):
    monkeypatch.setenv("ML_PAPER_CLASSIFIER_MODEL_ID", "other-org/custom-classifier")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ml.paper_classifier_model_id == "other-org/custom-classifier"


def test_classifier_resource_defaults(monkeypatch):
    monkeypatch.delenv("ML_CLASSIFIERS_REQUIRED", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ml.classifiers_required is False
    assert s.ml.paper_classifier_batch_size == 64
    assert s.ml.section_classifier_batch_size == 64
    assert s.ml.paper_classifier_batch_timeout_ms == 5.0
    assert s.ml.section_classifier_batch_timeout_ms == 5.0
    assert s.ml.classifier_vram_safety_reserve_mb == 2048
    assert s.ml.paper_classifier_estimated_peak_mb == 1536
    assert s.ml.section_classifier_estimated_peak_mb == 512


def test_upload_spool_memory_default(monkeypatch):
    monkeypatch.delenv("PIPELINE_UPLOAD_SPOOL_MEMORY_BYTES", raising=False)
    from bibr.config import GlobalSettings

    assert GlobalSettings().pipeline.upload_spool_memory_bytes == 1024 * 1024


def test_distributed_singleflight_defaults_on_and_is_opt_out(monkeypatch):
    monkeypatch.setenv("CACHE_DISTRIBUTED_SINGLEFLIGHT", "false")
    from bibr.config import GlobalSettings

    opted_out = GlobalSettings()
    assert opted_out.cache.distributed_singleflight is False
    monkeypatch.delenv("CACHE_DISTRIBUTED_SINGLEFLIGHT")
    defaults = GlobalSettings()
    assert defaults.cache.distributed_singleflight is True
    assert defaults.cache.singleflight_wait_seconds == 10.0
    assert defaults.cache.singleflight_lease_ttl_seconds == 120
    assert defaults.cache.singleflight_renew_interval_seconds == 30.0
    assert defaults.cache.singleflight_poll_interval_ms == 100


def test_ref_strategy_case_insensitive(monkeypatch):
    """Uppercase strategy values are accepted and normalized."""
    from bibr.config import GlobalSettings

    monkeypatch.setenv("REF_SEG_STRATEGY", "CRF")
    monkeypatch.setenv("REF_PARSE_STRATEGY", "NER")
    s = GlobalSettings()
    assert s.REF_SEG_STRATEGY == "crf"
    assert s.REF_PARSE_STRATEGY == "ner"


def test_dotenv_value_with_dollar_is_not_interpolated(tmp_path, monkeypatch):
    """A `.env` value literally containing ``${VAR}`` must be read verbatim.

    python-dotenv (used by pydantic-settings) interpolates ``${VAR}`` by
    default, which silently mangles secrets/URLs — e.g. a Redis password or API
    key containing ``${HOME}`` is shell-expanded on load, and bibr's own
    ``parse_env`` reads the literal value, so the loaded config diverges from
    what ``preset diff``/``show`` displays. bibr disables interpolation.
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("HOME", "/home/should-not-expand")
    env = tmp_path / ".env"
    env.write_text('LLM_API_KEY="abc${HOME}xyz"\n', encoding="utf-8")

    from bibr.config import LlmOptions

    opts = LlmOptions(_env_file=str(env))
    assert opts.api_key == "abc${HOME}xyz"


def test_dotenv_loading_can_be_disabled(tmp_path, monkeypatch):
    """``BIBR_DISABLE_DOTENV=1`` makes every section ignore ``.env`` files.

    Harnesses pin their variables in the environment; a developer's ``.env``
    in the checkout (or ``~/.bibr/.env``) must not supply the ones they forgot.
    The process environment still wins, and the knob applies to section
    models (``LlmOptions`` here), not only the top-level settings object.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("BIBR_DISABLE_DOTENV", raising=False)
    # conftest pins BIBR_ENV_FILE="" suite-wide; this test is about the default
    # chain, so it has to opt back into it.
    monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("LLM_MODEL=from-dotenv\n", encoding="utf-8")

    from bibr.config import GlobalSettings, dotenv_disabled, dotenv_files_present

    assert dotenv_files_present() == [tmp_path / ".env"]
    assert GlobalSettings().llm.model == "from-dotenv"

    monkeypatch.setenv("BIBR_DISABLE_DOTENV", "1")
    assert dotenv_disabled()
    assert GlobalSettings().llm.model != "from-dotenv"
    monkeypatch.setenv("LLM_MODEL", "from-env")
    assert GlobalSettings().llm.model == "from-env"


def test_crossref_email_from_flat_env(monkeypatch):
    """`CROSSREF_API_EMAIL=...` sets Settings.crossref.api_email."""
    monkeypatch.setenv("CROSSREF_API_EMAIL", "bot@example.com")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.crossref.api_email == "bot@example.com"


def test_llama_cpp_context_size_defaults():
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    # LLM ctx raised to fit the 24k-char input cap + 4k completion cap on 6 GB.
    assert s.llm.llama_cpp_context_size == 16384
    # OCR ctx stays modest (image tokens dominate, not text ctx).
    assert s.ocr.llama_cpp_context_size == 8192


def test_ocr_vision_defaults(monkeypatch):
    monkeypatch.setenv("OCR_BACKEND", "glm-llama")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr_vision.provider == "google"
    assert s.ocr_vision.model == "gemini-3-flash-preview"
    assert s.ocr_vision.base_url is None
    assert s.ocr_vision.rate_limit_rpm == 30
    assert s.ocr_vision.max_tokens == 16384
    assert s.ocr_vision.timeout_seconds == 60


def test_ocr_vision_env_override(monkeypatch):
    monkeypatch.setenv("OCR_BACKEND", "glm-llama")
    monkeypatch.setenv("OCR_VISION_PROVIDER", "openai")
    monkeypatch.setenv("OCR_VISION_MODEL", "gpt-4o")
    monkeypatch.setenv("OCR_VISION_RATE_LIMIT_RPM", "60")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr_vision.provider == "openai"
    assert s.ocr_vision.model == "gpt-4o"
    assert s.ocr_vision.rate_limit_rpm == 60


def test_ocr_concurrency_apple_silicon_default(monkeypatch):
    """Apple Silicon defaults to one OCR request to avoid queued prefill work."""
    monkeypatch.delenv("OCR_MAX_CONCURRENT_REGIONS", raising=False)
    monkeypatch.delenv("OCR_CONCURRENT_REGIONS_PER_FILE", raising=False)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr.max_concurrent_regions == 1
    assert s.ocr.concurrent_regions_per_file == 1


def test_ocr_concurrency_non_mac_default(monkeypatch):
    """On Linux/CUDA hosts defaults stay at the GPU-scaled values."""
    monkeypatch.delenv("OCR_MAX_CONCURRENT_REGIONS", raising=False)
    monkeypatch.delenv("OCR_CONCURRENT_REGIONS_PER_FILE", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr.max_concurrent_regions == 16 * s.ocr.local_gpus
    assert s.ocr.concurrent_regions_per_file == 6


def test_ocr_concurrency_explicit_override(monkeypatch):
    """Explicit env vars override the Apple-Silicon auto-default."""
    monkeypatch.setenv("OCR_MAX_CONCURRENT_REGIONS", "8")
    monkeypatch.setenv("OCR_CONCURRENT_REGIONS_PER_FILE", "4")
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ocr.max_concurrent_regions == 8
    assert s.ocr.concurrent_regions_per_file == 4


def test_ocr_user_set_concurrency_distinguishes_env_from_auto_tune(tmp_path, monkeypatch):
    """``compute_ocr_concurrency`` assigns the concurrency fields itself, and
    pydantic v2 adds assigned fields to ``model_fields_set`` — so membership
    alone cannot tell "user set this" from "the validator auto-tuned this".
    The validator must record the user-provided subset BEFORE assigning."""
    from bibr.config import GlobalSettings

    # GlobalSettings() reads OCR_* via the CWD-then-home dotenv fallback chain
    # (bibr.config._default_env_files); monkeypatch.delenv cannot neutralize a
    # dotenv-sourced value, so a dev machine with OCR_* in .env would fail this
    # test. Point both dotenv sources at empty dirs, matching
    # TestEnvFileFallbackChain's hermetic pattern below.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    monkeypatch.delenv("OCR_MAX_CONCURRENT_REGIONS", raising=False)
    monkeypatch.delenv("OCR_CONCURRENT_REGIONS_PER_FILE", raising=False)
    s = GlobalSettings()
    assert s.ocr.user_set_concurrency == frozenset()
    # The poisoning this exists to work around: the auto-tune assignment puts
    # the field in model_fields_set even though the user never set it.
    assert "max_concurrent_regions" in s.ocr.model_fields_set

    monkeypatch.setenv("OCR_MAX_CONCURRENT_REGIONS", "4")
    s2 = GlobalSettings()
    assert s2.ocr.user_set_concurrency == frozenset({"max_concurrent_regions"})
    assert s2.ocr.max_concurrent_regions == 4

    monkeypatch.setenv("OCR_CONCURRENT_REGIONS_PER_FILE", "3")
    s3 = GlobalSettings()
    assert s3.ocr.user_set_concurrency == frozenset(
        {"max_concurrent_regions", "concurrent_regions_per_file"}
    )


def test_api_key_aliases_work_without_prefix(monkeypatch):
    """Hoisted API-key survivors must resolve via bare env-var names."""
    for bare_name, field in [
        ("GOOGLE_API_KEY", "GOOGLE_API_KEY"),
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        ("LANGEXTRACT_API_KEY", "GOOGLE_API_KEY"),
        ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY"),
        ("GROQ_API_KEY", "GROQ_API_KEY"),
    ]:
        monkeypatch.setenv(bare_name, f"test-{bare_name}")
        from bibr.config import GlobalSettings

        s = GlobalSettings()
        assert getattr(s, field) == f"test-{bare_name}", f"{bare_name} did not resolve to {field}"
        monkeypatch.delenv(bare_name, raising=False)


def test_auth_options_default_is_no_token(monkeypatch):
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.auth.api_key is None
    assert s.auth.required is False


def test_auth_options_reads_env(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "sk_test_abc")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.auth.api_key == "sk_test_abc"
    assert s.auth.required is True


def test_auth_options_empty_string_is_treated_as_no_token(monkeypatch):
    monkeypatch.setenv("AUTH_API_KEY", "")
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.auth.required is False


def test_cors_origins_default_is_empty(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.cors.origins == []


class TestBehaviorFingerprint:
    """The serve cache namespace must change when behavior-affecting settings
    change (model, provider, ref strategies...), so a restart with new config
    can never serve stale results — but must NOT change on secrets."""

    def _fresh(self, monkeypatch, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from bibr.config import GlobalSettings

        return GlobalSettings()

    def test_changes_when_llm_model_changes(self, monkeypatch):
        from bibr.config import compute_behavior_fingerprint

        a = compute_behavior_fingerprint(self._fresh(monkeypatch, LLM_MODEL="model-a"))
        b = compute_behavior_fingerprint(self._fresh(monkeypatch, LLM_MODEL="model-b"))
        assert a != b

    def test_changes_when_ref_parse_strategy_changes(self, monkeypatch):
        from bibr.config import compute_behavior_fingerprint

        a = compute_behavior_fingerprint(self._fresh(monkeypatch, REF_PARSE_STRATEGY="ner"))
        b = compute_behavior_fingerprint(self._fresh(monkeypatch, REF_PARSE_STRATEGY="llm"))
        assert a != b

    def test_stable_across_secret_changes(self, monkeypatch):
        from bibr.config import compute_behavior_fingerprint

        a = compute_behavior_fingerprint(self._fresh(monkeypatch, LLM_API_KEY="secret-1"))
        b = compute_behavior_fingerprint(self._fresh(monkeypatch, LLM_API_KEY="secret-2"))
        assert a == b

    def test_stable_across_top_level_api_key_changes(self, monkeypatch):
        """Top-level GOOGLE/ANTHROPIC/GROQ_API_KEY must not sit in the fingerprint
        preimage (they were only filtered by section name, not key name) — else
        the plaintext key is hashed and rotation churns the cache (audit M3)."""
        from bibr.config import compute_behavior_fingerprint

        for var in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
            a = compute_behavior_fingerprint(self._fresh(monkeypatch, **{var: "AIzaSy-one"}))
            b = compute_behavior_fingerprint(self._fresh(monkeypatch, **{var: "AIzaSy-two-diff"}))
            assert a == b, var

    def test_cache_namespace_composes_version_and_fingerprint(self, monkeypatch):
        from bibr.config import cache_namespace, compute_behavior_fingerprint

        s = self._fresh(monkeypatch, CACHE_VERSION="v-test")
        assert cache_namespace(s) == f"bibr:v-test:{compute_behavior_fingerprint(s)}"


def test_production_env_does_not_break_settings_construction(monkeypatch):
    # `import bibr.config` on a production-flagged box must not crash even when
    # Redis/auth env is incomplete — otherwise `bibr --help` and `bibr setup`
    # (the tools that would fix the config) die at import. Hardening is
    # enforced at serve startup via validate_production_settings instead.
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.ENVIRONMENT == "production"


def test_cors_wildcard_in_production_raises(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_PASSWORD", "x")
    monkeypatch.setenv("AUTH_API_KEY", "k" * 32)
    monkeypatch.setenv("CORS_ORIGINS", '["*"]')
    from bibr.config import GlobalSettings, validate_production_settings

    with pytest.raises(ValueError, match="cors.origins"):
        validate_production_settings(GlobalSettings())


def test_auth_api_key_required_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_PASSWORD", "x")
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    from bibr.config import GlobalSettings, validate_production_settings

    with pytest.raises(ValueError, match="auth.api_key"):
        validate_production_settings(GlobalSettings())


def test_auth_api_key_must_have_minimum_length_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_PASSWORD", "x")
    monkeypatch.setenv("AUTH_API_KEY", "too-short")
    from bibr.config import GlobalSettings, validate_production_settings

    with pytest.raises(ValueError, match="at least 32"):
        validate_production_settings(GlobalSettings())


def test_redis_password_required_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.setenv("AUTH_API_KEY", "k")
    from bibr.config import GlobalSettings, validate_production_settings

    with pytest.raises(ValueError, match="redis.password"):
        validate_production_settings(GlobalSettings())


def test_validate_production_settings_noop_outside_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("AUTH_API_KEY", raising=False)
    from bibr.config import GlobalSettings, validate_production_settings

    validate_production_settings(GlobalSettings())  # must not raise


def test_cors_explicit_origins_pass_through(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", '["https://app.example.com"]')
    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.cors.origins == ["https://app.example.com"]


# --- Comma-separated list settings (audit [11]) ------------------------------


def test_mcp_url_allowed_hosts_accepts_the_documented_comma_form(monkeypatch):
    """This is the SSRF mitigation for chew_url, and docs/guides/mcp.md
    prescribes the comma-separated form. A bare list[str] field is
    JSON-decoded by pydantic-settings, so following the docs raised
    SettingsError on the first attribute access of Settings anywhere — taking
    bibr serve down with an unhandled traceback, which meant no deployment
    realistically had host allowlisting on."""
    monkeypatch.setenv("MCP_URL_ALLOWED_HOSTS", "arxiv.org,zenodo.org")
    from bibr.config import McpOptions

    assert McpOptions().url_allowed_hosts == ["arxiv.org", "zenodo.org"]


def test_mcp_url_allowed_hosts_lowercases_and_trims(monkeypatch):
    monkeypatch.setenv("MCP_URL_ALLOWED_HOSTS", " ArXiv.org , ZENODO.org ,")
    from bibr.config import McpOptions

    assert McpOptions().url_allowed_hosts == ["arxiv.org", "zenodo.org"]


def test_mcp_url_allowed_hosts_still_accepts_json(monkeypatch):
    monkeypatch.setenv("MCP_URL_ALLOWED_HOSTS", '["arxiv.org"]')
    from bibr.config import McpOptions

    assert McpOptions().url_allowed_hosts == ["arxiv.org"]


def test_cors_lists_accept_the_comma_form(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")
    monkeypatch.setenv("CORS_ALLOW_METHODS", "GET,POST")
    monkeypatch.setenv("CORS_ALLOW_HEADERS", "Authorization,Content-Type")
    from bibr.config import CorsOptions

    options = CorsOptions()
    assert options.origins == ["https://a.example", "https://b.example"]
    assert options.allow_methods == ["GET", "POST"]
    assert options.allow_headers == ["Authorization", "Content-Type"]


# --- Configuration errors must not print credentials (audit [10]) ------------


def test_model_level_validation_error_does_not_print_the_source_mapping(monkeypatch):
    """For a model-level validator pydantic reports loc == () and input ==
    the whole merged source mapping — every env/dotenv value matching a field
    on that model. The settings models' own redaction cannot apply, because
    what pydantic hands back is a plain dict."""
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("GOOGLE_API_KEY", "google-plaintext-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-plaintext-secret")
    monkeypatch.setenv("WTPSPLIT_BLOCK_SIZE", "512")
    monkeypatch.delenv("WTPSPLIT_STRIDE", raising=False)
    from bibr.config import GlobalSettings, _configuration_error

    with pytest.raises(ValidationError) as excinfo:
        GlobalSettings()
    message = str(_configuration_error(excinfo.value))

    assert "google-plaintext-secret" not in message
    assert "anthropic-plaintext-secret" not in message
    assert "must be set together" in message


def test_field_level_error_masks_a_secret_value():
    from pydantic import ValidationError

    from bibr.config import _configuration_error

    error = ValidationError.from_exception_data(
        "AuthOptions",
        [
            {
                "type": "string_too_short",
                "loc": ("api_key",),
                "input": "too-short-but-still-a-credential",
                "ctx": {"min_length": 32},
            }
        ],
    )

    assert "too-short-but-still-a-credential" not in str(_configuration_error(error))


# --- Cache fingerprint must see LLM budgets (audit [12]) ---------------------


def test_max_tokens_fields_are_not_treated_as_secrets():
    """An unanchored "token" substring match hid every *_max_tokens knob from
    compute_behavior_fingerprint, so changing a budget left the serve cache
    namespace byte-identical and old-budget results were re-served."""
    from bibr.config import _is_secret_name

    assert not _is_secret_name("LLM_MAX_TOKENS")
    assert not _is_secret_name("REF_PARSE_MAX_TOKENS")
    assert not _is_secret_name("section_max_tokens")

    assert _is_secret_name("GOOGLE_API_KEY")
    assert _is_secret_name("api_key")
    assert _is_secret_name("password")
    assert _is_secret_name("api_email")


def test_changing_a_max_tokens_budget_changes_the_behavior_fingerprint(monkeypatch):
    from bibr.config import GlobalSettings, compute_behavior_fingerprint

    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    before = compute_behavior_fingerprint(GlobalSettings())
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    after = compute_behavior_fingerprint(GlobalSettings())

    assert before != after


def test_rotating_a_credential_still_leaves_the_fingerprint_alone(monkeypatch):
    from bibr.config import GlobalSettings, compute_behavior_fingerprint

    monkeypatch.setenv("GOOGLE_API_KEY", "key-one")
    before = compute_behavior_fingerprint(GlobalSettings())
    monkeypatch.setenv("GOOGLE_API_KEY", "key-two")
    after = compute_behavior_fingerprint(GlobalSettings())

    assert before == after


def test_redis_url_left_none_when_neither_password_nor_url_set(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")  # avoid prod-validator path

    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.redis.url is None, f"expected None when nothing configured, got {s.redis.url!r}"


def test_redis_url_built_from_password_when_only_password_set(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_PASSWORD", "secret")
    monkeypatch.setenv("ENVIRONMENT", "development")

    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.redis.url == "redis://:secret@redis:6379/0"


def test_redis_url_passes_through_when_only_url_set(monkeypatch):
    """Cloud Redis: user supplies REDIS_URL with embedded creds (or none) and
    no separate REDIS_PASSWORD. The URL must pass through unchanged."""
    monkeypatch.setenv("REDIS_URL", "rediss://my-host:6380/0")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")

    from bibr.config import GlobalSettings

    s = GlobalSettings()
    assert s.redis.url == "rediss://my-host:6380/0"


def test_compute_code_hash_memoized_across_settings(monkeypatch):
    """Repeated calls reuse the module-level cache instead of re-walking the
    package tree."""
    import bibr.config as mod

    monkeypatch.setattr(mod, "_CODE_HASH_CACHED", None)

    walks = {"n": 0}
    real_rglob = Path.rglob

    def counting_rglob(self, pattern):
        walks["n"] += 1
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", counting_rglob)

    h1 = mod._compute_code_hash()
    h2 = mod._compute_code_hash()
    assert h1 == h2
    assert walks["n"] == 1, f"_compute_code_hash walked {walks['n']}x; should walk once"


def test_compute_code_hash_lazy_at_import(monkeypatch):
    """Default-factory must defer to first read; importing ``bibr.config``
    alone (without instantiating settings) must not walk the tree."""
    import bibr.config as mod

    monkeypatch.setattr(mod, "_CODE_HASH_CACHED", None)
    calls = {"n": 0}

    def spy():
        calls["n"] += 1
        return "auto-spied42"

    monkeypatch.setattr(mod, "_compute_code_hash", spy)

    # No call yet — module already imported, but no settings constructed
    # in this test scope.
    s = mod.GlobalSettings()
    v1 = s.cache.version
    assert calls["n"] == 1
    # Reading again on the same instance is O(1) — pydantic stores the
    # default value in the model.
    v2 = s.cache.version
    assert v1 == v2 == "auto-spied42"
    assert calls["n"] == 1


def test_crossref_cache_defaults_preserve_today_behavior():
    from bibr.config import CrossrefOptions

    opts = CrossrefOptions()
    assert opts.redis_cache is False
    assert opts.cache_redis_url is None
    assert opts.cache_ttl_seconds == 2_592_000


def test_crossref_cache_env_override(monkeypatch):
    monkeypatch.setenv("CROSSREF_REDIS_CACHE", "true")
    monkeypatch.setenv("CROSSREF_CACHE_REDIS_URL", "redis://cache:6379/0")
    monkeypatch.setenv("CROSSREF_CACHE_TTL_SECONDS", "3600")

    from bibr.config import CrossrefOptions

    opts = CrossrefOptions()
    assert opts.redis_cache is True
    assert opts.cache_redis_url == "redis://cache:6379/0"
    assert opts.cache_ttl_seconds == 3600


class TestEnvFileFallbackChain:
    """Running `bibr` from a directory with no `.env` must not silently lose
    all configuration — it should fall back to `~/.bibr/.env`. A CWD `.env`,
    when present, still wins; a real environment variable wins over both.

    Never touches the developer's real home directory: `Path.home` is
    monkeypatched to a `tmp_path` fixture for every test in this class.
    """

    @pytest.fixture(autouse=True)
    def _fake_home(self, tmp_path, monkeypatch):
        # conftest disables dotenv loading suite-wide (BIBR_ENV_FILE=""); this
        # class is the one place that must exercise the real chain, so it opts
        # back in — against a faked home, never the developer's.
        monkeypatch.delenv("BIBR_ENV_FILE", raising=False)
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        self.fake_home = fake_home

    def test_falls_back_to_home_env_when_cwd_has_none(self, tmp_path, monkeypatch):
        bibr_dir = self.fake_home / ".bibr"
        bibr_dir.mkdir()
        (bibr_dir / ".env").write_text("LLM_MODEL=from-home-env\n", encoding="utf-8")

        workdir = tmp_path / "workdir"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        from bibr.config import LlmOptions

        opts = LlmOptions()
        assert opts.model == "from-home-env"

    def test_cwd_env_wins_over_home_env(self, tmp_path, monkeypatch):
        bibr_dir = self.fake_home / ".bibr"
        bibr_dir.mkdir()
        (bibr_dir / ".env").write_text("LLM_MODEL=from-home-env\n", encoding="utf-8")

        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / ".env").write_text("LLM_MODEL=from-cwd-env\n", encoding="utf-8")
        monkeypatch.chdir(workdir)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        from bibr.config import LlmOptions

        opts = LlmOptions()
        assert opts.model == "from-cwd-env"

    def test_real_env_var_overrides_both_files(self, tmp_path, monkeypatch):
        bibr_dir = self.fake_home / ".bibr"
        bibr_dir.mkdir()
        (bibr_dir / ".env").write_text("LLM_MODEL=from-home-env\n", encoding="utf-8")

        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / ".env").write_text("LLM_MODEL=from-cwd-env\n", encoding="utf-8")
        monkeypatch.chdir(workdir)
        monkeypatch.setenv("LLM_MODEL", "from-real-env-var")

        from bibr.config import LlmOptions

        opts = LlmOptions()
        assert opts.model == "from-real-env-var"


class TestEnvFileOverride:
    """``BIBR_ENV_FILE`` replaces the dotenv chain outright.

    Empty means "load no dotenv file", which is how a process (the test suite,
    a container, CI) stays hermetic from whatever ``.env`` happens to sit in its
    CWD or home. Real env vars already outranked dotenv; this covers every
    setting they don't pin.
    """

    @staticmethod
    def _plant_env_files(tmp_path, monkeypatch):
        fake_home = tmp_path / "fake_home"
        (fake_home / ".bibr").mkdir(parents=True)
        (fake_home / ".bibr" / ".env").write_text("LLM_MODEL=from-home-env\n", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / ".env").write_text("LLM_MODEL=from-cwd-env\n", encoding="utf-8")
        monkeypatch.chdir(workdir)
        monkeypatch.delenv("LLM_MODEL", raising=False)

    def test_empty_override_ignores_cwd_and_home_env_files(self, tmp_path, monkeypatch):
        self._plant_env_files(tmp_path, monkeypatch)
        monkeypatch.setenv("BIBR_ENV_FILE", "")

        from bibr.config import LlmOptions

        assert LlmOptions().model not in {"from-cwd-env", "from-home-env"}

    def test_explicit_override_replaces_the_chain(self, tmp_path, monkeypatch):
        self._plant_env_files(tmp_path, monkeypatch)
        elsewhere = tmp_path / "elsewhere.env"
        elsewhere.write_text("LLM_MODEL=from-override\n", encoding="utf-8")
        monkeypatch.setenv("BIBR_ENV_FILE", str(elsewhere))

        from bibr.config import LlmOptions

        assert LlmOptions().model == "from-override"

    def test_explicit_env_file_argument_still_outranks_the_override(self, tmp_path, monkeypatch):
        self._plant_env_files(tmp_path, monkeypatch)
        chosen = tmp_path / "chosen.env"
        chosen.write_text("LLM_MODEL=from-argument\n", encoding="utf-8")
        monkeypatch.setenv("BIBR_ENV_FILE", "")

        from bibr.config import LlmOptions

        assert LlmOptions(_env_file=str(chosen)).model == "from-argument"


class TestSecretRedaction:
    """repr(Settings) / model_dump() must not leak plaintext secrets (audit M3)."""

    def test_repr_masks_secret_fields(self):
        from bibr.config import GlobalSettings

        s = GlobalSettings()
        s.llm.api_key = "AIzaSy-nested-secret-value"
        s.GOOGLE_API_KEY = "AIzaSy-top-level-secret"
        s.redis.password = "redis-secret-pw"
        text = repr(s) + repr(s.llm) + repr(s.redis)
        assert "AIzaSy-nested-secret-value" not in text
        assert "AIzaSy-top-level-secret" not in text
        assert "redis-secret-pw" not in text
        # A non-secret field is still visible for debugging.
        assert s.llm.provider in repr(s.llm)

    def test_model_dump_masks_secret_fields(self):
        from bibr.config import GlobalSettings

        s = GlobalSettings()
        s.crossref.api_key = "cr-secret-key-123"
        s.GROQ_API_KEY = "gsk-secret-top-level"
        dumped = s.model_dump()
        blob = str(dumped)
        assert "cr-secret-key-123" not in blob
        assert "gsk-secret-top-level" not in blob
        # Non-secret nested values survive round-trip.
        assert dumped["llm"]["provider"] == s.llm.provider
