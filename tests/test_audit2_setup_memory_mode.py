"""``bibr setup`` shows the memory mode a run will get and leaves choosing it to the run."""

from __future__ import annotations

import pytest

from bibr.setup_wizard import _build_recommended_setup
from tests.test_setup_wizard import _drive_easy_setup


def _fast(**machine):
    return _build_recommended_setup(system_memory_gb=32.0, fast=True, **machine)


def test_no_recommended_plan_writes_a_memory_mode():
    plans = [
        _fast(platform_key="cuda", accelerator_memory_gb=6.0, sys_platform="linux"),
        _build_recommended_setup(
            platform_key="cuda",
            accelerator_memory_gb=10.0,
            system_memory_gb=32.0,
            sys_platform="linux",
            machine="x86_64",
        ),
        _build_recommended_setup(system_memory_gb=32.0, allow_cloud=True),
        _build_recommended_setup(system_memory_gb=32.0),
    ]

    assert [plan.tier for plan in plans] == [
        "fast",
        "fully_local",
        "cloud_fallback",
        "private_server",
    ]
    assert not any("PIPELINE_MEMORY_MODE" in plan.env for plan in plans)


def test_a_fast_plan_counts_a_small_card_only_when_the_gpu_extra_goes_onto_it(monkeypatch):
    monkeypatch.setattr("bibr.ocr.registry._nvidia_compute_capability", lambda: 8.6)
    linux = _fast(
        platform_key="cuda", accelerator_memory_gb=6.0, sys_platform="linux", machine="x86_64"
    )
    windows = _fast(
        platform_key="cuda", accelerator_memory_gb=6.0, sys_platform="win32", machine="AMD64"
    )

    assert "gpu" in linux.extras and linux.memory_mode == "aggressive"
    assert "gpu" not in windows.extras and windows.memory_mode == "balanced"


def test_plans_with_remote_ocr_and_llm_follow_system_ram_not_a_small_card():
    weak_card = {"platform_key": "cuda", "accelerator_memory_gb": 4.0}

    cloud = _build_recommended_setup(system_memory_gb=32.0, allow_cloud=True, **weak_card)
    private = _build_recommended_setup(system_memory_gb=32.0, **weak_card)
    low_ram = _build_recommended_setup(system_memory_gb=8.0, **weak_card)

    assert (cloud.tier, cloud.memory_mode) == ("cloud_fallback", "balanced")
    assert (private.tier, private.memory_mode) == ("private_server", "balanced")
    assert low_ram.memory_mode == "aggressive"


def _memory_lines(env_path) -> list[str]:
    return [
        line
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if "PIPELINE_MEMORY_MODE" in line
    ]


@pytest.mark.parametrize("pinned", ["aggressive", "balanced"])
def test_a_merge_comments_out_the_mode_an_earlier_setup_wrote(monkeypatch, tmp_path, pinned):
    """On an 8 GB card setup wrote ``aggressive`` (no managed LLM) and, with one, the
    same: both are the rule's answers there, and ``balanced`` is on a 24 GB card."""
    hardware = ("cuda", 8.0) if pinned == "aggressive" else ("cuda", 24.0)
    existing = f"PIPELINE_MODE=fast\nPIPELINE_MEMORY_MODE={pinned}\nMY_OWN_SETTING=keep-me\n"

    wizard = _drive_easy_setup(
        monkeypatch,
        tmp_path,
        hardware=hardware,
        existing=existing,
        typed={f"{tmp_path / '.env'} already exists": "merge"},
    )

    assert _memory_lines(wizard.env_path) == [f"# PIPELINE_MEMORY_MODE={pinned}"]
    assert "MY_OWN_SETTING=keep-me" in wizard.env_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("hardware", "pinned"),
    [(("cuda", 24.0), "aggressive"), (("cuda", 8.0), "keep_all"), ((None, None), "keep_all")],
)
def test_a_merge_keeps_a_mode_the_rule_never_gives_on_this_machine(
    monkeypatch, tmp_path, hardware, pinned
):
    existing = f"PIPELINE_MODE=fast\nPIPELINE_MEMORY_MODE={pinned}\n"

    wizard = _drive_easy_setup(
        monkeypatch,
        tmp_path,
        hardware=hardware,
        existing=existing,
        typed={f"{tmp_path / '.env'} already exists": "merge"},
    )

    assert _memory_lines(wizard.env_path) == [f"PIPELINE_MEMORY_MODE={pinned}"]
