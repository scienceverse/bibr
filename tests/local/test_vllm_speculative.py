"""Speculative decoding for the managed vLLM LLM server (``LLM_VLLM_SPEC_DECODE``).

Runs on vLLM 0.32: MTP k=3 was quality-neutral where one LLM server ran for the
whole batch, and cut its LLM stage to 0.80x / 0.74x (RTX 3090 / 5090). Where
the server restarts every chunk to take turns with PaddleOCR-VL, each MTP
start took 51 s instead of 37 s and 12 and 16 GB batches ran 5% and 9%
slower, so ``auto`` leaves it off there. DFlash k=7 is opt-in.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from bibr.config import GlobalSettings
from bibr.local import vllm_share, vllm_speculative
from bibr.local.vllm_share import ManagedLlmVramError, managed_llm_floor_gb
from bibr.local.vllm_speculative import (
    choose_spec_decode,
    explicit_config,
    user_sets_speculative,
)
from bibr.ocr.registry import GpuMemory

BF16 = "scienceverse/bibr-extract-qwen35-4b"
W4A16 = "scienceverse/bibr-extract-qwen35-4b-W4A16"
TWO_B = "scienceverse/bibr-extract-qwen35-2b"
LOCAL = "/models/bibr-extract-qwen35-4b-W4A16"
CUDA_3090 = 23.56
MTP_ARGS = (
    "--speculative-config",
    '{"method":"mtp","num_speculative_tokens":3}',
    "--no-async-scheduling",
)


def _settings(mode: str = "auto", extra: str = "") -> GlobalSettings:
    settings = GlobalSettings()
    settings.llm.vllm_spec_decode = mode
    settings.llm.vllm_extra_args = extra
    return settings


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(vllm_speculative, "_unlisted_warned", set())
    monkeypatch.setattr(vllm_share, "_floor_setting_ignored_for", set())


class TestSetting:
    def test_auto_by_default(self):
        assert GlobalSettings().llm.vllm_spec_decode == "auto"

    @pytest.mark.parametrize(("raw", "mode"), [("MTP", "mtp"), (" DFlash ", "dflash")])
    def test_read_from_the_environment_in_any_case(self, monkeypatch, raw, mode):
        monkeypatch.setenv("LLM_VLLM_SPEC_DECODE", raw)
        assert GlobalSettings().llm.vllm_spec_decode == mode

    def test_unknown_values_are_rejected(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("LLM_VLLM_SPEC_DECODE", "eagle")
        with pytest.raises(ValidationError):
            GlobalSettings()


class TestAuto:
    @pytest.mark.parametrize(
        ("model", "floor"),
        [(BF16, 12.0), (W4A16, 7.25)],
    )
    def test_a_resident_server_at_or_above_the_mtp_floor_gets_mtp(self, model, floor):
        for budget in (floor, floor + 0.01, 20.03):
            choice = choose_spec_decode(_settings(), model, swaps_with_ocr=False, budget_gb=budget)
            assert choice.server_args() == MTP_ARGS, budget
            assert choice.describe().startswith("mtp k=3 (auto: the server stays up")

    @pytest.mark.parametrize(("model", "floor"), [(BF16, 12.0), (W4A16, 7.25)])
    def test_a_budget_below_the_mtp_floor_starts_without_it(self, model, floor):
        choice = choose_spec_decode(
            _settings(), model, swaps_with_ocr=False, budget_gb=floor - 0.01
        )
        assert choice.config is None
        assert f"below MTP's {floor:.2f} GiB floor" in choice.reason

    def test_a_share_rounded_to_the_card_meets_the_floor(self):
        # The share bibr launches for a 12.0 GiB floor on an RTX 3090: 0.51.
        choice = choose_spec_decode(
            _settings(), BF16, swaps_with_ocr=False, budget_gb=0.51 * CUDA_3090
        )
        assert choice.config is not None

    @pytest.mark.parametrize("model", [BF16, W4A16])
    def test_a_server_taking_turns_with_paddleocr_vl_gets_none(self, model):
        choice = choose_spec_decode(_settings(), model, swaps_with_ocr=True, budget_gb=20.03)
        assert choice.config is None
        assert "restarts every chunk" in choice.reason

    def test_an_unknown_budget_gets_none(self):
        choice = choose_spec_decode(_settings(), BF16, swaps_with_ocr=False, budget_gb=None)
        assert choice.config is None

    @pytest.mark.parametrize("model", [TWO_B, LOCAL, "org/unknown"])
    def test_models_without_a_measured_config_get_none(self, model):
        choice = choose_spec_decode(_settings(), model, swaps_with_ocr=False, budget_gb=20.0)
        assert choice.config is None
        assert "no MTP config" in choice.reason

    def test_auto_never_raises_the_floor(self):
        assert managed_llm_floor_gb(BF16, _settings()) == 11.0
        assert managed_llm_floor_gb(W4A16, _settings()) == 6.0

    def test_none_turns_it_off(self):
        choice = choose_spec_decode(_settings("none"), BF16, swaps_with_ocr=False, budget_gb=20.0)
        assert choice.config is None
        assert managed_llm_floor_gb(BF16, _settings("none")) == 11.0


class TestExplicit:
    def test_mtp_is_used_on_the_swap_path_too_and_raises_the_floor(self):
        choice = choose_spec_decode(_settings("mtp"), W4A16, swaps_with_ocr=True, budget_gb=None)
        assert choice.server_args() == MTP_ARGS
        assert managed_llm_floor_gb(W4A16, _settings("mtp")) == 7.25
        assert managed_llm_floor_gb(BF16, _settings("mtp")) == 12.0

    def test_dflash_pins_its_drafter_and_needs_its_floor(self):
        choice = choose_spec_decode(_settings("dflash"), BF16, swaps_with_ocr=True, budget_gb=None)
        assert choice.config is not None
        args = choice.server_args()
        assert '"model":"z-lab/Qwen3.5-4B-DFlash"' in args[1]
        assert '"revision":"9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"' in args[1]
        assert '"num_speculative_tokens":7' in args[1]
        assert args[2] == "--no-async-scheduling"
        assert managed_llm_floor_gb(BF16, _settings("dflash")) == 13.5

    def test_dflash_on_the_w4a16_build_is_ignored_with_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bibr.local.vllm_speculative"):
            for _ in range(2):
                choice = choose_spec_decode(
                    _settings("dflash"), W4A16, swaps_with_ocr=False, budget_gb=20.0
                )
                assert choice.config is None
        assert len(caplog.records) == 1
        assert "LLM_VLLM_SPEC_DECODE=dflash is ignored for" in caplog.records[0].getMessage()
        # Its floor stays the base build's.
        assert explicit_config(_settings("dflash"), W4A16) is None
        assert managed_llm_floor_gb(W4A16, _settings("dflash")) == 6.0

    def test_a_higher_floor_setting_still_wins(self):
        settings = _settings("mtp")
        settings.llm.local_min_vram_gb = 8.0
        assert managed_llm_floor_gb(W4A16, settings) == 8.0
        settings.llm.local_min_vram_gb = 7.0  # below MTP's 7.25: ignored
        assert managed_llm_floor_gb(W4A16, settings) == 7.25

    def test_the_precheck_and_the_start_ask_for_the_mtp_floor(self):
        # An MTP run on an emulated 12 GB card (12.0 GiB hog) where a chunk found
        # 8.96 GiB free: under MTP's 7.25 + 0.75 + 1.0 GiB, over the 7.75 the
        # base build needs.
        reading = GpuMemory(total_gb=CUDA_3090, free_gb=8.96, own_gb=2.60)
        base = vllm_share.managed_llm_share(
            reading, _settings(), W4A16, cap=0.85, swaps_with_ocr=True
        )
        assert base.reason == "floor"
        with pytest.raises(ManagedLlmVramError, match=r"needs 9\.00 GiB \(the model's 7\.25"):
            vllm_share.managed_llm_share(
                reading, _settings("mtp"), W4A16, cap=0.85, swaps_with_ocr=True
            )


class TestUserConfig:
    @pytest.mark.parametrize(
        "extra",
        [
            '--speculative-config \'{"method":"ngram","num_speculative_tokens":4}\'',
            '--speculative-config=\'{"method":"ngram"}\'',
            '--speculative_config \'{"method":"ngram"}\'',
        ],
    )
    def test_a_users_speculative_config_wins(self, extra):
        assert user_sets_speculative(extra)
        for mode in ("auto", "mtp", "dflash"):
            settings = _settings(mode, extra)
            choice = choose_spec_decode(settings, BF16, swaps_with_ocr=False, budget_gb=20.0)
            assert choice.config is None
            assert managed_llm_floor_gb(BF16, settings) == 11.0

    @pytest.mark.parametrize("extra", ["", "--max-num-seqs 32", "--enforce-eager"])
    def test_other_args_do_not_count(self, extra):
        assert not user_sets_speculative(extra)


def _mk_server(settings, **kwargs):
    from bibr.local import vllm_llm

    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch.object(vllm_llm.subprocess, "Popen") as popen,
        patch.object(vllm_llm.VllmLlmServer, "_wait_until_healthy"),
    ):
        popen.return_value = MagicMock()
        server = vllm_llm.VllmLlmServer(settings=settings, port=9998, **kwargs)
    server._process = None  # never signal the mocked process
    return popen.call_args[0][0]


class TestServerCommand:
    @pytest.fixture(autouse=True)
    def _card(self, monkeypatch):
        self.reading = GpuMemory(total_gb=CUDA_3090, free_gb=CUDA_3090 - 0.5, own_gb=0.5)
        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: self.reading)
        monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)

    def _settings(self, mode="auto", extra="", model=BF16):
        settings = _settings(mode, extra)
        settings.llm.local_model = model
        return settings

    def test_resident_sized_start_appends_mtp_after_the_registry_args(self, caplog):
        with caplog.at_level(logging.INFO, logger="bibr.local.vllm_llm"):
            cmd = _mk_server(self._settings(), mem_fraction=0.85, share_from_free_memory=True)
        assert cmd[-3:] == list(MTP_ARGS)
        assert cmd[cmd.index("--max-num-seqs") + 1] == "64"
        assert any(
            "speculative decoding: mtp k=3 (auto: the server stays up for the run and its "
            "16.73 GiB budget" in r.getMessage()
            for r in caplog.records
        )

    def test_swap_start_has_none(self, caplog):
        with caplog.at_level(logging.INFO, logger="bibr.local.vllm_llm"):
            cmd = _mk_server(
                self._settings(),
                mem_fraction=0.85,
                share_from_free_memory=True,
                swaps_with_ocr=True,
            )
        assert "--speculative-config" not in cmd
        assert "--no-async-scheduling" not in cmd
        assert any("speculative decoding: none (auto: " in r.getMessage() for r in caplog.records)

    def test_fixed_share_budget_is_read_from_the_card(self):
        cmd = _mk_server(self._settings(model=W4A16), mem_fraction=0.30)  # 7.07 GiB
        assert "--speculative-config" not in cmd
        cmd = _mk_server(self._settings(model=W4A16), mem_fraction=0.31)  # 7.30 GiB
        assert cmd[-3:] == list(MTP_ARGS)

    def test_a_users_share_in_extra_args_is_the_budget(self):
        # 0.40 of the card is 9.42 GiB, under the bf16 4B's 12.0 GiB MTP floor.
        cmd = _mk_server(self._settings(extra="--gpu-memory-utilization 0.40"), mem_fraction=0.85)
        assert "--speculative-config" not in cmd
        assert cmd[-2:] == ["--gpu-memory-utilization", "0.40"]

    def test_no_reading_has_no_budget(self, monkeypatch):
        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: None)
        cmd = _mk_server(self._settings(), mem_fraction=0.85)
        assert "--speculative-config" not in cmd

    def test_explicit_dflash_comes_before_user_args(self):
        cmd = _mk_server(
            self._settings("dflash", extra="--max-num-seqs 32"),
            mem_fraction=0.85,
            share_from_free_memory=True,
            swaps_with_ocr=True,
        )
        spec = cmd.index("--speculative-config")
        assert "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf" in cmd[spec + 1]
        assert cmd[spec + 2] == "--no-async-scheduling"
        assert cmd[-2:] == ["--max-num-seqs", "32"]
        # The share was sized against DFlash's 13.5 GiB floor.
        share = float(cmd[cmd.index("--gpu-memory-utilization") + 1])
        assert share * CUDA_3090 >= 13.5

    def test_a_users_config_is_the_only_one(self):
        extra = '--speculative-config \'{"method":"ngram","num_speculative_tokens":4}\''
        cmd = _mk_server(self._settings(extra=extra), mem_fraction=0.85)
        assert cmd.count("--speculative-config") == 1
        assert cmd[-1] == '{"method":"ngram","num_speculative_tokens":4}'
        assert "--no-async-scheduling" not in cmd
