"""The managed vLLM LLM server's share is sized from the VRAM free at each start.

At a fixed 0.85 of an RTX 3090 (23.56 GiB to CUDA) the server's restarts in
a 100-paper swap batch ran out of memory in CUDA graph capture once bibr's
own GPU sessions held more than about 2.2 GiB: the share passed vLLM's
startup check, and capture then took about 1.3 GiB beyond it.
"""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from bibr.config import Settings
from bibr.local import vllm_share
from bibr.local.vllm_share import (
    LlmShare,
    ManagedLlmVramError,
    llm_share_user_set,
    managed_llm_floor_gb,
    overshoot_reserve_gb,
    size_managed_llm_share,
)
from bibr.ocr import registry as _registry
from bibr.ocr.registry import GpuMemory

# Captured at collection, before the suite's conftest pins it for each test.
_REAL_PROBE = _registry.managed_gpu_memory

STUDENT_4B = "scienceverse/bibr-extract-qwen35-4b"
STUDENT_2B = "scienceverse/bibr-extract-qwen35-2b"
STUDENT_4B_W4A16 = "scienceverse/bibr-extract-qwen35-4b-W4A16"
# CUDA's total of an RTX 3090 (24,576 MiB less 453 MiB reserved) and of a
# 16 GB card (RTX 4080: 16,376 MiB less about 400 MiB).
CUDA_3090 = 23.56
CUDA_16GB = 15.6
MIB = 1 / 1024


def _bibr_alone(own_gb: float, total: float = CUDA_3090) -> GpuMemory:
    """The card as an LLM restart in a swap batch finds it: only bibr on it."""
    return GpuMemory(total_gb=total, free_gb=total - own_gb, own_gb=own_gb)


def _size(reading, *, cap=0.85, model=STUDENT_4B, swaps_with_ocr=True, extra=None) -> LlmShare:
    return size_managed_llm_share(
        reading,
        cap=cap,
        floor_gb=managed_llm_floor_gb(model),
        reserve_gb=overshoot_reserve_gb(model, extra),
        swaps_with_ocr=swaps_with_ocr,
    )


class TestOvershootReserve:
    """vLLM's peak beyond its share follows the capture sizes, not the share."""

    def test_students_capture_up_to_256(self):
        # 128 sequences: capture sizes up to 256, measured 0.75 GiB above 0.54.
        assert overshoot_reserve_gb(STUDENT_4B, None) == pytest.approx(1.0)
        assert overshoot_reserve_gb(STUDENT_2B, None) == pytest.approx(1.0)

    def test_vllm_default_captures_up_to_512(self):
        # 256 sequences: measured 1.17-1.29 GiB above 0.85.
        assert overshoot_reserve_gb("org/unknown", None) == pytest.approx(1.5)

    @pytest.mark.parametrize(
        ("extra", "reserve"),
        [
            ("--max-num-seqs 64", 0.75),
            ("--max-num-seqs=1024", 1.5),
            ("--max-num-seqs 256", 1.5),
            ("--enforce-eager", 0.5),
            ("--max-num-seqs lots", 1.5),
        ],
    )
    def test_user_args_win(self, extra, reserve):
        assert overshoot_reserve_gb(STUDENT_4B, extra) == pytest.approx(reserve)


class TestSwapRestartsBesideGrowingBibr:
    """E23 arm A: bibr held 504 MiB at the first LLM start and 2,262 at the fourth."""

    @pytest.mark.parametrize("own_mib", [504, 2262, 3100, 4500])
    def test_every_restart_gets_the_same_share_and_fits(self, own_mib):
        own = own_mib * MIB
        share = _size(_bibr_alone(own))
        assert share.reason == "free"
        assert share.share == 0.70
        reserve = overshoot_reserve_gb(STUDENT_4B, None)
        # Share, capture overshoot and bibr's 6 GB reserve all fit the card.
        assert share.share * CUDA_3090 + reserve + max(own, 6.0) <= CUDA_3090

    def test_bibr_beyond_its_reserve_keeps_a_minimum_of_room(self):
        own = 7.0
        share = _size(_bibr_alone(own))
        assert share.reason == "free"
        assert share.headroom_gb == 1.0
        assert share.share * CUDA_3090 + 1.0 + own + 1.0 <= CUDA_3090

    def test_describe_names_the_inputs(self):
        text = _size(_bibr_alone(2262 * MIB)).describe()
        assert "0.70 of 23.56 GiB (free)" in text
        assert "bibr 2.21 GiB" in text
        assert "floor 12.25 GiB" in text


class TestCapAndColocation:
    def test_roomy_card_keeps_the_configured_share(self):
        reading = _bibr_alone(0.5, total=78.4)  # 80 GB card
        share = _size(reading)
        assert (share.share, share.reason) == (0.85, "cap")

    def test_colocated_plan_keeps_its_share_beside_paddleocr(self):
        # The planned 0.54 with PaddleOCR-VL's 5 GB and its context up.
        reading = GpuMemory(total_gb=CUDA_3090, free_gb=CUDA_3090 - 5.4 - 0.5, own_gb=0.5)
        share = _size(reading, cap=0.54, swaps_with_ocr=False)
        assert (share.share, share.reason) == (0.54, "cap")

    def test_colocated_start_on_a_crowded_card_shrinks(self):
        # Free less the 1.0 GiB capture reserve and 1.0 GiB of room: 12.3 GiB.
        reading = GpuMemory(total_gb=CUDA_3090, free_gb=14.3, own_gb=3.0)
        share = _size(reading, cap=0.54, swaps_with_ocr=False)
        assert (share.share, share.reason) == (0.52, "free")

    def test_no_reading_keeps_the_configured_share(self):
        assert _size(None) == LlmShare(share=0.85, reason="unknown")


class TestLlmOnly:
    """No managed PaddleOCR-VL takes turns with the LLM: only bibr's sessions share the card.

    The swap headroom (bibr's 6 GB) dropped every such start on a 24 GB card
    from 0.85 to 0.70; the configured share is kept while it fits.
    """

    def test_free_card_keeps_the_configured_share(self):
        # E23 A's first LLM start: bibr held 504 MiB.
        reading = _bibr_alone(504 * MIB)
        share = _size(reading, swaps_with_ocr=False)
        assert (share.share, share.reason) == (0.85, "cap")
        assert share.headroom_gb == 1.0
        # The share, vLLM's overshoot reserve, bibr and 1 GiB of room fit.
        reserve = overshoot_reserve_gb(STUDENT_4B, None)
        assert 0.85 * CUDA_3090 + reserve + 504 * MIB + 1.0 <= CUDA_3090
        # The same reading with PaddleOCR-VL swapping in leaves bibr its 6 GB.
        assert _size(reading).share == 0.70

    def test_a_busier_card_shrinks_only_as_needed(self):
        # bibr at E23 A's 2,262 MiB: 21.35 GiB free, less 1.0 + 1.0 = 19.35.
        share = _size(_bibr_alone(2262 * MIB), swaps_with_ocr=False)
        assert (share.share, share.reason) == (0.82, "free")

    def test_16gb_card(self):
        share = _size(_bibr_alone(0.5, total=CUDA_16GB), swaps_with_ocr=False)
        assert (share.share, share.reason) == (0.83, "free")


class TestFloorRounding:
    """A target just above the floor must not round down to a share under it.

    On an RTX 3090 a 12.008 GiB target rounded down to 0.50 (11.78 GiB, under
    the 4B's floor of 12 GiB at the time), while a slightly smaller one took
    the floor path and got 0.51 (12.02 GiB). The window is [12.0, 0.51 x
    23.56) GiB. The floor is given here, so the window does not move with the
    registry's floors.
    """

    FLOOR = 12.0

    @classmethod
    def _size(cls, reading, *, swaps_with_ocr: bool) -> LlmShare:
        return size_managed_llm_share(
            reading,
            cap=0.85,
            floor_gb=cls.FLOOR,
            reserve_gb=overshoot_reserve_gb(STUDENT_4B, None),
            swaps_with_ocr=swaps_with_ocr,
        )

    @staticmethod
    def _reading_for_target(target: float, *, swaps_with_ocr: bool) -> GpuMemory:
        own = 0.5
        headroom = max(1.0, 6.0 - own) if swaps_with_ocr else 1.0
        free = target + overshoot_reserve_gb(STUDENT_4B, None) + headroom
        return GpuMemory(total_gb=CUDA_3090, free_gb=free, own_gb=own)

    @pytest.mark.parametrize("swaps_with_ocr", [True, False])
    @pytest.mark.parametrize("target", [12.0, 12.008, 12.015])
    def test_window_just_above_the_floor_takes_the_floor_path(self, target, swaps_with_ocr):
        assert target < 0.51 * CUDA_3090
        reading = self._reading_for_target(target, swaps_with_ocr=swaps_with_ocr)
        share = self._size(reading, swaps_with_ocr=swaps_with_ocr)
        assert (share.share, share.reason) == (0.51, "floor")
        assert share.budget_gb >= 12.0

    def test_just_past_the_window_is_sized_from_free_memory(self):
        reading = self._reading_for_target(0.51 * CUDA_3090 + 0.001, swaps_with_ocr=False)
        share = self._size(reading, swaps_with_ocr=False)
        assert (share.share, share.reason) == (0.51, "free")

    @pytest.mark.parametrize("swaps_with_ocr", [True, False])
    def test_no_target_launches_under_the_floor(self, swaps_with_ocr):
        # Every target from 11 to 13 GiB in 1 MiB steps: at the floor or refused.
        for step in range(2 * 1024 + 1):
            target = 11.0 + step * MIB
            reading = self._reading_for_target(target, swaps_with_ocr=swaps_with_ocr)
            try:
                share = self._size(reading, swaps_with_ocr=swaps_with_ocr)
            except ManagedLlmVramError:
                continue
            assert share.share * CUDA_3090 >= 12.0, (target, share)


class TestFloor:
    """A 16 GB card: the 4B's 12.25 GiB floor fits only while bibr is small."""

    def test_first_chunk_squeezes_to_the_floor(self):
        share = _size(_bibr_alone(0.5, total=CUDA_16GB))
        assert share.reason == "floor"
        assert share.share * CUDA_16GB >= 12.25
        assert share.share == 0.79
        assert share.headroom_gb == 1.0

    def test_a_grown_bibr_fails_with_a_typed_error(self):
        with pytest.raises(ManagedLlmVramError) as exc:
            _size(_bibr_alone(2.3, total=CUDA_16GB))
        message = str(exc.value)
        assert "Error in llm" in message
        assert "13.30 of 15.60 GiB free" in message
        assert "bibr itself holds 2.30 GiB" in message
        assert "needs 14.25 GiB (the model's 12.25 GiB floor, 1.00 GiB vLLM" in message
        assert "LLM_LOCAL_MEM_FRACTION" in message

    def test_the_error_is_an_upstream_failure(self):
        from bibr.exceptions import UpstreamServiceError

        assert issubclass(ManagedLlmVramError, UpstreamServiceError)

    def test_smaller_model_has_a_smaller_floor(self):
        share = _size(_bibr_alone(2.3, total=CUDA_16GB), model=STUDENT_2B)
        assert share.reason == "free"
        assert share.share * CUDA_16GB >= 6.25

    def test_cap_below_the_floor_is_the_floor(self):
        # 0.6 of a 16 GB card is 9.4 GiB, below the 4B's 12.25: the cap decides.
        share = _size(_bibr_alone(0.5, total=CUDA_16GB), cap=0.6)
        assert share.share == 0.6

    def test_unknown_model_has_no_floor(self):
        share = _size(_bibr_alone(2.3, total=CUDA_16GB), model="org/unknown")
        assert share.reason == "free"
        assert share.floor_gb is None
        with pytest.raises(ManagedLlmVramError):
            _size(GpuMemory(total_gb=CUDA_16GB, free_gb=1.0, own_gb=0.5), model="org/unknown")

    def test_registry_floors(self):
        # The measured budget floors, not the card-size tiers (20, 12 and 8).
        assert managed_llm_floor_gb(STUDENT_4B) == 12.25
        assert managed_llm_floor_gb(STUDENT_4B_W4A16) == 6.25
        assert managed_llm_floor_gb(STUDENT_2B) == 6.25
        assert managed_llm_floor_gb("scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0") is None
        assert managed_llm_floor_gb("org/unknown") is None

    def test_a_variant_without_a_budget_floor_falls_back_to_its_tier(self, monkeypatch):
        import dataclasses

        from bibr.local import llm_models

        model = llm_models.get_model("bibr-extract-2b")
        bare = dataclasses.replace(model.variants[0], min_budget_gb=None)
        patched = dataclasses.replace(model, variants=(bare, *model.variants[1:]))
        registry = tuple(patched if m.key == model.key else m for m in llm_models.REGISTRY)
        monkeypatch.setattr(llm_models, "REGISTRY", registry)
        assert managed_llm_floor_gb(STUDENT_2B) == 8.0


class TestUserSetShareWins:
    def test_default_is_not_user_set(self):
        from bibr.config import GlobalSettings

        assert llm_share_user_set(GlobalSettings()) is False

    def test_env_share_is_user_set(self, monkeypatch):
        from bibr.config import GlobalSettings

        monkeypatch.setenv("LLM_LOCAL_MEM_FRACTION", "0.6")
        assert llm_share_user_set(GlobalSettings()) is True

    def test_assigned_share_is_user_set(self):
        from bibr.config import GlobalSettings

        settings = GlobalSettings()
        settings.llm.local_mem_fraction = 0.6
        assert llm_share_user_set(settings) is True

    @pytest.mark.parametrize(
        "extra", ["--gpu-memory-utilization 0.6", "--gpu-memory-utilization=0.6"]
    )
    def test_extra_args_share_is_user_set(self, extra):
        from bibr.config import GlobalSettings

        settings = GlobalSettings()
        settings.llm.vllm_extra_args = extra
        assert llm_share_user_set(settings) is True


class TestSettledReading:
    def test_waits_while_free_memory_rises(self):
        readings = iter([10.0, 14.0, 18.0, 18.1])
        waits = []

        def probe():
            return GpuMemory(total_gb=CUDA_3090, free_gb=next(readings), own_gb=0.5)

        reading = vllm_share.settled_gpu_memory(probe, wait=waits.append)
        assert reading.free_gb == 18.1
        assert len(waits) == 3

    def test_no_reading_does_not_wait(self):
        waits = []
        assert vllm_share.settled_gpu_memory(lambda: None, wait=waits.append) is None
        assert waits == []

    def test_wait_can_stop_the_start(self):
        from bibr.local.http_runtime import StartupCancelled

        def stop(_seconds):
            raise StartupCancelled("vLLM startup was cancelled")

        with pytest.raises(StartupCancelled):
            vllm_share.settled_gpu_memory(lambda: _bibr_alone(0.5), wait=stop)


def _fake_nvidia_smi(monkeypatch, gpus: str, apps: str = "", *, reserved=True):
    """Answer nvidia-smi's GPU and compute-app queries."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        query = cmd[1]
        if query.startswith("--query-compute-apps"):
            return subprocess.CompletedProcess(cmd, 0, stdout=apps, stderr="")
        if "memory.reserved" in query and not reserved:
            return subprocess.CompletedProcess(
                cmd, 2, stdout="", stderr='Field "memory.reserved" is not a valid field'
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=gpus, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


class TestManagedGpuMemoryReading:
    A100 = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
    RTX3090 = "GPU-cccccccc-1111-2222-3333-444444444444"

    @pytest.fixture(autouse=True)
    def _real_probe(self, monkeypatch):
        # conftest pins the probe to None; put the module's own function back.
        monkeypatch.setattr(_registry, "managed_gpu_memory", _REAL_PROBE)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        self.registry = _registry

    def test_reads_cuda_total_free_and_own_use(self, monkeypatch):
        import os

        _fake_nvidia_smi(
            monkeypatch,
            f"{self.RTX3090}, 24576, 453, 21861\n",
            f"{os.getpid()}, {self.RTX3090}, 2262\n4242, {self.RTX3090}, 9000\n",
        )
        reading = self.registry.managed_gpu_memory()
        assert reading.total_gb == pytest.approx(23.558, abs=1e-3)
        assert reading.free_gb == pytest.approx(21861 / 1024)
        assert reading.own_gb == pytest.approx(2262 / 1024)

    def test_driver_without_reserved_counts_the_whole_card(self, monkeypatch):
        _fake_nvidia_smi(monkeypatch, f"{self.RTX3090}, 24576, 20000\n", reserved=False)
        reading = self.registry.managed_gpu_memory()
        assert reading.total_gb == 24.0
        assert reading.own_gb == 0.0

    def test_follows_cuda_visible_devices(self, monkeypatch):
        _fake_nvidia_smi(
            monkeypatch, f"{self.A100}, 81559, 500, 80000\n{self.RTX3090}, 24576, 453, 12000\n"
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", self.RTX3090[:12])
        reading = self.registry.managed_gpu_memory()
        assert reading.free_gb == pytest.approx(12000 / 1024)

    def test_cards_of_different_sizes_cannot_be_told_apart(self, monkeypatch):
        _fake_nvidia_smi(
            monkeypatch, f"{self.A100}, 81559, 500, 80000\n{self.RTX3090}, 24576, 453, 12000\n"
        )
        assert self.registry.managed_gpu_memory() is None

    def test_same_size_cards_take_the_least_free(self, monkeypatch):
        other = "GPU-dddddddd-1111-2222-3333-444444444444"
        _fake_nvidia_smi(
            monkeypatch, f"{self.RTX3090}, 24576, 453, 20000\n{other}, 24576, 453, 9000\n"
        )
        assert self.registry.managed_gpu_memory().free_gb == pytest.approx(9000 / 1024)

    @pytest.mark.parametrize("visible", ["", "NoDevFiles"])
    def test_nothing_visible(self, monkeypatch, visible):
        _fake_nvidia_smi(monkeypatch, f"{self.RTX3090}, 24576, 453, 20000\n")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        assert self.registry.managed_gpu_memory() is None

    def test_nvidia_smi_missing(self, monkeypatch):
        def missing(cmd, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(subprocess, "run", missing)
        assert self.registry.managed_gpu_memory() is None


def _mk_server(**kwargs):
    from bibr.local import vllm_llm

    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch.object(vllm_llm.subprocess, "Popen") as popen,
        patch.object(vllm_llm.VllmLlmServer, "_wait_until_healthy"),
    ):
        popen.return_value = MagicMock()
        server = vllm_llm.VllmLlmServer(**kwargs)
        return server, popen


def _launched_share(popen) -> str:
    cmd = popen.call_args[0][0]
    return cmd[cmd.index("--gpu-memory-utilization") + 1]


class TestServerStart:
    @pytest.fixture(autouse=True)
    def _fast_settle(self, monkeypatch):
        monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)

    def _probe(self, monkeypatch, reading):
        calls = []

        def probe():
            calls.append(True)
            return reading

        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", probe)
        return calls

    def test_sized_share_is_launched(self, monkeypatch, caplog):
        self._probe(monkeypatch, _bibr_alone(2262 * MIB))
        with caplog.at_level("INFO", logger="bibr.local.vllm_llm"):
            _server, popen = _mk_server(
                model=STUDENT_4B,
                port=9999,
                mem_fraction=0.85,
                share_from_free_memory=True,
                swaps_with_ocr=True,
            )
        assert _launched_share(popen) == "0.7"
        assert any("Managed vLLM LLM share 0.70" in r.getMessage() for r in caplog.records)

    def test_llm_only_start_keeps_the_configured_share(self, monkeypatch, caplog):
        self._probe(monkeypatch, _bibr_alone(504 * MIB))
        with caplog.at_level("INFO", logger="bibr.local.vllm_llm"):
            _server, popen = _mk_server(
                model=STUDENT_4B, port=9999, mem_fraction=0.85, share_from_free_memory=True
            )
        assert _launched_share(popen) == "0.85"
        assert any(
            "Managed vLLM LLM share 0.85 of 23.56 GiB (cap)" in r.getMessage()
            for r in caplog.records
        )

    def test_floor_that_does_not_fit_never_launches_vllm(self, monkeypatch):
        self._probe(monkeypatch, _bibr_alone(2.3, total=CUDA_16GB))
        from bibr.local import vllm_llm

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch.object(vllm_llm.subprocess, "Popen") as popen,
            pytest.raises(ManagedLlmVramError),
        ):
            vllm_llm.VllmLlmServer(
                model=STUDENT_4B, port=9999, mem_fraction=0.85, share_from_free_memory=True
            )
        popen.assert_not_called()

    def test_squeezed_start_warns(self, monkeypatch, caplog):
        self._probe(monkeypatch, _bibr_alone(0.5, total=CUDA_16GB))
        with caplog.at_level("WARNING", logger="bibr.local.vllm_llm"):
            _server, popen = _mk_server(
                model=STUDENT_4B,
                port=9999,
                mem_fraction=0.85,
                share_from_free_memory=True,
                swaps_with_ocr=True,
            )
        assert _launched_share(popen) == "0.79"
        assert any("little room left" in r.getMessage() for r in caplog.records)

    def test_fixed_share_never_reads_the_card(self, monkeypatch):
        calls = self._probe(monkeypatch, _bibr_alone(2.3, total=CUDA_16GB))
        _server, popen = _mk_server(model=STUDENT_4B, port=9999, mem_fraction=0.85)
        assert _launched_share(popen) == "0.85"
        assert calls == []

    def test_reused_server_is_not_sized(self, monkeypatch):
        from bibr.local import vllm_llm

        calls = self._probe(monkeypatch, _bibr_alone(2.3, total=CUDA_16GB))
        monkeypatch.setattr(vllm_llm, "guard_managed_server_port", lambda *a, **k: True)
        server, popen = _mk_server(
            model=STUDENT_4B, port=9999, mem_fraction=0.85, share_from_free_memory=True
        )
        popen.assert_not_called()
        assert calls == []

    def test_user_extra_share_still_comes_last(self, monkeypatch):
        monkeypatch.setattr(Settings.llm, "vllm_extra_args", "--gpu-memory-utilization 0.5")
        _server, popen = _mk_server(model=STUDENT_4B, port=9999, mem_fraction=0.85)
        cmd = popen.call_args[0][0]
        last = len(cmd) - 1 - cmd[::-1].index("--gpu-memory-utilization")
        assert cmd[last + 1] == "0.5"


W4A16_LOCAL = "/models/bibr-extract-qwen35-4b-W4A16"


def _w4a16_settings(floor: float | None = 6.25):
    """Settings for a local copy of the 4B W4A16 build, selected as documented."""
    import shlex

    from bibr.config import GlobalSettings
    from bibr.local.llm_models import QWEN35_4B_W4A16_SERVER_ARGS

    settings = GlobalSettings()
    settings.llm.local_model = W4A16_LOCAL
    settings.llm.vllm_extra_args = shlex.join(QWEN35_4B_W4A16_SERVER_ARGS)
    if floor is not None:
        settings.llm.local_min_vram_gb = floor
    return settings


def _hub_w4a16_settings():
    """Settings for the published W4A16 build: the registry supplies its args and floor."""
    from bibr.config import GlobalSettings

    settings = GlobalSettings()
    settings.llm.local_model = STUDENT_4B_W4A16
    return settings


def _hog(hog_gb: float, own_gb: float) -> GpuMemory:
    """A 3090 with another process holding *hog_gb*: a smaller card, emulated."""
    return GpuMemory(total_gb=CUDA_3090, free_gb=CUDA_3090 - hog_gb - own_gb, own_gb=own_gb)


class TestFloorSetting:
    """``LLM_LOCAL_MIN_VRAM_GB`` sets an unlisted model's floor and can raise a listed one's."""

    @pytest.fixture(autouse=True)
    def _fresh_warnings(self, monkeypatch):
        monkeypatch.setattr(vllm_share, "_floor_setting_ignored_for", set())

    def test_unset_by_default(self):
        from bibr.config import GlobalSettings

        assert GlobalSettings().llm.local_min_vram_gb is None

    def test_read_from_the_environment(self, monkeypatch):
        from bibr.config import GlobalSettings

        monkeypatch.setenv("LLM_LOCAL_MIN_VRAM_GB", "6.25")
        assert GlobalSettings().llm.local_min_vram_gb == 6.25

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_must_be_positive(self, monkeypatch, bad):
        from pydantic import ValidationError

        from bibr.config import GlobalSettings

        monkeypatch.setenv("LLM_LOCAL_MIN_VRAM_GB", bad)
        with pytest.raises(ValidationError):
            GlobalSettings()

    def test_covers_a_local_path(self):
        assert managed_llm_floor_gb(W4A16_LOCAL) is None
        assert managed_llm_floor_gb(W4A16_LOCAL, _w4a16_settings(floor=None)) is None
        assert managed_llm_floor_gb(W4A16_LOCAL, _w4a16_settings()) == 6.25

    def test_a_lower_setting_keeps_a_listed_floor_with_one_warning(self, caplog):
        # 7 left set after switching from a local build back to the bf16 4B.
        settings = _w4a16_settings(floor=7.0)
        with caplog.at_level(logging.WARNING, logger="bibr.local.vllm_share"):
            assert managed_llm_floor_gb(STUDENT_4B, settings) == 12.25
            assert managed_llm_floor_gb(STUDENT_4B, settings) == 12.25
            assert managed_llm_floor_gb(W4A16_LOCAL, settings) == 7.0
        warnings = [r.getMessage() for r in caplog.records]
        assert warnings == [
            f"LLM_LOCAL_MIN_VRAM_GB=7 is ignored for {STUDENT_4B}: bibr's model registry lists "
            "that model with a minimum of 12.25 GiB, and the setting can raise a listed model's "
            "minimum but not lower it. Unset it, or set LLM_LOCAL_MEM_FRACTION to choose the "
            "share yourself."
        ]  # once per listed model, not at every start
        assert managed_llm_floor_gb(STUDENT_4B, _w4a16_settings(floor=None)) == 12.25

    def test_a_higher_setting_raises_a_listed_floor_silently(self, caplog):
        with caplog.at_level(logging.WARNING, logger="bibr.local.vllm_share"):
            assert managed_llm_floor_gb(STUDENT_2B, _w4a16_settings(floor=7.0)) == 7.0
            assert managed_llm_floor_gb(STUDENT_4B, _w4a16_settings(floor=13.0)) == 13.0
            assert managed_llm_floor_gb(STUDENT_4B_W4A16, _w4a16_settings(floor=6.25)) == 6.25
        assert caplog.records == []

    def test_a_raised_floor_refuses_where_the_registry_floor_starts(self):
        settings = _hub_w4a16_settings()
        reading = _hog(12.0, 2.9)  # 8.66 GiB free: 6.25 + 0.75 + 1.0 fits
        assert (
            vllm_share.managed_llm_share(
                reading, settings, STUDENT_4B_W4A16, cap=0.85, swaps_with_ocr=True
            ).reason
            == "floor"
        )
        settings.llm.local_min_vram_gb = 7.0  # 7 + 0.75 + 1.0 does not
        with pytest.raises(ManagedLlmVramError, match=r"the model's 7\.00 GiB floor"):
            vllm_share.managed_llm_share(
                reading, settings, STUDENT_4B_W4A16, cap=0.85, swaps_with_ocr=True
            )

    def test_a_left_over_setting_never_starts_the_bf16_4b_below_its_floor(self):
        # The emulated 12 GB card where a local W4A16 copy starts at its 6.25
        # GiB floor: the bf16 4B is refused at its own 12.25, not squeezed.
        settings = _w4a16_settings()
        settings.llm.local_model = STUDENT_4B
        settings.llm.vllm_extra_args = ""
        with pytest.raises(ManagedLlmVramError, match=r"the model's 12\.25 GiB floor"):
            vllm_share.managed_llm_share(
                _hog(12.0, 0.50), settings, STUDENT_4B, cap=0.85, swaps_with_ocr=True
            )
        assert (
            vllm_share.managed_llm_share(
                _hog(12.0, 0.50), _w4a16_settings(), W4A16_LOCAL, cap=0.85, swaps_with_ocr=True
            ).reason
            == "floor"
        )

    def test_model_resolution(self):
        from bibr.config import GlobalSettings

        assert vllm_share.managed_llm_model(_w4a16_settings()) == W4A16_LOCAL
        assert vllm_share.managed_llm_model(_w4a16_settings(), "org/other") == "org/other"
        assert vllm_share.managed_llm_model(GlobalSettings()) == STUDENT_4B


class TestW4a16Floor:
    """The published 4B W4A16 build (6.25 GiB floor at 64 sequences) on emulated 16 and 12 GB cards.

    Readings are E25 V2's (an 8 GiB hog on the 3090): bibr 0.50 GiB before
    the first LLM start, 1.98-2.06 GiB before the later ones, which refused
    the bf16 4B at 13.48-13.57 GiB free.
    """

    def _share(self, reading):
        return vllm_share.managed_llm_share(
            reading, _hub_w4a16_settings(), STUDENT_4B_W4A16, cap=0.85, swaps_with_ocr=True
        )

    def test_constants(self):
        from bibr.local.llm_models import (
            QWEN35_4B_W4A16_BUDGET_GB,
            QWEN35_4B_W4A16_SERVER_ARGS,
            registry_server_args,
        )

        assert QWEN35_4B_W4A16_BUDGET_GB == 6.25
        # Same student args, with the last --max-num-seqs (which vLLM keeps) at 64.
        assert QWEN35_4B_W4A16_SERVER_ARGS[:-2] == registry_server_args(STUDENT_4B)[2:]
        assert QWEN35_4B_W4A16_SERVER_ARGS[-2:] == ("--max-num-seqs", "64")
        assert registry_server_args(STUDENT_4B_W4A16) == (
            "--revision",
            "a2af8f6e90e6481251af9eb07d02c034c30e3446",
            *QWEN35_4B_W4A16_SERVER_ARGS,
        )
        # 64 sequences capture up to 128: 0.5 + 1.0 x 128 / 512, from the
        # registry's args alone and from a local copy's extra args.
        assert overshoot_reserve_gb(STUDENT_4B_W4A16, None) == 0.75
        assert overshoot_reserve_gb(W4A16_LOCAL, _w4a16_settings().llm.vllm_extra_args) == 0.75

    @pytest.mark.parametrize(("free", "own"), [(15.04, 0.50), (13.57, 1.98), (13.48, 2.06)])
    def test_16gb_swap_path_starts_sized_from_free_memory(self, free, own):
        share = self._share(GpuMemory(total_gb=CUDA_3090, free_gb=free, own_gb=own))
        assert share.reason == "free"
        assert share.share == 0.37
        assert share.budget_gb >= 6.25

    def test_12gb_first_start_takes_the_floor(self):
        share = self._share(_hog(12.0, 0.50))
        assert share.reason == "floor"
        assert share.share == 0.27
        assert share.budget_gb >= 6.25
        assert share.headroom_gb == 1.0

    def test_12gb_grown_bibr_still_fits_until_3_5_gib(self):
        assert self._share(_hog(12.0, 3.50)).reason == "floor"
        with pytest.raises(ManagedLlmVramError) as exc:
            self._share(_hog(12.0, 3.60))
        message = str(exc.value)
        assert "the model's 6.25 GiB floor, 0.75 GiB vLLM takes beyond its share" in message
        assert "needs 8.00 GiB" in message

    def test_8gb_card_is_refused_even_with_bibr_off_the_gpu(self):
        # About 7.6 GiB to CUDA (an RTX 4060 reports 8,188 MiB): 6.25 + 0.75 +
        # 1.0 GiB is more than the whole card, so no start is sized. Both the
        # card's cap (0.85 x 7.6 = 6.46 GiB) and a 3090 behind a hog agree.
        for reading in (
            GpuMemory(total_gb=7.6, free_gb=7.6, own_gb=0.0),
            _hog(CUDA_3090 - 7.6, 0.0),
        ):
            with pytest.raises(ManagedLlmVramError, match="needs 8.00 GiB"):
                self._share(reading)

    def test_without_the_setting_a_local_path_has_no_floor(self):
        # Sized from free memory however little is left: vLLM would be launched
        # to fail instead of a typed refusal.
        share = vllm_share.managed_llm_share(
            _hog(12.0, 3.6), _w4a16_settings(floor=None), W4A16_LOCAL, cap=0.85, swaps_with_ocr=True
        )
        assert share.reason == "free"
        assert share.floor_gb is None
        assert share.budget_gb < 6.25

    def test_a_floorless_refusal_names_the_swap_room_it_needs(self):
        # E25 V2's second start, a local copy without the setting: on the swap
        # path bibr keeps 6 GiB less the 1.98 it holds, not the 1 GiB minimum.
        reading = GpuMemory(total_gb=CUDA_3090, free_gb=4.50, own_gb=1.98)
        with pytest.raises(ManagedLlmVramError) as exc:
            vllm_share.managed_llm_share(
                reading, _w4a16_settings(floor=None), W4A16_LOCAL, cap=0.85, swaps_with_ocr=True
            )
        assert (
            "it needs more than 4.77 GiB (the model has no floor; 0.75 GiB vLLM takes beyond its "
            "share while starting, and 4.02 GiB for bibr's GPU sessions)"
        ) in str(exc.value)
        assert "floor," not in str(exc.value)

    def test_the_bf16_4b_is_refused_where_w4a16_starts(self):
        reading = GpuMemory(total_gb=CUDA_3090, free_gb=13.57, own_gb=1.98)
        with pytest.raises(ManagedLlmVramError):
            _size(reading)
        assert self._share(reading).share == 0.37

    def test_server_launches_the_published_build_with_its_args(self, monkeypatch):
        monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)
        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: _hog(12.0, 0.5))
        server, popen = _mk_server(
            settings=_hub_w4a16_settings(),
            port=9999,
            mem_fraction=0.85,
            share_from_free_memory=True,
            swaps_with_ocr=True,
        )
        cmd = popen.call_args[0][0]
        assert cmd[cmd.index("--model") + 1] == STUDENT_4B_W4A16
        assert cmd[cmd.index("--revision") + 1] == "a2af8f6e90e6481251af9eb07d02c034c30e3446"
        assert _launched_share(popen) == "0.27"
        assert cmd[-2:] == ["--max-num-seqs", "64"]
        server._process = None

    def test_server_launches_a_local_copy_with_its_args(self, monkeypatch):
        monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)
        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: _hog(12.0, 0.5))
        server, popen = _mk_server(
            settings=_w4a16_settings(),
            port=9999,
            mem_fraction=0.85,
            share_from_free_memory=True,
            swaps_with_ocr=True,
        )
        cmd = popen.call_args[0][0]
        assert cmd[cmd.index("--model") + 1] == W4A16_LOCAL
        assert _launched_share(popen) == "0.27"
        assert cmd[-2:] == ["--max-num-seqs", "64"]
        server._process = None

    def test_server_refuses_below_the_floor_without_launching(self, monkeypatch):
        from bibr.local import vllm_llm

        monkeypatch.setattr(vllm_share, "_SETTLE_INTERVAL_S", 0.0)
        monkeypatch.setattr("bibr.ocr.registry.managed_gpu_memory", lambda: _hog(12.0, 3.6))
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch.object(vllm_llm.subprocess, "Popen") as popen,
            pytest.raises(ManagedLlmVramError),
        ):
            vllm_llm.VllmLlmServer(
                settings=_hub_w4a16_settings(),
                port=9999,
                mem_fraction=0.85,
                share_from_free_memory=True,
                swaps_with_ocr=True,
            )
        popen.assert_not_called()
