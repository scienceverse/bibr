"""The CUDA tier table: what ``bibr setup`` writes equals what the runtime picks.

For each card row, setup's recommended plan (``LLM_BACKEND``, ``LLM_LOCAL_MODEL``,
``PIPELINE_MEMORY_MODE``, the OCR its ``OCR_BACKEND`` resolves to, and the
managed vLLM server's speculative decoding) must equal the runtime's own
resolution with nothing set but ``--llm local``: the ``local`` alias, the
default model, the auto memory mode, the automatic ``paddle`` OCR chain and
``LLM_VLLM_SPEC_DECODE=auto``. The rows are the docs' tier table
(docs/guides/extraction-models.md).

That holds wherever no qualifying llama.cpp server is on PATH, which the suite
pins (tests/conftest.py). On a 10–11 GB Ampere card, setup runs the LLM and OCR
on a CUDA llama-server of build 11282 or newer when it finds one, while the
runtime never reads PATH to choose: there, the runtime given setup's ``.env``
equals setup's plan, and ``--llm local`` alone keeps the W4A16 build.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from bibr import setup_wizard
from bibr.config import GlobalSettings
from bibr.local import llama_cpp
from bibr.local.llm_models import default_local_model
from bibr.local.pipeline import _auto_memory_mode, resolve_llm_backend
from bibr.local.vllm_speculative import choose_spec_decode
from bibr.ocr import registry
from bibr.setup_wizard import _build_recommended_setup

_4B = "scienceverse/bibr-extract-qwen35-4b"
_W4A16 = "scienceverse/bibr-extract-qwen35-4b-W4A16"
_Q8 = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q8_0"
_Q4 = "scienceverse/bibr-extract-qwen35-4b-GGUF:Q4_K_M"

_V = "vllm"
_L = "llama-cpp"
_P = "paddle-vllm"
_G = "glm-llama"

#: (card, nvidia-smi memory.total MiB, compute capability) -> (LLM backend,
#: model, memory mode, OCR runtime, speculative decoding of the managed vLLM
#: LLM server; None for llama.cpp). Rows marked "ran" finished a batch with
#: this default on an RTX 3090 cut to that card's memory beside a 0.5 GiB
#: desktop, with no refused start and no failed paper (2026-10-24..26); the
#: others follow the same size rule. Turing and Pascal rows are the
#: compute-capability routing rule, not measured cards. Every vLLM row's LLM
#: server takes turns with PaddleOCR-VL, so ``auto`` leaves speculative
#: decoding off: with MTP, batches on 12 and 16 GB cards ran 5% and 9%
#: slower (on vLLM 0.32).
TIER_TABLE = [
    (("RTX 5090", 32607, 12.0), (_V, _4B, "balanced", _P, "none")),
    (("RTX 3090", 24576, 8.6), (_V, _4B, "balanced", _P, "none")),  # ran: 100 papers
    (("RTX 4000 Ada", 20475, 8.9), (_V, _4B, "balanced", _P, "none")),  # ran: 48
    # 16 GB ran at 16,380 MiB on 2026-10-26, on this change's code, with at
    # least 4.74 GiB to spare at every LLM start.
    (("RTX 4080", 16376, 8.9), (_V, _W4A16, "balanced", _P, "none")),  # ran: 48
    (("RTX 3060 12 GB", 12288, 8.6), (_V, _W4A16, "balanced", _P, "none")),
    (("RTX 5070", 12227, 12.0), (_V, _W4A16, "balanced", _P, "none")),  # ran: 100
    (("RTX 3080 10 GB", 10240, 8.6), (_V, _W4A16, "aggressive", _P, "none")),  # ran: 48
    (("RTX 3070", 8192, 8.6), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 4060", 8188, 8.9), (_L, _Q8, "aggressive", _G, None)),  # ran: 100
    # 6 GB also ran beside a 1.0 GiB desktop: 48 of 48, but 2 MiB free at the
    # minimum and 107 sentence-segmenter CPU fallbacks. No run beside a desktop
    # above 1 GiB.
    (("RTX 3050 6 GB", 6144, 8.6), (_L, _Q4, "aggressive", _G, None)),  # ran: 48
    # The Q4_K_M floor (its min_vram_gb, 5) follows the size rule: no 5 GB card ran.
    (("5 GB card", 5120, 8.6), (_L, _Q4, "aggressive", _G, None)),
    (("TITAN RTX", 24576, 7.5), (_L, _Q8, "balanced", _G, None)),
    (("RTX 2080 Ti", 11264, 7.5), (_L, _Q8, "aggressive", _G, None)),
    (("GTX 1080 Ti", 11264, 6.1), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 2060", 6144, 7.5), (_L, _Q4, "aggressive", _G, None)),
]


@pytest.fixture
def card(monkeypatch):
    def pin(mib: int, capability: float) -> float:
        gb = mib / 1024
        monkeypatch.setattr(registry.sys, "platform", "linux")
        monkeypatch.setattr(registry.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(registry, "_cuda_vram_gb", lambda: gb)
        monkeypatch.setattr(registry, "_nvidia_compute_capability", lambda: capability)
        monkeypatch.setattr("bibr.local.llm_models.detect_hardware", lambda: ("cuda", gb))
        monkeypatch.setattr(sys, "platform", "linux")
        return gb

    return pin


def _spec_decode(backend: str, model: str, ocr: str, gb: float, mode: str) -> str | None:
    """The speculative decoding choice of the plan's managed vLLM LLM starts.

    Its server swaps with a managed PaddleOCR-VL server unless both shares
    fit (``ResourceManager._llm_swaps_with_ocr``); the budget is the
    configured 0.85 of the card, the most a start takes.
    """
    if backend != "vllm":
        return None
    swaps = ocr == "paddle-vllm" and not registry.managed_vllm_shares_fit(
        0.85, registry.paddle_vllm_gpu_memory_utilization(vram_gb=gb), vram_gb=gb
    )
    settings = GlobalSettings()
    settings.llm.vllm_spec_decode = mode
    choice = choose_spec_decode(settings, model, swaps_with_ocr=swaps, budget_gb=0.85 * gb)
    return "none" if choice.config is None else choice.config.method


def _runtime(gb: float) -> tuple[str, str, str, str, str | None]:
    backend = resolve_llm_backend("local")
    model = default_local_model(backend)
    ocr = registry.automatic_backend_names(llm_backend=backend)[0]
    return (
        backend,
        model,
        _auto_memory_mode("cuda", gb, 32.0, llm_backend=backend),
        ocr,
        _spec_decode(backend, model, ocr, gb, GlobalSettings().llm.vllm_spec_decode),
    )


def _setup(gb: float) -> tuple[str, str, str, str, str | None]:
    env = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=gb,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    ).env
    return _plan(env, gb)


def _plan(env: dict[str, str], gb: float) -> tuple[str, str, str, str, str | None]:
    ocr = env["OCR_BACKEND"]
    if ocr == "paddle":
        ocr = registry.automatic_backend_names(llm_backend=env["LLM_BACKEND"])[0]
    # Setup writes no speculative decoding setting: the runtime's default applies.
    assert "LLM_VLLM_SPEC_DECODE" not in env
    backend, model = env["LLM_BACKEND"], env["LLM_LOCAL_MODEL"]
    return (
        backend,
        model,
        env["PIPELINE_MEMORY_MODE"],
        ocr,
        _spec_decode(backend, model, ocr, gb, "auto"),
    )


@pytest.mark.parametrize(("row", "expected"), TIER_TABLE, ids=[row[0][0] for row in TIER_TABLE])
def test_the_runtime_default_equals_the_setup_default(card, row, expected):
    _name, mib, capability = row
    gb = card(mib, capability)
    assert _runtime(gb) == expected
    assert _setup(gb) == expected


_VLLM_ROWS = [(row, expected) for row, expected in TIER_TABLE if expected[0] == "vllm"]


@pytest.mark.parametrize(("row", "expected"), _VLLM_ROWS, ids=[r[0] for r, _e in _VLLM_ROWS])
def test_every_vllm_default_takes_turns_with_paddleocr(card, monkeypatch, row, expected):
    """A managed vLLM LLM that takes turns with PaddleOCR-VL may be squeezed to
    the model's floor for a chunk; one that stays up for the run must leave
    bibr's sessions their whole reserve (``bibr.local.vllm_share``). Every vLLM
    default takes turns, as did every run that backs one, so that rule changes
    none of them. The whole card counts as CUDA's here, which only makes the
    shares fit together more easily."""
    from bibr.config import GlobalSettings
    from bibr.pipeline.resources import ResourceManager

    _name, mib, capability = row
    gb = card(mib, capability)
    monkeypatch.setattr(registry, "_managed_gpu_vram_gb", lambda: gb)
    env = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=gb,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    ).env
    settings = GlobalSettings()
    rm = ResourceManager(
        memory_mode=expected[2],
        ocr_backend=env["OCR_BACKEND"],
        managed_vllm_fraction=settings.llm.local_mem_fraction,
        llm_backend=env["LLM_BACKEND"],
        settings=settings,
    )
    assert rm._llm_swaps_with_ocr() is True


@pytest.mark.parametrize(("row", "expected"), TIER_TABLE, ids=[row[0][0] for row in TIER_TABLE])
def test_setup_installs_the_gpu_extra_on_every_ampere_row(card, row, expected):
    """Every run that backs a row had onnxruntime-gpu, so layout and the sentence
    segmenter ran on the GPU (and the NER parser in balanced mode); setup
    installs it on every Ampere or newer row. The older rows were never
    measured with it and keep the CPU build."""
    _name, mib, capability = row
    gb = card(mib, capability)
    extras = _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=gb,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
    ).extras
    assert ("gpu" in extras) is (capability >= 8.0)


# --- the llama-server dimension -------------------------------------------

_SERVER = "/opt/llama.cpp/llama-server"
_CUDA_DEVICES = (
    "ggml_cuda_init: found 1 CUDA devices:\n"
    "  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes\n"
    "Available devices:\n"
    "  CUDA0: NVIDIA GeForce RTX 3080 (10240 MiB, 9668 MiB free)\n"
)
_CUDA_INIT_FAILED = (
    "ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient "
    "for CUDA runtime version\n"
)
_VULKAN_DEVICES = (
    "ggml_vulkan: Found 1 Vulkan devices:\n"
    "ggml_vulkan: 0 = NVIDIA GeForce RTX 3080 (NVIDIA) | uma: 0 | fp16: 1\n"
    "Available devices:\n"
    "  Vulkan0: NVIDIA GeForce RTX 3080 (10240 MiB, 9668 MiB free)\n"
)


def _version(build) -> str:
    return f"version: {build} (1a2b3c4)\nbuilt with cc (GCC) 14.2.1 for x86_64-pc-linux-gnu\n"


#: What the server on PATH prints: (``--list-devices``, ``--version``); a
#: value of ``TimeoutError`` makes that probe time out. None: no server.
LLAMA_SERVERS = {
    "missing": None,
    "cuda b11282": (_CUDA_DEVICES, _version(11282)),
    "cuda b11281": (_CUDA_DEVICES, _version(11281)),
    "cuda, build unparsable": (_CUDA_DEVICES, "version: unknown\n"),
    "cuda, build 0": (_CUDA_DEVICES, _version(0)),
    "cuda, CUDA cannot start": (
        _CUDA_INIT_FAILED + "Available devices:\n",
        _CUDA_INIT_FAILED + _version(11282),
    ),
    "cpu": ("Available devices:\n", _version(11282)),
    "vulkan": (_VULKAN_DEVICES, _version(11282)),
    "probe timed out": (TimeoutError, TimeoutError),
    "build probe timed out": (_CUDA_DEVICES, TimeoutError),
}


@pytest.fixture
def llama_server(monkeypatch):
    """Put a fake llama.cpp server on PATH and let setup look for it.

    Returns a record of every probe command and of every PATH lookup.
    """
    record = SimpleNamespace(commands=[], lookups=0)
    caches = (
        llama_cpp._PROBE_RUN_CACHE,
        llama_cpp._BACKEND_KIND_CACHE,
    )
    saved = [dict(cache) for cache in caches]
    for cache in caches:
        cache.clear()
    monkeypatch.setattr(
        setup_wizard, "_probe_llama_server", lambda: llama_cpp.cuda_llama_server_for_setup()
    )

    def put(name: str, prefix=(_SERVER,)) -> SimpleNamespace:
        outputs = LLAMA_SERVERS[name]

        def find():
            record.lookups += 1
            return None if outputs is None else list(prefix)

        def run(cmd, **_kwargs):
            record.commands.append(list(cmd))
            out = outputs[0] if cmd[-1] == "--list-devices" else outputs[1]
            if out is TimeoutError:
                raise subprocess.TimeoutExpired(cmd, 8.0)
            return SimpleNamespace(stdout="", stderr=out, returncode=0)

        monkeypatch.setattr(llama_cpp, "find_llama_server", find)
        monkeypatch.setattr(llama_cpp.subprocess, "run", run)
        return record

    yield put
    for cache, before in zip(caches, saved, strict=True):
        cache.clear()
        cache.update(before)


_W10 = (_V, _W4A16, "aggressive", _P, "none")
_PAIR = (_L, _Q8, "aggressive", _G, None)
_REJECTED = {
    "cuda b11281": "build 11281 below 11282",
    "cuda, build unparsable": "build unreadable",
    "cuda, build 0": "build unreadable",
    "cuda, CUDA cannot start": (
        "CUDA could not start in this build (driver older than the build's CUDA?)"
    ),
    "cpu": "Vulkan/HIP/CPU build",
    "vulkan": "Vulkan/HIP/CPU build",
    "probe timed out": "probe timed out",
    "build probe timed out": "probe timed out",
}

#: (card, MiB, compute capability, server on PATH) -> setup's plan.
LLAMA_SERVER_TABLE = [
    (("RTX 3080 10 GB", 10240, 8.6, "cuda b11282"), _PAIR),
    # An 11 GB Ampere card, the band's top (an 11 GB card reports 11,264 MiB).
    (("11 GB card", 11264, 8.6, "cuda b11282"), _PAIR),
    *[(("RTX 3080 10 GB", 10240, 8.6, name), _W10) for name in ["missing", *_REJECTED]],
    # Outside the band nothing changes.
    (("RTX 5070", 12227, 12.0, "cuda b11282"), (_V, _W4A16, "balanced", _P, "none")),
    (("RTX 3060 12 GB", 12288, 8.6, "cuda b11282"), (_V, _W4A16, "balanced", _P, "none")),
    (("RTX 4060", 8188, 8.9, "cuda b11282"), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 3070", 8192, 8.6, "cuda b11282"), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 2080 Ti", 11264, 7.5, "cuda b11282"), (_L, _Q8, "aggressive", _G, None)),
    (("RTX 2080 Ti", 11264, 7.5, "vulkan"), (_L, _Q8, "aggressive", _G, None)),
]


def _setup_with_server(gb: float):
    """Setup as ``bibr setup`` runs it: it reads PATH for this card."""
    return _build_recommended_setup(
        platform_key="cuda",
        accelerator_memory_gb=gb,
        system_memory_gb=32.0,
        sys_platform="linux",
        machine="x86_64",
        llama_server=setup_wizard._llama_server_for_card("cuda", gb, "linux"),
    )


def _runtime_with_env(env: dict[str, str], gb: float) -> tuple[str, str, str, str, str | None]:
    """The runtime's resolution with setup's ``.env``: its LLM_BACKEND and model."""
    backend = resolve_llm_backend(env["LLM_BACKEND"])
    model = env.get("LLM_LOCAL_MODEL") or default_local_model(backend)
    assert model == default_local_model(backend)
    ocr = env["OCR_BACKEND"]
    if ocr == "paddle":
        ocr = registry.automatic_backend_names(llm_backend=backend)[0]
    return (
        backend,
        model,
        _auto_memory_mode("cuda", gb, 32.0, llm_backend=backend),
        ocr,
        _spec_decode(backend, model, ocr, gb, "auto"),
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    LLAMA_SERVER_TABLE,
    ids=[f"{r[0]}, {r[3]}" for r, _e in LLAMA_SERVER_TABLE],
)
def test_setup_with_a_llama_server_on_path(card, llama_server, row, expected):
    name, mib, capability, server = row
    gb = card(mib, capability)
    record = llama_server(server)

    setup = _setup_with_server(gb)

    assert _plan(setup.env, gb) == expected
    # The runtime given setup's .env serves setup's plan.
    assert _runtime_with_env(setup.env, gb) == expected
    pair_card = name in ("RTX 3080 10 GB", "11 GB card")
    if expected == _PAIR and pair_card:
        assert "vllm" not in setup.extras
        assert setup.extras == {"ml", "demo"}
        assert _SERVER in setup.runtime_summary
        assert "build 11282" in setup.runtime_summary
        assert "GLM-OCR" in setup.runtime_summary
        assert "148" in setup.runtime_summary and "85" in setup.runtime_summary
        assert "bibr setup --advanced" in setup.runtime_summary
        assert "W4A16" in setup.runtime_summary
        assert setup.fallback_hint
        assert setup.note == ""
    else:
        assert "bibr setup --advanced" not in setup.runtime_summary
        assert setup.fallback_hint == ""
    reason = _REJECTED.get(server) if pair_card else None
    if reason is None:
        assert setup.note == ""
    else:
        assert f"Found llama.cpp at {_SERVER} but not using it: {reason}." in setup.note
        assert "W4A16" in setup.note
    # Only the cards that may get the pair look at PATH.
    assert (record.lookups > 0) is pair_card


@pytest.mark.parametrize("server", ["cuda b11282", "vulkan", "missing"])
def test_llm_local_alone_never_reads_path(card, llama_server, server):
    """``--llm local`` without setup on the RTX 3080 keeps the W4A16 build on
    vLLM, whatever llama.cpp server is on PATH: the runtime never looks."""
    gb = card(10240, 8.6)
    record = llama_server(server)
    assert _runtime(gb) == _W10
    assert record.lookups == 0
    assert record.commands == []


def test_setup_reads_path_when_it_probes_the_machine_itself(card, llama_server, monkeypatch):
    gb = card(10240, 8.6)
    llama_server("cuda b11282")
    monkeypatch.setattr(setup_wizard, "detect_hardware", lambda: ("cuda", gb))
    monkeypatch.setattr("bibr.local.pipeline._get_system_memory_gb", lambda: 32.0)
    assert _plan(_build_recommended_setup().env, gb) == _PAIR


def test_a_described_machine_never_reads_path(card, llama_server):
    """Callers that pass the machine get no llama.cpp server unless they pass one."""
    gb = card(10240, 8.6)
    record = llama_server("cuda b11282")
    assert _setup(gb) == _W10
    assert record.lookups == 0


def test_the_suite_pins_setup_to_no_llama_server(card):
    gb = card(10240, 8.6)
    assert setup_wizard._llama_server_for_card("cuda", gb, "linux") == (
        setup_wizard.SetupLlamaServer()
    )


def test_one_list_devices_and_one_version_run_answer_setup(card, llama_server):
    gb = card(10240, 8.6)
    record = llama_server("cuda b11282")
    _setup_with_server(gb)
    _setup_with_server(gb)
    assert record.commands == [[_SERVER, "--list-devices"], [_SERVER, "--version"]]
    # The runtime's own probes of the same server reuse them.
    assert llama_cpp.probe_backend_kind([_SERVER]) == "cuda"
    assert llama_cpp.probe_build_number([_SERVER]) == 11282
    assert len(record.commands) == 2


def test_a_unified_llama_binary_qualifies_through_llama_serve(card, llama_server):
    gb = card(10240, 8.6)
    record = llama_server("cuda b11282", prefix=("/usr/local/bin/llama", "serve"))
    setup = _setup_with_server(gb)
    assert _plan(setup.env, gb) == _PAIR
    assert "/usr/local/bin/llama serve (build 11282)" in setup.runtime_summary
    assert record.commands == [
        ["/usr/local/bin/llama", "serve", "--list-devices"],
        ["/usr/local/bin/llama", "serve", "--version"],
    ]
