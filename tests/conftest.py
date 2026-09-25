import copy
import ipaddress
import logging
import os
import socket
import subprocess
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Guard: never let a test install packages into the environment running it.
#
# ``bibr setup`` installs extras for real: in a source checkout
# ``SetupWizard._install_selected_extras`` runs ``uv sync --inexact
# --extra=ml`` against the checkout's ``.venv``. Wizard tests that reached it
# unmocked installed torch partway through any run in a core-only venv —
# including CI's core-install job, which checks torch is absent only before
# the suite starts — while a full ``--extra all`` env skipped that branch and
# hid it. A test exercising an install path must stub ``subprocess.run`` in
# the module under test.
# ---------------------------------------------------------------------------
_real_subprocess_run = subprocess.run


def _changes_environment(args) -> bool:
    """True for the uv/pip commands that install into or remove from an env."""
    if isinstance(args, str):
        argv = args.split()
    elif isinstance(args, (bytes, os.PathLike)):
        argv = [os.fsdecode(args)]
    else:
        argv = [os.fsdecode(arg) for arg in args]
    if not argv:
        return False
    tool = Path(argv[0]).name.lower().removesuffix(".exe")
    rest = argv[1:]
    if rest[:2] == ["-m", "pip"]:
        tool, rest = "pip", rest[2:]
    words = [arg for arg in rest if not arg.startswith("-")]
    if tool == "uv":
        return words[:1] in (["sync"], ["add"], ["remove"]) or words[:2] in (
            ["pip", "install"],
            ["pip", "uninstall"],
            ["pip", "sync"],
        )
    return tool in ("pip", "pip3") and words[:1] in (["install"], ["uninstall"])


def _guarded_run(*popenargs, **kwargs):
    args = popenargs[0] if popenargs else kwargs.get("args", ())
    if _changes_environment(args):
        # pytest.fail raises a BaseException, so no ``except Exception`` in the
        # code under test can swallow it and let the test pass.
        pytest.fail(
            f"subprocess.run({args!r}) blocked: it would install into the "
            "environment running the tests. Stub subprocess.run in the module "
            "under test, or keep the test off the install path."
        )
    return _real_subprocess_run(*popenargs, **kwargs)


@pytest.fixture(autouse=True)
def _package_install_guard(monkeypatch):
    """Tests that stub ``subprocess.run`` themselves replace this for their duration."""
    monkeypatch.setattr(subprocess, "run", _guarded_run)


@pytest.fixture(autouse=True)
def _pin_cuda_probe_for_platform_tests(monkeypatch):
    """The automatic Linux OCR chain now asks the hardware before listing
    ``paddle-vllm``. Tests that pin ``sys.platform`` to linux were written
    against the old unconditional chain, so pin a roomy GPU here; tests that
    exercise the CPU-only / small-GPU paths patch the probe themselves."""
    monkeypatch.setattr("bibr.ocr.registry._cuda_vram_gb", lambda: 24.0)


@pytest.fixture(autouse=True)
def _skip_installed_onnxruntime_builds_check(monkeypatch):
    """Keep ``get_ort_providers()``'s installed-builds check off the test venv.

    It reads the installed distributions and warns, once per process, when
    ``onnxruntime`` and ``onnxruntime-gpu`` are both installed and the CPU
    build loads. In a venv with the ``gpu`` extra, the first test to mock a
    CPU-only onnxruntime would log that warning. Mark it checked;
    tests/test_onnx_providers_builds.py resets it with stubbed metadata.
    """
    monkeypatch.setattr("bibr.utils.onnx_providers._gpu_build_shadowed", False)


# ---------------------------------------------------------------------------
# Isolation: the process-global Settings must not leak between tests.
#
# ``Settings`` (bibr/config.py) is a process-global singleton whose nested
# sections are pydantic v2 models. Pydantic records every attribute assignment
# in ``model_fields_set``, and ``monkeypatch.setattr`` undoes itself with
# another assignment — so the field stays marked "user-set" after the test.
# About 15 production sites read ``model_fields_set`` as "the user explicitly
# set this" (llama_cpp.py, local/llm.py, ocr/registry.py, ...), which made the
# suite pass only in one file order (x-tests-1). Snapshot every section's
# values, ``model_fields_set`` and private attributes before each test and
# restore them in place afterwards (in place, so modules holding a section
# reference keep seeing the same object).
# ---------------------------------------------------------------------------


def _snapshot_settings():
    from pydantic import BaseModel

    from bibr.config import Settings

    inst = object.__getattribute__(Settings, "_instance")
    if inst is None:
        # The test would construct the singleton on first touch and any
        # pollution would then have no baseline to restore to. Construct it
        # now (once per session — later tests reuse the instance) so every
        # test starts from, and returns to, the same state. A failure here
        # means the pinned test env itself is invalid; fall back to no
        # snapshot rather than erroring the whole suite at setup.
        try:
            inst = Settings._get()
        except Exception:
            return None
    saved_sections = {}
    for name in type(inst).model_fields:
        value = getattr(inst, name, None)
        if isinstance(value, BaseModel):
            saved_sections[name] = (
                set(value.model_fields_set),
                value.model_copy(deep=True),
            )
    return {
        "fields_set": set(inst.model_fields_set),
        "extras": copy.deepcopy(inst.__pydantic_extra__),
        "private": copy.deepcopy(inst.__pydantic_private__),
        "root_values": {
            name: copy.deepcopy(getattr(inst, name, None))
            for name in type(inst).model_fields
            if name not in saved_sections
        },
        "sections": saved_sections,
    }


def _restore_settings(saved) -> None:
    from pydantic import BaseModel

    from bibr.config import Settings

    inst = object.__getattribute__(Settings, "_instance")
    if inst is None or saved is None:
        return
    for name, (fields_set, snapshot) in saved["sections"].items():
        current = getattr(inst, name, None)
        if isinstance(current, BaseModel) and type(current) is type(snapshot):
            for field in type(snapshot).model_fields:
                setattr(current, field, getattr(snapshot, field))
            current.model_fields_set.clear()
            current.model_fields_set.update(fields_set)
            if current.__pydantic_private__ is not None and snapshot.__pydantic_private__:
                current.__pydantic_private__.clear()
                current.__pydantic_private__.update(copy.deepcopy(snapshot.__pydantic_private__))
        else:
            setattr(inst, name, snapshot)
    for name, value in saved["root_values"].items():
        setattr(inst, name, value)
    inst.model_fields_set.clear()
    inst.model_fields_set.update(saved["fields_set"])
    if inst.__pydantic_extra__ is not None:
        inst.__pydantic_extra__.clear()
        if saved["extras"]:
            inst.__pydantic_extra__.update(saved["extras"])
    if inst.__pydantic_private__ is not None and saved["private"] is not None:
        inst.__pydantic_private__.clear()
        inst.__pydantic_private__.update(saved["private"])


@pytest.fixture(autouse=True)
def _isolate_global_settings(monkeypatch):
    """Each test sees pristine global Settings; its mutations die with it.

    Takes ``monkeypatch`` (the same function-scoped instance the test uses)
    so teardown can undo the test's attribute swaps *before* restoring the
    snapshot: ``monkeypatch`` undoes itself with another assignment, which
    pydantic records in ``model_fields_set`` — restoring first and letting
    the undo run afterwards would re-mark every touched field as user-set.
    ``MonkeyPatch.undo()`` is idempotent, so the later automatic teardown is
    a no-op.
    """
    saved = _snapshot_settings()
    yield
    monkeypatch.undo()
    _restore_settings(saved)


# ---------------------------------------------------------------------------
# Isolation: CLI-invoked logging must not leak between tests.
#
# ``bibr chew``/``doctor`` run ``logging.basicConfig`` and pin
# ``bibr.local``/``bibr.pipeline``/``bibr.structure``/``bibr.extract`` to
# INFO. Tests calling ``main()`` left those levels (and root handlers) set,
# which broke the serve log-level test whenever it ran afterwards (x-tests-9
# — not a production bug: ``bibr serve`` dispatches before that setup runs).
# Snapshot the root handlers/level and every existing logger's level, and
# reset any logger created mid-test to NOTSET afterwards.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_logging_state():
    import _pytest.logging

    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    before_levels = {
        name: logger.level
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    yield
    for handler in root.handlers[:]:
        if handler not in before_handlers and not isinstance(
            handler, _pytest.logging.LogCaptureHandler
        ):
            root.removeHandler(handler)
    root.setLevel(before_level)
    known = set(before_levels)
    for name, logger in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        if name in before_levels:
            logger.setLevel(before_levels[name])
        elif name not in known:
            logger.setLevel(logging.NOTSET)


# ---------------------------------------------------------------------------
# Guard: unit tests must not open non-loopback sockets.
#
# Three classifier tests used to POST header text to Google's API (conftest's
# ``test-google-key``) and pass only because the call failed; offline they
# each burned ~20 s of retry/backoff (x-tests-2). Those tests now stub the LLM
# tier. This guard sits next to the killpg/subprocess guards so the next leak
# fails loudly instead of slowing the suite or exfiltrating fixture text.
# Loopback stays open (serve TestClients, the wedged-Redis probe, spawned
# LitServe workers all talk to 127.0.0.1); Unix sockets are untouched.
# ---------------------------------------------------------------------------

_LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


def _is_loopback_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except (UnicodeDecodeError, AttributeError):
            return False
    if not isinstance(host, str):
        return False
    if not host or host in _LOOPBACK_NAMES:
        return True
    try:
        addr = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return addr.is_loopback or addr.is_unspecified


def _guarded_connect(real_connect):
    def connect(self, address, *args, **kwargs):
        if self.family not in (socket.AF_INET, socket.AF_INET6):
            return real_connect(self, address, *args, **kwargs)
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback_host(host):
            pytest.fail(
                f"socket.connect({address!r}) blocked: unit tests must not open "
                "non-loopback sockets. Stub the client (or the module's LLM tier) "
                "instead of letting the call reach the network."
            )
        return real_connect(self, address, *args, **kwargs)

    return connect


def _guarded_create_connection(real_create_connection):
    def create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback_host(host):
            pytest.fail(
                f"socket.create_connection({address!r}) blocked: unit tests must not "
                "open non-loopback sockets. Stub the client instead."
            )
        return real_create_connection(address, *args, **kwargs)

    return create_connection


def _guarded_getaddrinfo(real_getaddrinfo):
    def getaddrinfo(host, *args, **kwargs):
        if isinstance(host, str) and host not in _LOOPBACK_NAMES:
            try:
                ipaddress.ip_address(host.split("%", 1)[0])
            except ValueError:
                pytest.fail(
                    f"socket.getaddrinfo({host!r}) blocked: unit tests must not "
                    "resolve external names. Stub the client instead."
                )
        return real_getaddrinfo(host, *args, **kwargs)

    return getaddrinfo


@pytest.fixture(autouse=True)
def _block_non_loopback_sockets(monkeypatch):
    """Fail any test that tries a real non-loopback connect or DNS lookup."""
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect(socket.socket.connect))
    monkeypatch.setattr(
        socket, "create_connection", _guarded_create_connection(socket.create_connection)
    )
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo(socket.getaddrinfo))
