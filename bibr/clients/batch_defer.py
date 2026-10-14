"""Spool front-matter LLM calls for an Anthropic Message Batch, then count the replay.

``bibr batch --llm-batch`` (:mod:`bibr.batch.llm_batch`) runs a corpus in two
rounds around one Message Batch. In round 1 (:func:`collecting`) a deferred
call that misses the LLM response cache does not go out: its cache key and the
full request, as the live client would have sent it, are appended to a spool
file and :class:`~bibr.exceptions.LlmDeferred` fails the paper softly. The
batch answers are then stored in the cache under the spooled keys, and in
round 2 (:func:`observing`) the same calls hit; every lookup a round-1 call
of the same paper deferred is counted, so the run can say whether the batch
answers were actually used.

The hook is :func:`on_cache_lookup`, called by
``LLMClient._invoke_structured`` right after its cache lookup. Outside the two
modes it does nothing. The mode is process-wide (not a context variable)
because the local pipeline runs a paper's calls on worker threads.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.clients.batch import BatchRequest

logger = logging.getLogger(__name__)

# The calls deferred by ``--llm-batch front-matter``: the per-paper fan-out of
# the core-metadata step (``LLMClient.extract_core_metadata``), by the usage
# label the client gives each call.
FRONT_MATTER_LABELS = frozenset(
    {"extract_title_keywords", "extract_authors", "extract_paper_classification"}
)

SPOOL_DIRNAME = "llm_batch"
SPOOL_FILENAME = "spool.jsonl"
SPOOL_VERSION = 1

# The spool holds prompts built from papers that may be unpublished.
_DIR_MODE = 0o700
_FILE_MODE = 0o600

_KEY_RE = re.compile(r"^[0-9a-f]{32}$")


def spool_path(out_dir: Path) -> Path:
    """Where ``bibr batch --llm-batch`` spools the deferred requests of *out_dir*."""
    return Path(out_dir) / SPOOL_DIRNAME / SPOOL_FILENAME


@dataclass
class ReplayStats:
    """Cache lookups of round 2 for the calls round 1 deferred."""

    hits: int = 0
    misses: int = 0
    # Misses on a key the batch did not answer (an errored or expired row):
    # the expected live fallback, not a sign of a request mismatch.
    unanswered: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.lookups if self.lookups else None


@dataclass
class _Collect:
    path: Path
    labels: frozenset[str]
    lock: threading.Lock = field(default_factory=threading.Lock)
    spooled: int = 0
    write_failures: int = 0


@dataclass
class _Observe:
    expected: frozenset[tuple[str, str]]
    labels: frozenset[str]
    unanswered_keys: frozenset[str]
    on_update: Callable[[ReplayStats], None] | None
    stats: ReplayStats = field(default_factory=ReplayStats)
    lock: threading.Lock = field(default_factory=threading.Lock)


_mode_lock = threading.Lock()
_mode: _Collect | _Observe | None = None


@contextmanager
def _activate(mode: _Collect | _Observe) -> Iterator[None]:
    global _mode
    with _mode_lock:
        if _mode is not None:
            raise RuntimeError("an LLM batch round is already active in this process")
        _mode = mode
    try:
        yield
    finally:
        with _mode_lock:
            _mode = None


@contextmanager
def collecting(path: Path, labels: Iterable[str] = FRONT_MATTER_LABELS) -> Iterator[_Collect]:
    """Round 1: spool and defer cache misses of *labels* to *path*."""
    mode = _Collect(path=Path(path), labels=frozenset(labels))
    with _activate(mode):
        yield mode


@contextmanager
def observing(
    expected: Iterable[tuple[str, str]],
    *,
    labels: Iterable[str] = FRONT_MATTER_LABELS,
    unanswered_keys: Iterable[str] = (),
    on_update: Callable[[ReplayStats], None] | None = None,
) -> Iterator[ReplayStats]:
    """Round 2: count lookups of the ``(file hash, label)`` pairs round 1 deferred.

    A call of a deferred label from a paper with no file hash in context is
    counted too. *on_update* sees the stats after each counted lookup.
    """
    mode = _Observe(
        expected=frozenset(expected),
        labels=frozenset(labels),
        unanswered_keys=frozenset(unanswered_keys),
        on_update=on_update,
    )
    with _activate(mode):
        yield mode.stats


def _call_context() -> tuple[str | None, str | None]:
    """``(usage label, file hash)`` of the call being made, from the client's context."""
    from bibr.clients.llm import _usage_file_hash, _usage_label

    usage_key = _usage_file_hash.get()
    if not usage_key:
        return _usage_label.get(), None
    # ``<file hash>#<n>``: unique per pipeline invocation (new_usage_context_key).
    return _usage_label.get(), usage_key.rpartition("#")[0] or usage_key


def on_cache_lookup(
    client: Any,
    key: str,
    *,
    hit: bool,
    deferrable: bool,
    request: tuple[Any, list[dict], str, str | None, int | None],
) -> None:
    """The client's hook after one cache lookup; see the module docstring.

    *deferrable* is false for calls that must not wait for a batch (a JSON-mode
    re-roll, a native backend). *request* is ``(response_model, messages,
    system_prompt, reasoning_effort, max_tokens)`` as ``_invoke_structured``
    received them. Raises :class:`~bibr.exceptions.LlmDeferred` when the call
    was spooled.
    """
    mode = _mode
    if mode is None or not deferrable:
        return
    label, file_hash = _call_context()
    if label not in mode.labels:
        return
    if isinstance(mode, _Observe):
        _observe(mode, key, hit=hit, label=label, file_hash=file_hash)
        return
    if hit:
        return
    if client._settings.llm.provider.strip().lower() != "anthropic":
        return
    if not _spool(mode, client, key, label=label, file_hash=file_hash, request=request):
        return
    from bibr.exceptions import LlmDeferred

    raise LlmDeferred(f"{label} deferred to an Anthropic Message Batch (key {key})")


def _observe(mode: _Observe, key: str, *, hit: bool, label: str, file_hash: str | None) -> None:
    if file_hash is not None and (file_hash, label) not in mode.expected:
        return  # not a call round 1 deferred (a later fallback call, say)
    with mode.lock:
        if hit:
            mode.stats.hits += 1
        elif key in mode.unanswered_keys:
            mode.stats.unanswered += 1
        else:
            mode.stats.misses += 1
        snapshot = ReplayStats(mode.stats.hits, mode.stats.misses, mode.stats.unanswered)
    if mode.on_update is not None:
        mode.on_update(snapshot)


def response_model_path(model: type) -> str:
    return f"{model.__module__}:{model.__qualname__}"


def spool_row(
    client: Any,
    key: str,
    *,
    label: str,
    file_hash: str | None,
    request: tuple[Any, list[dict], str, str | None, int | None],
) -> dict:
    """One spool line: the cache key and the request as the live client sends it."""
    from bibr.clients.llm import _build_call_kwargs

    response_model, messages, system_prompt, reasoning_effort, max_tokens = request
    settings = client._settings
    return {
        "v": SPOOL_VERSION,
        "key": key,
        "label": label,
        "file_hash": file_hash,
        "provider": settings.llm.provider.strip().lower(),
        "model": settings.llm.model,
        "schema": getattr(response_model, "__name__", str(response_model)),
        "response_model": response_model_path(response_model),
        "system": system_prompt,
        # With the prompt builders' content parts, so the batch request keeps
        # the prompt-cache breakpoints the live call would have had.
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "call_kwargs": _build_call_kwargs(
            reasoning_effort=reasoning_effort, max_tokens=max_tokens, settings=settings
        ),
    }


def _spool(
    mode: _Collect,
    client: Any,
    key: str,
    *,
    label: str,
    file_hash: str | None,
    request: tuple[Any, list[dict], str, str | None, int | None],
) -> bool:
    """Append one row; ``False`` (the call then goes live) when it cannot be written."""
    try:
        if load_response_model(response_model_path(request[0])) is not request[0]:
            raise ValueError(f"{request[0]!r} cannot be loaded back by its import path")
        line = json.dumps(
            spool_row(client, key, label=label, file_hash=file_hash, request=request),
            ensure_ascii=False,
        )
        with mode.lock:
            mode.path.parent.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
            fd = os.open(mode.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _FILE_MODE)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            mode.spooled += 1
    except (OSError, TypeError, ValueError, ImportError, AttributeError) as exc:
        with mode.lock:
            mode.write_failures += 1
        logger.warning("Could not spool %s for a Message Batch (%s); calling live", label, exc)
        return False
    return True


def read_spool(path: Path) -> list[dict]:
    """The spool's rows, first occurrence per key; malformed lines are skipped."""
    rows: dict[str, dict] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            logger.warning("Skipping unreadable spool line %d of %s", number, path)
            continue
        if (
            not isinstance(row, dict)
            or row.get("v") != SPOOL_VERSION
            or not isinstance(row.get("key"), str)
            or not _KEY_RE.match(row["key"])
            or not isinstance(row.get("messages"), list)
        ):
            logger.warning("Skipping malformed spool line %d of %s", number, path)
            continue
        rows.setdefault(row["key"], row)
    return list(rows.values())


def load_response_model(path: str) -> type:
    """The pydantic model a spool row names; only bibr's own models load."""
    from pydantic import BaseModel

    module_name, _, qualname = path.partition(":")
    if not qualname or not (module_name == "bibr" or module_name.startswith("bibr.")):
        raise ValueError(f"not a bibr response model: {path!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ValueError(f"not a response model: {path!r}")
    return obj


def request_for(row: dict) -> BatchRequest:
    """The batch request of one spool row, with the row's key as its ``custom_id``."""
    from bibr.clients.batch import BatchRequest
    from bibr.clients.llm import _flatten_content_parts

    messages = tuple(row["messages"])
    flat = _flatten_content_parts(list(messages))
    return BatchRequest(
        custom_id=row["key"],
        schema=load_response_model(row["response_model"]),
        system=row["system"],
        user_text="".join(m["content"] for m in flat if isinstance(m.get("content"), str)),
        messages=messages,
        max_tokens=row.get("max_tokens"),
        reasoning_effort=row.get("reasoning_effort"),
        call_kwargs=row.get("call_kwargs"),
    )
