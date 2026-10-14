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

``LLM_POINTER_OUTPUTS`` needs nothing of its own here. Its title/keywords and
authors calls run under the same usage labels, so :data:`FRONT_MATTER_LABELS`
is the same in both modes, and the hook sees the request they send: the
pointer prompt and response model are spooled, and the cache key (schema,
prompts, request parameters; not the flag) is the one the live pointer call
looks up. The cache holds the pointer answer as the model gave it, and the
client copies the abstract and the affiliations back after the lookup, from a
hit as from a live answer, so a batch answer and a live one end the same way.
When bibr cannot use a pointer answer it asks for the verbatim answer under
the same label. Round 1 never gets there (the pointer call was deferred), so
in round 2 that call goes live; it is counted as a fallback
(:attr:`ReplayStats.fallbacks`), not as a miss, because no batch was asked
for it.
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
from dataclasses import dataclass, field, replace
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
    # Misses of a deferred call's paper and label in a form round 1 did not
    # spool: under LLM_POINTER_OUTPUTS, the verbatim call bibr makes when it
    # cannot use the batch's pointer answer. Live, and outside the hit rate.
    fallbacks: int = 0

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
    # ``(file hash, label, schema)`` of every call round 1 deferred.
    expected: frozenset[tuple[str, str, str]]
    labels: frozenset[str]
    unanswered_keys: frozenset[str]
    on_update: Callable[[ReplayStats], None] | None
    stats: ReplayStats = field(default_factory=ReplayStats)
    lock: threading.Lock = field(default_factory=threading.Lock)
    calls: frozenset[tuple[str, str]] = field(init=False)
    forms: frozenset[tuple[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.calls = frozenset((file_hash, label) for file_hash, label, _ in self.expected)
        self.forms = frozenset((label, schema) for _, label, schema in self.expected)


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
    expected: Iterable[tuple[str, str, str]],
    *,
    labels: Iterable[str] = FRONT_MATTER_LABELS,
    unanswered_keys: Iterable[str] = (),
    on_update: Callable[[ReplayStats], None] | None = None,
) -> Iterator[ReplayStats]:
    """Round 2: count lookups of the ``(file hash, label, schema)`` calls round 1 deferred.

    A call of a deferred label and schema from a paper with no file hash in
    context is counted too. A miss of a deferred paper and label under another
    schema is a fallback (:attr:`ReplayStats.fallbacks`). *on_update* sees the
    stats after each counted lookup.
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
        schema = _schema_name(request[0])
        _observe(mode, key, hit=hit, label=label, file_hash=file_hash, schema=schema)
        return
    if hit:
        return
    if client._settings.llm.provider.strip().lower() != "anthropic":
        return
    if not _spool(mode, client, key, label=label, file_hash=file_hash, request=request):
        return
    from bibr.exceptions import LlmDeferred

    raise LlmDeferred(f"{label} deferred to an Anthropic Message Batch (key {key})")


def _observe(
    mode: _Observe, key: str, *, hit: bool, label: str, file_hash: str | None, schema: str
) -> None:
    if file_hash is not None and (file_hash, label) not in mode.calls:
        return  # round 1 deferred no call of this label for this paper
    if file_hash is None:
        deferred = (label, schema) in mode.forms
    else:
        deferred = (file_hash, label, schema) in mode.expected
    if not deferred and hit:
        return  # another form of the call, answered by the cache: nothing paid
    with mode.lock:
        if not deferred:
            # The deferred call's verbatim re-ask after a pointer answer bibr
            # could not use (LLM_POINTER_OUTPUTS): no batch was asked for it.
            mode.stats.fallbacks += 1
        elif hit:
            mode.stats.hits += 1
        elif key in mode.unanswered_keys:
            mode.stats.unanswered += 1
        else:
            mode.stats.misses += 1
        snapshot = replace(mode.stats)
    if mode.on_update is not None:
        mode.on_update(snapshot)


def _schema_name(response_model: Any) -> str:
    """The schema a spool row records for *response_model* (``spool_row``'s ``schema``)."""
    return getattr(response_model, "__name__", str(response_model))


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
        "schema": _schema_name(response_model),
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


def _parse_row(line: bytes) -> dict | None:
    """One spool line as a row, or ``None`` when it is not a usable row."""
    try:
        row = json.loads(line)
    except ValueError:
        return None
    if (
        not isinstance(row, dict)
        or row.get("v") != SPOOL_VERSION
        or not isinstance(row.get("key"), str)
        or not _KEY_RE.match(row["key"])
        or not isinstance(row.get("messages"), list)
    ):
        return None
    return row


def iter_spool(path: Path) -> Iterator[tuple[int, dict]]:
    """``(byte offset, row)`` of each usable spool line, read one line at a time.

    A corpus-scale spool holds gigabytes of prompts; a reader keeps what it
    needs (keys and offsets, say) and loads a row again by its offset with
    :func:`read_row`. Malformed lines are skipped.
    """
    path = Path(path)
    if not path.is_file():
        return
    with path.open("rb") as handle:
        offset = 0
        for number, line in enumerate(handle, start=1):
            start, offset = offset, offset + len(line)
            if not line.strip():
                continue
            row = _parse_row(line)
            if row is None:
                logger.warning("Skipping malformed spool line %d of %s", number, path)
                continue
            yield start, row


def read_row(handle: Any, offset: int) -> dict:
    """The row at *offset* of a spool opened in binary mode (see :func:`iter_spool`)."""
    handle.seek(offset)
    row = _parse_row(handle.readline())
    if row is None:
        raise ValueError(f"no spool row at byte {offset}")
    return row


def read_spool(path: Path, *, unique: bool = True) -> list[dict]:
    """The spool's rows, by default only the first per key; malformed lines are skipped."""
    rows: dict[str, dict] = {}
    every: list[dict] = []
    for _, row in iter_spool(path):
        if unique:
            rows.setdefault(row["key"], row)
        else:
            every.append(row)
    return list(rows.values()) if unique else every


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
