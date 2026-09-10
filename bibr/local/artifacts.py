"""Local durable artifact writer used by explicit CLI file outputs."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import uuid
from pathlib import Path

from bibr.pipeline.artifacts import (
    ArtifactDisposition,
    EnrichmentSidecar,
    RunState,
    canonical_json_sha256,
)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP, errno.EPERM}:
            return
        raise
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in {errno.EBADF, errno.EINVAL, errno.ENOTSUP, errno.EPERM}:
                raise
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: object, **json_kwargs) -> None:
    """Atomically replace *path* with flushed, fsynced UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp)
    options = {"ensure_ascii": False, "allow_nan": False, **json_kwargs}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, **options)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


class LocalArtifactSink:
    """Artifact sink rooted at one caller-requested result filename."""

    def __init__(self, destination: Path, *, json_kwargs: dict | None = None) -> None:
        self.requested_path = Path(destination)
        self._json_kwargs = dict(json_kwargs or {})
        self._attempts: list[dict] = []
        self._events: list[dict[str, str]] | None = None

    def destination_path(self, fs) -> Path:
        disposition = fs.artifact_disposition or ArtifactDisposition.PROMOTABLE
        if disposition == ArtifactDisposition.PROMOTABLE:
            return self.requested_path
        return (
            self.requested_path.parent / "_quarantine" / str(disposition) / self.requested_path.name
        )

    def sidecar_path(self, fs) -> Path:
        destination = self.destination_path(fs)
        return destination.with_name(f"{destination.name}.enrichment.json")

    def core_path(self, fs) -> Path:
        destination = self.destination_path(fs)
        if destination.suffix:
            return destination.with_suffix(f".core{destination.suffix}")
        return destination.with_name(f"{destination.name}.core.json")

    def receipt_path(self, fs) -> Path:
        # Stable across disposition resolution: STARTED is recorded before
        # Validate, while quarantine placement is not known until post-parse.
        return self.requested_path.with_name(f"{self.requested_path.name}.receipt.json")

    def write_core(self, fs, payload: dict) -> str:
        # The sibling core remains immutable when the public destination is
        # later materialized with a verified enrichment sidecar.
        atomic_write_json(self.core_path(fs), payload, **self._json_kwargs)
        return canonical_json_sha256(payload)

    def materialize(self, fs, payload: dict) -> None:
        """Atomically publish an already verified payload to its routed destination."""

        atomic_write_json(self.destination_path(fs), payload, **self._json_kwargs)

    def read_core(self, fs) -> dict:
        with self.core_path(fs).open(encoding="utf-8") as handle:
            return json.load(handle)

    def write_enrichment(self, fs, sidecar: EnrichmentSidecar) -> None:
        atomic_write_json(self.sidecar_path(fs), sidecar.to_dict(), **self._json_kwargs)

    def read_enrichment(self, fs) -> EnrichmentSidecar:
        with self.sidecar_path(fs).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("enrichment sidecar must be a JSON object")
        return EnrichmentSidecar.from_dict(value)

    def _load_attempt_history(self, fs) -> None:
        if self._events is not None:
            return
        receipt_path = self.receipt_path(fs)
        existing: dict = {}
        try:
            with receipt_path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing = loaded
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        attempts = existing.get("attempts")
        if isinstance(attempts, list):
            self._attempts = list(attempts)
        elif isinstance(existing.get("events"), list) and existing["events"]:
            self._attempts = [
                {
                    "attempt_id": existing.get("attempt_id", "legacy"),
                    "events": list(existing["events"]),
                }
            ]
        attempt = {"attempt_id": uuid.uuid4().hex, "events": []}
        self._attempts.append(attempt)
        self._events = attempt["events"]

    @staticmethod
    def _terminal(events: list[dict[str, str]]) -> bool:
        if not events:
            return False
        last = events[-1]
        return last["state"] == str(RunState.ENRICHMENT_COMPLETE) or (
            last["state"] == str(RunState.CORE_WRITTEN)
            and last.get("detail") == "enrichment_not_requested"
        )

    @staticmethod
    def _accept_transition(events: list[dict[str, str]], state: RunState) -> bool:
        if not events:
            return True
        previous = events[-1]["state"]
        if previous in {
            str(RunState.ENRICHMENT_COMPLETE),
            str(RunState.FAILED),
            str(RunState.CUTOFF_INTERRUPTED),
        }:
            return False
        if (
            previous == str(RunState.CORE_WRITTEN)
            and events[-1].get("detail") == "enrichment_not_requested"
        ):
            return False
        if previous == str(state):
            return False
        rank = {
            str(RunState.STARTED): 0,
            str(RunState.CORE_WRITTEN): 1,
            str(RunState.ENRICHMENT_PARTIAL): 2,
            str(RunState.ENRICHMENT_COMPLETE): 3,
            str(RunState.FAILED): 3,
            str(RunState.CUTOFF_INTERRUPTED): 3,
        }
        return rank[str(state)] >= rank.get(previous, -1)

    def record(self, fs, state: RunState, *, detail: str | None = None) -> None:
        self._load_attempt_history(fs)
        assert self._events is not None
        event = {"state": str(state)}
        if detail:
            event["detail"] = " ".join(str(detail).split())[:512]
        # Receipts are a monotonic state log. Repeated callbacks are harmless,
        # and a completed attempt cannot be regressed by late cancellation.
        if self._accept_transition(self._events, state):
            self._events.append(event)
        disposition = fs.artifact_disposition or ArtifactDisposition.PROMOTABLE
        terminal = self._terminal(self._events)
        retryable = disposition != ArtifactDisposition.PROMOTABLE or not terminal
        receipt = {
            "schema_version": "1",
            "source_path": str(fs.path),
            "destination_path": str(self.destination_path(fs)),
            "disposition": str(disposition),
            "core_sha256": fs.core_sha256,
            "retryable": retryable,
            "events": list(self._events),
            "attempts": self._attempts,
        }
        atomic_write_json(self.receipt_path(fs), receipt, **self._json_kwargs)
