"""Offline structured batch LLM layer (Anthropic Message Batches, v1).

Deliberately separate from bibr/clients/llm.py (the live, per-request path).
Turns a list of independent structured-extraction requests into one provider
batch job, polls to completion, and returns validated pydantic results keyed
by a caller custom_id. Structured output is a forced tool whose input_schema
is the request schema's model_json_schema() sanitized to strict-tool-legal
form. The `anthropic` SDK is imported lazily (optional `batch` extra).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from bibr.clients.prompts import PROMPTS, prompt_text

_STRICT_DROP_KEYS = frozenset({"title", "default"})


def _sanitize_strict(schema: dict) -> dict:
    """Rewrite a pydantic JSON schema into an Anthropic strict-tool-legal one.

    Every object node gets ``additionalProperties: false`` and ``required`` =
    all its property keys; ``title``/``default`` are dropped everywhere. Recurses
    through ``properties``, ``items``, ``anyOf``/``allOf``/``oneOf`` and ``$defs``.
    Nullable-enum unions (``anyOf: [{enum...}, {type: null}]``) pass through intact.
    """
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k not in _STRICT_DROP_KEYS}
    if out.get("type") == "object" or "properties" in out:
        props = {k: _sanitize_strict(v) for k, v in out.get("properties", {}).items()}
        out["properties"] = props
        out["required"] = list(props.keys())
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = _sanitize_strict(out["items"])
    for comb in ("anyOf", "allOf", "oneOf"):
        if comb in out:
            out[comb] = [_sanitize_strict(s) for s in out[comb]]
    if "$defs" in out:
        out["$defs"] = {k: _sanitize_strict(v) for k, v in out["$defs"].items()}
    return out


@dataclass(frozen=True)
class BatchRequest:
    """One unit of batch work: a caller-unique custom_id, the pydantic schema to
    validate the result into, and the resolved system + flattened user text."""

    custom_id: str
    schema: type
    system: str
    user_text: str

    @classmethod
    def from_spec(cls, custom_id, spec_name, *, schema=None, **build_kwargs):
        spec = PROMPTS[spec_name]
        parts = spec.build_user(**build_kwargs)
        return cls(
            custom_id=custom_id,
            schema=schema or spec.response_model,
            system=spec.system,
            user_text=prompt_text(parts),
        )


@dataclass(frozen=True)
class BatchError:
    """A non-succeeded batch row (errored / canceled / expired), or a
    post-hoc validation failure of a succeeded row."""

    custom_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BatchStatus:
    batch_id: str
    processing_status: str
    counts: dict
    ended: bool


def _tool_name(schema) -> str:
    # Anthropic tool names must match ^[a-zA-Z0-9_-]{1,64}$; class names satisfy it.
    return schema.__name__


def _tool_use_input(message):
    """First tool_use block's `.input` from an Anthropic Message, or None."""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    return None


# If the SDK/API rejects strict tools (needs a beta header on your SDK version),
# flip to False — forced tool_choice + client-side pydantic validation still
# guarantees a schema-valid result; strict only adds grammar-level enforcement.
_STRICT_TOOLS = True


class AnthropicBatchAdapter:
    """Anthropic Message Batches adapter (forced-tool structured output)."""

    def __init__(
        self,
        model="claude-haiku-4-5",
        max_tokens=1024,
        *,
        client=None,
        settings=None,
    ):
        from bibr.config import snapshot_settings

        self.model = model
        self.max_tokens = max_tokens
        self._client = client  # inject a fake in tests; lazily built otherwise
        self._settings = settings if settings is not None else snapshot_settings()

    def _build_request(self, req: BatchRequest) -> dict:
        tool_name = _tool_name(req.schema)
        tool = {
            "name": tool_name,
            "description": f"Emit the {tool_name} structured result.",
            "input_schema": _sanitize_strict(req.schema.model_json_schema()),
            "strict": _STRICT_TOOLS,
        }
        return {
            "custom_id": req.custom_id,
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": req.system,
                "messages": [{"role": "user", "content": req.user_text}],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool_name},
            },
        }

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy: optional `batch` extra

            self._client = anthropic.Anthropic(api_key=self._settings.ANTHROPIC_API_KEY)
        return self._client

    @staticmethod
    def _verify_manifest(state, requests, state_path) -> None:
        """Refuse to resume a batch that was submitted for different work.

        ``retrieve`` skips any ``custom_id`` it does not recognise, so resuming
        against a state file from another run returned a quietly partial (or
        entirely empty) result set that looked like a successful batch. The
        manifest recorded at submit time is what makes that detectable — it was
        written and then never read.

        A manifest entry with no matching request is fine (the caller resumed
        with a subset, e.g. after writing some results durably). A *request*
        that the manifest does not vouch for is not.
        """
        manifest = state.get("manifest")
        if not isinstance(manifest, dict):
            # Pre-manifest state file; nothing to check against.
            return
        mismatched = [
            r.custom_id
            for r in requests
            if manifest.get(r.custom_id) != hashlib.sha256(r.user_text.encode()).hexdigest()[:16]
        ]
        if mismatched:
            raise ValueError(
                f"Batch state {state_path} was submitted for different requests: "
                f"{len(mismatched)} of {len(requests)} do not match its manifest "
                f"(first: {mismatched[0]!r}). Delete the state file to submit a new batch."
            )

    def retrieve(self, batch_id, requests):
        by_id = {r.custom_id: r.schema for r in requests}
        out: dict = {}
        for row in self.client.messages.batches.results(batch_id):
            cid = row.custom_id
            if cid not in by_id:
                continue  # already-durable / stale id — resume case
            result = row.result
            if result.type != "succeeded":
                err = getattr(result, "error", None)
                out[cid] = BatchError(
                    custom_id=cid,
                    error_type=getattr(err, "type", result.type),
                    message=str(getattr(err, "message", result.type)),
                )
                continue
            payload = _tool_use_input(result.message)
            if payload is None:
                out[cid] = BatchError(cid, "no_tool_use", "no tool_use block in response")
                continue
            try:
                out[cid] = by_id[cid].model_validate(payload)
            except Exception as e:  # noqa: BLE001 - surface as a row error, don't abort the batch
                out[cid] = BatchError(cid, "validation_error", str(e)[:500])
        return out

    def submit(self, requests) -> str:
        payload = [self._build_request(r) for r in requests]
        batch = self.client.messages.batches.create(requests=payload)
        return batch.id

    def poll(self, batch_id) -> BatchStatus:
        b = self.client.messages.batches.retrieve(batch_id)
        status = b.processing_status
        rc = b.request_counts
        counts = {
            k: getattr(rc, k, 0)
            for k in ("processing", "succeeded", "errored", "canceled", "expired")
        }
        return BatchStatus(batch_id, status, counts, ended=status == "ended")

    def run(self, requests, *, state_path=None, poll_interval=60, timeout_s=86400, progress=None):
        state_path = Path(state_path) if state_path else None
        batch_id = None
        if state_path and state_path.exists():
            state = json.loads(state_path.read_text())
            batch_id = state.get("batch_id")
            if batch_id is not None:
                self._verify_manifest(state, requests, state_path)
        if batch_id is None:
            batch_id = self.submit(requests)
            if state_path:
                manifest = {
                    r.custom_id: hashlib.sha256(r.user_text.encode()).hexdigest()[:16]
                    for r in requests
                }
                state_path.write_text(
                    json.dumps(
                        {
                            "provider": "anthropic",
                            "model": self.model,
                            "batch_id": batch_id,
                            "manifest": manifest,
                        }
                    )
                )
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.poll(batch_id)
            if progress:
                progress(
                    f"{status.counts['succeeded']}/{len(requests)} succeeded "
                    f"(status={status.processing_status})"
                )
            if status.ended:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"batch {batch_id} did not end within {timeout_s}s")
            if poll_interval:
                time.sleep(poll_interval)
        return self.retrieve(batch_id, requests)
