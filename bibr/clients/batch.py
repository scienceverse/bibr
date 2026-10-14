"""Anthropic Message Batches client: structured calls at half the price, answered later.

Deliberately separate from bibr/clients/llm.py (the live, per-request path),
but each request is built the way that path builds its call to Anthropic: the
:class:`~bibr.clients.providers.anthropic.AnthropicProvider` call parameters
(temperature or none, thinking budget, the non-streaming output cap) and
prompt-cache transform, then Instructor's own tool request for the response
schema. A succeeded row is parsed by Instructor's response parser, so a batch
answer validates exactly as the live call would have. ``bibr batch --llm-batch``
(:mod:`bibr.batch.llm_batch`) uses this to fill the LLM response cache before
it replays a corpus. The ``anthropic`` SDK is imported lazily (``batch`` extra).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from bibr.clients.prompts import PROMPTS, prompt_text

# Anthropic's limits for one batch: 100,000 requests or 256 MB of requests.
# The byte budget leaves room for the JSON envelope around the requests.
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 256 * 1024 * 1024
_BATCH_BYTE_BUDGET = MAX_BATCH_BYTES - 4 * 1024 * 1024

# What Instructor adds to the system prompt itself whenever it leaves the tool
# choice to the model (extended thinking), so the answer is the tool call alone.
_TOOL_ONLY_INSTRUCTION = "Return only the tool call and no additional text."

_T = TypeVar("_T")


def partition(sized: Iterable[tuple[_T, int]]) -> Iterator[list[_T]]:
    """Cut ``(item, request bytes)`` pairs, in order, into batches within Anthropic's limits.

    Lazy, so a caller can measure one request at a time and keep only the
    items (cache keys, say) rather than every built request.
    """
    current: list[_T] = []
    size = 0
    for item, n in sized:
        if current and (len(current) >= MAX_BATCH_REQUESTS or size + n > _BATCH_BYTE_BUDGET):
            yield current
            current, size = [], 0
        current.append(item)
        size += n
    if current:
        yield current


@dataclass(frozen=True)
class BatchRequest:
    """One unit of batch work.

    ``custom_id`` is caller-unique (at most 64 characters of ``[A-Za-z0-9_-]``);
    ``schema`` is the pydantic response model. ``messages`` are the
    conversation after the system prompt as the live client receives them,
    with the prompt builders' content parts (see :func:`bibr.clients.prompts.part`)
    still in place, so the provider can mark its prompt-cache breakpoints;
    without them a single user turn of ``user_text`` is sent. ``user_text``
    is the flattened prompt, which the resume manifest fingerprints.

    ``call_kwargs`` are the provider call parameters as the live client
    resolved them (``max_tokens``, ``temperature`` or ``thinking``); without
    them they are resolved from the settings for ``max_tokens`` (the task's
    output cap) and ``reasoning_effort``.
    """

    custom_id: str
    schema: type
    system: str
    user_text: str
    messages: tuple[dict, ...] | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    call_kwargs: Mapping[str, Any] | None = None

    @classmethod
    def from_spec(
        cls,
        custom_id,
        spec_name,
        *,
        schema=None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        **build_kwargs,
    ):
        spec = PROMPTS[spec_name]
        parts = spec.build_user(**build_kwargs)
        return cls(
            custom_id=custom_id,
            schema=schema or spec.response_model,
            system=spec.system,
            user_text=prompt_text(parts) if isinstance(parts, list) else str(parts),
            messages=({"role": "user", "content": parts},),
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
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


def _manifest_digest(request: BatchRequest) -> str:
    return hashlib.sha256(request.user_text.encode()).hexdigest()[:16]


def _instructor_tools():
    """Instructor's Anthropic TOOLS mode, request builder and response parser."""
    from instructor.v2.core.mode import Mode
    from instructor.v2.core.providers import Provider
    from instructor.v2.core.response import handle_response_model, process_response

    return Mode.TOOLS, Provider.ANTHROPIC, handle_response_model, process_response


def transport_problem() -> str | None:
    """Why this environment cannot build batch requests, or ``None``.

    The requests come from Instructor's v2 request builder, which older
    Instructor releases do not ship; checked up front so a corpus run can
    refuse before any paper is processed.
    """
    try:
        _instructor_tools()
    except ImportError as exc:
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = f"instructor {version('instructor')}"
        except PackageNotFoundError:
            installed = "no instructor"
        return (
            "Message Batches requests are built with Instructor's v2 request builder, "
            f"which {installed} does not provide ({exc}): upgrade instructor."
        )
    return None


class AnthropicBatchAdapter:
    """Anthropic Message Batches adapter (the live path's tool requests).

    ``model`` defaults to ``LLM_BATCH_MODEL``. An answer can only stand in
    for a live one when it came from the model the live call would use, so
    the cache filler passes ``LLM_MODEL``. ``max_tokens`` is the output cap
    of a request that carries neither ``max_tokens`` nor ``call_kwargs``;
    ``None`` keeps the provider's (``LLM_MAX_TOKENS`` within the
    non-streaming cap).
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        *,
        client=None,
        settings=None,
    ):
        from bibr.config import snapshot_settings

        self._settings = settings if settings is not None else snapshot_settings()
        self.model = model or self._settings.llm.batch_model
        self.max_tokens = max_tokens
        self._client = client  # inject a fake in tests; lazily built otherwise
        self._provider_instance = None

    def _provider(self):
        """The live Anthropic adapter, configured for ``self.model``."""
        if self._provider_instance is None:
            from bibr.clients.providers.anthropic import AnthropicProvider

            settings = self._settings
            if settings.llm.model != self.model:
                llm = settings.llm.model_copy(update={"model": self.model})
                settings = settings.model_copy(update={"llm": llm})
            self._provider_instance = AnthropicProvider(settings=settings)
        return self._provider_instance

    def build_params(self, req: BatchRequest) -> dict:
        """The Messages API parameters of *req*, as the live client sends them."""
        from bibr.clients.llm import _flatten_content_parts
        from bibr.clients.providers.anthropic import rejects_forced_tool_choice

        mode, provider_name, handle_response_model, _ = _instructor_tools()
        provider = self._provider()
        if req.call_kwargs is not None:
            # A JSON round trip: the caller's mapping must not alias the request.
            kwargs = json.loads(json.dumps(dict(req.call_kwargs)))
        else:
            kwargs = provider.call_kwargs(req.reasoning_effort, req.max_tokens or self.max_tokens)
        turns = req.messages or ({"role": "user", "content": req.user_text},)
        messages = [{"role": "system", "content": req.system}, *(dict(m) for m in turns)]
        messages = _flatten_content_parts(provider.transform_messages(messages))
        _, params = handle_response_model(
            req.schema,
            mode=mode,
            provider=provider_name,
            model=self.model,
            messages=messages,
            **kwargs,
        )
        choice = params.get("tool_choice")
        if (
            rejects_forced_tool_choice(self.model)
            and isinstance(choice, dict)
            and choice.get("type") in ("tool", "any")
        ):
            # These models answer a forced tool choice with a 400. Leave the
            # choice to the model and ask for the tool call alone, as
            # Instructor does when extended thinking rules out forcing it.
            params["tool_choice"] = {"type": "auto"}
            system = params.get("system")
            extra = {"type": "text", "text": _TOOL_ONLY_INSTRUCTION}
            if isinstance(system, list):
                params["system"] = [*system, extra]
            elif isinstance(system, str) and system:
                params["system"] = [{"type": "text", "text": system}, extra]
            else:
                params["system"] = [extra]
        return params

    def _build_request(self, req: BatchRequest) -> dict:
        return {"custom_id": req.custom_id, "params": self.build_params(req)}

    def request_bytes(self, req: BatchRequest) -> int:
        """The size *req* adds to a batch's request body."""
        return len(json.dumps(self._build_request(req)).encode("utf-8")) + 1

    def split(self, requests: Sequence[BatchRequest]) -> list[list[BatchRequest]]:
        """*requests* in order, cut into batches within Anthropic's per-batch limits."""
        return list(partition((req, self.request_bytes(req)) for req in requests))

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy: optional `batch` extra

            from bibr.clients.providers.anthropic import _check_sdk_version

            _check_sdk_version()
            api_key = self._settings.llm.api_key or self._settings.ANTHROPIC_API_KEY
            self._client = anthropic.Anthropic(api_key=api_key or None)
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
            r.custom_id for r in requests if manifest.get(r.custom_id) != _manifest_digest(r)
        ]
        if mismatched:
            raise ValueError(
                f"Batch state {state_path} was submitted for different requests: "
                f"{len(mismatched)} of {len(requests)} do not match its manifest "
                f"(first: {mismatched[0]!r}). Delete the state file to submit a new batch."
            )

    @staticmethod
    def _parse(message, schema, prepared: dict):
        """Validate one succeeded row's message the way the live call does."""
        mode, provider_name, handle_response_model, process_response = _instructor_tools()
        model = prepared.get(schema)
        if model is None:
            model, _ = handle_response_model(schema, mode=mode, provider=provider_name)
            prepared[schema] = model
        return process_response(
            message,
            response_model=model,
            stream=False,
            # Instructor's create(), which the live call goes through, defaults
            # to strict validation (no "1" for an int, no "true" for a bool).
            # process_response() defaults to lax, which would cache answers
            # the live path rejects under the live cache key.
            strict=True,
            mode=mode,
            provider=provider_name,
        )

    def retrieve(self, batch_id, requests):
        by_id = {r.custom_id: r.schema for r in requests}
        prepared: dict = {}
        out: dict = {}
        for row in self.client.messages.batches.results(batch_id):
            cid = row.custom_id
            if cid not in by_id:
                continue  # already-durable / stale id — resume case
            result = row.result
            if result.type != "succeeded":
                err = getattr(result, "error", None)
                # The SDK nests the API error one level down (``error.error``).
                inner = getattr(err, "error", None) or err
                out[cid] = BatchError(
                    custom_id=cid,
                    error_type=str(getattr(inner, "type", None) or result.type),
                    message=str(getattr(inner, "message", None) or result.type),
                )
                continue
            content = getattr(result.message, "content", None) or []
            if not any(getattr(block, "type", None) == "tool_use" for block in content):
                out[cid] = BatchError(cid, "no_tool_use", "no tool_use block in response")
                continue
            try:
                out[cid] = self._parse(result.message, by_id[cid], prepared)
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

    def ensure_submitted(self, requests, *, state_path=None) -> str:
        """The batch of *requests*: the one *state_path* records, else a new one.

        A new batch's id and manifest go to *state_path*, so a later call (or
        :meth:`run`) collects it instead of paying for it again.
        """
        state_path = Path(state_path) if state_path else None
        if state_path and state_path.exists():
            state = json.loads(state_path.read_text())
            batch_id = state.get("batch_id")
            if batch_id is not None:
                self._verify_manifest(state, requests, state_path)
                return batch_id
        batch_id = self.submit(requests)
        if state_path:
            manifest = {r.custom_id: _manifest_digest(r) for r in requests}
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
        return batch_id

    def run(self, requests, *, state_path=None, poll_interval=60, timeout_s=86400, progress=None):
        batch_id = self.ensure_submitted(requests, state_path=state_path)
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
            remaining = deadline - time.monotonic()
            if remaining < 0:
                raise TimeoutError(f"batch {batch_id} did not end within {timeout_s}s")
            if poll_interval:
                # No longer than the time left, so a short time box ends on time.
                time.sleep(min(poll_interval, remaining))
        return self.retrieve(batch_id, requests)
