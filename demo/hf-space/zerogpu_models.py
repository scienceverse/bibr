"""In-process OpenAI-compatible model endpoints for bibr on a ZeroGPU Space.

bibr reaches its OCR model, metadata LLM and reference parser over
OpenAI-compatible HTTP, normally served by vLLM. A ZeroGPU Space has no
long-running GPU process: CUDA exists only inside a ``@spaces.GPU`` call. So
this module answers those requests in the Space's own process:

* every ``httpx.AsyncClient`` gets a mount that sends ``http://127.0.0.1:PORT``
  to a small ASGI app instead of the network (bibr's OCR client, its ref-llm
  client and the OpenAI SDK all use httpx);
* the app queues requests per model and runs each queue as a batch through a
  ``@spaces.GPU`` function that calls ``transformers`` ``generate``.

``httpx.ASGITransport`` awaits the app in the caller's task, so each request
keeps the context of the Gradio event that started the paper. ZeroGPU reads
the visitor from that context, so a paper is billed to the visitor's quota,
not to the Space.

Structured outputs (``response_format: json_schema``) use XGrammar, the
backend bibr pins for vLLM, through its transformers logits processor.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("ZEROGPU_SHIM_PORT", "18765"))
BASE_URL = f"http://127.0.0.1:{PORT}"

try:  # ``spaces`` is a no-op outside ZeroGPU, but keep local runs import-light.
    import spaces
except ImportError:  # pragma: no cover - local runs without the package
    spaces = None


def gpu(duration: Callable[..., int] | int):
    """``spaces.GPU`` when available, otherwise the function unchanged."""

    def wrap(fn):
        if spaces is None:
            return fn
        return spaces.GPU(duration=duration)(fn)

    return wrap


# --------------------------------------------------------------------------
# Request and response shapes


@dataclass
class ChatRequest:
    """One OpenAI chat-completions request, reduced to what the models need."""

    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float = 0.0
    repetition_penalty: float | None = None
    json_schema: dict | None = None
    chat_template_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_body(cls, body: dict, default_max_tokens: int) -> ChatRequest:
        schema = None
        fmt = body.get("response_format") or {}
        if fmt.get("type") == "json_schema":
            schema = (fmt.get("json_schema") or {}).get("schema")
        elif fmt.get("type") == "json_object":
            schema = {"type": "object"}
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        return cls(
            messages=body.get("messages") or [],
            max_tokens=int(max_tokens or default_max_tokens),
            temperature=float(body.get("temperature") or 0.0),
            repetition_penalty=body.get("repetition_penalty"),
            json_schema=schema,
            chat_template_kwargs=dict(body.get("chat_template_kwargs") or {}),
        )


@dataclass
class Completion:
    text: str
    finish_reason: str  # "stop" or "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0


# --------------------------------------------------------------------------
# Backends. Each one loads its weights at import time (ZeroGPU emulates CUDA
# outside @spaces.GPU, and placing models on "cuda" at module level is the
# documented way to load them) and generates a whole batch per call.


class Backend:
    """A served model alias. Subclasses implement ``generate``."""

    alias: str
    max_batch: int = 8
    default_max_tokens: int = 1024
    #: GPU seconds requested per batch: ``base + per_item * len(batch)``.
    duration_base: int = 5
    duration_per_item: int = 4
    duration_cap: int = 60

    def duration_for(self, batch: Sequence[ChatRequest]) -> int:
        return min(self.duration_cap, self.duration_base + self.duration_per_item * len(batch))

    def generate(self, batch: list[ChatRequest]) -> list[Completion]:
        raise NotImplementedError


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    """bf16 everywhere by default; ``ZEROGPU_SHIM_DTYPE=float32`` for CPUs without bf16."""
    import torch

    return getattr(torch, os.environ.get("ZEROGPU_SHIM_DTYPE", "bfloat16"))


def _decode_image(url: str):
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError("only data: image URLs are accepted")
    payload = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _hf_messages(messages: list[dict]) -> tuple[list[dict], list]:
    """OpenAI messages -> transformers chat messages, plus the images in order."""
    out, images = [], []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": message["role"], "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for part in content or []:
            if part.get("type") == "image_url":
                image = _decode_image(part["image_url"]["url"])
                images.append(image)
                parts.append({"type": "image", "image": image})
            elif part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
        out.append({"role": message["role"], "content": parts})
    return out, images


def _text_messages(messages: list[dict]) -> list[dict]:
    """OpenAI messages -> plain-text chat messages for a text-only template."""
    out = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content or [] if p.get("type") == "text")
        out.append({"role": message["role"], "content": content})
    return out


class _HFGenerator(Backend):
    """Shared batched ``generate`` over a transformers model and processor."""

    model: Any
    processor: Any  # an AutoProcessor or a tokenizer
    eos_ids: set[int]

    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _encode(self, batch: list[ChatRequest]):
        raise NotImplementedError

    def _logits_processors(self, batch: list[ChatRequest]):
        if not any(r.json_schema for r in batch):
            return None
        return _json_logits_processor(self._tokenizer(), batch, self.model.config)

    def generate(self, batch: list[ChatRequest]) -> list[Completion]:
        import torch

        inputs = self._encode(batch).to(self.model.device)
        for key, value in inputs.items():  # pixel values follow the weights' dtype
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        max_new = max(r.max_tokens for r in batch)
        tokenizer = self._tokenizer()
        # Qwen3.5 checkpoints set generation_config.eos_token_id to
        # <|endoftext|>, but chat turns end with <|im_end|>: stop on both, or
        # every row runs to max_new_tokens.
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": False,
            "eos_token_id": sorted(self.eos_ids),
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else min(self.eos_ids),
        }
        penalties = {r.repetition_penalty for r in batch if r.repetition_penalty}
        if penalties:
            kwargs["repetition_penalty"] = max(penalties)
        processors = self._logits_processors(batch)
        if PROGRESS_EVERY:
            from transformers import LogitsProcessorList

            processors = LogitsProcessorList([*(processors or []), _Progress(self.alias, PROGRESS_EVERY)])
        if processors is not None:
            kwargs["logits_processor"] = processors
        logger.info(
            "%s: batch of %d, prompt tokens %s, max_tokens %s",
            self.alias,
            len(batch),
            inputs["attention_mask"].sum(dim=1).tolist(),
            [r.max_tokens for r in batch],
        )
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for i, (row, request) in enumerate(zip(output, batch, strict=True)):
            new = row[prompt_len:].tolist()
            # Trim at the first EOS / pad, then cap at this request's own budget.
            cut = next((i for i, t in enumerate(new) if t in self.eos_ids), None)
            stopped = cut is not None and cut < request.max_tokens
            new = new[: cut if cut is not None else len(new)][: request.max_tokens]
            results.append(
                Completion(
                    text=tokenizer.decode(new, skip_special_tokens=True),
                    finish_reason="stop" if stopped else "length",
                    prompt_tokens=int(inputs["attention_mask"][i].sum()),
                    completion_tokens=len(new),
                )
            )
        return results

    def _collect_eos(self) -> set[int]:
        ids: set[int] = set()
        tokenizer = self._tokenizer()
        for source in (self.model.generation_config.eos_token_id, tokenizer.eos_token_id):
            if isinstance(source, int):
                ids.add(source)
            elif source:
                ids.update(source)
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
                ids.add(token_id)
        if tokenizer.pad_token_id is not None:
            ids.add(tokenizer.pad_token_id)
        return ids


class PaddleOcrBackend(_HFGenerator):
    """PaddleOCR-VL-1.6 element recognition (bibr's ``paddle-http`` contract)."""

    alias = "paddle-ocr-vl-1.6"
    max_batch = 16
    default_max_tokens = 2048
    duration_per_item = 2

    def __init__(self, model_id: str = "PaddlePaddle/PaddleOCR-VL-1.6", revision: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self._tokenizer().padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=_dtype()
        ).to(_device())
        self.model.eval()
        self.eos_ids = self._collect_eos()

    def _encode(self, batch):
        conversations = [_hf_messages(r.messages)[0] for r in batch]
        return self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )


class ChatLlmBackend(_HFGenerator):
    """A text chat model, optionally with a LoRA adapter merged in."""

    def __init__(
        self,
        alias: str,
        model_id: str,
        revision: str | None = None,
        adapter_dir: str | None = None,
        max_batch: int = 4,
        default_max_tokens: int = 2048,
        duration_per_item: int = 10,
        duration_cap: int = 90,
        template_defaults: dict | None = None,
    ):
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.alias = alias
        # Chat-template kwargs a request may override, e.g. thinking off.
        self.template_defaults = dict(template_defaults or {})
        self.max_batch = max_batch
        self.default_max_tokens = default_max_tokens
        self.duration_per_item = duration_per_item
        self.duration_cap = duration_cap
        # The adapter bundle ships the tokenizer and chat template it was trained with.
        self.processor = AutoTokenizer.from_pretrained(
            adapter_dir or model_id, revision=None if adapter_dir else revision
        )
        self.processor.padding_side = "left"
        # Qwen3.5 checkpoints are conditional-generation models; bibr's adapter
        # keys target ``model.language_model.layers``, which this class loads.
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, revision=revision, dtype=_dtype()
            )
        except ValueError:  # a text-only checkpoint
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=_dtype())
        if adapter_dir:
            from peft import PeftModel

            # PEFT loads onto CUDA when torch says it is available, which ZeroGPU
            # fakes outside @spaces.GPU, so load and merge the adapter on the CPU.
            model = PeftModel.from_pretrained(model, adapter_dir, torch_device="cpu").merge_and_unload()
        self.model = model.to(_device())
        self.model.eval()
        self.eos_ids = self._collect_eos()

    def _encode(self, batch):
        texts = [
            self.processor.apply_chat_template(
                _text_messages(r.messages),
                tokenize=False,
                add_generation_prompt=True,
                **{**self.template_defaults, **r.chat_template_kwargs},
            )
            for r in batch
        ]
        return self.processor(texts, return_tensors="pt", padding=True, add_special_tokens=False)


#: Debugging slow (CPU) runs: log every N decode steps of a batch.
PROGRESS_EVERY = int(os.environ.get("ZEROGPU_SHIM_PROGRESS_EVERY", "0"))
#: Debugging slow (CPU) runs: cap every backend's batch size.
MAX_BATCH_CAP = int(os.environ.get("ZEROGPU_SHIM_MAX_BATCH", "0"))


class _Progress:
    def __init__(self, alias: str, every: int):
        self.alias, self.every, self.step, self.started = alias, every, 0, time.monotonic()

    def __call__(self, input_ids, scores):
        if self.step % self.every == 0:
            logger.info("%s: decode step %d, %.1fs", self.alias, self.step, time.monotonic() - self.started)
        self.step += 1
        return scores


def _json_logits_processor(tokenizer, batch: list[ChatRequest], config):
    """One XGrammar matcher per row: its schema, or any JSON when it has none."""
    import xgrammar as xgr
    from transformers import LogitsProcessorList

    vocab_size = getattr(getattr(config, "text_config", config), "vocab_size", None)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = _compiler_cache.setdefault(id(tokenizer), xgr.GrammarCompiler(info))
    grammars = [
        compiler.compile_json_schema(json.dumps(r.json_schema))
        if r.json_schema
        else compiler.compile_builtin_json_grammar()
        for r in batch
    ]
    return LogitsProcessorList([xgr.contrib.hf.LogitsProcessor(grammars)])


_compiler_cache: dict[int, Any] = {}


# --------------------------------------------------------------------------
# Batching. Requests for one alias that arrive within ``window`` seconds go to
# the GPU together; bibr sends OCR regions and flagged references
# concurrently, and every @spaces.GPU call has a scheduling cost.


class Batcher:
    def __init__(self, backend: Backend, run: Callable[[str, list[ChatRequest]], list[Completion]], window: float = 0.05):
        self.backend = backend
        self._run = run
        self._window = window
        self._pending: list[tuple[ChatRequest, asyncio.Future]] = []
        self._draining = False

    async def submit(self, request: ChatRequest) -> Completion:
        future = asyncio.get_running_loop().create_future()
        self._pending.append((request, future))
        if not self._draining:
            self._draining = True
            # create_task copies this request's context: the drain runs, and
            # ZeroGPU bills, as the visitor whose paper sent the request.
            asyncio.get_running_loop().create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        try:
            await asyncio.sleep(self._window)
            while self._pending:
                # Constrained and free-text requests go in separate batches:
                # the logits processor would force JSON on a free-text row.
                constrained = self._pending[0][0].json_schema is not None
                batch = [p for p in self._pending if (p[0].json_schema is not None) == constrained]
                batch = batch[: min(self.backend.max_batch, MAX_BATCH_CAP or self.backend.max_batch)]
                taken = {id(p) for p in batch}
                self._pending = [p for p in self._pending if id(p) not in taken]
                live = [(r, f) for r, f in batch if not f.done()]
                if not live:
                    continue
                try:
                    called = time.monotonic()
                    results = await asyncio.to_thread(
                        self._run, self.backend.alias, [r for r, _ in live]
                    )
                    wall = time.monotonic() - called
                    elapsed = 0.0
                    if isinstance(results, tuple):
                        results, elapsed = results
                    stats = STATS.setdefault(
                        self.backend.alias,
                        {"batches": 0, "requests": 0, "seconds": 0.0, "wall": 0.0},
                    )
                    stats["batches"] += 1
                    stats["requests"] += len(live)
                    stats["seconds"] += elapsed
                    # GPU call as the caller saw it: queueing and attach included.
                    stats["wall"] += wall
                except Exception as exc:  # noqa: BLE001 - surfaced per request
                    for _, f in live:
                        if not f.done():
                            f.set_exception(exc)
                    continue
                for (_, f), result in zip(live, results, strict=True):
                    if not f.done():
                        f.set_result(result)
                await asyncio.sleep(0)  # let late arrivals join the next batch
        finally:
            self._draining = False


# --------------------------------------------------------------------------
# The GPU entry point and the ASGI app.

BACKENDS: dict[str, Backend] = {}
#: Per alias: GPU batches, requests, seconds spent generating, and wall seconds
#: of the GPU calls including ZeroGPU's queueing and attach (this process).
STATS: dict[str, dict[str, float]] = {}


def _gpu_duration(alias: str, batch: list[ChatRequest]) -> int:
    return BACKENDS[alias].duration_for(batch)


@gpu(duration=_gpu_duration)
def run_batch(alias: str, batch: list[ChatRequest]) -> tuple[list[Completion], float]:
    """Generate one batch on the GPU. Runs in ZeroGPU's forked GPU worker."""
    started = time.monotonic()
    results = BACKENDS[alias].generate(batch)
    elapsed = time.monotonic() - started
    logger.info(
        "%s: %d request(s) in %.1fs, %d tokens out, finish %s",
        alias,
        len(batch),
        elapsed,
        sum(r.completion_tokens for r in results),
        [r.finish_reason for r in results],
    )
    return results, elapsed


def build_app(
    backends: dict[str, Backend],
    runner: Callable[[str, list[ChatRequest]], list[Completion]] | None = None,
):
    """The OpenAI-compatible ASGI app serving *backends* by alias."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    batchers = {alias: Batcher(b, runner or run_batch) for alias, b in backends.items()}

    async def models(_: Request):
        return JSONResponse(
            {"object": "list", "data": [{"id": a, "object": "model", "owned_by": "bibr-space"} for a in backends]}
        )

    async def chat(request: Request):
        body = await request.json()
        alias = body.get("model")
        if alias not in batchers:
            return JSONResponse({"error": {"message": f"unknown model {alias!r}"}}, status_code=404)
        backend = backends[alias]
        result = await batchers[alias].submit(ChatRequest.from_body(body, backend.default_max_tokens))
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": alias,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }
        )

    async def health(_: Request):
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/health", health),
        ]
    )


# bibr's own clients use httpx; recent OpenAI SDKs use its fork httpx2. Both
# get the same mount.
_HTTP_LIBS = [httpx]
try:
    import httpx2

    _HTTP_LIBS.append(httpx2)
except ImportError:
    pass

_app_holder: dict[str, Any] = {"app": None}


def _make_transport(lib):
    class AppTransport(lib.AsyncBaseTransport):
        """An ASGI transport whose app can be replaced after clients are built."""

        async def handle_async_request(self, request):
            app = _app_holder["app"]
            if app is None:
                raise lib.ConnectError("zerogpu_models.install() has not been called", request=request)
            # ASGITransport ignores httpx timeouts; apply the read timeout here
            # (it includes time queued for the GPU, as with a real server).
            read = (request.extensions.get("timeout") or {}).get("read")
            call = lib.ASGITransport(app=app).handle_async_request(request)
            if read is None:
                return await call
            try:
                return await asyncio.wait_for(call, read)
            except TimeoutError as exc:
                raise lib.ReadTimeout("model shim timed out", request=request) from exc

    return AppTransport()


_patched: set[str] = set()


def install(app) -> None:
    """Route requests for BASE_URL from every new async httpx client to *app*.

    Calling it again swaps the app; clients already built follow the swap.
    """
    _app_holder["app"] = app
    for lib in _HTTP_LIBS:
        if lib.__name__ in _patched:
            continue
        transport = _make_transport(lib)
        original_init = lib.AsyncClient.__init__

        def patched_init(self, *args, _orig=original_init, _t=transport, **kwargs):
            mounts = dict(kwargs.pop("mounts", None) or {})
            mounts.setdefault(f"all://127.0.0.1:{PORT}", _t)
            _orig(self, *args, mounts=mounts, **kwargs)

        lib.AsyncClient.__init__ = patched_init
        _patched.add(lib.__name__)
