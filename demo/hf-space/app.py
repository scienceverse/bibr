"""bibr + metacheck demo for a Hugging Face ZeroGPU Space.

Upload a paper; bibr extracts its metadata and references, and metacheck
(pytacheck) runs a few quick checks on the result.

bibr's CPU models (layout, sentence segmenter, classifiers, reference NER)
run in this process through ONNX Runtime. The three GPU models are served
in-process by ``zerogpu_models`` and run only on ZeroGPU's GPU:

* OCR for tables, formulas and scanned pages: PaddleOCR-VL-1.6
* front-page metadata: bibr's fine-tuned Qwen3.5-4B student (``DEMO_LLM=local``),
  or a cloud API
* review of uncertain references: bibr's Qwen3.5-0.8B LoRA adapter

ZeroGPU's per-request token expires about a minute into a request, so GPU
calls made later in a paper fail. By default (``DEMO_GPU_MODE=paper``) the whole
paper runs in one ``@spaces.GPU`` window requested at the start, sized by its
page count. ``DEMO_GPU_MODE=batch`` keeps one GPU call per batch of requests.

Settings (Space variables; API keys are Space secrets):

DEMO_GPU_MODE       paper (default) | batch
DEMO_GPU_SECONDS    longest GPU window a paper may ask for (default 240)
DEMO_REQUIRE_LOGIN  1 (default): visitors sign in with Hugging Face first, since
                    an anonymous visitor's 2 GPU minutes a day can't fit a paper
DEMO_LLM            local (default) | anthropic | google | none
DEMO_EXTRACTOR      4b (default) | 2b, the student DEMO_LLM=local serves
DEMO_REF_REVIEW     1 (default) loads the reference adapter; 0 keeps NER only
DEMO_MAX_PAGES      pages read per paper (default 15), bounds GPU time
DEMO_MAX_FILE_SIZE_MB   upload limit (default 10)
CROSSREF_API_EMAIL  turns Crossref reference matching on
ANTHROPIC_API_KEY / GEMINI_API_KEY   for DEMO_LLM=anthropic / google
"""

from __future__ import annotations

# ``spaces`` must be imported before torch so ZeroGPU can patch CUDA.
try:
    import spaces  # noqa: F401
except ImportError:  # local runs without the package
    pass

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import gradio as gr

import zerogpu_models as zm

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bibr-space")
# At INFO, httpx logs ZeroGPU scheduler URLs, which carry the visitor's token.
for _name in ("httpx", "httpx2"):
    logging.getLogger(_name).setLevel(logging.WARNING)

HERE = Path(__file__).parent
DEMO_LLM = os.environ.get("DEMO_LLM", "local").lower()
REF_REVIEW = os.environ.get("DEMO_REF_REVIEW", "1") == "1"
MAX_PAGES = int(os.environ.get("DEMO_MAX_PAGES", "15"))
GPU_MODE = os.environ.get("DEMO_GPU_MODE", "paper").lower()
GPU_SECONDS_MAX = int(os.environ.get("DEMO_GPU_SECONDS", "240"))
# Window per paper: model loads and the front page, then OCR and reference
# review per page. Measured once (2026-10-29): a 7-page born-digital paper
# used 141 GPU s. ZeroGPU billed what was used, past the window asked, but
# the window is what it checks against the visitor's quota, so err high.
GPU_SECONDS_BASE = 60
GPU_SECONDS_PER_PAGE = 12
# Only on Spaces: Gradio's local OAuth mock needs a Hugging Face login.
REQUIRE_LOGIN = os.environ.get("DEMO_REQUIRE_LOGIN", "1") == "1" and bool(os.environ.get("SPACE_ID"))
MAX_FILE_MB = int(os.environ.get("DEMO_MAX_FILE_SIZE_MB", "10"))

# Pinned weights. The students' revisions are the ones bibr's vLLM registry pins
# (bibr#179): merged Qwen3.5 checkpoints fine-tuned on bibr's 0.7.0 prompts, the
# prompts this Space's bibr sends.
OCR_MODEL = ("PaddlePaddle/PaddleOCR-VL-1.6", os.environ.get("OCR_MODEL_REVISION"))
EXTRACTORS = {
    "4b": ("scienceverse/bibr-extract-qwen35-4b", "6ab47a4e4e8b1b76c1ab234aec66f360683ff9fc"),
    "2b": ("scienceverse/bibr-extract-qwen35-2b", "81d0442d0ecadc20e1b9587d54bff42242123870"),
}
EXTRACTOR = EXTRACTORS[os.environ.get("DEMO_EXTRACTOR", "4b").lower()]
#: Thinking off, as in the students' training and evals.
NO_THINKING = {"enable_thinking": False}
REF_BASE = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
REF_ALIAS = "bibr-ref-qwen35-08b-balanced"
REF_ADAPTER_DIR = Path(os.environ.get("REF_ADAPTER_DIR", HERE / "models" / "bibr-ref-qwen35-08b-balanced-v1"))

# ---------------------------------------------------------------------------
# bibr settings, set before bibr is imported.

os.environ.setdefault("ML_RUNTIME", "onnx")
os.environ.setdefault("FIGURE_IMAGES", "false")
# Generous timeouts (a request may wait in the ZeroGPU queue), and enough
# concurrency that OCR regions and flagged references reach the shim together
# and run as one GPU batch.
for key, value in {
    "OCR_REQUEST_TIMEOUT": "180",
    "OCR_MAX_CONCURRENT_REGIONS": "16",
    "OCR_CONCURRENT_REGIONS_PER_FILE": "16",
    "REF_LLM_MAX_CONCURRENCY": "32",
    "REF_LLM_TIMEOUT_SECONDS": "120",
    "REF_LLM_TOTAL_TIMEOUT_SECONDS": "240",
}.items():
    os.environ.setdefault(key, value)
if DEMO_LLM == "local":
    os.environ.update(
        LLM_BACKEND="cloud",
        LLM_PROVIDER="openai",
        LLM_BASE_URL=f"{zm.BASE_URL}/v1",
        LLM_MODEL=EXTRACTOR[0],
        LLM_CHAT_TEMPLATE_KWARGS=json.dumps(NO_THINKING),
    )
    os.environ.setdefault("LLM_TIMEOUT_SECONDS", "180")
elif DEMO_LLM == "anthropic":
    os.environ.setdefault("LLM_PROVIDER", "anthropic")
    os.environ.setdefault("LLM_MODEL", "claude-haiku-5-5")
elif DEMO_LLM == "google":
    os.environ.setdefault("LLM_PROVIDER", "google")
if REF_REVIEW:
    os.environ.update(
        REF_PARSE_STRATEGY="ner",
        REF_LLM_BASE_URL=f"{zm.BASE_URL}/v1",
        REF_LLM_MODEL=REF_ALIAS,
        REF_NER_LLM_FALLBACK="true",
    )
else:
    os.environ["REF_NER_LLM_FALLBACK"] = "false"

# ---------------------------------------------------------------------------
# GPU models, loaded at import (ZeroGPU emulates CUDA here; see zerogpu_models).


def _ensure_ref_adapter() -> Path:
    from bibr.local.ref_model import download

    # Downloads bibr's release asset once, and verifies its pinned SHA-256s.
    return download(REF_ADAPTER_DIR)


def _memory() -> str:
    """This process's resident memory and the container's limit, for the logs."""
    rss = limit = "?"
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = f"{int(line.split()[1]) / 2**20:.1f} GiB"
    except OSError:
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            limit = raw if raw == "max" else f"{int(raw) / 2**30:.1f} GiB"
            break
        except (OSError, ValueError):
            continue
    return f"RSS {rss}, limit {limit}"


def _loaded(name: str, make):
    logger.info("loading %s (%s)", name, _memory())
    started = time.monotonic()
    backend = make()
    logger.info("loaded %s in %.0fs (%s)", name, time.monotonic() - started, _memory())
    return backend


def load_backends() -> dict[str, zm.Backend]:
    backends: dict[str, zm.Backend] = {}
    ocr = _loaded(OCR_MODEL[0], lambda: zm.PaddleOcrBackend(*OCR_MODEL))
    backends[ocr.alias] = ocr
    if DEMO_LLM == "local":
        backends[EXTRACTOR[0]] = _loaded(EXTRACTOR[0], lambda: zm.ChatLlmBackend(
            EXTRACTOR[0],
            EXTRACTOR[0],
            revision=EXTRACTOR[1],
            max_batch=4,
            default_max_tokens=4096,
            template_defaults=NO_THINKING,
        ))
    if REF_REVIEW:
        adapter_dir = str(_ensure_ref_adapter())
        backends[REF_ALIAS] = _loaded(REF_ALIAS, lambda: zm.ChatLlmBackend(
            REF_ALIAS,
            REF_BASE[0],
            revision=REF_BASE[1],
            adapter_dir=adapter_dir,
            max_batch=16,
            default_max_tokens=512,
            duration_per_item=1,
            duration_cap=45,
            template_defaults=NO_THINKING,
        ))
    return backends


if os.environ.get("DEMO_FAKE_MODELS") != "1":  # tests install their own backends
    zm.BACKENDS.update(load_backends())
    # In paper mode the shim runs inside the paper's GPU window, so it calls
    # the models directly instead of asking ZeroGPU for a GPU per batch.
    runner = zm.generate_batch if GPU_MODE == "paper" else None
    zm.install(zm.build_app(zm.BACKENDS, runner=runner))

# bibr's CPU stages stay off CUDA. ZeroGPU reports CUDA as available in the main
# process, and bibr's device probes (get_device_capability, mem_get_info) would
# initialise it for real; after that every @spaces.GPU worker, forked from this
# process, fails with "Cannot re-initialize CUDA in forked subprocess". In paper
# mode the probes also run inside the forked worker, where the forks inherit
# this override and keep bibr's ONNX/torch classifiers on the CPU.
import bibr.pipeline.classifier_resources as _bibr_classifiers  # noqa: E402
import bibr.utils.device as _bibr_device  # noqa: E402

_bibr_device.cuda_incompatibility = lambda: "the GPU is only attached inside @spaces.GPU calls"
_bibr_classifiers._cuda_memory_info = lambda: (False, None, None)

# ---------------------------------------------------------------------------
# The pipeline (CPU stages in this process; GPU stages via the shim).

from bibr.demo.local_app import (  # noqa: E402 - after the settings above
    _build_authors_data,
    _build_references_data,
    _build_summary_md,
    _GradioProgress,
    _parse_json_response,
)
from bibr.local.pipeline import LocalPipeline  # noqa: E402

_pipeline: LocalPipeline | None = None
_lock = asyncio.Lock()


def get_pipeline() -> LocalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LocalPipeline(
            ocr_backend="paddle-http",
            ocr_url=zm.BASE_URL,
            crossref=bool(os.environ.get("CROSSREF_API_EMAIL")),
            end_page=MAX_PAGES,
            no_llm=DEMO_LLM == "none",
        )
    return _pipeline


# metacheck modules that run offline, without an LLM, in about a second.
CHECKS = [
    "stat_check",
    "marginal",
    "stat_p_exact",
    "stat_effect_size",
    "coi_check",
    "funding_check",
    "ethics_check",
    "ref_consistency",
]
_LIGHTS = {"green": "🟢", "yellow": "🟡", "red": "🔴", "na": "⚪", "info": "🔵", "fail": "⚫"}


def run_checks(json_path: str) -> str:
    try:
        import pytacheck as pc
    except ImportError:
        return "*metacheck is not installed in this Space.*"
    paper = pc.read(json_path)
    rows = ["| | Check | Result |", "|---|---|---|"]
    for name in CHECKS:
        try:
            out = pc.module_run(paper, name)
            light = _LIGHTS.get(str(out.traffic_light), "⚪")
            title = getattr(out, "title", None) or name
            summary = (out.summary_text or "").replace("\n", " ").replace("|", "\\|")
        except Exception as exc:  # noqa: BLE001 - one failing check must not hide the rest
            logger.warning("check %s failed: %s", name, exc)
            light, title, summary = "⚫", name, f"Could not run ({type(exc).__name__})."
        rows.append(f"| {light} | {title} | {summary} |")
    return "\n".join(rows)


def _gpu_totals(stats=None) -> tuple[float, float, float]:
    stats = (zm.STATS if stats is None else stats).values()
    return (
        sum(s["seconds"] for s in stats),
        sum(s.get("wall", 0.0) for s in stats),
        sum(s["batches"] for s in stats),
    )


def _page_count(path: str) -> int:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(path)
    except Exception:  # noqa: BLE001 - not a PDF pypdfium2 opens; bibr reports why
        return MAX_PAGES
    try:
        return len(pdf)
    finally:
        pdf.close()


def paper_seconds(file_path: str) -> int:
    """The GPU window to ask ZeroGPU for: the quota a visitor needs left."""
    pages = min(_page_count(file_path), MAX_PAGES)
    return min(GPU_SECONDS_MAX, GPU_SECONDS_BASE + GPU_SECONDS_PER_PAGE * pages)


_paper_loop: asyncio.AbstractEventLoop | None = None


@zm.gpu(duration=paper_seconds)
def process_on_gpu(file_path: str) -> tuple[dict, dict]:
    """Run the whole paper in one GPU window (``DEMO_GPU_MODE=paper``).

    Runs in ZeroGPU's forked GPU worker, so the model stats come back with the
    result. The pipeline and its event loop are kept in case ZeroGPU reuses
    the worker for the next paper.
    """
    global _paper_loop
    if _paper_loop is None or _paper_loop.is_closed():
        _paper_loop = asyncio.new_event_loop()
    zm.STATS.clear()
    paper_json = _paper_loop.run_until_complete(get_pipeline().process_file(file_path))
    return paper_json, dict(zm.STATS)


async def chew(file_path, oauth_token: gr.OAuthToken | None = None, progress=gr.Progress()):
    # Gradio fills oauth_token from the LoginButton's session, or from an API
    # caller's Client(..., oauth_token=...). Hugging Face's proxy strips the
    # Authorization header, so a plain API token does not reach the app.
    if REQUIRE_LOGIN and oauth_token is None:
        raise gr.Error(
            "Sign in with Hugging Face first (button at the top). A paper needs more GPU "
            "time than ZeroGPU gives anonymous visitors."
        )
    if not file_path:
        raise gr.Error("Upload a PDF first.")
    if os.path.getsize(file_path) > MAX_FILE_MB * 1024 * 1024:
        raise gr.Error(f"Files up to {MAX_FILE_MB} MB, please.")
    progress(0.0, desc="Reading the paper...")
    async with _lock:
        if GPU_MODE == "paper":
            window = paper_seconds(file_path)
            progress(0.1, desc=f"Running on the GPU (up to {window} s of your quota)...")
            # to_thread copies this request's context, which ZeroGPU reads to
            # bill the visitor.
            called = time.monotonic()
            paper_json, stats = await asyncio.to_thread(process_on_gpu, file_path)
            gpu_s, _, gpu_calls = _gpu_totals(stats)
            gpu_wall = time.monotonic() - called
            logger.info("paper: %d s window asked, %.0f s used, %.1f s generating", window, gpu_wall, gpu_s)
        else:
            before = _gpu_totals()
            paper_json = await get_pipeline().process_file(file_path, progress=_GradioProgress(progress))
            gpu_s, gpu_wall, gpu_calls = (a - b for a, b in zip(_gpu_totals(), before))
            logger.info("paper used %.1f GPU s generating, %.1f s in %d GPU calls", gpu_s, gpu_wall, gpu_calls)
    result = _parse_json_response(paper_json)
    out = Path(tempfile.mkdtemp()) / (Path(file_path).stem + ".bibr.json")
    out.write_text(json.dumps(paper_json, ensure_ascii=False, indent=2))
    progress(0.95, desc="Running metacheck...")
    checks_md = await asyncio.to_thread(run_checks, str(out))
    if GPU_MODE == "paper":
        gpu_line = f"\n\n*GPU: {gpu_wall:.0f} s, {gpu_s:.0f} s of it generating.*"
    else:
        gpu_line = f"\n\n*GPU: {gpu_wall:.0f} s in {gpu_calls:.0f} calls.*" if gpu_calls else ""
    return (
        _build_summary_md(result) + gpu_line,
        _build_authors_data(result),
        _build_references_data(result),
        checks_md,
        gr.DownloadButton(value=str(out), visible=True),
    )


INTRO = f"""\
# bibr 🦫 + metacheck

Upload a scientific paper (PDF). [bibr](https://bibr.org) extracts its metadata and
references, and [metacheck](https://github.com/scienceverse/pytacheck) runs a few
quick checks on the result. The first {MAX_PAGES} pages are read.

Sign in with Hugging Face to run a paper. Each paper uses up to
{GPU_SECONDS_MAX} s of your ZeroGPU quota (free accounts get 5 minutes a day, PRO 40),
so a free account runs about two short papers a day. Papers are deleted from the Space
after processing.
"""

with gr.Blocks(title="bibr demo", delete_cache=(600, 3600)) as demo:
    gr.Markdown(INTRO)
    if REQUIRE_LOGIN:
        gr.LoginButton()
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Paper (PDF)", file_types=[".pdf"], type="filepath")
            run = gr.Button("Extract", variant="primary")
            download = gr.DownloadButton("Download bibr JSON", visible=False)
            examples = sorted(str(p) for p in (HERE / "examples").glob("*.pdf"))
            if examples:
                gr.Examples(examples=examples, inputs=file_input)
        with gr.Column(scale=2):
            summary = gr.Markdown("*Results appear here.*")
    with gr.Tabs():
        with gr.Tab("References"):
            refs = gr.Dataframe(
                headers=["#", "Authors", "Title", "Year", "Container", "DOI", "Type", "Match", "Score"],
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Authors"):
            authors = gr.Dataframe(
                headers=["Given Name", "Family Name", "Affiliation", "ORCID", "Email", "Corresponding"],
                interactive=False,
                wrap=True,
            )
        with gr.Tab("metacheck"):
            checks = gr.Markdown()
    run.click(chew, inputs=file_input, outputs=[summary, authors, refs, checks, download])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch(max_file_size=f"{MAX_FILE_MB}mb")
