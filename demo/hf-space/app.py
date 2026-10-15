"""bibr + metacheck demo for a Hugging Face ZeroGPU Space.

Upload a paper; bibr extracts its metadata and references, and metacheck
(pytacheck) runs a few quick checks on the result.

bibr's CPU models (layout, sentence segmenter, classifiers, reference NER)
run in this process through ONNX Runtime. The three GPU models are served
in-process by ``zerogpu_models`` and run only inside ``@spaces.GPU`` batches:

* OCR for tables, formulas and scanned pages: PaddleOCR-VL-1.6
* front-page metadata: bibr's fine-tuned Qwen3.5-4B student (``DEMO_LLM=local``),
  or a cloud API
* review of uncertain references: bibr's Qwen3.5-0.8B LoRA adapter

Settings (Space variables; API keys are Space secrets):

DEMO_LLM            local (default) | anthropic | google | none
DEMO_EXTRACTOR      4b (default) | 2b, the student DEMO_LLM=local serves
DEMO_REF_REVIEW     1 (default) loads the reference adapter; 0 keeps NER only
DEMO_MAX_PAGES      pages read per paper (default 30), bounds GPU time
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
MAX_PAGES = int(os.environ.get("DEMO_MAX_PAGES", "30"))
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
    import time

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
    zm.install(zm.build_app(zm.BACKENDS))

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


def _gpu_totals() -> tuple[float, float, float]:
    stats = zm.STATS.values()
    return (
        sum(s["seconds"] for s in stats),
        sum(s.get("wall", 0.0) for s in stats),
        sum(s["batches"] for s in stats),
    )


async def chew(file_path, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Upload a PDF first.")
    if os.path.getsize(file_path) > MAX_FILE_MB * 1024 * 1024:
        raise gr.Error(f"Files up to {MAX_FILE_MB} MB, please.")
    progress(0.0, desc="Reading the paper...")
    async with _lock:
        before = _gpu_totals()
        paper_json = await get_pipeline().process_file(file_path, progress=_GradioProgress(progress))
        gpu_s, gpu_wall, gpu_calls = (a - b for a, b in zip(_gpu_totals(), before))
    logger.info("paper used %.1f GPU s generating, %.1f s in %d GPU calls", gpu_s, gpu_wall, gpu_calls)
    result = _parse_json_response(paper_json)
    out = Path(tempfile.mkdtemp()) / (Path(file_path).stem + ".bibr.json")
    out.write_text(json.dumps(paper_json, ensure_ascii=False, indent=2))
    progress(0.95, desc="Running metacheck...")
    checks_md = await asyncio.to_thread(run_checks, str(out))
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

GPU time comes from your Hugging Face ZeroGPU quota; signing in gives you more.
Papers are deleted from the Space after processing.
"""

with gr.Blocks(title="bibr demo", delete_cache=(600, 3600)) as demo:
    gr.Markdown(INTRO)
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
