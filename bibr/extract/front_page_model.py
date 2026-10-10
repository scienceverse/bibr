"""Page-level front-matter model (``ML_FRONT_PAGE_MODEL_MODE``).

A small vision-language model (a Qwen3.5 LoRA trained separately from
publisher JATS projected onto cached OCR regions) reads the first page or two
as a whole: the page image, plus every layout region with its id, label, box
and OCR text. It answers which regions belong to *the article this PDF is
about*, and how many articles start on the page, so a masthead, a journal
banner or the tail of the previous article cannot root the record.

This module is the contract shared with the trainer: the region listing, the
prompt, the response schema and the parser live here so the training targets
are rendered by exactly the code that reads the served model's answers.

The model is served behind any OpenAI-compatible ``/v1/chat/completions``
endpoint (vLLM with the LoRA, in practice) and consulted from the parse stage
while the raw OCR regions are still resident. Its prediction is evidence for
:func:`bibr.extract.front_matter.resolve_front_matter`, which decides what it
may override. Every failure is soft: an unreachable server, a timeout or a
malformed answer logs and leaves the heuristics alone.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

FRONT_PAGE_SCHEMA_VERSION = "front_page_v1"
FRONT_PAGE_MODES = ("off", "arbiter", "primary")
# Characters of OCR text shown per region. An abstract is identified by its
# opening, and the cap keeps a dense two-page prompt well inside 8k tokens.
MAX_REGION_CHARS = 320
# Non-text regions carry no text the record could own; they are listed by
# label only so the model still sees the page's furniture.
_NON_TEXT_LABELS = frozenset(
    {"image", "chart", "figure", "table", "seal", "formula", "header_image"}
)
_REGION_ID_RE = re.compile(r"^p(\d+)r(\d+)$")

SYSTEM_PROMPT = (
    "You read the first pages of a scholarly PDF. The user lists every layout region as "
    "`[id] label (x0,y0,x1,y1): text`, with boxes on a 0-1000 page grid, and may attach the "
    "page images. Several articles can start on one page, and journal mastheads, running "
    "heads, licence boxes and the end of a previous article are not part of any record. "
    "Identify the target article: the one this PDF is about, normally the article whose "
    "full text follows. Answer with JSON only."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["articles_on_page", "target_regions", "title", "authors", "affiliations", "doi"],
    "properties": {
        "articles_on_page": {"type": "integer", "minimum": 0, "maximum": 20},
        "target_regions": {"type": "array", "items": {"type": "string"}, "maxItems": 64},
        "title": {"type": ["string", "null"]},
        "authors": {
            "type": "array",
            "maxItems": 200,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "markers"],
                "properties": {
                    "name": {"type": "string"},
                    "markers": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                },
            },
        },
        "affiliations": {
            "type": "array",
            "maxItems": 100,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["marker", "text"],
                "properties": {
                    "marker": {"type": ["string", "null"]},
                    "text": {"type": "string"},
                },
            },
        },
        "doi": {"type": ["string", "null"]},
    },
}


def region_id(page: int, index: int) -> str:
    """``p<page>r<index>``: page is 1-based within the processed pages, as in
    ``RegionSummary.page`` and ``FrontRolePredictions``."""
    return f"p{int(page)}r{int(index)}"


def parse_region_id(value: str) -> tuple[int, int] | None:
    match = _REGION_ID_RE.match(value.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class PageRegion:
    """One OCR layout region as the page model sees it."""

    page: int
    index: int
    label: str
    bbox: tuple[int, int, int, int]
    text: str

    @property
    def key(self) -> tuple[int, int]:
        return self.page, self.index

    @property
    def region_id(self) -> str:
        return region_id(self.page, self.index)


def _attr(region: Any, name: str, default=None):
    if isinstance(region, Mapping):
        return region.get(name, region.get(f"_{name}", default))
    return getattr(region, name, default)


def _grid_box(bbox: Any) -> tuple[int, int, int, int]:
    if not bbox or len(bbox) != 4:
        return (0, 0, 0, 0)
    try:
        values = [round(float(v)) for v in bbox]
    except (TypeError, ValueError):
        return (0, 0, 0, 0)
    x0, y0, x1, y1 = (max(0, min(1000, v)) for v in values)
    return (x0, y0, x1, y1)


def page_regions_from_ocr(
    pages: Sequence[Sequence[Any]], *, max_pages: int = 2
) -> list[PageRegion]:
    """Reading-order regions of the first *max_pages* processed pages.

    ``pages`` is the OCR stage's per-page region list (typed regions or dicts
    with ``bbox_2d`` already on the 0-1000 grid). Page numbers are 1-based
    within the processed pages, the key ``RegionSummary`` carries.
    """
    out: list[PageRegion] = []
    for page_idx, regions in enumerate(pages[: max(0, max_pages)]):
        for pos, region in enumerate(regions):
            index = _attr(region, "index", pos)
            label = _attr(region, "label") or _attr(region, "native_label") or "text"
            out.append(
                PageRegion(
                    page=page_idx + 1,
                    index=int(index if index is not None else pos),
                    label=str(label),
                    bbox=_grid_box(_attr(region, "bbox_2d")),
                    text=str(_attr(region, "content") or ""),
                )
            )
    return out


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= MAX_REGION_CHARS:
        return flat
    return flat[: MAX_REGION_CHARS - 1].rstrip() + "…"


def render_region_listing(regions: Sequence[PageRegion]) -> str:
    lines: list[str] = []
    page: int | None = None
    for region in regions:
        if region.page != page:
            page = region.page
            lines.append(f"# page {page}")
        x0, y0, x1, y1 = region.bbox
        head = f"[{region.region_id}] {region.label} ({x0},{y0},{x1},{y1})"
        if region.label in _NON_TEXT_LABELS:
            lines.append(head)
        else:
            lines.append(f"{head}: {_clip(region.text)}")
    return "\n".join(lines)


def encode_page_image(image: Any, *, max_side: int) -> bytes:
    """PNG bytes of a PIL page image, downscaled so its long side is *max_side*."""
    img = image
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_messages(
    regions: Sequence[PageRegion], images: Sequence[bytes] | None = None
) -> list[dict[str, Any]]:
    """OpenAI chat messages for one paper (images are PNG bytes, page order)."""
    content: list[dict[str, Any]] = []
    for png in images or ():
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
            }
        )
    content.append({"type": "text", "text": render_region_listing(regions)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


@dataclass(frozen=True)
class PageAuthor:
    name: str
    markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageAffiliation:
    marker: str | None
    text: str


@dataclass(frozen=True)
class PageRecordPrediction:
    """The page model's answer for one paper."""

    target: frozenset[tuple[int, int]]
    articles_on_page: int
    title: str | None = None
    authors: tuple[PageAuthor, ...] = ()
    affiliations: tuple[PageAffiliation, ...] = ()
    doi: str | None = None
    model_version: str = FRONT_PAGE_SCHEMA_VERSION
    # Region ids the model named that the page does not have; any is a sign
    # the answer is unreliable and the resolver ignores the prediction.
    unknown_region_ids: tuple[str, ...] = ()


def render_target(
    *,
    target: Sequence[tuple[int, int]],
    articles_on_page: int,
    title: str | None,
    authors: Sequence[PageAuthor],
    affiliations: Sequence[PageAffiliation],
    doi: str | None,
) -> str:
    """The canonical JSON completion — what the trainer teaches the model to emit."""
    payload = {
        "articles_on_page": int(articles_on_page),
        "target_regions": [region_id(p, i) for p, i in sorted(set(target))],
        "title": title,
        "authors": [{"name": a.name, "markers": list(a.markers)} for a in authors],
        "affiliations": [{"marker": a.marker, "text": a.text} for a in affiliations],
        "doi": doi,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_prediction(
    text: str,
    *,
    known: frozenset[tuple[int, int]] | None = None,
    model_version: str = FRONT_PAGE_SCHEMA_VERSION,
) -> PageRecordPrediction | None:
    """Parse a completion; ``None`` when it is not the schema's JSON."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{") :] if "{" in raw else raw
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    regions = data.get("target_regions")
    count = data.get("articles_on_page")
    if not isinstance(regions, list) or not isinstance(count, int) or isinstance(count, bool):
        return None
    target: set[tuple[int, int]] = set()
    unknown: list[str] = []
    for item in regions:
        key = parse_region_id(item) if isinstance(item, str) else None
        if key is None or (known is not None and key not in known):
            unknown.append(str(item))
            continue
        target.add(key)
    authors: list[PageAuthor] = []
    for item in data.get("authors") or ():
        if isinstance(item, dict) and (name := _str_or_none(item.get("name"))):
            markers = tuple(str(m).strip() for m in item.get("markers") or () if str(m).strip())
            authors.append(PageAuthor(name=name, markers=markers))
    affiliations: list[PageAffiliation] = []
    for item in data.get("affiliations") or ():
        if isinstance(item, dict) and (aff := _str_or_none(item.get("text"))):
            affiliations.append(PageAffiliation(marker=_str_or_none(item.get("marker")), text=aff))
    return PageRecordPrediction(
        target=frozenset(target),
        articles_on_page=max(0, count),
        title=_str_or_none(data.get("title")),
        authors=tuple(authors),
        affiliations=tuple(affiliations),
        doi=_str_or_none(data.get("doi")),
        model_version=model_version,
        unknown_region_ids=tuple(unknown),
    )


class FrontPageModelClient:
    """Blocking client for the served page model; safe to share across threads."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout: float,
        max_pages: int,
        image_max_side: int,
        send_images: bool,
        api_key: str | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        import httpx

        from bibr.ocr.http_security import normalize_ocr_base_url, ocr_request_headers

        self._base_url = normalize_ocr_base_url(base_url)
        headers = ocr_request_headers(
            self._base_url, api_key, allow_insecure_http=allow_insecure_http
        )
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self.model = model
        self.max_pages = max_pages
        self.image_max_side = image_max_side
        self.send_images = send_images
        self.version = f"{FRONT_PAGE_SCHEMA_VERSION}:{model}"

    def predict(
        self, ocr_pages: Sequence[Sequence[Any]], page_images: Sequence[Any] | None = None
    ) -> PageRecordPrediction | None:
        regions = page_regions_from_ocr(ocr_pages, max_pages=self.max_pages)
        if not any(region.label not in _NON_TEXT_LABELS for region in regions):
            return None
        images = (
            [
                encode_page_image(image, max_side=self.image_max_side)
                for image in list(page_images)[: self.max_pages]
            ]
            if self.send_images and page_images
            else None
        )
        body = {
            "model": self.model,
            "messages": build_messages(regions, images),
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": FRONT_PAGE_SCHEMA_VERSION, "schema": RESPONSE_SCHEMA},
            },
        }
        response = self._client.post(f"{self._base_url}/v1/chat/completions", json=body)
        response.raise_for_status()
        choices = response.json().get("choices") or []
        text = (choices[0].get("message") or {}).get("content") if choices else None
        if not isinstance(text, str):
            return None
        prediction = parse_prediction(
            text,
            known=frozenset(region.key for region in regions),
            model_version=self.version,
        )
        if prediction is None:
            logger.warning("Front-page model returned an answer outside its schema; ignoring it")
        return prediction

    def close(self) -> None:
        self._client.close()


_CACHE: dict[tuple, FrontPageModelClient | None] = {}
_CACHE_LOCK = threading.Lock()


def front_page_mode(settings: GlobalSettings | None) -> str:
    if settings is None:
        return "off"
    mode = str(getattr(settings.ml, "front_page_model_mode", "off") or "off")
    return mode if mode in FRONT_PAGE_MODES else "off"


def load_front_page_model(settings: GlobalSettings) -> FrontPageModelClient | None:
    """The configured client, or ``None`` when the mode is off or no URL is set."""
    ml = settings.ml
    base_url = getattr(ml, "front_page_model_base_url", None)
    if front_page_mode(settings) == "off" or not base_url:
        return None
    api_key = getattr(ml, "front_page_model_api_key", None)
    if api_key is not None and hasattr(api_key, "get_secret_value"):
        api_key = api_key.get_secret_value()
    key = (
        str(base_url),
        str(ml.front_page_model_name),
        float(ml.front_page_model_timeout),
        int(ml.front_page_model_pages),
        int(ml.front_page_model_image_max_side),
        bool(ml.front_page_model_send_images),
        bool(api_key),
    )
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        try:
            client: FrontPageModelClient | None = FrontPageModelClient(
                str(base_url),
                model=str(ml.front_page_model_name),
                timeout=float(ml.front_page_model_timeout),
                max_pages=int(ml.front_page_model_pages),
                image_max_side=int(ml.front_page_model_image_max_side),
                send_images=bool(ml.front_page_model_send_images),
                api_key=api_key or None,
                allow_insecure_http=bool(getattr(settings.ocr, "allow_insecure_http", False)),
            )
        except Exception as exc:  # noqa: BLE001 - optional evidence, never fatal
            logger.warning("Front-page model unavailable (%s); using heuristics alone", exc)
            client = None
        _CACHE[key] = client
        return client


def predict_front_page(
    settings: GlobalSettings,
    ocr_pages: Sequence[Sequence[Any]],
    page_images: Sequence[Any] | None = None,
) -> PageRecordPrediction | None:
    """Run the page model for one paper; ``None`` on any failure."""
    client = load_front_page_model(settings)
    if client is None or not ocr_pages:
        return None
    try:
        return client.predict(ocr_pages, page_images)
    except Exception:  # noqa: BLE001 - optional evidence must never fail parsing
        logger.warning("Front-page model call failed; continuing without it", exc_info=True)
        return None


def reset_front_page_cache() -> None:
    """Test hook: drop cached clients."""
    with _CACHE_LOCK:
        for client in _CACHE.values():
            if client is not None:
                client.close()
        _CACHE.clear()


__all__ = [
    "FRONT_PAGE_MODES",
    "FRONT_PAGE_SCHEMA_VERSION",
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "FrontPageModelClient",
    "PageAffiliation",
    "PageAuthor",
    "PageRecordPrediction",
    "PageRegion",
    "build_messages",
    "encode_page_image",
    "front_page_mode",
    "load_front_page_model",
    "page_regions_from_ocr",
    "parse_prediction",
    "parse_region_id",
    "predict_front_page",
    "region_id",
    "render_region_listing",
    "render_target",
    "reset_front_page_cache",
]
