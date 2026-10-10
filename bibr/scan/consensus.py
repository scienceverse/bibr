"""Two-recognizer consensus on scanned pages, with escalation on disagreement.

bibr ships two recognizers, PaddleOCR-VL and GLM-OCR. On a scanned page every
text region read by the primary recognizer is read again by a second one, and
the normalized edit distance between the two readings is the region's
disagreement (the agreement signal of Consensus Entropy, CVPR 2026: where
independent recognizers disagree, at least one of them is wrong). The regions
that disagree most are escalated:

- With an escalation recognizer configured (a stronger model, or one strong
  on rare scripts), it reads the region too, and the reading closest to the
  other two (the medoid) is kept.
- Without one, the primary reading is kept, unless it is empty and the second
  one is not, and the region is flagged.

Every compared region carries its score in ``_ocr_consensus`` so evaluations
can calibrate the threshold. Failures of the second or escalation recognizer
never fail the paper: the primary reading stands.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rapidfuzz.distance import Levenshtein

from bibr.ocr.normalization import normalize_ocr_output

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.ocr.backend import OcrBackend
    from bibr.ocr.profiles import OcrProfile

logger = logging.getLogger(__name__)

# The OCR backends a consensus or escalation recognizer may use: clients of a
# server someone else runs, or a cloud vision API. A managed local runtime
# would start a second engine next to the primary one on the same GPU.
CONSENSUS_BACKENDS = frozenset({"glm-http", "paddle-http", "gemini", "openai", "anthropic"})
_CLOUD_BACKENDS = frozenset({"gemini", "openai", "anthropic"})

# Layout labels read as plain text and compared. Tables and formulas are left
# out: the two families emit different markup (OTSL vs HTML, LaTeX spacing),
# so their edit distance measures format, not reading.
COMPARED_LABELS = frozenset(
    {
        "abstract",
        "content",
        "doc_title",
        "figure_title",
        "footnote",
        "paragraph_title",
        "reference",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
    }
)

# Characters one family emits as markup and the other does not.
_MARKUP_RE = re.compile(r"[#*_`\x00-\x08\x0b-\x1f]")
_SPACE_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")


def comparable(text: str) -> str:
    """The text as the two recognizers' readings are compared: NFKC, no markup, one space."""
    text = unicodedata.normalize("NFKC", text)
    text = _MARKUP_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return _SPACE_RE.sub(" ", text).strip()


def disagreement(a: str, b: str) -> float:
    """Normalized edit distance between two readings, 0 (same) to 1 (nothing shared)."""
    left, right = comparable(a), comparable(b)
    if not left and not right:
        return 0.0
    return float(Levenshtein.normalized_distance(left, right))


def medoid(readings: Sequence[tuple[str, str]]) -> str:
    """The source of the reading closest to all others; the last one wins ties.

    ``readings`` is ``(source, text)`` in order of increasing trust, so the
    escalation recognizer, listed last, decides a three-way tie.
    """
    best_source, best_cost = readings[-1][0], math.inf
    for source, text in reversed(readings):
        cost = sum(disagreement(text, other) for _, other in readings)
        if cost < best_cost:
            best_source, best_cost = source, cost
    return best_source


@dataclass
class Recognizer:
    """An OCR backend with the profile its prompts and output follow."""

    role: str
    backend: OcrBackend
    profile: OcrProfile
    concurrency: int = 4
    _sem: asyncio.Semaphore | None = field(default=None, init=False, repr=False)

    async def read(self, image: Any, task: str = "text") -> str:
        if self._sem is None:
            self._sem = asyncio.Semaphore(max(1, self.concurrency))
        async with self._sem:
            raw = await self.backend.recognize(image, self.profile.prompt_for(task))
        return normalize_ocr_output(self.profile, task, raw).content

    async def shutdown(self) -> None:
        try:
            await self.backend.shutdown()
        except Exception:  # noqa: BLE001 - a closing client must not fail the stage
            logger.debug("Closing the %s recognizer failed", self.role, exc_info=True)


@dataclass
class ConsensusConfig:
    threshold: float = 0.15
    escalate_share: float = 0.07
    page_kinds: frozenset[str] = frozenset({"scan"})

    @classmethod
    def from_settings(cls, settings: GlobalSettings) -> ConsensusConfig:
        ocr = settings.ocr
        return cls(
            threshold=ocr.consensus_threshold,
            escalate_share=ocr.consensus_escalate_share,
            page_kinds=frozenset(ocr.consensus_page_kinds),
        )


@dataclass
class ConsensusReport:
    compared: int = 0
    disagreeing: int = 0
    escalated: int = 0
    replaced: int = 0
    failed: int = 0
    # (page, region index, score) of the escalated regions, 0-based page.
    escalated_regions: list[tuple[int, int, float]] = field(default_factory=list)


def _crop(page_img: Any, bbox: Sequence[float]) -> Any:
    try:
        from bibr.ocr.image_processing import crop_image_region

        return crop_image_region(page_img, list(bbox))
    except Exception:  # noqa: BLE001 - opencv missing: plain PIL crop
        w, h = page_img.size
        return page_img.crop(
            (
                int(bbox[0] * w / 1000),
                int(bbox[1] * h / 1000),
                int(bbox[2] * w / 1000),
                int(bbox[3] * h / 1000),
            )
        )


def _comparable_region(region: dict[str, Any]) -> bool:
    label = region.get("native_label") or region.get("label")
    return (
        label in COMPARED_LABELS
        and not region.get("_native_text_used")
        and bool(region.get("bbox_2d"))
    )


async def apply_consensus(
    pages: list[list[dict[str, Any]]],
    page_images: Sequence[Any],
    page_indices: Sequence[int],
    page_kinds: dict[int, str],
    *,
    second: Recognizer,
    escalation: Recognizer | None,
    config: ConsensusConfig,
    is_outage: Callable[[BaseException], bool] = lambda _exc: False,
) -> ConsensusReport:
    """Compare the primary readings of scan regions with a second recognizer, in place.

    ``pages`` holds the OCR stage's per-page region dicts (window-relative,
    aligned with ``page_images`` and ``page_indices``). Regions of pages whose
    class is in ``config.page_kinds`` are compared; the rest are untouched.
    """
    report = ConsensusReport()
    targets: list[tuple[int, dict[str, Any], Any]] = []
    for page_idx, page_img, regions in zip(page_indices, page_images, pages, strict=True):
        if page_kinds.get(page_idx) not in config.page_kinds:
            continue
        for region in regions:
            if _comparable_region(region):
                targets.append((page_idx, region, page_img))
    if not targets:
        return report

    outage: list[BaseException] = []

    async def _second_reading(page_img: Any, region: dict[str, Any]) -> str | None:
        if outage:
            return None
        try:
            return await second.read(_crop(page_img, region["bbox_2d"]))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the primary reading stands
            if is_outage(exc):
                outage.append(exc)
            logger.debug("Second recognizer failed on a region: %s", exc)
            return None

    seconds = await asyncio.gather(*(_second_reading(img, r) for _, r, img in targets))
    if outage:
        logger.warning(
            "The consensus recognizer is unreachable; scan regions keep the primary reading: %s",
            outage[0],
        )

    scored: list[tuple[float, int, dict[str, Any], Any, str]] = []
    for (page_idx, region, page_img), second_text in zip(targets, seconds, strict=True):
        if second_text is None:
            report.failed += 1
            continue
        primary = region.get("content") or ""
        score = disagreement(primary, second_text)
        report.compared += 1
        region["_ocr_consensus"] = {"score": round(score, 4), "second": second.role}
        if score >= config.threshold:
            report.disagreeing += 1
            scored.append((score, page_idx, region, page_img, second_text))

    if not scored:
        return report
    budget = max(1, math.ceil(config.escalate_share * report.compared))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[:budget]

    async def _escalate(item: tuple[float, int, dict[str, Any], Any, str]) -> None:
        score, page_idx, region, page_img, second_text = item
        primary = region.get("content") or ""
        meta = region["_ocr_consensus"]
        meta["escalated"] = True
        readings = [("primary", primary), ("second", second_text)]
        if escalation is not None:
            try:
                readings.append(
                    ("escalation", await escalation.read(_crop(page_img, region["bbox_2d"])))
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - fall back to the two readings
                logger.debug("Escalation recognizer failed on a region: %s", exc)
        if len(readings) == 3:
            source = medoid(readings)
        else:
            # Two readings and no tie-breaker: keep the primary unless it is empty.
            source = "second" if not primary.strip() and second_text.strip() else "primary"
        meta["chosen"] = source
        if source != "primary":
            region["content"] = dict(readings)[source]
            region["_raw_ocr_content"] = primary
            report.replaced += 1
        report.escalated += 1
        report.escalated_regions.append((page_idx, region.get("index", -1), score))

    await asyncio.gather(*(_escalate(item) for item in chosen))
    return report


def _build_recognizer(
    role: str,
    backend: str,
    *,
    url: str | None,
    model: str | None,
    profile: str | None,
    settings: GlobalSettings,
) -> Recognizer:
    from bibr.ocr import registry
    from bibr.ocr.profiles import resolve_ocr_profile

    if backend not in CONSENSUS_BACKENDS:
        raise ValueError(f"{backend!r} cannot serve as a consensus recognizer")
    if backend in _CLOUD_BACKENDS:
        from bibr.local import ocr_cloud  # noqa: F401 - registers the cloud clients
    else:
        from bibr.local import ocr as _ocr  # noqa: F401 - registers the HTTP clients

    resolved = resolve_ocr_profile(
        # Cloud vision clients follow the GLM prompts; their output needs no normalizing.
        explicit=profile or ("glm" if backend in _CLOUD_BACKENDS else None),
        backend=backend,
        model=model or "",
        max_tokens=settings.ocr.generation_max_tokens,
        temperature=settings.ocr.generation_temperature,
    )
    client = registry.create(
        backend,
        base_url=url,
        model=model,
        profile=resolved,
        settings=settings,
    )
    return Recognizer(
        role=role,
        backend=client,
        profile=resolved,
        concurrency=settings.ocr.concurrent_regions_per_file,
    )


async def start_recognizers(
    settings: GlobalSettings,
) -> tuple[Recognizer, Recognizer | None] | None:
    """The consensus and escalation recognizers the settings name, ready to read.

    None when consensus is off, or when the consensus recognizer cannot start;
    an escalation recognizer that cannot start leaves consensus flag-only.
    Never raises: the scan path is an addition to OCR, not a dependency of it.
    """
    ocr = settings.ocr
    if not ocr.consensus_backend:
        return None
    try:
        second = _build_recognizer(
            "consensus",
            ocr.consensus_backend,
            url=ocr.consensus_url,
            model=ocr.consensus_model,
            profile=ocr.consensus_profile,
            settings=settings,
        )
        await second.backend.wait_for_server()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR consensus disabled: the consensus recognizer did not start: %s", exc)
        return None
    escalation = None
    if ocr.escalation_backend:
        try:
            escalation = _build_recognizer(
                "escalation",
                ocr.escalation_backend,
                url=ocr.escalation_url,
                model=ocr.escalation_model,
                profile=ocr.escalation_profile,
                settings=settings,
            )
            await escalation.backend.wait_for_server()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OCR escalation disabled: the escalation recognizer did not start: %s", exc
            )
            escalation = None
    return second, escalation


def identity(settings: GlobalSettings) -> str:
    """What decides consensus output, for the OCR cache key ("off" when disabled)."""
    ocr = settings.ocr
    if not ocr.consensus_backend:
        return "off"
    parts = [
        ocr.consensus_backend,
        ocr.consensus_model or "",
        ocr.consensus_profile or "",
        f"{ocr.consensus_threshold}",
        f"{ocr.consensus_escalate_share}",
        "+".join(sorted(ocr.consensus_page_kinds)),
        ocr.escalation_backend or "",
        ocr.escalation_model or "",
        ocr.escalation_profile or "",
    ]
    return ":".join(parts)


def describe(report: ConsensusReport, *, escalation: bool) -> str:
    """One sentence on what consensus found, for the export warning."""
    pages = sorted({page + 1 for page, _, _ in report.escalated_regions})
    where = ", ".join(str(p) for p in pages)
    if escalation:
        return (
            f"The two OCR recognizers disagreed on {report.disagreeing} of {report.compared} "
            f"scanned regions; {report.escalated} were escalated and {report.replaced} "
            f"re-read (pages {where})"
        )
    return (
        f"The two OCR recognizers disagreed on {report.disagreeing} of {report.compared} "
        f"scanned regions; the {report.escalated} worst are flagged (pages {where})"
    )
