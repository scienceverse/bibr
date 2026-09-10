"""Conservative native-line repair plans, with stable typed source spans.

No recognizer confidence is treated as proof of fidelity. Geometry that cannot
establish ownership falls back to the existing region recognition path.
"""

from __future__ import annotations

import re
import unicodedata

from bibr.ocr.native_source import overlap
from bibr.ocr.native_text import (
    DEFAULT_ELIGIBLE_LABELS,
    _is_native_text_usable,
)

CAPTIONS = frozenset({"figure_title", "table_title", "chart_title"})


def _box(chars):
    boxes = [c["bbox_2d"] for c in chars if "bbox_2d" in c]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def native_lines(source):
    """Group native reading-order characters, splitting explicit breaks and column gaps."""
    lines, current = [], []
    last_box = None
    for char in source["characters"]:
        box = char.get("bbox_2d")
        split = char["text"] in {"\r", "\n"}
        if box and last_box:
            height = max(last_box[3] - last_box[1], box[3] - box[1], 1)
            split |= abs((box[1] + box[3] - last_box[1] - last_box[3]) / 2) > height * 0.8
            split |= box[0] - last_box[2] > height * 3
        if split:
            if current:
                lines.append(current)
            current, last_box = [], None
        if char["text"] not in {"\r", "\n"}:
            current.append(char)
            if box:
                last_box = box
    if current:
        lines.append(current)
    return [line for line in lines if any(c.get("bbox_2d") for c in line)]


def _reason(text, ratio):
    if any(unicodedata.category(c) == "Co" for c in text):
        return "private_use"
    if not text.strip() or not _is_native_text_usable(text, ratio):
        return "encoding"
    return None


def plan_native_repairs(regions, source, *, min_chars, min_printable_ratio, native_captions=False):
    lines = native_lines(source)
    for region in regions:
        label, bbox = region.get("label"), region.get("bbox_2d")
        if label not in DEFAULT_ELIGIBLE_LABELS | CAPTIONS or not bbox:
            continue
        caption = label in CAPTIONS
        if caption and not native_captions:
            continue
        if source.get("page_rotation"):
            region.pop("_native_text_used", None)
            region["_native_text_candidate"] = region.get("content", "")
            region["content"] = ""
            region["_native_text_rejection_reason"] = "rotated_page"
            continue
        owner = region["_source_region_id"]
        selected = [line for line in lines if overlap(_box(line), bbox) > 0.5]
        if not selected:
            continue
        formulas = [
            r
            for r in regions
            if r.get("label") == "inline_formula"
            and r.get("bbox_2d")
            and overlap(r["bbox_2d"], bbox) >= 0.98
        ]
        allowed = {owner, *(r["_source_region_id"] for r in formulas)}
        # Every visible glyph must have one leaf owner, and complete native lines
        # must fit. This deliberately refuses uncertain clipped/overlapping crops.
        if any(
            overlap(_box(line), bbox) < 0.98
            or any(
                not c["text"].isspace()
                and (
                    len(c["owners"]) != 1
                    or c["owners"][0] not in allowed
                    or c.get("duplicate_of")
                    or c.get("angle", 0) not in (None, 0.0)
                )
                for c in line
            )
            for line in selected
        ):
            region.pop("_native_text_used", None)
            region["_native_text_candidate"] = region.get("content", "")
            region["content"] = ""
            region["_native_text_rejection_reason"] = "ownership"
            continue
        candidate = "\n".join("".join(c["text"] for c in line) for line in selected)
        if caption and (
            len(selected) != 1
            or not re.match(r"\s*(?:Fig(?:ure)?\.?|Table|Chart)\s+\w", candidate, re.I)
        ):
            continue
        if len(candidate.strip()) < min_chars and not region.get("_native_text_used"):
            continue
        if not formulas and not caption and region.get("_native_text_used"):
            continue
        spans, consumed_formulas = [], []
        invalid = False
        for line_index, line in enumerate(selected):
            line_box = _box(line)
            matched = [
                f
                for f in formulas
                if min(f["bbox_2d"][3], line_box[3]) > max(f["bbox_2d"][1], line_box[1])
            ]
            if any(f in consumed_formulas for f in matched):
                invalid = True
                break
            matched.sort(key=lambda f: f["bbox_2d"][0])
            chunks = [[] for _ in range(len(matched) + 1)]
            formula_chars = [[] for _ in matched]
            for char in line:
                box = char.get("bbox_2d")
                if box is None:
                    # PDF separators without geometry retain their order in the
                    # current text run. Visible unlocated glyphs failed ownership.
                    chunks[0].append(char)
                    continue
                which = next(
                    (i for i, f in enumerate(matched) if overlap(box, f["bbox_2d"]) > 0.5), None
                )
                if which is not None:
                    formula_chars[which].append(char)
                else:
                    index = sum((box[0] + box[2]) / 2 > f["bbox_2d"][2] for f in matched)
                    chunks[index].append(char)
            for index, chunk in enumerate(chunks):
                if chunk:
                    text = "".join(c["text"] for c in chunk)
                    reason = _reason(text, min_printable_ratio) if text.strip() else None
                    spans.append(
                        {
                            "type": "text",
                            "candidate": text,
                            "content": text if reason is None else "",
                            "status": "native" if reason is None else "pending",
                            "reason": reason,
                            "bbox_2d": _box(chunk)
                            if any(c.get("bbox_2d") for c in chunk)
                            else line_box,
                            "source_ids": [c["source_id"] for c in chunk],
                        }
                    )
                if index < len(matched):
                    formula = matched[index]
                    spans.append(
                        {
                            "type": "inline_formula",
                            "candidate": "".join(c["text"] for c in formula_chars[index]),
                            "content": "",
                            "status": "pending",
                            "reason": "formula_structure",
                            "bbox_2d": formula["bbox_2d"],
                            "formula_owner": formula["_source_region_id"],
                            "source_ids": [c["source_id"] for c in formula_chars[index]],
                        }
                    )
                    consumed_formulas.append(formula)
            if line_index < len(selected) - 1:
                spans.append(
                    {"type": "separator", "content": "\n", "status": "native", "source_ids": []}
                )
        if invalid or len(consumed_formulas) != len(formulas) or not spans:
            continue
        for i, span in enumerate(spans):
            span["span_id"] = f"{owner}:s{i}"
        region["_native_spans"] = spans
        region["_native_text_candidate"] = candidate
        region["_native_text_used"] = all(s["status"] == "native" for s in spans)
        region["content"] = (
            "".join(s["content"] for s in spans) if region["_native_text_used"] else ""
        )
        for formula in consumed_formulas:
            formula["_native_formula_parent"] = owner


async def recognize_native_spans(
    page_img,
    region,
    page_idx,
    filename,
    ocr_fn,
    ocr_sem,
    *,
    settings,
    profile,
    warning_sink,
    owned_crops,
):
    """Recognize only pending crops through the existing OCR lifecycle and error handling."""
    from bibr.pipeline.stages.ocr import _ocr_page_regions_impl

    pending = [span for span in region["_native_spans"] if span["status"] == "pending"]
    crops = []
    for index, span in enumerate(pending):
        box = span["bbox_2d"]
        pad = min(2.0, max(0.5, (box[3] - box[1]) * 0.1))
        crops.append(
            {
                "index": index,
                "label": span["type"],
                "task_type": "formula" if span["type"] == "inline_formula" else "text",
                "bbox_2d": [
                    max(0, box[0] - pad),
                    max(0, box[1] - pad),
                    min(1000, box[2] + pad),
                    min(1000, box[3] + pad),
                ],
            }
        )
    results = await _ocr_page_regions_impl(
        page_img,
        crops,
        page_idx,
        filename,
        ocr_fn,
        ocr_sem,
        settings=settings,
        profile=profile,
        warning_sink=warning_sink,
        owned_crops=owned_crops,
    )
    for span, result in zip(pending, results, strict=True):
        text = result.get("content", "")
        span["raw_response"] = result.get("_raw_ocr_content", text)
        span["finish_reason"] = result.get("_ocr_finish_reason")
        if span["type"] == "inline_formula":
            # LaTeX uses syntax (e.g. ^ and backslashes) outside the prose gate.
            bad = any(
                unicodedata.category(c) in {"Co", "Cs"}
                or c == "\ufffd"
                or (unicodedata.category(c) == "Cc" and not c.isspace())
                for c in text
            )
            reason = "encoding" if bad or not text.strip() else None
        else:
            reason = _reason(text, settings.ocr.native_text_min_printable_ratio)
        if reason or span["finish_reason"] == "length":
            span["status"] = "unresolved"
            span["repair_failure"] = reason or "truncated"
            span["content"] = (
                "[unresolved formula]" if span["type"] == "inline_formula" else "[unresolved text]"
            )
            if warning_sink is not None:
                warning_sink(
                    f"Native repair unresolved: {span['span_id']} ({span['repair_failure']})"
                )
        else:
            span["status"] = "recognized_unverified"
            if span["type"] == "inline_formula":
                text = text.strip()
                if text.startswith((r"\[", r"\(")) and text.endswith((r"\]", r"\)")):
                    text = text[2:-2]
                span["content"] = "$" + text.strip().strip("$").strip() + "$"
            else:
                candidate = span["candidate"]
                leading = candidate[: len(candidate) - len(candidate.lstrip())]
                trailing = candidate[len(candidate.rstrip()) :]
                span["content"] = leading + text.strip() + trailing
    return "".join(span["content"] for span in region["_native_spans"])
