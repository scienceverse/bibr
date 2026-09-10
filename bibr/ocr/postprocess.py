"""OCR post-processing utilities.

Extracted from bibr._vendor.glmocr — standalone functions for merging
formula numbers, hyphenated text blocks, bullet-point inference, and
OCR content cleaning (repeated-content / hallucination removal, \t
stripping, punctuation collapsing, numbered-list normalisation).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# OCR content cleaning (ported from vendored ResultFormatter._clean_content
# and result_postprocess_utils)
# ---------------------------------------------------------------------------

# Numbered-list normalization: "(1)text" or "(A)text" → "(1) text"
_NUMBERED_PAREN_RE = re.compile(r"^(\(|\uff08)(\d+|[A-Za-z])(\)|\uff09)(.+)$")
# "1.text" or "1)text" or "A)text" → "1. text" etc.
_NUMBERED_DOT_RE = re.compile(r"^(\d+|[A-Za-z])(\.|\)|\uff09)(.+)$")


def _find_consecutive_repeat(
    s: str,
    min_unit_len: int = 10,
    min_repeats: int = 10,
) -> str | None:
    """Find and truncate consecutive repeated patterns.

    Returns the string with the repetition replaced by a single occurrence,
    or ``None`` if no repeats were found.
    """
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None
    # Cap input length to avoid catastrophic backtracking with the
    # backreference + non-greedy quantifier pattern on near-repetitive text.
    _MAX_SEARCH_LEN = 50_000
    search_s = s[:_MAX_SEARCH_LEN] if n > _MAX_SEARCH_LEN else s
    pattern = re.compile(
        r"(.{"
        + str(min_unit_len)
        + ","
        + str(max_unit_len)
        + r"}?)\1{"
        + str(min_repeats - 1)
        + ",}",
        re.DOTALL,
    )
    match = pattern.search(search_s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def _clean_repeated_content(
    content: str,
    min_len: int = 10,
    min_repeats: int = 10,
    line_threshold: int = 10,
) -> str:
    """Remove repeated content (both consecutive and line-level).

    Catches hallucinated output where the OCR model repeats a phrase or line
    dozens of times.

    Return contract: when a repeat is detected and content is trimmed, the
    result is whitespace-stripped so both detection paths emit a consistent
    surface (the consecutive-repeat path used to strip implicitly via its
    operation on ``content.strip()``, while the line-level path returned
    original-with-leading-whitespace). When no repeat is detected, *content*
    is returned unchanged.
    """
    stripped = content.strip()
    if not stripped:
        return content

    # 1. Consecutive repeat detection
    if len(stripped) > min_len * min_repeats:
        result = _find_consecutive_repeat(stripped, min_unit_len=min_len, min_repeats=min_repeats)
        if result is not None:
            return result.strip()

    # 2. Line-level repeat detection
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / total_lines) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(
                        1 for j in range(i, min(i + 3, len(lines))) if lines[j] == common
                    )
                    if consecutive >= 3:
                        original_lines = content.split("\n")
                        non_empty_count = 0
                        for idx, orig_line in enumerate(original_lines):
                            if orig_line.strip():
                                non_empty_count += 1
                                if non_empty_count == i + 1:
                                    return "\n".join(original_lines[: idx + 1]).strip()
                        break
    return content


def clean_ocr_content(content: str) -> str:
    r"""Clean raw OCR output text.

    Ported from vendored ``ResultFormatter._clean_content``:

    * Strip leading/trailing literal ``\t`` sequences.
    * Collapse runs of repeated punctuation (``....`` → ``...``).
    * Remove hallucinated repeated content for long strings (>2048 chars).
    * Normalise numbered-list items (``(1)text`` → ``(1) text``).
    """
    if not content:
        return content

    # 1. Strip leading/trailing literal \t
    result = re.sub(r"^(\\t)+", "", content).lstrip()
    result = re.sub(r"(\\t)+$", "", result).rstrip()

    # 2. Collapse repeated punctuation
    result = re.sub(r"(\.)\1{2,}", r"\1\1\1", result)
    result = re.sub(r"(\u00b7)\1{2,}", r"\1\1\1", result)  # middle dot
    result = re.sub(r"(_)\1{2,}", r"\1\1\1", result)
    result = re.sub(r"(\\_)\1{2,}", r"\1\1\1", result)

    # 3. Remove hallucinated repeated content (long strings only)
    if len(result) >= 2048:
        result = _clean_repeated_content(result)

    # 4. Normalise numbered-list formatting
    m = _NUMBERED_PAREN_RE.match(result)
    if m:
        _, symbol, _, rest = m.groups()
        result = f"({symbol}) {rest.lstrip()}"
    else:
        m = _NUMBERED_DOT_RE.match(result)
        if m:
            symbol, sep, rest = m.groups()
            sep = ")" if sep == "\uff09" else sep
            result = f"{symbol}{sep} {rest.lstrip()}"

    return result.strip()


def clean_formula_number(number_content: str) -> str:
    """Clean up formula number by removing parentheses.

    Examples: "(1)" → "1", "（2.1）" → "2.1", "3" → "3".
    """
    number_clean = number_content.strip()
    if (
        number_clean.startswith("(")
        and number_clean.endswith(")")
        or number_clean.startswith("\uff08")
        and number_clean.endswith("\uff09")
    ):
        number_clean = number_clean[1:-1]
    return number_clean


def _inherit_sources(target, *regions):
    source_ids = list(
        dict.fromkeys(
            source
            for region in regions
            for source in (
                region.get("_source_region_ids")
                or ([region["_source_region_id"]] if region.get("_source_region_id") else [])
            )
        )
    )
    if source_ids:
        target["_source_region_ids"] = source_ids
    for key in ("_native_spans", "_formula_proposals"):
        values = [item for region in regions for item in region.get(key, [])]
        if values:
            target[key] = values


def merge_formula_numbers(json_page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r"""Merge formula_number into adjacent formula block using \tag{}.

    Handles formula→formula_number and formula_number→formula adjacency.
    """
    if not json_page_results:
        return json_page_results

    merged_results: list[dict[str, Any]] = []
    skip_indices: set[int] = set()

    for i, block in enumerate(json_page_results):
        if i in skip_indices:
            continue

        native_label = block.get("native_label", "")

        # Case 1: formula_number followed by formula
        if native_label == "formula_number":
            if i + 1 < len(json_page_results):
                next_block = json_page_results[i + 1]
                if next_block.get("label") == "formula":
                    number_content = block.get("content", "").strip()
                    number_clean = clean_formula_number(number_content)

                    formula_content = next_block.get("content", "")
                    merged_block = next_block.copy()

                    if formula_content.endswith("\n$$"):
                        merged_block["content"] = (
                            formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                        )
                    elif formula_content.rstrip().endswith("$$"):
                        # Handle formulas ending with $$ without a preceding newline
                        stripped = formula_content.rstrip()
                        merged_block["content"] = stripped[:-2] + f" \\tag{{{number_clean}}}$$"

                    _inherit_sources(merged_block, block, next_block)
                    merged_results.append(merged_block)
                    skip_indices.add(i + 1)
                    continue
            # Orphan formula_number (last on page, or not adjacent to a
            # formula): pass it through instead of dropping the equation
            # number from the body text.
            merged_results.append(block)
            continue

        # Case 2: formula followed by formula_number
        if block.get("label") == "formula":
            if i + 1 < len(json_page_results):
                next_block = json_page_results[i + 1]
                if next_block.get("native_label") == "formula_number":
                    number_content = next_block.get("content", "").strip()
                    number_clean = clean_formula_number(number_content)

                    formula_content = block.get("content", "")
                    merged_block = block.copy()

                    if formula_content.endswith("\n$$"):
                        merged_block["content"] = (
                            formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                        )
                    elif formula_content.rstrip().endswith("$$"):
                        stripped = formula_content.rstrip()
                        merged_block["content"] = stripped[:-2] + f" \\tag{{{number_clean}}}$$"

                    _inherit_sources(merged_block, block, next_block)
                    merged_results.append(merged_block)
                    skip_indices.add(i + 1)
                    continue

            merged_results.append(block)
            continue

        merged_results.append(block)

    for idx, block in enumerate(merged_results):
        block["index"] = idx

    return merged_results


def merge_text_blocks(json_page_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge hyphenated text blocks if the combined word is valid.

    Uses Zipf frequency >= 2.5 as the validity threshold for merged words.
    """
    from wordfreq import zipf_frequency

    if not json_page_results:
        return json_page_results

    merged_results: list[dict[str, Any]] = []
    skip_indices: set[int] = set()

    for i, block in enumerate(json_page_results):
        if i in skip_indices:
            continue

        if block.get("label") != "text":
            merged_results.append(block)
            continue

        content = block.get("content", "")
        if not isinstance(content, str):
            merged_results.append(block)
            continue

        content_stripped = content.rstrip()
        if not content_stripped:
            merged_results.append(block)
            continue

        if not content_stripped.endswith("-"):
            merged_results.append(block)
            continue

        merged = False
        for j in range(i + 1, len(json_page_results)):
            if json_page_results[j].get("label") == "text":
                next_content = json_page_results[j].get("content", "")
                if isinstance(next_content, str):
                    next_stripped = next_content.lstrip()
                    if next_stripped and next_stripped[0].islower():
                        words_before = content_stripped[:-1].split()
                        next_words = next_stripped.split()

                        if words_before and next_words:
                            word_fragment_before = words_before[-1]
                            word_fragment_after = next_words[0]
                            merged_word = word_fragment_before + word_fragment_after

                            zipf_score = zipf_frequency(merged_word.lower(), "en")
                            if zipf_score >= 2.5:
                                merged_content = content_stripped[:-1] + next_content.lstrip()
                                merged_block = block.copy()
                                merged_block["content"] = merged_content

                                _inherit_sources(merged_block, block, json_page_results[j])
                                merged_results.append(merged_block)
                                skip_indices.add(j)
                                merged = True
                    break
            else:
                # Any non-text region between the hyphenated word and the
                # candidate text block invalidates the merge — the words
                # don't actually neighbour each other in reading order.
                break

        if not merged:
            merged_results.append(block)

    for idx, block in enumerate(merged_results):
        block["index"] = idx

    return merged_results


def format_bullet_points(
    json_page_results: list[dict[str, Any]],
    left_align_threshold: float = 10.0,
) -> list[dict[str, Any]]:
    """Detect and add missing bullet points to list items.

    If a text block sits between two bullet-pointed items and is
    left-aligned with them, prepend ``- `` to it.
    """
    if len(json_page_results) < 3:
        return json_page_results

    for i in range(1, len(json_page_results) - 1):
        current_block = json_page_results[i]
        prev_block = json_page_results[i - 1]
        next_block = json_page_results[i + 1]

        if current_block.get("native_label") != "text":
            continue
        if prev_block.get("native_label") != "text" or next_block.get("native_label") != "text":
            continue

        current_content = current_block.get("content", "")
        if current_content.startswith("- "):
            continue

        prev_content = prev_block.get("content", "")
        next_content = next_block.get("content", "")

        if not (prev_content.startswith("- ") and next_content.startswith("- ")):
            continue

        current_bbox = current_block.get("bbox_2d", [])
        prev_bbox = prev_block.get("bbox_2d", [])
        next_bbox = next_block.get("bbox_2d", [])

        if not (current_bbox and prev_bbox and next_bbox):
            continue

        current_left = current_bbox[0]
        prev_left = prev_bbox[0]
        next_left = next_bbox[0]

        if (
            abs(current_left - prev_left) <= left_align_threshold
            and abs(current_left - next_left) <= left_align_threshold
        ):
            current_block["content"] = "- " + current_content

    return json_page_results
