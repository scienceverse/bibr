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

# A leading literal "\t" is a GLM-OCR tab artifact in prose, but in a formula
# it is the start of \theta, \tau, \text, \tilde or \times.  Formula regions
# strip a leading one only when no command name follows.  A trailing one
# cannot start a command, so every region strips it the same way.
_LEADING_TAB_RE = re.compile(r"^(\\t)+")
_TRAILING_TAB_RE = re.compile(r"(\\t)+$")
_FORMULA_LEADING_TAB_RE = re.compile(r"^(?:\\t(?![A-Za-z]))+")

# Punctuation that attaches to the token before it: a space in front of it
# would split "(1)–(3)" or "(a), (b)".
_ATTACHED_PUNCTUATION = frozenset(",.;:!?)]}…-–—_^/\\%")


def _is_list_marker_rest(rest: str, sep: str) -> bool:
    """Whether *rest* can follow a list marker the OCR glued to its text.

    "1.text", "(1)12 participants" and "(a)“Always”" are list items.  After a
    period, though, a digit continues a number ("3.14", "10.1038/…",
    "0.5\\sum") and a letter carrying its own period is an abbreviation
    ("U.S. Adults", "e.g. the", "i.e., the"); a space would split those.  A
    closing parenthesis starts neither, so a word, digit or quote may follow
    it.  Punctuation that attaches to the marker ("(1)–(3)") is never split
    off, and whitespace after the marker is only normalised to one space.
    """
    first = rest[0]
    if first.isspace():
        return True
    if first in _ATTACHED_PUNCTUATION:
        return False
    if sep == ".":
        return not first.isdigit() and not (first.isalpha() and rest[1:2] == ".")
    return True


def _has_repeated_ngram(s: str, unit_len: int, min_repeats: int) -> bool:
    """Exact O(n) precondition for the consecutive-repeat search below.

    A unit of at least *unit_len* characters repeated *min_repeats* times
    contains its own leading *unit_len*-gram once per repetition. So if no
    gram of that length occurs *min_repeats* times, the backreference pattern
    cannot match and running it is pure waste — which is the common case: on
    131,199 real OCR regions, 737 of the 943 regions over 2,048 characters
    produced no repeat at all, at a median 24 ms and a p99 of 110 ms each.
    """
    if len(s) < unit_len * min_repeats:
        return False
    counts = Counter(s[i : i + unit_len] for i in range(len(s) - unit_len + 1))
    return max(counts.values(), default=0) >= min_repeats


def _find_consecutive_repeat(
    s: str,
    min_unit_len: int = 10,
    min_repeats: int = 10,
) -> str | None:
    """Find and collapse the first run of consecutive repeated patterns.

    Returns the string with the repetition replaced by a single occurrence —
    the text before and after the run is kept — or ``None`` if no repeats
    were found.
    """
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    # Bound the input the backreference + non-greedy quantifier pattern sees:
    # its cost is superlinear in the searched length (2,048 chars ≈ 16 ms;
    # 32,768 ≈ 4.3 s).
    _MAX_SEARCH_LEN = 50_000
    search_s = s[:_MAX_SEARCH_LEN] if n > _MAX_SEARCH_LEN else s
    # Derive the unit ceiling from what is actually searched. Taking it from
    # the full length left the cap ineffective — a 100,000-character input
    # cost twice a 50,000-character one (23 s measured) despite both
    # searching 50,000 characters.
    max_unit_len = len(search_s) // min_repeats
    if max_unit_len < min_unit_len:
        return None
    if not _has_repeated_ngram(search_s, min_unit_len, min_repeats):
        return None
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
    if match is None:
        return None
    # Keep the text after the run: dropping it cut a dot-leader table of
    # contents to its first entry. A decode loop can run on past the searched
    # window, so its copies beyond the window go too, and so does a final
    # partial copy where the model hit its token cap.
    unit = match.group(1)
    end = match.end()
    while s.startswith(unit, end):
        end += len(unit)
    if unit.startswith(s[end:]):
        end = n
    return s[: match.start()] + unit + s[end:]


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


def clean_ocr_content(content: str, *, formula: bool = False) -> str:
    r"""Clean raw OCR output text.

    Ported from vendored ``ResultFormatter._clean_content``:

    * Strip leading/trailing literal ``\t`` sequences.
    * Collapse runs of repeated punctuation (``....`` → ``...``).
    * Remove hallucinated repeated content for long strings (>2048 chars).
    * Normalise numbered-list items (``(1)text`` → ``(1) text``).

    Only OCR output needs this: text taken from the PDF text layer carries
    none of these artifacts. ``formula=True`` marks LaTeX from a formula
    region. There a leading ``\t`` is only stripped when no command name
    follows it (``\theta``, ``\text``), and the punctuation and list-marker
    steps are skipped: they repair prose rendering, and in LaTeX they only
    change bytes (``(a)_{n}`` → ``(a) _{n}``). The repeat trimmer still runs,
    since formula decoding loops too.
    """
    if not content:
        return content

    # 1. Strip leading/trailing literal \t
    leading = _FORMULA_LEADING_TAB_RE if formula else _LEADING_TAB_RE
    result = leading.sub("", content).lstrip()
    result = _TRAILING_TAB_RE.sub("", result).rstrip()

    # 2. Collapse repeated punctuation
    if not formula:
        result = re.sub(r"(\.)\1{2,}", r"\1\1\1", result)
        result = re.sub(r"(\u00b7)\1{2,}", r"\1\1\1", result)  # middle dot
        result = re.sub(r"(_)\1{2,}", r"\1\1\1", result)
        result = re.sub(r"(\\_)\1{2,}", r"\1\1\1", result)

    # 3. Remove hallucinated repeated content (long strings only)
    if len(result) >= 2048:
        result = _clean_repeated_content(result)

    # 4. Normalise numbered-list formatting
    if formula:
        return result.strip()
    #    Full-width parentheses become ASCII whether or not a space goes in.
    m = _NUMBERED_PAREN_RE.match(result)
    if m:
        _, symbol, _, rest = m.groups()
        marker, sep = f"({symbol})", ")"
    elif m := _NUMBERED_DOT_RE.match(result):
        symbol, sep, rest = m.groups()
        sep = ")" if sep == "\uff09" else sep
        marker = f"{symbol}{sep}"
    if m:
        result = (
            f"{marker} {rest.lstrip()}" if _is_list_marker_rest(rest, sep) else f"{marker}{rest}"
        )

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
                                # The merged text is from the text layer only
                                # when both halves are.
                                if not json_page_results[j].get("_native_text_used"):
                                    merged_block.pop("_native_text_used", None)

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
