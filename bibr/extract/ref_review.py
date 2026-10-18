"""Source checks for bounded NER-to-fine-tuned reference review.

These are observable routing signals, not calibrated confidence. A review
replaces the entire primary parse only when it improves those checks without
discarding supported title, author or year evidence.
"""

from __future__ import annotations

import re

from bibr.models import PaperReference

_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_YEAR = re.compile(r"(?<!\d)(?:1[5-9]|20)\d\d(?!\d)")
# Han, kana and Hangul. _WORDS reads an unspaced CJK run as one word, so a parse
# that lumps authors, title and journal into one span still passes coverage.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
# Signals of the input rather than the parse: no parser can clear them.
_INPUT_FLAGS = frozenset({"long_reference", "cjk_script"})
_FIELDS = ("authors", "title", "container", "editors", "publisher", "edition")


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold().replace("–", "-").replace("—", "-")


def coverage(reference: str, parsed: PaperReference | None) -> float:
    words = {word for word in _WORDS.findall(fold(reference)) if len(word) > 1}
    if not words or parsed is None:
        return 0.0
    claimed = {word for key in _FIELDS for word in _WORDS.findall(fold(getattr(parsed, key) or ""))}
    return len(words & claimed) / len(words)


def reasons(
    reference: str, parsed: PaperReference | None, *, min_coverage: float, long_chars: int
) -> tuple[str, ...]:
    if parsed is None:
        return ("empty_parse",)
    flags = []
    source = fold(reference)
    if not parsed.title:
        flags.append("missing_title")
    if parsed.year is None and not parsed.is_in_press and _YEAR.search(reference):
        flags.append("missing_printed_year")
    if parsed.year is not None:
        suffix = re.search(rf"(?<!\d){parsed.year}([a-z])(?!\w)", reference)
        if suffix and parsed.year_suffix != suffix[1]:
            flags.append("missing_year_suffix")
    for key in ("title", "authors", "container"):
        value = getattr(parsed, key)
        if value and fold(value) not in source:
            flags.append(f"unsupported_{key}")
    if coverage(reference, parsed) < min_coverage:
        flags.append("unassigned_words")
    if len(reference) > long_chars:
        flags.append("long_reference")
    if _CJK.search(reference):
        flags.append("cjk_script")
    return tuple(flags)


def improves(
    reference: str,
    current: PaperReference | None,
    candidate: PaperReference,
    *,
    min_coverage: float,
    long_chars: int,
) -> bool:
    if current is None:
        return bool(candidate.title or candidate.authors)
    source = fold(reference)
    for key in ("title", "authors"):
        value = getattr(current, key)
        if value and fold(value) in source and not getattr(candidate, key):
            return False
    if current.year is not None and str(current.year) in reference and candidate.year is None:
        return False
    before = set(reasons(reference, current, min_coverage=min_coverage, long_chars=long_chars))
    after = set(reasons(reference, candidate, min_coverage=min_coverage, long_chars=long_chars))
    cjk = "cjk_script" in before
    before -= _INPUT_FLAGS
    after -= _INPUT_FLAGS
    old_coverage, new_coverage = coverage(reference, current), coverage(reference, candidate)
    if new_coverage < old_coverage:
        return False
    if cjk:
        # Coverage cannot tell CJK fields apart, so keep the review when no
        # source check gets worse.
        return after <= before
    return len(after) < len(before) or new_coverage > old_coverage
