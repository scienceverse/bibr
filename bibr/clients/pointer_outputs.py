"""Turn pointer answers back into the verbatim fields the rest of bibr reads.

With ``LLM_POINTER_OUTPUTS`` on, two front-matter calls stop retyping text the
prompt already holds. Their latency is set by how many tokens they write, and
the abstract and the per-author affiliations are most of those tokens:

* title/keywords returns the abstract's first and last words
  (:class:`~bibr.schemas.TitleKeywordsPointerLLM`). :func:`locate_abstract`
  finds them in the exact text the model was shown and :func:`resolve_title_keywords`
  copies what lies between them, unless the last words close a sentence the
  paragraph goes on after, or lie past the keywords or a box printed after the
  abstract (the model misjudged where the abstract ends).
* authors lists each affiliation once and gives every author the numbers of
  its affiliations (:class:`~bibr.schemas.AuthorsPointerLLM`).
  :func:`expand_authors` writes each author's affiliation string back out,
  joined with "; " as the verbatim prompt asks for. A stand-in written for an
  affiliation the text does not print ("Affiliation 3") is left empty.

Both return the verbatim response models (:class:`~bibr.schemas.TitleKeywordsLLM`,
:class:`~bibr.schemas.AuthorsLLM`), so nothing downstream changes.

Anchors are matched on letters and digits only (case, accents, spaces,
hyphens and punctuation ignored), which absorbs the ways a model "tidies" a
quote: rejoining a word hyphenated at a line end, straightening quotes, or
dropping a line break. A fuzzy pass then absorbs a corrected OCR typo.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from bibr.input.consolidate_text import fix_ocr_artifacts
from bibr.schemas import AuthorLLM, AuthorsLLM, TitleKeywordsLLM

if TYPE_CHECKING:
    from bibr.schemas import AuthorsPointerLLM, TitleKeywordsPointerLLM

logger = logging.getLogger(__name__)

# Fewer letters and digits than this cannot place an anchor reliably.
_MIN_ANCHOR_CHARS = 8
# rapidfuzz partial-ratio floor for an anchor that has no exact match. Anchors
# are a few words, so one or two changed characters still clear it.
_FUZZY_MIN_SCORE = 90
# A copied span longer than this means the anchors matched the wrong places.
# The longest structured abstracts run to about 5,000 characters.
MAX_ABSTRACT_CHARS = 15_000
# Punctuation that belongs to the abstract's first or last word.
_OPENING_MARKS = "([{\"'“‘«¿¡"
_CLOSING_MARKS = ".,;:!?)]}\"'”’»%…"
# An "Abstract" heading the opening anchor took in. Stripped only when a line
# break or a colon, period or dash follows it, or it is printed in capitals, so
# an abstract that opens "Abstract reasoning ..." keeps its first word.
_LEADING_LABEL_RE = re.compile(
    r"^(?:(?i:abstract|summary)[ \t]*(?:[:.\u2013\u2014-][ \t]*|\r?\n\s*)|(?:ABSTRACT|SUMMARY)\s+)"
)
# Blocks printed beside the abstract that reading order can drop inside it: a
# correspondence or "To cite" sidebar, a copyright line, submission dates.
_FURNITURE_BLOCK_RE = re.compile(
    r"^\W*(?:correspondence(?:\s+to\b|\s*:)|(?:the\s+)?corresponding\s+author\b"
    r"|to\s+cite(?:\s*:|\s+this\b)|©|copyright\b|(?:received|revised|accepted)\b:?\s*\d"
    r"|published\s+(?:online|by)\b|reprints\s+and\s+permissions\b|article\s+reuse\s+guidelines\b)",
    re.IGNORECASE,
)
# What opens a block printed after the abstract and never inside it: the
# keyword list, or a box such as "Research in context". An abstract span that
# runs into one ended on the wrong last words.
_AFTER_ABSTRACT_RE = re.compile(
    r"^\W*(?:key\s*words?|index\s+terms|research\s+in\s+context|statement\s+of\s+relevance"
    r"|relevance\s+for\s+practice|relevantie\s+voor\s+de\s+praktijk|significance\s+statement"
    r"|plain[\s-]+language\s+summary)\b",
    re.IGNORECASE,
)
# A keyword list printed without a label: the answer's keywords make up at
# least this share of the block's letters and digits.
_KEYWORD_LIST_SHARE = 0.6
# A fuzzy match of the closing anchor ends at the last place this many of the
# anchor's own final letters and digits occur in it.
_TAIL_PROBE = 6
# A block break. A PDF text layer wraps lines inside a block with "\r\n".
_BLOCK_BREAK_RE = re.compile(r"(?<!\r)\n")
# A doubled text layer prints the end of one block again at the start of the
# next: at least this many letters and digits of it, behind at most
# _REPEAT_LEAD garbled ones.
_REPEAT_PROBE = 60
_REPEAT_LEAD = 40
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A stand-in the model wrote for an affiliation the text does not print. Matched
# on the whole entry, so an institution's own name is never caught: "Affiliation 3",
# "Affiliation 1 (Jane Doe, John Smith)", "N/A".
_STAND_IN_RE = re.compile(
    r"affiliation\s*#?\s*\d+(?:\s*\(.*\))?|unknown|n/?a|none|null|\W*",
    re.IGNORECASE | re.DOTALL,
)
# ... or a note saying so: "Affiliation 3 not printed on this page".
_NOT_PRINTED_RE = re.compile(
    r"\bnot\s+(?:printed|given|provided|shown|listed|stated|specified|available)\b",
    re.IGNORECASE,
)
# A conflict-of-interest footnote that the page prints inside the abstract.
# Each pattern is one kind of statement about the authors. One kind alone also
# turns up in abstracts about employment or company ownership; two do not.
_DISCLOSURE_KINDS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:is|are|was|were)\s+(?:an?\s+)?(?:employees?\s+of|employed\s+by)\b",
        r"\breceive[sd]?\s+royalties\b|\broyalties\s+from\b",
        r"\b(?:is|are)\s+(?:also\s+)?(?:an?\s+)?(?:share|stock)holders?\s+(?:in|of)\b",
        r"\b(?:personal|consulting|speaker|speaking)\s+fees\b|\bhonoraria\b",
        r"\b(?:advisory|scientific)\s+boards?\s+(?:of|for)\b"
        r"|\bserved?\s+as\s+(?:an?\s+)?(?:paid\s+)?consultant\b",
        r"\b(?:declares?|reports?|ha(?:ve|s))\s+no\s+(?:conflicts?\s+of\s+interest|competing\s+interests?)\b",
    )
)
# A word broken at a line end with a non-breaking or Unicode hyphen, which the
# OCR cleanup's dehyphenation (ASCII "-" only) would otherwise leave as is.
_WRAP_HYPHEN_RE = re.compile(r"[\u2010\u2011](?=[ \t]*\r?\n)")
# Spaces other than line breaks (no-break, thin, figure ...) become plain spaces,
# as a model writing the abstract out would type them.
_ODD_SPACE_RE = re.compile(r"[^\S\r\n ]")


def _squash(text: str) -> tuple[str, list[int]]:
    """The letters and digits of *text*, case- and accent-folded, with the
    offset in *text* each one came from."""
    chars: list[str] = []
    offsets: list[int] = []
    for offset, char in enumerate(text):
        for folded in unicodedata.normalize("NFKD", char).casefold():
            if folded.isalnum():
                chars.append(folded)
                offsets.append(offset)
    return "".join(chars), offsets


def _exact_spans(haystack: str, needle: str, start: int = 0) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = haystack.find(needle, start)
    while position != -1:
        spans.append((position, position + len(needle)))
        position = haystack.find(needle, position + 1)
    return spans


def _fuzzy_spans(
    haystack: str, needle: str, start: int = 0, end: int | None = None
) -> list[tuple[int, int]]:
    """Every place *needle* nearly matches ``haystack[start:end]``, in order."""
    end = len(haystack) if end is None else end
    if end - start < len(needle):
        return []
    alignment = fuzz.partial_ratio_alignment(
        needle, haystack[start:end], score_cutoff=_FUZZY_MIN_SCORE
    )
    if alignment is None:
        return []
    found = (start + alignment.dest_start, start + alignment.dest_end)
    return [
        *_fuzzy_spans(haystack, needle, start, found[0]),
        found,
        *_fuzzy_spans(haystack, needle, found[1], end),
    ]


@dataclass(frozen=True)
class AbstractSpan:
    """Where the abstract sits in the source text: ``source[start:end]``."""

    start: int
    end: int
    fuzzy: bool


def locate_abstract(source: str, first_words: str, last_words: str) -> AbstractSpan | None:
    """Find the abstract that opens with *first_words* and closes with
    *last_words* in *source*, or ``None`` when the anchors do not place it.

    The earliest opening that has a closing after it wins. A closing phrase can
    recur inside the abstract (a first and last sentence that both end "... for
    people affected by rare diseases"), so among the closings after the opening
    the one that looks most like the abstract's end wins, the earliest on a tie
    (:func:`_closing_fit`). Exact matches are tried before fuzzy ones.
    """
    squashed, offsets = _squash(source)
    head = _squash(first_words)[0]
    tail = _squash(last_words)[0]
    if len(head) < _MIN_ANCHOR_CHARS or len(tail) < _MIN_ANCHOR_CHARS:
        return None
    closing = last_words.rstrip()[-1:]
    mark = closing if closing and not closing.isalnum() else None

    heads = _exact_spans(squashed, head)
    fuzzy = not heads
    if fuzzy:
        heads = _fuzzy_spans(squashed, head)
    for head_start, head_end in heads:
        tails = [span for span in _exact_spans(squashed, tail, head_start) if span[1] >= head_end]
        tail_fuzzy = not tails
        if tail_fuzzy:
            found = (
                _tighten(squashed, tail, span) for span in _fuzzy_spans(squashed, tail, head_start)
            )
            tails = [span for span in found if span[1] >= head_end]
        ends = [offsets[tail_end - 1] + 1 for _, tail_end in tails]
        # sorted() is stable: the earliest closing wins among equal fits.
        for end in sorted(ends, key=lambda end: -_closing_fit(source, end, mark)):
            span = _widen(source, offsets[head_start], end, snap=fuzzy or tail_fuzzy)
            if span[1] - span[0] <= MAX_ABSTRACT_CHARS:
                return AbstractSpan(*span, fuzzy=fuzzy or tail_fuzzy)
    return None


def _tighten(squashed: str, tail: str, span: tuple[int, int]) -> tuple[int, int]:
    """*span*, a fuzzy match of *tail*, ending where the anchor's own last
    letters and digits do. A match is as long as the anchor, so one quoted a
    word longer than printed runs on past the abstract (into a section number)."""
    probe = tail[-_TAIL_PROBE:]
    stop = squashed.rfind(probe, span[0], span[1])
    return span if stop == -1 else (span[0], stop + len(probe))


def _closing_fit(source: str, end: int, mark: str | None) -> int:
    """How much ``source[:end]`` looks like the end of an abstract: a point for
    the closing anchor's own final punctuation mark, and a point for a line
    break (or the end of the text) right after it."""
    stop = end
    while stop < len(source) and source[stop] in _CLOSING_MARKS:
        stop += 1
    fit = 1 if mark is not None and mark in source[end:stop] else 0
    rest = source[stop:].lstrip(" \t")
    return fit + (1 if not rest or rest[0] in "\r\n" else 0)


def _widen(source: str, start: int, end: int, *, snap: bool) -> tuple[int, int]:
    """Take in the punctuation of the first and last words. A fuzzy match can
    start or stop inside a word, so *snap* first moves each end to the nearer
    edge of that word."""
    if snap:
        start, end = _word_edge(source, start), _word_edge(source, end)
    while start > 0 and source[start - 1] in _OPENING_MARKS:
        start -= 1
    while end < len(source) and source[end] in _CLOSING_MARKS:
        end += 1
    return start, end


def _word_edge(source: str, position: int) -> int:
    """*position*, or the nearer edge of the word it falls inside."""
    left = right = position
    while left > 0 and source[left - 1].isalnum():
        left -= 1
    while right < len(source) and source[right].isalnum():
        right += 1
    if position in (left, right):
        return position
    return left if position - left <= right - position else right


def abstract_text(source: str, span: AbstractSpan) -> str:
    """The abstract *span* places in *source*, without what a model copying it
    out would have left behind.

    The anchors fix the first and last blocks as abstract text. A block between
    them is dropped when it is page furniture printed beside the abstract (a
    correspondence sidebar, an email address, a copyright line, submission
    dates, a conflict-of-interest footnote), and a block that opens by repeating
    the end of the one before it, as a doubled text layer does, loses the
    repeat. A leading "Abstract" heading is dropped, words broken at a line end
    are rejoined as the OCR cleanup rejoins them, and unusual spaces become
    plain ones.
    """
    blocks = _BLOCK_BREAK_RE.split(_LEADING_LABEL_RE.sub("", source[span.start : span.end]))
    kept: list[str] = []
    for index, block in enumerate(blocks):
        inner = 0 < index < len(blocks) - 1
        if inner and (
            _FURNITURE_BLOCK_RE.match(block) or _EMAIL_RE.search(block) or _is_disclosure(block)
        ):
            continue
        kept.append(_without_repeat(kept[-1], block) if kept else block)
    text = fix_ocr_artifacts(_WRAP_HYPHEN_RE.sub("-", "\n".join(kept)))
    return _ODD_SPACE_RE.sub(" ", text).strip()


def _is_disclosure(block: str) -> bool:
    """Whether *block* reads as a conflict-of-interest statement."""
    return sum(1 for kind in _DISCLOSURE_KINDS if kind.search(block)) >= 2


def _without_repeat(previous: str, block: str) -> str:
    """*block* without a repeat of the end of *previous* at its start."""
    ending = _squash(previous)[0]
    tail = ending[-_REPEAT_PROBE:]
    if len(tail) < _REPEAT_PROBE:
        return block
    squashed, offsets = _squash(block)
    end = squashed.find(tail) + len(tail)
    if end < len(tail):
        return block
    leads = range(min(_REPEAT_LEAD, end - len(tail)) + 1)
    if any(ending.endswith(squashed[lead:end]) for lead in leads):
        return block[offsets[end - 1] + 1 :].lstrip(_CLOSING_MARKS + " \t\r\n")
    return block


@dataclass(frozen=True)
class ResolvedTitleKeywords:
    """A pointer answer turned back into a verbatim one.

    ``status`` says what happened: ``copied``, ``copied_fuzzy`` or ``absent``
    (the model found no abstract). ``response`` is ``None`` when the anchors
    cannot be trusted, and ``status`` is a key of :data:`UNUSED_ANCHORS`.
    """

    response: TitleKeywordsLLM | None
    status: str


# Why a pointer answer's abstract was not used, by status.
UNUSED_ANCHORS = {
    "unplaced": "The abstract's first and last words were not found in the front matter",
    "cut_short": "The abstract's last words close a sentence its paragraph goes on after",
    "overran": "The abstract's last words lie past the keywords or a box printed after it",
}


def resolve_title_keywords(answer: TitleKeywordsPointerLLM, source: str) -> ResolvedTitleKeywords:
    """Copy the abstract *answer* points at out of *source*, the exact text
    the model was shown, and return the verbatim response.

    Last words that close a sentence inside the abstract, or that lie past
    the keyword list or a box printed after it, mark a wrong span; the copy is
    not made and the status says why."""
    fields = answer.model_dump(exclude={"abstract_first_words", "abstract_last_words"})
    first, last = answer.abstract_first_words, answer.abstract_last_words
    if first is None and last is None:
        # An explicit null keeps its meaning, the model found no abstract; an
        # omitted or blank answer stays omitted, as in the verbatim response.
        if answer._abstract_explicitly_absent:
            fields["abstract"] = None
        return ResolvedTitleKeywords(TitleKeywordsLLM.model_validate(fields), "absent")
    span = locate_abstract(source, first or "", last or "")
    if span is not None and _runs_on(source, span.end):
        return ResolvedTitleKeywords(None, "cut_short")
    if span is not None and _runs_past(source, span, answer.keywords):
        return ResolvedTitleKeywords(None, "overran")
    abstract = abstract_text(source, span) if span is not None else ""
    if span is None or not abstract:
        return ResolvedTitleKeywords(None, "unplaced")
    return ResolvedTitleKeywords(
        TitleKeywordsLLM.model_validate({**fields, "abstract": abstract}),
        "copied_fuzzy" if span.fuzzy else "copied",
    )


def _runs_on(source: str, end: int) -> bool:
    """Whether the paragraph that holds the abstract's end goes on after *end*
    with more of a sentence: a lowercase word (the last words stopped inside a
    sentence), or a capitalised word and a lowercase one (another sentence). A
    copyright, citation or keyword line can follow on the same line."""
    rest = _BLOCK_BREAK_RE.split(source[end:], maxsplit=1)[0].lstrip()
    if _FURNITURE_BLOCK_RE.match(rest) or _AFTER_ABSTRACT_RE.match(rest):
        return False
    words = [word.strip(",;:.") for word in rest.split(maxsplit=2)[:2]]
    if not words or not words[0].isalpha():
        return False
    if words[0].islower():
        return True
    return len(words) == 2 and words[0][0].isupper() and words[1].isalpha() and words[1].islower()


def _runs_past(source: str, span: AbstractSpan, keywords: list[str]) -> bool:
    """Whether a block of *span* after its first opens the keyword list or a
    box printed after the abstract, or is the keyword list without a label."""
    blocks = _BLOCK_BREAK_RE.split(source[span.start : span.end])[1:]
    return any(
        _AFTER_ABSTRACT_RE.match(block) or _is_keyword_list(block, keywords) for block in blocks
    )


def _is_keyword_list(block: str, keywords: list[str]) -> bool:
    """Whether *block* is mostly the answer's *keywords*."""
    squashed = _squash(block)[0]
    found = {word for word in (_squash(keyword)[0] for keyword in keywords) if len(word) >= 3}
    found = {word for word in found if word in squashed}
    return len(found) >= 2 and sum(map(len, found)) >= _KEYWORD_LIST_SHARE * len(squashed)


def _is_stand_in(entry: str) -> bool:
    return bool(_STAND_IN_RE.fullmatch(entry.strip()) or _NOT_PRINTED_RE.search(entry))


def expand_authors(answer: AuthorsPointerLLM) -> AuthorsLLM | None:
    """Write each author's affiliations back out from the numbered list.

    Numbers count from 1. A list that uses 0 and never the last number was
    counted from 0, so it is shifted. A number outside the list is dropped, and
    so is one that points at a stand-in the model wrote for an affiliation the
    text does not print ("Affiliation 3", "N/A"): the verbatim answer leaves
    such an affiliation empty too.

    ``None`` when authors give numbers but the list is empty: the answer lost
    its affiliations, and the verbatim call is asked instead.
    """
    numbers = [n for author in answer.authors for n in author.affiliation_ids]
    if numbers and not answer.affiliations:
        return None
    names = ["" if _is_stand_in(name) else name.strip() for name in answer.affiliations]
    stand_ins = sum(1 for name in answer.affiliations if name.strip() and _is_stand_in(name))
    shift = 1 if 0 in numbers and len(names) not in numbers else 0
    authors: list[AuthorLLM] = []
    dropped = 0
    for author in answer.authors:
        affiliation: list[str] = []
        for number in author.affiliation_ids:
            index = number + shift - 1
            if not 0 <= index < len(names):
                dropped += 1
            elif names[index] and names[index] not in affiliation:
                affiliation.append(names[index])
        authors.append(
            AuthorLLM(
                given=author.given,
                family=author.family,
                affiliation="; ".join(affiliation),
                email=author.email,
                corresponding=author.corresponding,
                orcid=author.orcid,
            )
        )
    if dropped:
        logger.warning(
            "Author answer named %d affiliation number(s) outside its list of %d",
            dropped,
            len(names),
        )
    if stand_ins:
        logger.warning(
            "Author answer wrote %d stand-in(s) for affiliations the text does not print; "
            "left empty",
            stand_ins,
        )
    return AuthorsLLM(authors=authors)
