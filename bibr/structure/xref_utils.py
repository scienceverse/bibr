"""Shared cross-reference and URL detection utilities.

Used by PDFParser to detect table/figure/supplementary/equation/section
cross-references and plain-text URLs in sentences.
"""

import logging
import re

from bibr.paper_contents import PaperFigure, PaperSentence, PaperTable, PaperXref
from bibr.structure.float_labels import (
    FIGURE_WORD,
    LETTERED_LABEL,
    NUMBERED_LABEL,
    SUPPLEMENT_WORD,
    TABLE_WORD,
    is_supplementary_label,
    normalize_label,
    printed_label,
)
from bibr.utils.text import URL_RE as URL_RE  # URL_RE re-exported here
from bibr.utils.text import collapse_ws

logger = logging.getLogger(__name__)

# Widest ``N-M`` span that is filled in element-by-element. A compound
# reference ("Tables 1-3", "[4-7]") is expanded so every id gets its own
# link, and the input is document text — so an unbounded fill lets one
# crafted or badly-OCR'd sentence ("Table 1-999999999") materialise a
# billion ints, tens of GB of RAM, from a single region. Real documents
# never enumerate more than a handful; a wider span is treated as a typo or
# a false positive and only its endpoints are kept.
MAX_INT_RANGE_SPAN = 64


def expand_int_range(lo: int, hi: int) -> list[int]:
    """Fill ``lo..hi`` inclusive, refusing to materialise an implausible span."""
    if hi - lo > MAX_INT_RANGE_SPAN:
        logger.debug("Refusing to expand implausible numeric range %d-%d", lo, hi)
        return [lo, hi]
    return list(range(lo, hi + 1))


# ── Number separator used by supplementary/equation/section compound refs ──
# Matches: "1, 2 & 3", "1 and 2", "1-3"
_NUM_LIST = r"(?:\s*(?:[,&]|\band\b)\s*\d+)"
_NUM_RANGE = r"(?:\s*[-–]\s*\d+)?"

# ── Table / Figure xref patterns ──────────────────────────────────────────
# A mention is the word and one or more printed labels (see
# ``bibr.structure.float_labels``): "Table 3", "Tables 3.1 and 3.2", "Fig. 2a",
# "Figs. S1–S3", "Tables II and III", "Supplementary Table 4". A list holds
# numbered or lettered labels, not both, so the panels of "Fig. 2A, B" are not
# read as a figure "B".
# The letter lookbehind blocks substring matches — author names
# ("Constable 2014" → "table 2014"), suffix words with glued footnote
# markers ("stable11" → "table11"), URL path segments
# ("ditctab20121en.pdf" → "tab20121"), "config 3" → "fig 3", "unstable S1" —
# while still allowing an OCR-flattened marker glued to the word
# ("9Table 1"). Mirrors EQUATION_XREF_RE.
_LABEL_SEP = r"\s*(?:[-–]|,?\s*(?:&|\band\b)|,)\s*"


def _float_xref_re(word: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z])(?P<supplement>{SUPPLEMENT_WORD})?(?P<word>{word})\s*"
        rf"(?P<labels>{NUMBERED_LABEL}(?:{_LABEL_SEP}{NUMBERED_LABEL})*"
        rf"|{LETTERED_LABEL}(?:{_LABEL_SEP}{LETTERED_LABEL})*)",
        re.IGNORECASE,
    )


TABLE_XREF_RE = _float_xref_re(TABLE_WORD)
FIGURE_XREF_RE = _float_xref_re(FIGURE_WORD)

# Re-reads a mention's label list one label at a time, the separator before
# each one telling a range ("1–3") from a list ("1 and 3").
_FIRST_NUMBERED_RE = re.compile(NUMBERED_LABEL, re.IGNORECASE)
_FIRST_LETTERED_RE = re.compile(LETTERED_LABEL, re.IGNORECASE)
_NEXT_NUMBERED_RE = re.compile(rf"(?P<sep>{_LABEL_SEP})(?P<label>{NUMBERED_LABEL})", re.IGNORECASE)
_NEXT_LETTERED_RE = re.compile(rf"(?P<sep>{_LABEL_SEP})(?P<label>{LETTERED_LABEL})", re.IGNORECASE)

# A caption that marks its float as a later piece of an earlier one: a
# bracketed marker near the label ("Table 3 (continued)", "FIGURE 2 (Cont.)"),
# or nothing but the label and the word ("Table 3 continued", "Figure 3.
# Cont."). Such a piece repeats the label when it was not merged into the first.
_CONTINUED_CAPTION_RE = re.compile(
    r"[(\[]\s*cont(?:inued|'d|d|\.)?\s*[)\]]"
    r"|^\s*(?:tables?|fig(?:ure)?s?\.?)\s*\S+?\s*[.:]?\s*cont(?:inued|'d|d|\.)\s*$",
    re.IGNORECASE,
)
# A label split into what precedes its last number and that number:
# "S12" → ("S", "12"), "3.1" → ("3.", "1").
_LABEL_NUMBER_RE = re.compile(r"(?P<head>.*?)(?P<number>\d+)")
_LETTER_PREFIX_RE = re.compile(r"[A-Za-z][.-]?")

# How a figure or table xref was resolved, exported as
# ``extraction.diagnostics.xref_tier``: by the float's printed label, or — when
# no float of that kind has a label — by taking the printed number as the
# float's position.
LABEL_TIER = "label"
POSITION_TIER = "position"

# ── Supplementary xref patterns ───────────────────────────────────────────
# Matches: Supplementary Material, Supplemental Data 2, Supplementary Tables.
# A supplement named by a table or figure label ("Table S1", "Supplementary
# Figure 3") is read by the table/figure patterns above instead.
# Supp ids are unvalidated against real ids, so the lookbehind is the only
# defense against substring matches.
SUPP_NAMED_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?:(?:Online\s+)?Supplementa(?:ry|l))\s+"
    r"(?:Tables?|Figures?|Fig\.?|Materials?|Information|Data|Methods?|Results?|Appendix)"
    r"(?:\s*(\d+" + _NUM_RANGE + _NUM_LIST + r"*))?",
    re.IGNORECASE,
)

# ── Equation xref patterns ────────────────────────────────────────────────
# Matches: Equation 3, Eq. 5, Eqs. 1-3, Equation (3), eq.(1), Eq. (2.3)
# The letter lookbehind blocks substring matches inside raw LaTeX and URLs
# ("\leq 1" → "eq 1", "osf.io/geq9x" → "eq9") while still allowing an
# OCR-flattened footnote marker glued to the word ("9Equations (1)").
# A hyphen before a lowercase short form is rejected too: unit spellings
# such as "CO2-eq. (39.1%)" would otherwise match from the "eq". A hyphen
# before longhand "Equation" is a range dash ("Equation 5-Equation 7"),
# so only the lowercase "eq" shape is refused. (An en dash is still
# allowed everywhere: longhand ranges print "Equation 5–Equation 7".)
# Dotted ids are captured whole so "Eq. (2.3)" isn't truncated to
# "Eq. (2".  The close paren is consumed only when the open paren was
# matched, so an enclosing parenthetical — "(see Equations 1 and 7)" —
# keeps its own ")". A trailing percent sign is rejected after the numbers
# ("eq. (39.1%)" is a share, not an equation).
# The match's word is captured so detect_xrefs can drop version-printed
# software ("EQS 6", "EQS 6.1", see _is_software_version_mention).
_EQ_NUM = r"\d+(?:\.\d+)*"
_EQ_NUM_RANGE = rf"(?:\s*[-–]\s*{_EQ_NUM})?"
_EQ_NUM_LIST = rf"(?:\s*(?:[,&]|\band\b)\s*{_EQ_NUM}{_EQ_NUM_RANGE})*"
EQUATION_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?<!-(?=(?-i:eq)))(?P<word>(?:Equations?|Eqs?\.?))\s*(?P<open>\()?"
    rf"(?P<nums>{_EQ_NUM}{_EQ_NUM_RANGE}{_EQ_NUM_LIST})"
    r"(?(open)\)?)"
    r"(?![\d.]*\s*%)",
    re.IGNORECASE,
)

# ── Section xref patterns ─────────────────────────────────────────────────
# Matches: Section 3, Sections 2 and 3, Subsection 2.4, §2.1, §§3.1-3.3
# The lookbehind blocks substring matches (journal name "Intersections 4:
# 26–50" → "sections 4"), but "Subsection 2.4" is a genuine section
# reference, so the "Sub" prefix is allowed explicitly.
SECTION_XREF_RE = re.compile(
    r"(?:(?<![A-Za-z])(?:Sub)?Sections?|§§?)\s*"
    r"(\d+(?:\.\d+)*"
    r"(?:\s*[-–]\s*\d+(?:\.\d+)*)?"
    r"(?:\s*(?:[,&]|\band\b)\s*\d+(?:\.\d+)*)*)",
    re.IGNORECASE,
)

# Every pattern above requires one of these literals (case-insensitively):
# "Table"/"Tab."→tab, "Tbl."→tbl, "Figure"/"Fig."→fig, "Supplementary"/
# "Supplemental"→supplementa, "Equation"/"Eq."→eq, "Section"/"Subsection"→
# section, and the section sign. A necessary condition, so a sentence that
# fails it can skip every pass — ~6.5 ms/paper of the shared serve event
# loop, verified match-identical on 200 real papers.
_XREF_PRESCAN_RE = re.compile(r"tab|tbl|fig|supplementa|eq|section|§", re.IGNORECASE)

# Helper to parse number references like "1, 2, 3" or "1-3"
NUM_SEP_RE = re.compile(r"\d+")


def _normalize_xref_text(text: str) -> str:
    """Collapse internal whitespace (line breaks, NBSPs) inside xref content."""
    return collapse_ws(text)


def _expand_nums(num_str: str) -> list[int]:
    """Expand a number sequence like ``"5-7"`` or ``"1, 3"`` into individual ints.

    Unlike bare ``NUM_SEP_RE.findall`` this properly fills in ranges so that
    ``"5-7"`` yields ``[5, 6, 7]`` rather than ``[5, 7]``. A reversed range
    (``"5-3"``) yields ``[]``: it is read as a typo or false positive, not
    filled ascending (that recovery lives with the citation linker).
    """
    result: list[int] = []
    # Split on list separators first (comma, ampersand, "and")
    parts = re.split(r"\s*(?:[,&]|\band\b)\s*", num_str)
    for part in parts:
        part = part.strip()
        if "." in part:
            # Hierarchical ids like "2.3" (or "2.1-2.3"): keep the major
            # number, mirroring section xref semantics; no range fill.
            for m in re.finditer(r"(\d+)(?:\.\d+)+", part):
                major = int(m.group(1))
                if major not in result:
                    result.append(major)
            continue
        # Each part may be a single number or a range like "5-7"
        range_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", part.strip())
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            result.extend(expand_int_range(lo, hi))
        else:
            digits = NUM_SEP_RE.findall(part)
            result.extend(int(d) for d in digits)
    return result


def _label_range(first: str, last: str) -> list[str]:
    """The labels from *first* to *last*: "1"–"3", "S1"–"S3" (and "S1"–"3"),
    "3.1"–"3.3". Any other pair — or an implausible span — keeps its ends."""
    lo = _LABEL_NUMBER_RE.fullmatch(first)
    hi = _LABEL_NUMBER_RE.fullmatch(last)
    if lo is None or hi is None:
        return [first, last]
    head = lo.group("head")
    hi_head = hi.group("head")
    if not hi_head and _LETTER_PREFIX_RE.fullmatch(head):
        hi_head = head  # "Tables S1–3"
    lo_number, hi_number = int(lo.group("number")), int(hi.group("number"))
    if hi_head.casefold() != head.casefold() or lo_number >= hi_number:
        return [first, last]
    return [f"{head}{number}" for number in expand_int_range(lo_number, hi_number)]


def _mention_labels(match: re.Match[str]) -> list[tuple[str, bool]] | None:
    """The labels a table/figure mention prints, with ranges filled in.

    Each label comes with whether it can only name a supplement: after
    "Supplementary" a label that is neither a number (read as "S<n>") nor
    "S"-prefixed ("Supplementary Table A1"). ``None`` for a lettered label
    after a lowercase word ("the table I made"), which is prose.
    """
    text = match.group("labels")
    first = _FIRST_NUMBERED_RE.match(text)
    following = _NEXT_NUMBERED_RE
    if first is None:
        if match.group("word")[0].islower():
            return None
        first = _FIRST_LETTERED_RE.match(text)
        following = _NEXT_LETTERED_RE
    if first is None:  # the patterns above always leave one; typing guard only
        return None
    labels = [printed_label(first.group(0))]
    position = first.end()
    while (step := following.match(text, position)) is not None:
        label = printed_label(step.group("label"))
        if step.group("sep").strip() in {"-", "–"}:
            labels[-1:] = _label_range(labels[-1], label)
        else:
            labels.append(label)
        position = step.end()
    if not match.group("supplement"):
        return [(label, False) for label in labels]
    return [
        (
            printed_label(label, supplement=True),
            not label[:1].isdigit() and not is_supplementary_label(label),
        )
        for label in labels
    ]


def _label_candidates(label: str) -> list[str]:
    """*label*, then — for a letter after a number ("2a") — the label without
    it, so a panel mention ("Figure 2a") finds figure "2"."""
    if len(label) > 1 and label[-1].isalpha() and label[-2].isdecimal():
        return [label, label[:-1]]
    return [label]


def _printed_number(label: str) -> int:
    """The first number *label* prints ("S2" → 2), or 0: what a supplementary
    xref records, which no exported row is keyed by."""
    number = NUM_SEP_RE.search(label)
    return int(number.group()) if number else 0


def _is_software_version_mention(match: re.Match[str]) -> bool:
    """Whether an equation-pattern match names versioned software, not an equation.

    Structural-equation software is cited bare and versioned ("EQS 6.1",
    "EQS 6", Bentler): an all-caps short plural with no period. Real plural
    references print "Eqs." or "eqs", and real bare short forms ("eq 5",
    "eqs 4 and 8", "Eq (1)") take plain integers — so the caps-no-period
    plural is dropped whatever the number shape, while the dotted-number
    rule still covers the singular caps form ("EQ 2.3").
    """
    word = match.group("word")
    if word == "EQS":
        return True
    if word == "EQ":
        return "." in match.group("nums")
    return False


class _FloatIndex:
    """Where mentions of one kind of float (tables, or figures) point.

    ``by_label`` maps each normalized printed label to the ids of the floats
    printed with it. ``by_position`` lists the ids by page (a float without a
    page first), then in the order the parser produced them.
    """

    def __init__(self, floats: list[tuple[int, str | None, int | None, str | None]]) -> None:
        self.by_label: dict[str, list[int]] = {}
        self._continued: set[int] = set()
        for float_id, label, _page, caption in floats:
            if label:
                self.by_label.setdefault(normalize_label(label), []).append(float_id)
            if caption and _CONTINUED_CAPTION_RE.search(caption[:80]):
                self._continued.add(float_id)
        ordered = sorted(floats, key=lambda item: item[2] or 0)
        self.by_position = [item[0] for item in ordered]

    def _named(self, ids: list[int]) -> int:
        """The float a label names: the only one printed with it or, when
        unmerged continuation pieces repeat the label, the only one that is not
        a continuation; else 0."""
        if len(ids) == 1:
            return ids[0]
        firsts = [float_id for float_id in ids if float_id not in self._continued]
        return firsts[0] if len(firsts) == 1 else 0

    def resolve(self, label: str, kind: str) -> tuple[str, int, str | None]:
        """``(xref_type, xref_id, tier)`` for a mention of *kind* printing *label*.

        When any float of the kind has a label, only the label decides: one
        float printed with it is the target; none, or two (ambiguous), leave
        the xref without one (``xref_id`` 0, exported as a null
        ``target_id``). When none has, a plain number is the float's position.
        An "S"-prefixed label that no float carries names a supplement.
        """
        supplementary = is_supplementary_label(label)
        if self.by_label:
            ids: list[int] = next(
                (
                    found
                    for candidate in _label_candidates(label)
                    if (found := self.by_label.get(normalize_label(candidate)))
                ),
                [],
            )
            if ids or not supplementary:
                return kind, self._named(ids) if ids else 0, LABEL_TIER
        elif not supplementary:
            number = next(
                (int(candidate) for candidate in _label_candidates(label) if candidate.isdecimal()),
                None,
            )
            in_range = number is not None and 1 <= number <= len(self.by_position)
            return kind, self.by_position[number - 1] if in_range else 0, POSITION_TIER
        return "supplementary", _printed_number(label), None


def detect_xrefs(
    sentences: list[PaperSentence],
    tables: list[PaperTable],
    figures: list[PaperFigure],
) -> list[PaperXref]:
    """Detect cross-references to tables, figures, supplementary materials,
    equations, and sections in sentences.

    Scans sentence text for patterns like "Table 1", "Figure 3", "Table S1",
    "Eq. 5", "Section 2", etc. and creates PaperXref objects linking each
    mention to the referenced item.

    A table or figure mention resolves by the floats' printed labels
    (``PaperTable.label``/``PaperFigure.label``): "Table 3.1" links the table
    captioned "Table 3.1", whatever its id. Only when no float of that kind
    has a label is the printed number taken as a position: "Figure 2" is then
    the second figure by page. Every mention yields a row, with ``xref_id`` 0
    when it names no float or two, and ``tier`` saying which way it resolved.

    Footnote xrefs are created separately by ``PDFParser.create_content_sections``.

    Parameters
    ----------
    sentences : list[PaperSentence]
        All sentences to scan for cross-references.
    tables : list[PaperTable]
        Tables available for cross-referencing.
    figures : list[PaperFigure]
        Figures available for cross-referencing.

    Returns
    -------
    list[PaperXref]
        Detected cross-references.
    """
    xrefs: list[PaperXref] = []

    float_patterns = (
        (
            "table",
            TABLE_XREF_RE,
            _FloatIndex([(t.table_id, t.label, t.page_number, t.caption) for t in tables]),
        ),
        (
            "figure",
            FIGURE_XREF_RE,
            _FloatIndex([(f.figure_id, f.label, f.page_number, f.caption) for f in figures]),
        ),
    )

    for sent in sentences:
        # Display formulas are exported as "[equation]" placeholders — any
        # match inside the raw math (e.g. "\leq 1", "\tag{2}") would anchor
        # an xref to text the consumer never sees.
        if sent.is_display_formula:
            continue

        if not _XREF_PRESCAN_RE.search(sent.text):
            continue

        # Table and figure xrefs, supplements named by a label included
        # ("Table S1", "Supplementary Figure 2"): one row per printed label.
        float_spans: list[tuple[int, int]] = []
        for kind, pattern, index in float_patterns:
            for m in pattern.finditer(sent.text):
                labels = _mention_labels(m)
                if labels is None:
                    continue
                float_spans.append(m.span())
                for label, supplement_only in labels:
                    xref_type, xref_id, tier = (
                        ("supplementary", _printed_number(label), None)
                        if supplement_only
                        else index.resolve(label, kind)
                    )
                    xrefs.append(
                        PaperXref(
                            xref_id=xref_id,
                            xref_type=xref_type,
                            contents=_normalize_xref_text(m.group(0)),
                            text_id=sent.text_id,
                            tier=tier,
                            start=m.start(),
                            end=m.end(),
                        )
                    )

        # Supplementary named refs (Supplemental Material, Supplementary Data 2)
        # that no table/figure mention above already covers.
        for m in SUPP_NAMED_XREF_RE.finditer(sent.text):
            if any(start < m.end() and m.start() < end for start, end in float_spans):
                continue
            if m.group(1):
                nums = _expand_nums(m.group(1))
                for num in nums:
                    xrefs.append(
                        PaperXref(
                            xref_id=num,
                            xref_type="supplementary",
                            contents=_normalize_xref_text(m.group(0)),
                            text_id=sent.text_id,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
            else:
                xrefs.append(
                    PaperXref(
                        xref_id=0,
                        xref_type="supplementary",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Equation xrefs — no validation
        for m in EQUATION_XREF_RE.finditer(sent.text):
            if _is_software_version_mention(m):
                continue
            nums = _expand_nums(m.group("nums"))
            for num in nums:
                xrefs.append(
                    PaperXref(
                        xref_id=num,
                        xref_type="equation",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Section xrefs — no validation
        for m in SECTION_XREF_RE.finditer(sent.text):
            # For section refs like "§2.1", extract the major number as xref_id
            nums = [int(n) for n in re.findall(r"\d+", m.group(1).split(",")[0].split("&")[0])]
            if nums:
                xrefs.append(
                    PaperXref(
                        xref_id=nums[0],
                        xref_type="section",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
                        start=m.start(),
                        end=m.end(),
                    )
                )

    return xrefs
