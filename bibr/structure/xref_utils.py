"""Shared cross-reference and URL detection utilities.

Used by PDFParser to detect table/figure/supplementary/equation/section
cross-references and plain-text URLs in sentences.
"""

import logging
import re

from bibr.paper_contents import PaperFigure, PaperSentence, PaperTable, PaperXref
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


# ── Number separator used by table/figure/equation/section compound refs ──
# Matches: "1, 2 & 3", "1 and 2", "1-3"
_NUM_LIST = r"(?:\s*(?:[,&]|\band\b)\s*\d+)"
_NUM_RANGE = r"(?:\s*[-–]\s*\d+)?"

# ── Table / Figure xref patterns ──────────────────────────────────────────
# The letter lookbehind blocks substring matches — author names
# ("Constable 2014" → "table 2014"), suffix words with glued footnote
# markers ("stable11" → "table11"), URL path segments
# ("ditctab20121en.pdf" → "tab20121"), "config 3" → "fig 3" — while
# still allowing an OCR-flattened marker glued to the word ("9Table 1").
# Mirrors EQUATION_XREF_RE.

TABLE_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?:Tables?|Tab\.?|Tbl\.?)\s*"
    r"(\d+" + _NUM_RANGE + _NUM_LIST + r"*)",
    re.IGNORECASE,
)

FIGURE_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?:Figures?|Fig\.?|Figs\.?)\s*"
    r"(\d+(?:[a-z](?![a-z]))?"
    r"(?:\s*[-–]\s*\d+(?:[a-z](?![a-z]))?)?"
    r"(?:\s*(?:[,&]|\band\b)\s*\d+(?:[a-z](?![a-z]))?)*)",
    re.IGNORECASE,
)

# ── Supplementary xref patterns ───────────────────────────────────────────
# Matches: Table S1, Fig. S2, Supplementary Table 1, Supplemental Figure 3,
#          Supplementary Material, Supplemental Material, Online Supplement
# Supp/section ids are unvalidated against real ids, so the lookbehind is
# the only defense against substring matches ("unstable S1", "config S2").
SUPP_TABLE_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?:Tables?|Tab\.?|Tbl\.?)\s*S(\d+" + _NUM_RANGE + _NUM_LIST + r"*)",
    re.IGNORECASE,
)

SUPP_FIGURE_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?:Figures?|Fig\.?|Figs\.?)\s*S(\d+(?:[a-z](?![a-z]))?"
    r"(?:\s*[-–]\s*\d+(?:[a-z](?![a-z]))?)?"
    r"(?:\s*(?:[,&]|\band\b)\s*\d+(?:[a-z](?![a-z]))?)*)",
    re.IGNORECASE,
)

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
# Dotted ids are captured whole so "Eq. (2.3)" isn't truncated to
# "Eq. (2".  The close paren is consumed only when the open paren was
# matched, so an enclosing parenthetical — "(see Equations 1 and 7)" —
# keeps its own ")".
_EQ_NUM = r"\d+(?:\.\d+)*"
_EQ_NUM_RANGE = rf"(?:\s*[-–]\s*{_EQ_NUM})?"
_EQ_NUM_LIST = rf"(?:\s*(?:[,&]|\band\b)\s*{_EQ_NUM}{_EQ_NUM_RANGE})*"
EQUATION_XREF_RE = re.compile(
    r"(?<![A-Za-z])(?:Equations?|Eqs?\.?)\s*(?P<open>\()?"
    rf"(?P<nums>{_EQ_NUM}{_EQ_NUM_RANGE}{_EQ_NUM_LIST})"
    r"(?(open)\)?)",
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

# Helper to parse number references like "1, 2, 3" or "1-3"
NUM_SEP_RE = re.compile(r"\d+")


def _normalize_xref_text(text: str) -> str:
    """Collapse internal whitespace (line breaks, NBSPs) inside xref content."""
    return collapse_ws(text)


def _expand_nums(num_str: str) -> list[int]:
    """Expand a number sequence like ``"5-7"`` or ``"1, 3"`` into individual ints.

    Unlike bare ``NUM_SEP_RE.findall`` this properly fills in ranges so that
    ``"5-7"`` yields ``[5, 6, 7]`` rather than ``[5, 7]``.
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

    # Build lookup: table_id / figure_id → validate that the ID exists.
    # Papers reference tables/figures by their ID (e.g., "Table 3" means table_id=3),
    # not by ordinal position in the list.
    tbl_id_set = {tbl.table_id for tbl in tables}
    fig_id_set = {fig.figure_id for fig in figures}

    for sent in sentences:
        # Display formulas are exported as "[equation]" placeholders — any
        # match inside the raw math (e.g. "\leq 1", "\tag{2}") would anchor
        # an xref to text the consumer never sees.
        if sent.is_display_formula:
            continue

        # Table xrefs
        for m in TABLE_XREF_RE.finditer(sent.text):
            nums = _expand_nums(m.group(1))
            for num in nums:
                if num in tbl_id_set:
                    xrefs.append(
                        PaperXref(
                            xref_id=num,
                            xref_type="table",
                            contents=_normalize_xref_text(m.group(0)),
                            text_id=sent.text_id,
                        )
                    )

        # Figure xrefs
        for m in FIGURE_XREF_RE.finditer(sent.text):
            nums = _expand_nums(m.group(1))
            for num in nums:
                if num in fig_id_set:
                    xrefs.append(
                        PaperXref(
                            xref_id=num,
                            xref_type="figure",
                            contents=_normalize_xref_text(m.group(0)),
                            text_id=sent.text_id,
                        )
                    )

        # Supplementary table xrefs (Table S1, etc.) — no validation
        for m in SUPP_TABLE_XREF_RE.finditer(sent.text):
            nums = _expand_nums(m.group(1))
            for num in nums:
                xrefs.append(
                    PaperXref(
                        xref_id=num,
                        xref_type="supplementary",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
                    )
                )

        # Supplementary figure xrefs (Fig. S1, etc.) — no validation
        for m in SUPP_FIGURE_XREF_RE.finditer(sent.text):
            nums = _expand_nums(m.group(1))
            for num in nums:
                xrefs.append(
                    PaperXref(
                        xref_id=num,
                        xref_type="supplementary",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
                    )
                )

        # Supplementary named refs (Supplementary Table 1, Supplemental Material)
        for m in SUPP_NAMED_XREF_RE.finditer(sent.text):
            if m.group(1):
                nums = _expand_nums(m.group(1))
                for num in nums:
                    xrefs.append(
                        PaperXref(
                            xref_id=num,
                            xref_type="supplementary",
                            contents=_normalize_xref_text(m.group(0)),
                            text_id=sent.text_id,
                        )
                    )
            else:
                xrefs.append(
                    PaperXref(
                        xref_id=0,
                        xref_type="supplementary",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
                    )
                )

        # Equation xrefs — no validation
        for m in EQUATION_XREF_RE.finditer(sent.text):
            nums = _expand_nums(m.group("nums"))
            for num in nums:
                xrefs.append(
                    PaperXref(
                        xref_id=num,
                        xref_type="equation",
                        contents=_normalize_xref_text(m.group(0)),
                        text_id=sent.text_id,
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
                    )
                )

    return xrefs
