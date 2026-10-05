"""The outline guard: whether a PDF's outline is a usable heading tree.

A port of the junk-outline guard the popo evaluation froze on 2026-10-03
(``outline-guard-v1``, scripts/outline_guard.py), with its rules, patterns and
order unchanged. The evaluation ran it on what pypdfium2 reads: bibr's own
outline reader (``bibr.input.pdf_outline.extract_pdf_outline``, on pypdfium2's
table of contents) and each page's ``get_text_bounded()``. Here it runs on the
layer's outline (pdfium's bookmarks, walked by :mod:`bibr.document.outline`) and
the layer's own text.

Entry cleanup drops an entry (its children are not re-parented: the guard only
counts the entries that stay):

- C1 blank: the title has no letter or digit
- C2 page: a page label ("12", "iv", "Page 3", "3 of 10")
- C3 nav: a navigation or container label ("Cover", "Contents", "Untitled") or
  a file name ("x.pdf", "Microsoft Word - x.docx")
- C4 float: a float bookmark ("Figure 2", "Table 1", "Box 2")
- C5 title: the paper's own title (the /Title, letters and digits only, at
  least 10 of them)
- C6 wrapper: after C1-C5, a single first entry that is the only one at the top
  level with everything else under it (a title or journal wrapper)

The outline is rejected when:

- R1 too few: fewer than 3 entries are left
- R2 targets: more than half the entries have no page in the document, or the
  PDF has more than 2 pages and every entry points at the same page
- R3 ungrounded: the text layer holds at least 2,000 letters and digits and
  fewer than half the entries are printed on their target page or the page
  either side (an entry without a page: anywhere in the document); this
  catches issue tables of contents and renamed bookmarks

Differences from the evaluation, which a reader comparing numbers must know:

- The grounding key. The evaluation grounded an entry on both the letters and
  digits of its title and those of its title with the leading numbering marker
  stripped (``bibr.input.pdf_outline._strip_marker``: "1.2 ", "Section 3",
  "(a)"); this port grounds on the title's letters and digits alone. An entry
  whose printed form lacks the marker its bookmark carries is grounded there and
  not here, so the port can reject (R3) an outline the evaluation kept.
- No pipeline title. The evaluation's C5 also dropped an entry that repeats the
  title the pipeline extracted; the layer is built before any title is
  extracted, so the PDF's /Title is the only title it knows, and an entry that
  repeats the extracted title but not the /Title stays.
- The input. The text is the layer's records, not pypdfium2's page text, and
  the entries are the layer's own walk of the bookmarks (every depth, none
  filtered), not ``extract_pdf_outline``'s.
- The scope. The layer gives a verdict only where it holds the text of every
  page of the document. On a layer built for a page range (or with a page it
  could not read) R3 would see too little: of the gate192 outlines that pass on
  the whole document it rejected 30 of 97 with the first half of the pages built
  and 47 of 90 with the first five, and the 2,000-character gate lets a junk
  outline through. Such a layer keeps a rejection by R1 or R2, which read no
  text, and leaves a pass at None.

The evaluation's measured precision is therefore the evaluation's: measure the
guard again on the layer's outline before quoting it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from bibr.document.model import Decided, OutlineEntry, OutlineGuard, Page, PageColumns

# The rule version is the one the popo evaluation froze as outline-guard-v1.
GUARD_VERSION = "outline_guard/1"

_PAGE_RX = re.compile(
    r"^\s*(?:(?:page|pages|pg|pp|p|seite|pagina|página)\.?\s*)?"
    r"(?:\d+|(?=[ivxlcdm]+\b)m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3}))"
    r"(?:\s*(?:of|/|-|–)\s*\d+)?\s*$",
    re.I,
)
_NAV = {
    "cover",
    "front cover",
    "back cover",
    "cover page",
    "title page",
    "titlepage",
    "title",
    "front matter",
    "frontmatter",
    "back matter",
    "backmatter",
    "contents",
    "content",
    "table of contents",
    "toc",
    "bookmarks",
    "bookmark",
    "untitled",
    "document",
    "article",
    "full text",
    "fulltext",
    "pdf",
    "copyright",
    "copyright page",
    "blank page",
    "blank",
    "start",
    "top",
    "home",
}
_FILE_RX = re.compile(
    r"\.(pdf|docx?|tex|rtf|odt|indd)\s*$|^\s*microsoft\s+(word|powerpoint)\b", re.I
)
_FLOAT_RX = re.compile(
    r"^\s*(?:supplementary\s+|suppl\.?\s+)?(?:fig(?:ure)?s?|tables?|tab|scheme|chart|plate|box|exhibit|graph)"
    r"\.?\s*[a-z]?\d+",
    re.I,
)

# Whatever is not a letter or a digit; the underscore is the one character \w
# adds to those, so [\W_] is exactly the complement of str.isalnum.
_NOT_ALNUM = re.compile(r"[\W_]+")


def alnum(text: str | None) -> str:
    """*text* folded (NFKC, case) down to its letters and digits."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _NOT_ALNUM.sub("", folded)


def _nav_key(title: str) -> str:
    folded = unicodedata.normalize("NFKC", title or "").casefold()
    return re.sub(r"[\W_]+", " ", folded).strip()


def entry_rule(title: str, title_keys: set[str]) -> str | None:
    """The cleanup rule that drops an entry titled *title*, or None."""
    if not alnum(title):
        return "C1_blank"
    if _PAGE_RX.match(title):
        return "C2_page"
    if _nav_key(title) in _NAV or _FILE_RX.search(title):
        return "C3_nav"
    if _FLOAT_RX.match(title):
        return "C4_float"
    folded = alnum(title)
    if len(folded) >= 10 and folded in title_keys:
        return "C5_title"
    return None


class Text(Protocol):
    """What R3 asks of the document's text."""

    @property
    def chars(self) -> int:
        """Letters and digits in the whole document."""

    def grounded(self, entry: OutlineEntry) -> bool:
        """Whether *entry*'s title is printed on its target page or the page either side."""


class PageText:
    """The text of a layer's pages for R3, folded to letters and digits and read on first use."""

    def __init__(self, pages: Sequence[Page], n_pages: int) -> None:
        self._pages = pages
        self._n_pages = n_pages
        self._folded: list[str] | None = None
        self._whole: str | None = None

    @property
    def folded(self) -> list[str]:
        """Each page's folded text by 0-based page index (empty for a page the layer lacks)."""
        if self._folded is None:
            folded = [""] * self._n_pages
            for page in self._pages:
                if page.cols is not None and 0 <= page.index < self._n_pages:
                    folded[page.index] = alnum(_records_text(page.cols))
            self._folded = folded
        return self._folded

    @property
    def whole(self) -> str:
        if self._whole is None:
            self._whole = "".join(self.folded)
        return self._whole

    @property
    def chars(self) -> int:
        return sum(len(text) for text in self.folded)

    def grounded(self, entry: OutlineEntry) -> bool:
        # The evaluation also probed the title without its leading numbering marker; see the
        # module docstring.
        keys = {key for key in {alnum(entry.title)} - {""} if len(key) >= 3}
        probes = keys | {key[:30] for key in keys if len(key) > 30}
        if entry.page is None:
            return any(probe in self.whole for probe in probes)
        # The target page and the page before and after it (0-based indices).
        pages = self.folded
        window = "".join(pages[max(0, entry.page - 1) : min(len(pages), entry.page + 2)])
        return any(probe in window for probe in probes)


def _records_text(cols: PageColumns) -> str:
    """The page's reading records as one string."""
    # One code point per record; a record of several code points is kept in rec_text.
    text = np.asarray(cols.rec_cp, dtype="<u4").tobytes().decode("utf-32-le", "replace")
    if cols.rec_text:
        chars = list(text)
        for index, string in cols.rec_text.items():
            chars[index] = string
        text = "".join(chars)
    return text


def judge(
    entries: Sequence[OutlineEntry],
    *,
    meta_title: str | None,
    n_pages: int,
    text: Text | None,
) -> OutlineGuard:
    """The guard's verdict on *entries* of a document of *n_pages* pages.

    *meta_title* is the PDF's /Title; *text* is the document's text, None when
    there is none to ground the titles in (R3 is then not applied).
    """
    title_keys = {key for key in (alnum(meta_title),) if len(key) >= 10}
    kept: list[OutlineEntry] = []
    dropped: list[tuple[int, str]] = []
    for entry in entries:
        rule = entry_rule(entry.title, title_keys)
        if rule:
            dropped.append((entry.idx, rule))
        else:
            kept.append(entry)
    if len(kept) >= 2:
        top = min(entry.level for entry in kept)
        roots = [i for i, entry in enumerate(kept) if entry.level == top]
        if roots == [0]:
            dropped.append((kept[0].idx, "C6_wrapper"))
            kept = kept[1:]
    reject = None
    score = None
    ungrounded: tuple[str, ...] = ()
    n = len(kept)
    if n < 3:
        reject = "R1_too_few"
    else:
        no_page = sum(1 for entry in kept if entry.page is None)
        pages = {entry.page for entry in kept if entry.page is not None}
        if no_page * 2 > n or (n_pages > 2 and len(pages) == 1 and no_page == 0):
            reject = "R2_targets"
        elif text is not None and text.chars >= 2000:
            misses = [entry for entry in kept if not text.grounded(entry)]
            score = (n - len(misses)) / n
            ungrounded = tuple(entry.entry_id for entry in misses)
            if (n - len(misses)) * 2 < n:
                reject = "R3_ungrounded"
    return OutlineGuard(
        passed=reject is None,
        reject=reject,
        dropped=tuple(dropped),
        decided=Decided(
            "outline_guard", GUARD_VERSION, score=score, calibrated=False, evidence=ungrounded
        ),
    )
