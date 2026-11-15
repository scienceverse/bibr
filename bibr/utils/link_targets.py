"""What a document's own links say about a line break inside a printed link.

A PDF prints a long link over two or more lines, and the text keeps only the
line break, which reading turns into a space ("https://osf.io/ 5s8kv"). The
shape rules in :mod:`bibr.input.consolidate_text` judge such a break from the
words on either side of it, and a short identifier with prose after it
("osf.io/" + "5s8kv), the primary"), a word in capitals ("/" + "FSA") or a host
split at its dot without a scheme ("osf." + "io/2y34r/") reads to them like the
end of the link.

A clickable link names its target in the PDF's link annotation, whatever the
page prints. :class:`LinkTargets` holds the printed forms of a document's
targets (with and without the scheme, ``www.`` and percent-encoding, and a DOI
link's bare DOI), and :meth:`LinkTargets.spans` says whether the text on both
sides of a break runs on through one of them. It only ever says a break is
inside a link: a break no target crosses is left to the shape rules, so a link
without an annotation (a bare "osf.io/abcde" in a Word PDF, a scan) reads as it
did. Two targets of which one ends at the break and another runs on through it
say nothing either.

Matching ignores case and needs the text to agree with a target over at least
``_MIN_AGREEMENT`` characters around the break, both sides of it included.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import unquote

# A target shorter than this is too short to prove anything ("a.co").
_MIN_FORM = 8
# The text before and after a break must agree with a target over this many
# characters, the break between them.
_MIN_AGREEMENT = 8
# The index keys a target by up to this many characters before each position.
_KEY_LEFT = 3
# HTML entities a publisher left in the URI, sometimes escaped twice.
_MAX_UNESCAPES = 3
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_DOI_HOST_RE = re.compile(r"^(?:dx\.|www\.)?doi\.org/", re.IGNORECASE)
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
# What a link holds and two words of prose do not: a path, scheme or query
# character, or a dot inside it ("osf.io").
_LINK_SHAPE_RE = re.compile(r"[/:?=&#%@~_]|\.[^\W_]")
# A decoded form holding white space, a NUL or a replacement character is not
# printed in a way a line break can split.
_UNPRINTABLE_RE = re.compile(r"[\s\x00\ufffd]")
# Closing punctuation after a link's last character ("5s8kv),").
_CLOSERS = ".,;:!?)]}>\"'”’"


def _unescape(uri: str) -> str:
    text = uri.strip()
    for _ in range(_MAX_UNESCAPES):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def printed_forms(uri: str) -> set[str]:
    """The ways a page may print the link to *uri*, case folded.

    The target itself; without its scheme, and without ``www.``; percent-decoded;
    without a trailing "/"; and, for a doi.org target, the bare DOI. A ``mailto:``
    or other scheme without "//" gives nothing: its address is no link a line
    break splits in this way.
    """
    text = _unescape(uri)
    if not text or any(ch.isspace() for ch in text) or not _SCHEME_RE.match(text):
        return set()
    out: set[str] = set()
    for variant in {text, unquote(text)}:
        rest = _SCHEME_RE.sub("", variant)
        hostless_www = rest[4:] if rest.lower().startswith("www.") else rest
        candidates = {variant, rest, hostless_www}
        if _DOI_HOST_RE.match(rest):
            candidates.add(_DOI_HOST_RE.sub("", rest))
        for form in candidates:
            form = form.rstrip("/").casefold()
            if len(form) >= _MIN_FORM and not _UNPRINTABLE_RE.search(form):
                out.add(form)
    return {f for f in out if _DOI_RE.match(f) or "/" in f or "." in f}


class LinkTargets:
    """The printed forms of a document's link targets, indexed by where a break may fall."""

    __slots__ = ("_forms", "_index")

    def __init__(self, uris: Iterable[str]) -> None:
        forms: set[str] = set()
        for uri in uris:
            if isinstance(uri, str):
                forms |= printed_forms(uri)
        self._forms = frozenset(forms)
        # (up to _KEY_LEFT characters before a position, the character at it) ->
        # (form, position): the places a break can fall inside each form.
        index: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
        for form in self._forms:
            for p in range(1, len(form)):
                index[(form[max(0, p - _KEY_LEFT) : p], form[p])].append((form, p))
        self._index = dict(index)

    @classmethod
    def from_uri_dicts(cls, links: Iterable[Mapping[str, Any]] | None) -> LinkTargets | None:
        """Targets from the ``{"uri": ...}`` records the PDF inspection keeps, or None."""
        if not isinstance(links, list | tuple):
            return None
        uris = [link.get("uri") for link in links if isinstance(link, Mapping)]
        targets = cls(u for u in uris if isinstance(u, str))
        return targets if targets else None

    def __bool__(self) -> bool:
        return bool(self._forms)

    def __len__(self) -> int:
        return len(self._forms)

    def spans(self, before: str, after: str, *, line_ends: bool = False) -> bool:
        """Whether a line break between *before* and *after* lies inside one target.

        *before* is the text on the line above the break, back to its last white
        space (a link already joined over earlier breaks included); *after* the
        first word of the line below, and *line_ends* whether that word ends its
        line (or the text). True when, from a point in *before* where a link may
        start (the start of the word, or after a character that is no letter or
        digit), the text joined across the break runs on as a target does:

        - up to the target's end, which then falls inside *after* with only
          closing punctuation or a "/" behind it; or
        - through all of *after*, the target going on past it. A printed link
          holds no space, so this needs *after* to end its line, and the joined
          text to look like a link: a "/", ":", "?", "=", "&", "#", "%", "@",
          "~" or "_" in it, or a dot with a letter or digit after it. Two words
          of prose that spell the start of a host ("Creative" + "Commons" before
          creativecommons.org, "The" + "Guardian.") are neither.

        A break at which one target ends and another runs on proves nothing, nor
        does agreement over fewer than ``_MIN_AGREEMENT`` characters.
        """
        if not before or not after or not self._forms:
            return False
        left = before.casefold()
        right = after.casefold()
        seam = len(left)
        joined = left + right
        # A key holds up to _KEY_LEFT characters before the break: fewer only where
        # the form starts there, so each size finds the forms whose link starts
        # that far back in *before*.
        found = any(
            _may_start_link(left, seam - p)
            and left[seam - p :] == form[:p]
            and _runs_on(joined, seam - p, form, seam, line_ends)
            for size in range(min(_KEY_LEFT, seam), 0, -1)
            for form, p in self._index.get((left[-size:], right[0]), ())
        )
        # One target ending at the break while another runs on proves nothing.
        return found and not self._ends_at(left)

    def _ends_at(self, left: str) -> bool:
        """Whether some target ends exactly at the end of *left* (a link may end there)."""
        for start in range(len(left)):
            if _may_start_link(left, start) and left[start:].rstrip("/") in self._forms:
                return True
        return False


def _may_start_link(text: str, start: int) -> bool:
    """Whether a link may begin at *start* of the token *text*: at its start, or
    after a character that is no letter or digit ("(https", "doi:10.", "/github")."""
    return start == 0 or (start > 0 and not text[start - 1].isalnum())


def _runs_on(joined: str, start: int, form: str, seam: int, line_ends: bool) -> bool:
    """Whether the text from *start* agrees with *form* across the break at *seam*.

    It does when it runs to the form's end, with nothing, closing punctuation or
    a "/" after it; or when it ends while the form goes on, at the end of its
    line and shaped like a link (see :meth:`LinkTargets.spans`). Either way the
    agreement passes the break and covers at least ``_MIN_AGREEMENT`` characters.
    """
    text = joined[start:]
    if text.startswith(form):
        agreed = len(form)
        tail = text[agreed:]
        if tail and tail[0] not in _CLOSERS and tail[0] != "/":
            return False  # the word goes on with letters the target does not have
    elif form.startswith(text):
        agreed = len(text)
        if not line_ends or not _LINK_SHAPE_RE.search(text.rstrip(_CLOSERS)):
            return False  # prose that spells the start of a target
    else:
        return False
    return start + agreed > seam and agreed >= _MIN_AGREEMENT
