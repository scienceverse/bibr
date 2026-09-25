"""OCR artifact correction for post-OCR text processing.

Fixes common OCR artifacts (ligatures, soft hyphens, markdown/LaTeX wrappers)
in text content.
"""

import logging
import re

from bibr.utils.text import CONTROL_CHAR_RE, DOI_BODY, URL_RE, normalize_unicode

logger = logging.getLogger(__name__)

# Common OCR artifacts: character substitutions
# (STX soft-hyphen marks are handled contextually by _resolve_stx_marks.)
_OCR_ARTIFACT_MAP: dict[str, str] = {
    "\u00ad": "-",  # soft hyphen → regular hyphen
    "\ufb01": "fi",  # fi ligature
    "\ufb02": "fl",  # fl ligature
    "\ufb00": "ff",  # ff ligature
    "\ufb03": "ffi",  # ffi ligature
    "\ufb04": "ffl",  # ffl ligature
    "\uff08": "(",  # full-width left parenthesis
    "\uff09": ")",  # full-width right parenthesis
    "\uff0c": ",",  # full-width comma
}

# DOI/URL starters: inside such a token a printed line-end hyphen is a literal
# part of the identifier, not hyphenation.  Public — also used by PDFParser's
# carry-over join to detect region boundaries that fall inside a URL.
DOI_URL_CONTEXT_RE = re.compile(r"https?://|www\.|(?:dx\.)?doi\.org/|10\.\d{4,9}/", re.IGNORECASE)


# Zipf frequency at/above which a fragment or joined form counts as a real
# word \u2014 same threshold the OCR block merger uses (ocr/postprocess.py).
_ZIPF_VALID_WORD = 2.5

# Unicode letter class (no digits, no underscore) \u2014 same class the word-linewrap
# patterns below use. ASCII-only fragments match exactly what "[A-Za-z]+" did.
_STX_ALPHA_BEFORE_RE = re.compile(r"[^\W\d_]+$")
_STX_ALPHA_AFTER_RE = re.compile(r"[^\W\d_]+")

# Script \u2192 the wordfreq lexicons that can actually adjudicate it.  Nothing
# upstream knows the paper's language (``fix_ocr_artifacts`` sees one text
# region at a time), so it is derived from the fragments' own characters.
# Getting it wrong is safe: an unconsulted lexicon scores 0.0 for everything,
# and a lexicon that has never seen either form now KEEPS the printed hyphen.
_SCRIPT_LANG_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"[\u0456\u0457\u0454\u0491]", re.IGNORECASE), ("uk", "ru")),
    (re.compile(r"[\u0400-\u052f]"), ("ru",)),
    (re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]"), ("el",)),
    (re.compile(r"[\u0590-\u05ff]"), ("he",)),
    (re.compile(r"[\u0600-\u06ff]"), ("ar",)),
    (re.compile(r"[\u0900-\u097f]"), ("hi",)),
)

# Latin-script diacritics that (near-)uniquely name one wordfreq lexicon.
# Checked before the ambiguous table below, so a single "\u00df" or "\u0142" anywhere in
# the region pins the language exactly.
_LATIN_DIACRITIC_LANGS: dict[str, tuple[str, ...]] = (
    dict.fromkeys("\u011f\u0131", ("tr",))
    | dict.fromkeys("\u0142\u0105\u0119\u017c\u017a\u0144\u015b", ("pl",))
    | dict.fromkeys("\u00e3\u00f5", ("pt",))
    | dict.fromkeys("\u00f1", ("es",))
    | dict.fromkeys("\u00df", ("de",))
    | dict.fromkeys("\u0153", ("fr",))
    | dict.fromkeys("\u010d\u0159\u017e\u011b\u016f\u0161\u00fd\u0165\u010f\u0148", ("cs",))
    | dict.fromkeys("\u0219\u021b", ("ro",))
    | dict.fromkeys("\u0151\u0171", ("hu",))
    | dict.fromkeys("\u00e5\u00f8\u00e6", ("da", "nb", "sv"))
)

# Accents shared across several languages. They cannot pin one lexicon, but
# they DO rule English out, and leaving them undecided is not neutral: it
# re-hyphenates every French wrap whose fragment carries an "\u00e9". Each bundle
# stays narrow because every member costs one lazily-loaded frequency table.
_LATIN_AMBIGUOUS_LANGS: dict[str, tuple[str, ...]] = (
    dict.fromkeys(
        "\u00e8\u00ea\u00e0\u00e2\u00f9\u00fb\u00ee\u00ef\u00f4\u00e7", ("fr", "it", "pt", "es")
    )
    | dict.fromkeys("\u00e9", ("fr", "pt", "es", "hu", "cs"))
    | dict.fromkeys("\u00e4\u00f6\u00fc", ("de", "sv", "fi", "tr", "hu"))
    | dict.fromkeys("\u00e1\u00ed\u00f3\u00fa", ("es", "pt", "hu", "cs"))
)


_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")

# Scanned in order, so an exact pin ("ß", "ł") anywhere beats an ambiguous
# accent that happens to appear earlier in the region.
_LATIN_LANG_SCANS: tuple[tuple[re.Pattern[str], dict[str, tuple[str, ...]]], ...] = tuple(
    (re.compile(f"[{re.escape(''.join(table))}]"), table)
    for table in (_LATIN_DIACRITIC_LANGS, _LATIN_AMBIGUOUS_LANGS)
)


def _detect_langs(text: str) -> tuple[str, ...] | None:
    """Lexicons *text*'s own characters positively name, or ``None``."""
    for pattern, langs in _SCRIPT_LANG_RULES:
        if pattern.search(text):
            return langs
    lowered = text.lower()
    for pattern, table in _LATIN_LANG_SCANS:
        m = pattern.search(lowered)
        if m is not None:
            return (*table[m.group()], "en")
    return None


def _lang_candidates(fragment: str, context_langs: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Pick the wordfreq lexicons to consult for *fragment*.

    *context_langs* is the surrounding region's detection (see
    :func:`_detect_langs`); it is used only when the fragment itself names no
    language, because a wrap usually falls between two undiacriticked
    syllables ("wypowie|dzenie") whose language only the region reveals.
    English is always kept alongside it — a Latin word inside a Cyrillic
    abstract must still be adjudicable.

    Returns an empty tuple when no lexicon plausibly covers the fragment: it
    carries non-ASCII letters but neither it nor the region names a language.
    English's weak opinion about such a fragment must not be trusted — it
    scores "santé" 2.18 and "publique" 2.13, just under the word threshold,
    while French scores both above 5.0. Every lookup then returns 0.0 and the
    printed hyphen is kept.
    """
    langs = _detect_langs(fragment)
    if langs is not None:
        return langs
    if context_langs:
        return tuple(dict.fromkeys((*context_langs, "en")))
    return () if _NON_ASCII_RE.search(fragment) else ("en",)


def _max_zipf(word: str, langs: tuple[str, ...]) -> float:
    """Strongest zipf opinion any of *langs* holds about *word* (0.0 = unknown)."""
    from wordfreq import zipf_frequency

    best = 0.0
    for lang in langs:
        try:
            best = max(best, zipf_frequency(word, lang))
        except (LookupError, ValueError):  # unsupported language code
            continue
    return best


def _stx_alpha_keep_hyphen(before: str, after: str, langs: tuple[str, ...] | None = None) -> bool:
    """Decide whether an STX between two word fragments is a literal hyphen.

    Camel compounds ("Cross\\x02National") always keep it \u2014 hyphenation never
    resumes at an uppercase letter. Otherwise the lexicon adjudicates: if the
    joined form is a real word ("off\\x02line" \u2192 "offline") the mark was
    hyphenation; if it is not but BOTH fragments are real words the printed
    hyphen was a literal compound ("approach\\x02related"). Rare joined forms
    with a non-word fragment re-join ("psychophysio\\x02logical" \u2014 zipf 1.65,
    under the word threshold but still a lexicon opinion).

    When no consulted lexicon has ever seen the joined form *or* either
    fragment AND the fragments carry characters those lexicons demonstrably do
    not cover, nothing is proven and the printed hyphen is KEPT. Deleting it is
    the destructive branch, and it used to fire for every script the English
    lexicon does not cover (Cyrillic, Greek, Turkish, accented Latin).

    The keep-by-default rule does not extend to plain-ASCII fragments. An
    absent dictionary entry alone does not identify an unsupported script;
    ambiguous compounds without script evidence remain a limitation.
    """
    if after[0].isupper() and before[-1].islower():
        return True
    if langs is None:
        langs = _lang_candidates(before + after)
    joined = _max_zipf((before + after).lower(), langs)
    if joined >= _ZIPF_VALID_WORD:
        return False
    before_zipf = _max_zipf(before.lower(), langs)
    after_zipf = _max_zipf(after.lower(), langs)
    if (
        len(before) >= 2
        and len(after) >= 2
        and before_zipf >= _ZIPF_VALID_WORD
        and after_zipf >= _ZIPF_VALID_WORD
    ):
        return True
    if joined or before_zipf or after_zipf:
        return False
    return bool(_NON_ASCII_RE.search(before + after))


def _resolve_stx_marks(text: str) -> str:
    """Resolve GLM-OCR STX (``\\x02``) line-end soft-hyphen marks.

    GLM-OCR emits STX at line-end soft-hyphen positions without joining the
    broken word. Plain hyphenated words re-join without the hyphen
    ("off\\x02line" \u2192 "offline"). Inside a DOI/URL token the printed
    hyphen is literal and is restored \u2014 stripping it corrupts the identifier
    ("annurev-psych\\x02113011" must become "annurev-psych-113011").  The same
    holds between digits: numbers are never syllable-hyphenated, so the
    printed hyphen in bare ORCID iDs, year ranges, and page ranges is literal
    ("0000-0002\\x020247" must become "0000-0002-0247").  Between word
    fragments, :func:`_stx_alpha_keep_hyphen` adjudicates literal compound
    hyphens ("Cross\\x02National", "approach\\x02related") against true
    hyphenation.
    """
    if "\x02" not in text:
        return text
    context_langs = _detect_langs(text) or ()
    parts: list[str] = []
    pos = 0
    while (i := text.find("\x02", pos)) != -1:
        token_start = max(text.rfind(" ", 0, i), text.rfind("\n", 0, i), text.rfind("\t", 0, i)) + 1
        in_doi_or_url = bool(DOI_URL_CONTEXT_RE.search(text[token_start:i]))
        between_digits = i > 0 and text[i - 1].isdigit() and text[i + 1 : i + 2].isdigit()
        keep = in_doi_or_url or between_digits
        if not keep:
            before = _STX_ALPHA_BEFORE_RE.search(text[token_start:i])
            after = _STX_ALPHA_AFTER_RE.match(text, i + 1)
            if before and after:
                keep = _stx_alpha_keep_hyphen(
                    before.group(),
                    after.group(),
                    _lang_candidates(before.group() + after.group(), context_langs),
                )
        parts.append(text[pos:i])
        parts.append("-" if keep else "")
        pos = i + 1
    parts.append(text[pos:])
    return "".join(parts)


# Literal line-wrap hyphens: GLM-OCR usually marks wrap hyphens with STX but
# sometimes emits them literally, either at line end ("Lak-\nens") or — APA
# style, which breaks URLs *before* punctuation — at line start ("Lak\n-ens").
# ``(?=\S)`` keeps paragraph breaks and markdown bullets ("\n- item") intact.
_URL_LINEWRAP_RE = re.compile(r"-[ \t]*\r?\n[ \t]*(?=\S)|[ \t]*\r?\n[ \t]*-(?=\S)")

# Cheap prefilter: a hyphen touching a line break on either side (spaces/tabs
# allowed).  Strict superset of what _URL_LINEWRAP_RE can match.
_HYPHEN_AT_LINEBREAK_RE = re.compile(r"-[ \t]*\r?\n|\n[ \t]*-")


def _bridge_url_linewraps(text: str) -> str:
    """Re-join literal line wraps at hyphens inside URL/DOI/numeric tokens.

    Counterpart of :func:`_resolve_stx_marks` for wraps the OCR did not
    STX-mark.  Same keep-hyphen policy: inside a URL/DOI token the printed
    hyphen is treated as literal ("Lak-\\nens" → "Lak-ens"), and a hyphen
    between digits is always literal — numbers are never syllable-hyphenated
    (bare ORCID iDs, year ranges, page ranges).  Plain-word wraps are left
    untouched.  Runs repeated passes so a URL wrapped across several lines
    re-joins fully (the lookback can only see past the previous wrap once it
    has been bridged).
    """
    if not _HYPHEN_AT_LINEBREAK_RE.search(text):
        return text
    while "\n" in text:

        def _bridge(m: re.Match[str], s: str = text) -> str:
            # Hyphen between digits across the wrap is always literal.
            pre = s[m.start() - 1 : m.start()]
            post = s[m.end() : m.end() + 1]
            if pre.isdigit() and post.isdigit():
                return "-"
            token_start = (
                max(
                    s.rfind(" ", 0, m.start()),
                    s.rfind("\n", 0, m.start()),
                    s.rfind("\t", 0, m.start()),
                )
                + 1
            )
            if DOI_URL_CONTEXT_RE.search(s[token_start : m.start()]):
                return "-"
            return m.group(0)

        bridged = _URL_LINEWRAP_RE.sub(_bridge, text)
        if bridged == text:
            return text
        text = bridged
    return text


_WORD_LINEWRAP_RE = re.compile(
    r"(?P<left>[^\W\d_]{2,})-[ \t]*\r?\n[ \t]*(?P<right>[^\W\d_]{2,})",
    re.UNICODE,
)

_ALPHANUMERIC_LINEWRAP_RE = re.compile(
    r"(?P<alpha>[^\W\d_]+)-[ \t]*\r?\n[ \t]*(?P<digits>\d+)"
    r"|(?P<short_number>\b\d{1,2})-[ \t]*\r?\n[ \t]*(?P<word>[^\W\d_]{2,})",
    re.UNICODE,
)


def _bridge_alphanumeric_linewraps(text: str) -> str:
    """Keep literal hyphens in scientific alphanumeric compounds."""
    # Strict superset of what _ALPHANUMERIC_LINEWRAP_RE can match — both its
    # alternatives require "-[ \t]*\r?\n". Its two neighbours already take
    # this shortcut; without it the two-alternative sub ran over every OCR
    # region, and on 13,248 real regions the rule fires on 0.02% of them.
    if not _HYPHEN_AT_LINEBREAK_RE.search(text):
        return text

    def _bridge(match: re.Match[str]) -> str:
        if match.group("alpha") is not None:
            return f"{match.group('alpha')}-{match.group('digits')}"
        return f"{match.group('short_number')}-{match.group('word')}"

    return _ALPHANUMERIC_LINEWRAP_RE.sub(_bridge, text)


def _dehyphenate_word_linewraps(text: str) -> str:
    """Remove within-region word line wraps while retaining real hyphens."""
    # Strict superset of what _WORD_LINEWRAP_RE can match — skips the language
    # scan and the substitution entirely for regions with no wrapped hyphen.
    if not _HYPHEN_AT_LINEBREAK_RE.search(text):
        return text

    context_langs = _detect_langs(text) or ()

    def _join(match: re.Match[str]) -> str:
        left = match.group("left")
        right = match.group("right")
        token_start = (
            max(
                text.rfind(" ", 0, match.start()),
                text.rfind("\n", 0, match.start()),
                text.rfind("\t", 0, match.start()),
            )
            + 1
        )
        token_end_candidates = [
            pos
            for pos in (
                text.find(" ", match.end()),
                text.find("\n", match.end()),
                text.find("\t", match.end()),
            )
            if pos >= 0
        ]
        token_end = min(token_end_candidates, default=len(text))
        token = text[token_start:token_end]
        identifier_context = any(marker in token for marker in ("@", "/", "\\", "_"))
        langs = _lang_candidates(left + right, context_langs)
        joined_score = _max_zipf((left + right).lower(), langs)
        compound_score = _max_zipf(f"{left}-{right}".lower(), langs)
        keep_hyphen = identifier_context or compound_score >= joined_score + 0.5
        if not keep_hyphen and joined_score < _ZIPF_VALID_WORD:
            keep_hyphen = _stx_alpha_keep_hyphen(left, right, langs)
        return f"{left}-{right}" if keep_hyphen else left + right

    return _WORD_LINEWRAP_RE.sub(_join, text)


# Mid-word DOI line-wraps: a DOI token split mid-word with NO hyphen at the
# break ("10.1016/j.neu\nron.2015.11.028" → "10.1016/j.neuron.2015.11.028").
# These reach the LLM as "j.neu ron" after whitespace flattening and get
# mis-reconstructed (the "j." is dropped). Joining the fragments directly is
# correct ONLY for a genuine mid-word split, so the pattern is tightly gated:
#   - the DOI body ends in TWO letters right before the wrap (mid-word) — this
#     rejects a COMPLETE DOI whose wrap precedes a new token, because complete
#     DOIs end in a digit or a separator+single-char ("…00550.x", "…00742-z");
#   - the continuation starts lowercase/digit AND carries a dot or digit — this
#     rejects joining into a following reference ("…\nvan der Berg");
#   - registrant 10.1146 (Annual Reviews) is excluded: its DOIs are
#     "annurev-{subject}-…" and the text layer drops the structural hyphen at
#     the wrap, so joining would fabricate a wrong DOI.
_DOI_MIDWORD_WRAP_RE = re.compile(
    r"(10\.(?!1146/)\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z][A-Za-z])"
    r"[ \t]*\r?\n[ \t]*"
    r"(?=[a-z0-9][-._;()/:A-Za-z0-9]*[.\d])"
)


def _bridge_doi_midword_wraps(text: str) -> str:
    """Re-join mid-word line-wraps inside a DOI token (no hyphen at the break).

    See :data:`_DOI_MIDWORD_WRAP_RE` for the gating that keeps complete DOIs,
    following references, and hyphen-structured prefixes (Annual Reviews)
    untouched. Repeated passes handle a DOI wrapped across several lines.
    """
    if "\n" not in text or "10." not in text:
        return text
    prev: str | None = None
    while prev != text:
        prev = text
        text = _DOI_MIDWORD_WRAP_RE.sub(r"\1", text)
    return text


# A permissive DOI continuation can absorb adjacent prose or a URL. Keep the documented
# continuation grammar rather than narrowing it in ways that reject legitimate suffixes.
_DOI_DOT_WRAP_RE = re.compile(
    r"(10\.(?!1146/)\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z0-9]\.)"
    r"[ \t]*\r?\n?[ \t]*"
    r"(?!\d{1,4}[.,]\s+[A-Z])"
    r"(?!\d{1,2}\.\d{1,2}\s+[A-Z])"
    r"(?=[a-z0-9][-._;()/:a-z0-9]*[.\d])"
)


def _bridge_doi_dot_wraps(text: str) -> str:
    """Re-join a DOI wrap that falls right after an internal period.

    See :data:`_DOI_DOT_WRAP_RE` for the gating that accepts both a newline and
    a plain-space separator while rejecting prose, numbered references, and
    numbered headings. Repeated passes handle a DOI wrapped at two internal
    dots (e.g. "journal." then "pone." then "0279511"), matching the sibling
    bridges.
    """
    if "10." not in text:
        return text
    prev: str | None = None
    while prev != text:
        prev = text
        text = _DOI_DOT_WRAP_RE.sub(r"\1", text)
    return text


# OCR sometimes serializes a wrap as "10.1136/ vetrec-2018-105253" — a space
# straight after the registrant slash. Close the gap only when the next token
# carries a digit, dot or hyphen (DOI tails do; prose words don't).
_DOI_SLASH_SPACE_RE = re.compile(r"(10\.\d{4,9}/)[ \t]+(?=[A-Za-z0-9][-._;()/:A-Za-z0-9]*[.\d-])")


def _bridge_doi_slash_space(text: str) -> str:
    """Re-join a DOI registrant slash from a following token split by a space.

    See :data:`_DOI_SLASH_SPACE_RE` for the gating that requires the following
    token to carry DOI-tail structure (digit/dot/hyphen), never plain prose.
    """
    if "10." not in text:
        return text
    return _DOI_SLASH_SPACE_RE.sub(r"\1", text)


# CRLF line-wraps inside a URL/DOI token at positions _bridge_doi_midword_wraps
# does NOT cover: registrant-internal ("doi.org/1\n0.1038"), after-digit
# ("acr.21\n979"), after-slash ("rheumatology/\nket317"), host/scheme-split
# ("https:/\n/doi.org", "…science\n.org"), and www host ("www.sea\nledenvelope").
# A bare \r\n (no space before it) inside URL context, line-start indentation
# allowed after it.
_URL_PATH_CHARS = r"A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-"
_URL_CRLF_WRAP_RE = re.compile(rf"(?<=[{_URL_PATH_CHARS}])\r?\n[ \t]*(?=[{_URL_PATH_CHARS}])")
_CAP_WORD_RE = re.compile(r"[A-Z][a-z]")
_DOI_BODY_RE = re.compile(DOI_BODY)
_URL_STRUCT_AHEAD_RE = re.compile(r"[./]")
_HOST_SCHEME_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_DOMAINISH_START_RE = re.compile(r"[A-Za-z0-9-]")


def _join_url_crlf(m: "re.Match[str]") -> str:
    """Decide whether a CRLF wrap sits inside a URL/DOI token and should join."""
    s = m.string
    lstart = (
        max(
            s.rfind(" ", 0, m.start()),
            s.rfind("\n", 0, m.start()),
            s.rfind("\t", 0, m.start()),
            s.rfind("\r", 0, m.start()),
        )
        + 1
    )
    left = s[lstart : m.start()]
    rend = m.end()
    while rend < len(s) and s[rend] not in " \t\r\n":
        rend += 1
    right = s[m.end() : rend]
    if not left or not right:
        return m.group(0)
    # URL/DOI context on the JOINED form (handles scheme-splits like https:/ //doi.org).
    if not DOI_URL_CONTEXT_RE.search(left[-40:] + right[:8]):
        return m.group(0)
    # Veto prose: a capitalized word continuation is a name/sentence, not a URL.
    if _CAP_WORD_RE.match(right):
        return m.group(0)
    last, first = left[-1], right[0]
    in_doi_body = bool(_DOI_BODY_RE.search(left))
    # Never join a 10.NNNN/ DOI-body letter->letter wrap: that is
    # _bridge_doi_midword_wraps's domain (incl. the annurev structural-hyphen
    # guard). Make the invariant explicit rather than relying on fall-through.
    if in_doi_body and last.isalpha() and first.isalpha():
        return m.group(0)
    # Rule 1: digit -> digit (mid-number / registrant-internal). A complete DOI
    # ending ".x" has a letter before the wrap, so the page-range guard is free.
    if last.isdigit() and first.isdigit():
        return ""
    # Rule 2: continuation starts with a URL-structural char.
    if first in "./:)":
        return ""
    # Rule 2b: left ends with a path/scheme separator.
    if last in "/:":
        return ""
    # Rule 2c: host-separator wrap — the previous line ends with a bare "."
    # that is a domain-label separator ("blog." in "https://blog.\nopenai.com"),
    # not sentence punctuation. Gated tightly: OUTSIDE a 10.NNNN/ DOI body
    # (sentence-final periods after complete DOIs are excluded via
    # in_doi_body), the token must still be inside the URL's host — i.e. no
    # path "/" has appeared yet after the scheme/www — since a period after
    # the URL has already reached its path is prose punctuation, not a host
    # separator. The next line must start with a domain/path-ish char.
    if last == "." and not in_doi_body and _DOMAINISH_START_RE.match(first):
        scheme_m = _HOST_SCHEME_RE.search(left)
        if scheme_m and "/" not in left[scheme_m.end() :]:
            return ""
    # Rule 3: host continuation (letter->letter) OUTSIDE a 10.NNNN/ DOI body,
    # only when the continuation carries a host '.' or '/'.
    if (
        not in_doi_body
        and last.isalpha()
        and first.isalpha()
        and _URL_STRUCT_AHEAD_RE.search(right[:20])
    ):
        return ""
    return m.group(0)


def _bridge_doi_url_crlf_wraps(text: str) -> str:
    """Re-join CRLF wraps inside a URL/DOI token at positions the mid-word DOI
    bridge does not cover. Additive to :func:`_bridge_doi_midword_wraps`; never
    joins a 10.NNNN/ DOI-body letter->letter wrap (that function's domain, incl.
    the annurev structural-hyphen guard). Repeated passes handle multi-line
    wraps."""
    if "\n" not in text or not DOI_URL_CONTEXT_RE.search(text):
        return text
    prev: str | None = None
    while prev != text:
        prev = text
        text = _URL_CRLF_WRAP_RE.sub(_join_url_crlf, text)
    return text


# Space-serialized DOI wrap: glmocr sometimes emits a printed DOI line-wrap as a
# literal space instead of a newline ("…0033-295X .113.4.842"), so the CRLF
# bridges above never see it and the LLM truncates the DOI at the space
# (dropping the ".113.4.842" suffix). Re-join only "<doi-fragment> .<digit>":
# a wrapped DOI suffix continues with ".<digit>", whereas prose after a DOI
# reads "…00550.x Smith" / "…2020. 4 studies" — the period sits BEFORE the
# space, never after it — so the space-before-period shape is DOI-specific. A
# following page range ("…295X 113-185") has no leading dot and is left alone.
_DOI_SPACE_WRAP_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z0-9])[ \t](?=\.\d)")


def _bridge_doi_space_wraps(text: str) -> str:
    """Re-join a DOI whose line-wrap was serialized as a literal space.

    See :data:`_DOI_SPACE_WRAP_RE` for the ``<doi-body> .<digit>`` gating that
    keeps sentence boundaries and following page ranges untouched. Repeated
    passes handle a DOI wrapped across several lines (each emitted as a space).
    """
    if "10." not in text:
        return text
    prev: str | None = None
    while prev != text:
        prev = text
        text = _DOI_SPACE_WRAP_RE.sub(r"\1", text)
    return text


# Matches ```markdown ... ``` or ``` ... ``` fenced code blocks
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:markdown)?\s*\n(.*?)\n?\s*```$", re.DOTALL)

# Matches $$\text{...}$$ — LaTeX text wrappers OCR produces for non-formula regions.
# Captures the plain text inside one or more \text{} commands.
_LATEX_TEXT_WRAP_RE = re.compile(r"^\$\$\s*((?:\\text\s*\{[^}]*\}\s*)+)\s*\$\$$")
# Prefix-only: locate ``\text{`` and let the balanced-brace helper find the matching ``}``.
# Stdlib ``re`` can't do recursive groups, so a manual depth counter handles
# nested braces (e.g. ``\text{a {b} c}``).
_LATEX_TEXT_PREFIX_RE = re.compile(r"\\text\s*\{")

# Matches lone $$ display-math blocks that contain broken/garbled LaTeX
# (e.g., \begin{array} with spaced-out characters from OCR misrecognition).
_BROKEN_LATEX_RE = re.compile(r"^\$\$\s*\\begin\{array\}", re.DOTALL)
# Matches trailing \end{array}$$ fragments (split across text entries)
_TRAILING_LATEX_RE = re.compile(r"^\s*\\end\{array\}\s*\$\$\s*$", re.DOTALL)

# Matches inline math $ ... $ (single dollar, NOT $$ display math).
# Used to strip OCR-inserted inline LaTeX delimiters from body text.
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)\s*(.+?)\s*\$(?!\$)")

# Matches LaTeX superscript affiliation markers that may leak into author names:
#   ^{1}  ^{1,2}  ^{1, 2, 3}  ^{†}  ^{*}  $ ^{1} $  $ ^{1,2} $
_AFFILIATION_MARKER_RE = re.compile(
    r"(?<!\$)\$\s*\^(?:\{[^}]*\}|[^\s$])\s*\$|\^(?:\{[^}]*\}|[^\s])"
)

# Matches affiliation lines: superscript marker + \text{} in dollar delimiters.
# OCR labels these as formulas; they should be discarded (affiliation info is
# extracted separately by the LLM metadata step).
#   $ ^1 \text{ Eindhoven University of Technology} $
#   $$ ^{1,2} \text{MIT} $$
_AFFILIATION_LINE_RE = re.compile(
    r"^\s*\${1,2}\s*\^(?:\{[^}]*\}|\S)\s*\\text\s*\{[^}]*\}\s*\${1,2}\s*$"
)

# ---------------------------------------------------------------------------
# LaTeX command stripping for body text
# ---------------------------------------------------------------------------

# Unwrap text-mode LaTeX commands: \mathrm{...}, \text{...}, \textit{...}, etc.
# Captures the content inside the braces.
_LATEX_TEXT_CMD_RE = re.compile(
    r"\\(?:mathrm|textrm|text|textit|textbf|textsc|textsf|mathit|mathbf|mathsf|mathcal)\s*\{([^}]*)\}"
)

# Flatten subscripts _{...} and superscripts ^{...} to just the content.
_LATEX_SUB_RE = re.compile(r"_\{([^}]*)\}")
_LATEX_SUP_RE = re.compile(r"\^\{([^}]*)\}")
# Also handle single-char subscripts/superscripts without braces: _x, ^x
_LATEX_SUB_SINGLE_RE = re.compile(r"_([A-Za-z0-9])")
_LATEX_SUP_SINGLE_RE = re.compile(r"\^([A-Za-z0-9])")

# Common LaTeX Greek letter and symbol commands -> Unicode equivalents.
_LATEX_SYMBOL_MAP: dict[str, str] = {
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\delta": "\u03b4",
    r"\epsilon": "\u03b5",
    r"\varepsilon": "\u03b5",
    r"\zeta": "\u03b6",
    r"\eta": "\u03b7",
    r"\theta": "\u03b8",
    r"\vartheta": "\u03b8",
    r"\iota": "\u03b9",
    r"\kappa": "\u03ba",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\nu": "\u03bd",
    r"\xi": "\u03be",
    r"\pi": "\u03c0",
    r"\rho": "\u03c1",
    r"\sigma": "\u03c3",
    r"\tau": "\u03c4",
    r"\upsilon": "\u03c5",
    r"\phi": "\u03c6",
    r"\varphi": "\u03c6",
    r"\chi": "\u03c7",
    r"\psi": "\u03c8",
    r"\omega": "\u03c9",
    r"\Gamma": "\u0393",
    r"\Delta": "\u0394",
    r"\varDelta": "\u0394",
    r"\Theta": "\u0398",
    r"\Lambda": "\u039b",
    r"\Xi": "\u039e",
    r"\Pi": "\u03a0",
    r"\Sigma": "\u03a3",
    r"\Phi": "\u03a6",
    r"\Psi": "\u03a8",
    r"\Omega": "\u03a9",
    r"\infty": "\u221e",
    r"\pm": "\u00b1",
    r"\times": "\u00d7",
    r"\cdot": "\u00b7",
    r"\leq": "\u2264",
    r"\geq": "\u2265",
    # Matched longest first, so neither prints as "≤slant"
    r"\leqslant": "\u2264",
    r"\geqslant": "\u2265",
    r"\neq": "\u2260",
    r"\approx": "\u2248",
    r"\sim": "~",
    r"\partial": "\u2202",
    r"\nabla": "\u2207",
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\sqrt": "\u221a",
}

# Single compiled regex for all known LaTeX symbol commands (longest first).
_LATEX_SYMBOL_RE = re.compile(
    "|".join(re.escape(cmd) for cmd in sorted(_LATEX_SYMBOL_MAP, key=len, reverse=True))
)

# Regex matching any backslash command (e.g., \varDelta, \mathrm).
# Used as a fallback to strip unrecognized commands.
_LATEX_BACKSLASH_CMD_RE = re.compile(r"\\([a-zA-Z]+)")


def fix_ocr_artifacts(text: str) -> str:
    """Replace common OCR artifacts with correct characters.

    Handles ligatures (ff, fi, fl, ffi, ffl), soft hyphens, and the
    full-width punctuation GLM-OCR occasionally substitutes into Latin text.

    Args:
        text: Input text potentially containing OCR artifacts.

    Returns:
        Cleaned text.
    """
    text = normalize_unicode(text)
    text = _resolve_stx_marks(text)
    text = CONTROL_CHAR_RE.sub("", text)
    for artifact, replacement in _OCR_ARTIFACT_MAP.items():
        text = text.replace(artifact, replacement)
    text = _bridge_url_linewraps(text)
    text = _bridge_doi_url_crlf_wraps(text)
    text = _bridge_doi_midword_wraps(text)
    text = _bridge_doi_dot_wraps(text)
    text = _bridge_doi_slash_space(text)
    text = _bridge_doi_space_wraps(text)
    text = _bridge_alphanumeric_linewraps(text)
    return normalize_unicode(_dehyphenate_word_linewraps(text))


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping plain text.

    The OCR engine sometimes wraps ordinary text in
    ``\\`\\`\\`markdown ... \\`\\`\\``` blocks.  This strips the fences and
    returns the inner content.
    """
    m = _MARKDOWN_FENCE_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text


def _extract_latex_text(s: str) -> str:
    r"""Extract the inner content of the FIRST ``\text{...}`` in ``s``.

    Handles balanced nested braces, unlike a plain ``[^}]*`` regex. Returns
    the inner string; if no ``\text{...}`` is found the input is returned
    unchanged. If the opening brace is unbalanced, returns everything from
    the start of the inner content to the end of the string as a best
    effort.
    """
    m = _LATEX_TEXT_PREFIX_RE.search(s)
    if m is None:
        return s
    start = m.end()  # one past the opening `{`
    depth = 1
    i = start
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i]
        i += 1
    # Unbalanced — return everything to end of string as a best-effort.
    return s[start:]


def _iter_latex_text_inners(s: str) -> list[str]:
    r"""Return inner contents of all ``\text{...}`` blocks in ``s``.

    Uses :func:`_extract_latex_text` repeatedly with balanced-brace handling
    so a single nested-brace argument doesn't truncate at the first ``}``.
    """
    parts: list[str] = []
    pos = 0
    while True:
        m = _LATEX_TEXT_PREFIX_RE.search(s, pos)
        if m is None:
            return parts
        # Re-use the single-extraction helper on the slice from the match
        # onward; it walks from the prefix found at position 0 of the slice.
        inner = _extract_latex_text(s[m.start() :])
        parts.append(inner)
        # Advance past this block: the helper consumed prefix + inner + 1 (`}`).
        # Slice-relative consumed length = (m.end() - m.start()) + len(inner) + 1.
        # Convert back to absolute string offset.
        consumed = (m.end() - m.start()) + len(inner) + 1
        pos = m.start() + consumed


def unwrap_latex_text(text: str) -> str:
    r"""Unwrap ``$$\text{...}$$`` into plain text.

    When the OCR engine classifies a non-formula region as a formula, it
    wraps the content in ``\text{}``.  This extracts the plain text.
    Returns the original string if the pattern doesn't match.
    """
    m = _LATEX_TEXT_WRAP_RE.match(text.strip())
    if m:
        parts = _iter_latex_text_inners(m.group(1))
        return " ".join(p.strip() for p in parts if p.strip())
    return text


def strip_inline_math(text: str) -> str:
    r"""Strip inline math delimiters ``$ ... $`` from body text.

    The OCR engine wraps inline statistics like ``t(97.7)=2.9`` in single-dollar
    math delimiters: ``$ t(97.7)=2.9 $``.  This strips the delimiters and keeps
    the content as plain text.  Double-dollar display math (``$$...$$``) is not
    affected.

    A ``$`` immediately followed by a digit is a currency amount, not a math
    delimiter — otherwise two amounts in one sentence (``$5 … $10``) would pair
    up as a math span and the text between them would be fused.  When a
    currency ``$`` is skipped, scanning resumes right after it so a later real
    math span can still match.
    """
    out: list[str] = []
    pos = 0
    while True:
        m = _INLINE_MATH_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        if text[m.start() + 1].isdigit():
            # Currency, not math: keep the ``$`` and let its closing ``$``
            # candidate re-open the next search.
            out.append(text[pos : m.start() + 1])
            pos = m.start() + 1
            continue
        out.append(text[pos : m.start()])
        out.append(m.group(1))
        pos = m.end()


def strip_affiliation_markers(text: str) -> str:
    r"""Strip LaTeX superscript affiliation markers from text.

    Removes patterns like ``^{1}``, ``^{1,2}``, ``$ ^{1} $``, ``^{†}``
    that OCR engines produce for author–affiliation superscripts.
    Collapses any resulting extra whitespace.
    """
    cleaned = _AFFILIATION_MARKER_RE.sub("", text)
    return " ".join(cleaned.split())


def strip_latex_commands(text: str) -> str:
    r"""Strip residual LaTeX commands from body text.

    Handles common patterns that leak through OCR into plain text:

    * ``\mathrm{...}``, ``\text{...}``, ``\textit{...}``, ``\textbf{...}``
      -- unwrapped to their content.
    * ``\Delta``, ``\alpha``, ``\varDelta``, etc. -- converted to Unicode.
    * ``_{...}`` and ``^{...}`` -- flattened to just the content inside braces.
    * Any remaining ``\command`` -- the backslash prefix is removed.
    """
    result = text

    # 1. Unwrap text-mode commands: \mathrm{n=5} -> n=5
    result = _LATEX_TEXT_CMD_RE.sub(r"\1", result)

    # 2. Flatten subscripts and superscripts: _{c} -> c, ^{2} -> 2
    result = _LATEX_SUB_RE.sub(r"\1", result)
    result = _LATEX_SUP_RE.sub(r"\1", result)
    result = _LATEX_SUB_SINGLE_RE.sub(r"\1", result)
    result = _LATEX_SUP_SINGLE_RE.sub(r"\1", result)

    # 3. Replace known LaTeX symbol commands with Unicode equivalents.
    result = _LATEX_SYMBOL_RE.sub(lambda m: _LATEX_SYMBOL_MAP[m.group()], result)

    # 4. Strip any remaining \command sequences (remove backslash + command name).
    result = _LATEX_BACKSLASH_CMD_RE.sub(r"\1", result)

    # 5. Clean up leftover empty brace pairs and extra whitespace
    result = result.replace("{}", "")
    result = result.replace("{ }", "")
    result = " ".join(result.split())

    return result


def clean_text_content(text: str) -> str | None:
    """Early-phase cleaning of body text — safe for pre-extraction use.

    Applies only cleanups that do NOT destroy patterns needed by downstream
    extractors (citation linker, equation extractor, xref detection):
    markdown fences, LaTeX ``\\text{}`` wrappers, and broken LaTeX blocks.

    Destructive cleanups (inline math delimiter stripping, LaTeX command
    flattening) are deferred to :func:`clean_text_content_late`, which
    must run **after** all extraction stages.

    Returns ``None`` if the text should be discarded entirely.
    """
    result = strip_markdown_fences(text)
    result = unwrap_latex_text(result)

    stripped = result.strip()
    if not stripped:
        return None

    # Discard broken LaTeX blocks that ended up labeled as text
    if _BROKEN_LATEX_RE.match(stripped):
        return None
    if _TRAILING_LATEX_RE.match(stripped):
        return None

    return result


# URL mask placeholders: NUL-delimited indices survive every late-phase
# transform (no whitespace, no LaTeX metacharacters).
_URL_MASK_RE = re.compile("\x00(\\d+)\x00")


def _mask_urls(text: str) -> tuple[str, list[str]]:
    """Replace URL spans with ``\\x00<i>\\x00`` placeholders."""
    urls: list[str] = []

    def _mask(m: re.Match[str]) -> str:
        urls.append(m.group(0))
        return f"\x00{len(urls) - 1}\x00"

    return URL_RE.sub(_mask, text), urls


def _restore_urls(text: str, urls: list[str]) -> str:
    if not urls:
        return text
    return _URL_MASK_RE.sub(lambda m: urls[int(m.group(1))], text)


def clean_text_content_late(text: str) -> str:
    """Late-phase cleaning — strips inline math delimiters and LaTeX commands.

    Must run **after** citation linking, equation extraction, and xref
    detection, because those stages depend on patterns like ``^{3}`` and
    ``$...$`` that this function removes.

    URL spans are masked during cleaning — LaTeX stripping would otherwise
    corrupt them (``_LATEX_SUB_SINGLE_RE`` eats the underscores in
    ``…/to_err_is_human``, flattening it to ``…/toerrishuman``).
    """
    masked, urls = _mask_urls(text)
    result = strip_inline_math(masked)
    result = strip_latex_commands(result)
    # Collapse OCR character-spacing in regular text (e.g. "9 0 6" → "906").
    # This handles spaced text that came from formula OCR regions merged into
    # body text, or from the text OCR prompt on dense statistical notation.
    result = _collapse_bare_spaced_runs(result)
    return _restore_urls(result, urls)


_OPERATORNAME_RE = re.compile(r"\\operatorname\s*\{([^}]*)\}")

# Matches \text{...}, \mathrm{...}, \textrm{...} — commands that wrap readable text
# inside formulas and are prone to the same OCR character-spacing issue.
_LATEX_TEXT_BLOCK_RE = re.compile(r"\\(text|mathrm|textrm)\s*\{([^}]*)\}")


def _collapse_spaced_chars(inner: str) -> str | None:
    """If *inner* is all single characters separated by spaces, join them.

    Returns the collapsed string, or ``None`` if the pattern doesn't match.
    """
    tokens = inner.split()
    if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
        return "".join(tokens)
    return None


def _fix_operatorname_spacing(text: str) -> str:
    r"""Collapse OCR-inserted spaces inside ``\operatorname{...}`` blocks.

    OCR engines often insert spaces between individual characters in
    mathematical operator names, producing
    ``\operatorname{A t t e n t i o n}`` instead of
    ``\operatorname{Attention}``.  This detects the pattern (all tokens
    are single characters separated by spaces) and collapses them.
    """

    def _collapse(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        collapsed = _collapse_spaced_chars(inner)
        if collapsed is not None:
            return r"\operatorname{" + collapsed + "}"
        return str(m.group(0))

    return _OPERATORNAME_RE.sub(_collapse, text)


def _fix_text_spacing(text: str) -> str:
    r"""Collapse OCR-inserted spaces inside ``\text{...}``, ``\mathrm{...}``, etc.

    Same issue as ``\operatorname{}``: OCR formula recognition spaces out
    individual characters, producing ``\text{g e n d e r}`` instead of
    ``\text{gender}``.

    .. note::
       For multi-word content the word boundaries are lost because the OCR
       uses uniform single-character spacing.  ``\text{u n e m p l o y m e n t
       a t a g e}`` collapses to ``\text{unemploymentatage}``.  This is still
       preferable to per-character spacing for downstream consumers.
    """

    def _collapse(m: re.Match[str]) -> str:
        cmd = m.group(1)
        inner = m.group(2).strip()
        collapsed = _collapse_spaced_chars(inner)
        if collapsed is not None:
            return "\\" + cmd + "{" + collapsed + "}"
        return str(m.group(0))

    return _LATEX_TEXT_BLOCK_RE.sub(_collapse, text)


_MATH_OPERATORS = frozenset("=+<>~≤≥≠±∓")


def _collapse_bare_spaced_runs(text: str) -> str:
    """Collapse bare character-spaced sequences from OCR formula recognition.

    OCR engines sometimes space out every character in statistical notation,
    producing ``F ( 1, 2 6 )=1 7. 9 0 6`` instead of ``F(1,26)=17.906``.

    Strategy: split on spaces, find maximal runs of single-character tokens
    (length == 1), and collapse runs of 3+ that contain at least one digit
    and no mathematical operators.  The operator check preserves intentional
    spacing in expressions like ``x = 5`` or ``a + b``.
    """
    tokens = text.split(" ")
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1:
            # Accumulate a run of single-char tokens
            run_start = i
            while i < len(tokens) and len(tokens[i]) == 1:
                i += 1
            run = tokens[run_start:i]
            has_digit = any(c.isdigit() for c in run)
            has_operator = any(c in _MATH_OPERATORS for c in run)
            if len(run) >= 3 and has_digit and not has_operator:
                result.append("".join(run))
            else:
                result.extend(run)
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def clean_formula_text(text: str) -> str | None:
    """Clean formula text that is actually garbled OCR output.

    Returns ``None`` if the text is a broken LaTeX block that should be
    discarded entirely (e.g., ``\\begin{array}`` with spaced-out characters).
    Otherwise returns the (possibly cleaned) text.
    """
    stripped = text.strip()

    # Discard broken \begin{array} blocks with spaced-out characters
    if _BROKEN_LATEX_RE.match(stripped):
        return None

    # Discard trailing \end{array}$$ fragments
    if _TRAILING_LATEX_RE.match(stripped):
        return None

    # Discard affiliation lines: $ ^1 \text{University} $
    # OCR classifies these as formulas but they're just author affiliations.
    if _AFFILIATION_LINE_RE.match(stripped):
        return None

    # Unwrap $$\text{...}$$ to plain text
    unwrapped = unwrap_latex_text(stripped)
    if unwrapped != stripped:
        return unwrapped

    # Fix spaced-out characters from OCR inside LaTeX commands
    # e.g. \operatorname{A t t e n t i o n} → \operatorname{Attention}
    #      \text{g e n d e r} → \text{gender}
    result = _fix_operatorname_spacing(text)
    result = _fix_text_spacing(result)

    # Fix bare character-spaced runs from formula OCR
    # e.g. F ( 1, 2 6 )=1 7. 9 0 6 → F(1,26)=17.906
    result = _collapse_bare_spaced_runs(result)

    return result
