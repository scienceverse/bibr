"""OCR artifact correction for post-OCR text processing.

Fixes common OCR artifacts (ligatures, soft hyphens, markdown/LaTeX wrappers)
in text content.
"""

import logging
import re
from collections.abc import Iterable, Iterator

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
# The lookbehind starts a match only where a run of letters starts: tried from
# inside a long run that does not reach the mark, ``[^\W\d_]+$`` backtracks
# quadratically.
_STX_ALPHA_BEFORE_RE = re.compile(r"(?<![^\W\d_])[^\W\d_]+$")
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
    # The token (back to the last space, newline or tab) is tracked across
    # marks, and each stretch between two marks searched once: re-scanning the
    # token for every mark is quadratic in a long one. Neither a DOI/URL
    # starter nor a letter run can span a mark.
    token_start = 0
    in_doi_or_url = False
    while (i := text.find("\x02", pos)) != -1:
        space = max(text.rfind(" ", pos, i), text.rfind("\n", pos, i), text.rfind("\t", pos, i))
        if space >= 0:
            token_start = space + 1
            in_doi_or_url = False
        lo = max(token_start, pos)
        in_doi_or_url = in_doi_or_url or bool(DOI_URL_CONTEXT_RE.search(text, lo, i))
        between_digits = i > 0 and text[i - 1].isdigit() and text[i + 1 : i + 2].isdigit()
        keep = in_doi_or_url or between_digits
        if not keep:
            before = _STX_ALPHA_BEFORE_RE.search(text, lo, i)
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


# A printed hyphen at a line break: ASCII hyphen-minus, U+2010 HYPHEN or U+2011
# NON-BREAKING HYPHEN. The text layer of some typesets prints U+2010 where others
# print "-"; all three follow the same join/keep rules below.
_HYPHEN = "[-\u2010\u2011]"

# Literal line-wrap hyphens: GLM-OCR usually marks wrap hyphens with STX but
# sometimes emits them literally, either at line end ("Lak-\nens") or — APA
# style, which breaks URLs *before* punctuation — at line start ("Lak\n-ens").
# ``(?=\S)`` keeps paragraph breaks and markdown bullets ("\n- item") intact.
_URL_LINEWRAP_RE = re.compile(rf"{_HYPHEN}[ \t]*\r?\n[ \t]*(?=\S)|[ \t]*\r?\n[ \t]*{_HYPHEN}(?=\S)")

# Cheap prefilter: a hyphen touching a line break on either side (spaces/tabs
# allowed).  Strict superset of what _URL_LINEWRAP_RE can match.
_HYPHEN_AT_LINEBREAK_RE = re.compile(rf"{_HYPHEN}[ \t]*\r?\n|\n[ \t]*{_HYPHEN}")


def _bridge_url_linewraps(text: str) -> str:
    """Re-join literal line wraps at hyphens inside URL/DOI/numeric tokens.

    Counterpart of :func:`_resolve_stx_marks` for wraps the OCR did not
    STX-mark.  Same keep-hyphen policy: inside a URL/DOI token the printed
    hyphen is treated as literal ("Lak-\\nens" → "Lak-ens"), and a hyphen
    between digits is always literal — numbers are never syllable-hyphenated
    (bare ORCID iDs, year ranges, page ranges).  Plain-word wraps are left
    untouched.  A URL wrapped across several lines re-joins fully: the token
    a wrap looks back over runs through the wraps bridged before it.
    """
    if not _HYPHEN_AT_LINEBREAK_RE.search(text):
        return text
    parts: list[str] = []
    pos = 0
    # One pass carries the token (back to the last space, newline or tab of
    # the text as bridged so far) and its DOI/URL context from wrap to wrap,
    # so each stretch between two wraps is searched once; repeated passes over
    # the whole text bridged one more wrap of a long token each. No DOI/URL
    # starter has a "-", so none spans a bridged wrap.
    in_doi_or_url = False
    for m in _URL_LINEWRAP_RE.finditer(text):
        start, end = m.span()
        space = max(
            text.rfind(" ", pos, start), text.rfind("\n", pos, start), text.rfind("\t", pos, start)
        )
        if space >= 0:
            in_doi_or_url = False
        lo = max(space + 1, pos)
        in_doi_or_url = in_doi_or_url or bool(DOI_URL_CONTEXT_RE.search(text, lo, start))
        parts.append(text[pos:start])
        # Hyphen between digits across the wrap is always literal.
        if in_doi_or_url or (text[start - 1 : start].isdigit() and text[end : end + 1].isdigit()):
            parts.append("-")
        else:
            parts.append(m.group(0))
            in_doi_or_url = False  # the line break it keeps ends the token
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


_WORD_LINEWRAP_RE = re.compile(
    rf"(?P<left>[^\W\d_]{{2,}}){_HYPHEN}[ \t]*\r?\n[ \t]*(?P<right>[^\W\d_]{{2,}})",
    re.UNICODE,
)

_ALPHANUMERIC_LINEWRAP_RE = re.compile(
    rf"(?P<alpha>[^\W\d_]+){_HYPHEN}[ \t]*\r?\n[ \t]*(?P<digits>\d+)"
    rf"|(?P<short_number>\b\d{{1,2}}){_HYPHEN}[ \t]*\r?\n[ \t]*(?P<word>[^\W\d_]{{2,}})",
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
#     the wrap, so joining would fabricate a wrong DOI. The older dotted form
#     "annurev.{subject}.…" has no hyphen to lose and is bridged;
#   - the continuation may open upper-case when it carries a digit and is no
#     PubMed marker ("2018J" + "G004669", "STROK" + "EAHA.119.025015"), and a
#     single letter right after the registrant slash or a later slash ("nar/g" +
#     "kq873") may end the first line;
#   - a number and a full stop with a capitalized word after it ("2. Smith") is the
#     number of the next reference, and the join callback in reference text refuses an
#     upper-case continuation after a lower-case letter ("abc" + "COVID-19") and a
#     continuation whose second word is a word of prose.
_DOI_MIDWORD_WRAP_RE = re.compile(
    r"(10\.(?!1146/(?!annurev\.))\d{4,9}/"
    r"(?:[-._;()/:A-Za-z0-9]*[A-Za-z][A-Za-z]|(?:[-._;()/:A-Za-z0-9]*/)?[A-Za-z]))"
    r"[ \t]*\r?\n[ \t]*"
    r"(?!\d{1,4}[.,]\s+(?!PMID|PMCID|PubMed)[A-Z])"
    r"(?=[a-z0-9][-._;()/:A-Za-z0-9]*[.\d]|(?!PM)[A-Z](?![a-z])[-._;()/A-Za-z0-9]*\d)"
)


# What the mid-word bridge joined in the releases before it learned the shapes above. Text that
# is no reference entry keeps these.
_DOI_MIDWORD_WRAP_PLAIN_RE = re.compile(
    r"(10\.(?!1146/)\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z][A-Za-z])"
    r"[ \t]*\r?\n[ \t]*"
    r"(?=[a-z0-9][-._;()/:A-Za-z0-9]*[.\d])"
)


def _bridge_doi_midword_wraps(text: str, reference: bool = True) -> str:
    """Re-join mid-word line-wraps inside a DOI token (no hyphen at the break).

    See :data:`_DOI_MIDWORD_WRAP_RE` for the gating that keeps complete DOIs,
    following references, and hyphen-structured prefixes (Annual Reviews)
    untouched. Repeated passes handle a DOI wrapped across several lines. Text that is
    no reference entry (*reference* False) keeps the plainer
    :data:`_DOI_MIDWORD_WRAP_PLAIN_RE`.
    """
    if "\n" not in text or "10." not in text:
        return text
    prev: str | None = None
    while prev != text:
        prev = text
        if reference:
            text = _DOI_MIDWORD_WRAP_RE.sub(_midword_join, text)
        else:
            text = _DOI_MIDWORD_WRAP_PLAIN_RE.sub(r"\1", text)
    return text


def _midword_join(m: "re.Match[str]") -> str:
    """The text of a mid-word wrap after the break is joined, unless it starts something new.

    A continuation in capitals does not follow a lower-case word of the same DOI
    ("abc" + "COVID-19"), and a word of prose after it shows that a name or a code of the
    next sentence was read.
    """
    s, end = m.string, m.end()
    if s[end : end + 1].isupper() and m.group(1)[-1].islower():
        return m.group(0)
    if _PROSE_WORD_RE.match(_word_after_first(s, end)):
        return m.group(0)
    return m.group(1)


# A permissive DOI continuation can absorb adjacent prose or a URL. Keep the documented
# continuation grammar rather than narrowing it in ways that reject legitimate suffixes.
_DOI_DOT_WRAP_RE = re.compile(
    r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z0-9]\.)"
    r"[ \t]*\r?\n?[ \t]*"
    r"(?!\d{1,4}[.,]\s+(?!PMID|PMCID|PubMed)[A-Z])"
    r"(?!\d{1,2}\.\d{1,2}\s+[A-Z])"
    r"(?!https?:|www\.)"
    r"(?=[a-z0-9][-._;()/:a-z0-9]*[.\d])"
)


# What the dot bridge joined before it learned Annual Reviews DOIs and the PubMed marker
# after a number. Text that is no reference entry keeps these.
_DOI_DOT_WRAP_PLAIN_RE = re.compile(
    r"(10\.(?!1146/)\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z0-9]\.)"
    r"[ \t]*\r?\n?[ \t]*"
    r"(?!\d{1,4}[.,]\s+[A-Z])"
    r"(?!\d{1,2}\.\d{1,2}\s+[A-Z])"
    r"(?!https?:|www\.)"
    r"(?=[a-z0-9][-._;()/:a-z0-9]*[.\d])"
)


def _bridge_doi_dot_wraps(text: str, reference: bool = True) -> str:
    """Re-join a DOI wrap that falls right after an internal period.

    See :data:`_DOI_DOT_WRAP_RE` for the gating that accepts both a newline and
    a plain-space separator while rejecting prose, numbered references, and
    numbered headings. Repeated passes handle a DOI wrapped at two internal
    dots (e.g. "journal." then "pone." then "0279511"), matching the sibling
    bridges. Text that is no reference entry (*reference* False) keeps the plainer
    :data:`_DOI_DOT_WRAP_PLAIN_RE`.
    """
    if "10." not in text:
        return text
    pattern = _DOI_DOT_WRAP_RE if reference else _DOI_DOT_WRAP_PLAIN_RE
    prev: str | None = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1", text)
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
# A line that ends on the registrant of a DOI: "10." (the registrant number follows) or
# "10.NNNN/" (the suffix follows). The suffix opens with a word that carries a digit, a
# dot or a bracket ("Nmat2630", "PhysRevLett.78.1190"); a first word with none of these is
# a name or a word of prose.
_DOI_REGISTRANT_END_RE = re.compile(r"(?<![\w.-])10\.(?:\d{4,9}/)?$")
_DOI_REGISTRANT_NUMBER_RE = re.compile(r"\d{4,9}(?:/|%2[Ff]|$)")
_DOI_SUFFIX_START_RE = re.compile(r"\d|\(|[A-Za-z<\[]\S*?[\d.(<\[]")
# What makes the first word of the next line look like text of a link: a character that a
# path or a query holds ("/", "_", "%", "=", "?", "&", "#"), a file name or a host name
# inside it ("Register.aspx"), an identifier made of letters and a run of three digits or
# more ("NCT02319005"), of twelve letters and digits ("44695c678bdb44f19172cae4ef842381"),
# a number of several parts ("OM:2025:89:FIN", "B.1,5521.1"), or a slug of three words or
# more in lower case, five in any case. A capitalized word, a code such as "COVID-19",
# "H1N1" or "V2", an address and a word with one hyphen are none of these.
_URL_PATH_TOKEN_RE = re.compile(r"[/_%=?&#]")
_URL_FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9]\.[A-Za-z0-9]{2,}")
_URL_ID_TOKEN_RE = re.compile(
    r"(?=\S*[A-Za-z])\S*?\d{3,}|(?=\S*\d)(?=\S*[A-Za-z])[A-Za-z0-9]{12,}"
    r"|(?:[^:\s]*:){2,}|(?=\S*\d)(?:[^.\s]*\.){2,}"
)
# A slug of five words or more, whatever their case ("Quality-statement-3-Referring-people").
_URL_LONG_SLUG_RE = re.compile(r"[A-Za-z0-9]+(?:[-\u2010\u2011\u2013][A-Za-z0-9]+){4,}[.,;:)]*$")
_EMAIL_RE = re.compile(r"\w@\w")
# A lower-case slug: words and numbers joined by hyphens ("sults-pbr-2012-13").
_URL_SLUG_RE = re.compile(r"[a-z0-9]+(?:[-\u2010\u2011\u2013][a-z0-9]+)+(?:\.[a-z]{2,5})?[.,;:)]*$")
_HYPHEN_SPLIT_RE = re.compile(r"[-\u2010\u2011\u2013]")
# A file extension after a wrap that follows a dot ("index." + "html"). A top-level domain is
# also an English word ("org members"), so it counts only with a path, a query or a further
# label after it ("org/publications", "com.br").
_URL_EXTENSION_RE = re.compile(
    r"(?:(?:html?|xhtml|shtml|php|aspx?|jsp|cfm|pdf|docx?|xlsx?|pptx?|txt|xml|json|csv|zip)"
    r"(?![A-Za-z])|(?:org|com|net|edu|gov)(?=[/?#]|\.[A-Za-z0-9]))"
)
# A DOI whose registrant is itself split by the wrap ("10.\n1016/...", "10\n.1016/..."):
# the text has no "10.NNNN/" until the wrap is joined, so it must still reach the bridge.
_DOI_SPLIT_START_RE = re.compile(r"10\.?\d{0,8}[ \t]*\r?\n")
# How far back a wrap looks for the scheme or "10.NNNN/" that opens its token, and how far
# the earlier releases looked: a wrap whose context lies beyond the second is a long URL's.
# A token longer than the first is not followed back any further, which bounds the work a
# wrap costs however many lines one token is printed over.
_URL_TOKEN_WINDOW = 1000
_URL_NARROW_WINDOW = 40
_HOST_SCHEME_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_DOMAINISH_START_RE = re.compile(r"[A-Za-z0-9-]")
_URL_BRIDGE_PASSES = 3

# The first word of a line after a URL or a DOI that does not continue it: a new
# identifier ("doi:10.5194/..." under a journal link), a label ("DOI:", "E-ISSN", "e-mail:",
# "@"), an acronym ("ISSN"), a name followed by a comma ("Smith,"), or the number or page
# range that starts the next reference ("112.", "12-34") before a capitalized word.
_NEW_IDENTIFIER_RE = re.compile(r"(?:https?:|www\.|doi\b)", re.IGNORECASE)
# The word of a citation that follows a link: "accessed 30 June 2018", "Retrieved from".
_CITATION_WORD_RE = re.compile(
    r"(?:accessed|retrieved|available|cited|visited|viewed)[.,;:]*$", re.IGNORECASE
)
# The same word, opening what follows the first word of the line: "entresto (accessed 2024)".
_CITATION_AFTER_RE = re.compile(
    r"[(\[]?(?:accessed|retrieved|cited|visited|viewed|last)\b", re.IGNORECASE
)
_LABEL_TOKEN_RE = re.compile(
    r"(?:(?:[A-Za-z][A-Za-z-]*:|@|[A-Za-z]-[A-Za-z]+[.:]?)$|[A-Za-z]-[A-Z]{2,}:)"
)
_ACRONYM_TOKEN_RE = re.compile(r"[A-Z]{2,}[.,;:)]*$")
_NAME_TOKEN_RE = re.compile(r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*[,;]$")
_NUMBER_TOKEN_RE = re.compile(r"\d{1,4}(?:[-–]\d{1,4})?[.,;:)]*$")
# A year in brackets opens what follows a DOI ("(2019). Jones K."), not the suffix of one.
_YEAR_IN_BRACKETS_RE = re.compile(r"\((?:19|20)\d\d\)[.,;:]*$")
# A lower-case word of two letters or more after the end of an identifier is prose.
_PROSE_WORD_RE = re.compile(r"[a-z]{2,}(?![\w:])")

# The rest of an Elsevier-style suffix after a bracketed group: "S0959-4388(96)" + "80080-8"
# (four to six digits, a hyphen, a check character, or the rest of the digits), a suffix
# of three numbers joined by hyphens ("(2015)" + "01-0151-06"), and the rest of a SICI
# suffix: a number followed by a separator or bracket ("1097-4679(1999", "41:7<1196").
# A page range ("12-34") and a range of years ("2020-2021") are none of these.
_DOI_BRACKET_TAIL_RE = re.compile(
    r"\d{4,6}-[0-9A-Za-z](?![\w-])|\d{5,6}-\d{1,3}(?![\w-])"
    r"|(?!(?:19|20)\d\d-(?:19|20)\d\d)\d{4}-\d{3}[\dXx](?![\w-])"
    r"|\d{1,6}-\d{1,6}-\d{1,6}(?![\w-])|\d[0-9A-Za-z\-]*[:<\[(]\w"
)
# A bracketed group opening the next line: "S0304-3940" + "(03)00880-2".
_DOI_BRACKET_OPEN_RE = re.compile(r"\(\d{2,8}\)\d")
# After a dot, an upper-case continuation made of letters and digits: "jbc." + "M705403200",
# "CSHPERSPECT." + "A031583", "JBJS." + "CC.23.00060". Not a PubMed or PMC identifier, which
# follow a DOI as text.
_DOI_DOT_UPPER_RE = re.compile(
    r"(?!PM|S\d{4}[-‐-―]\d{3}[\dXx][(\[])[A-Z]{1,8}[.-]?\d[A-Za-z0-9.\-;()/:]*"
)
# What opens the next reference or sentence after an identifier: a capitalized word
# ("Smith, J.", "Genomics"), other than the words a record adds to a DOI.
_NEW_TEXT_RE = re.compile(r"(?!PubMed|Epub|Online)[A-Z][a-z]+")


# The spaces that end a word of the text: a no-break or thin space is white space in a PDF's
# text layer, and no part of a link.
_LINE_SPACE = " \t\xa0\u2002\u2003\u2009\u200a\u202f"
_WORD_END = _LINE_SPACE + "\r\n"


def _next_token(s: str, pos: int) -> str:
    """The word after *pos* on the same line, or ""."""
    end = len(s)
    while pos < end and s[pos] in _LINE_SPACE:
        pos += 1
    stop = pos
    while stop < end and s[stop] not in _WORD_END:
        stop += 1
    return s[pos:stop]


def _word_after_first(s: str, pos: int) -> str:
    """The second word of the line that starts at *pos*, or ""."""
    stop = pos
    while stop < len(s) and s[stop] not in _WORD_END:
        stop += 1
    return _next_token(s, stop)


def _is_prose_after(right: str, following: str, last: str) -> bool:
    """Whether the first word of the next line, with the word after it, is no identifier.

    *right* is the first word of the line a break leads to, *following* the next one on
    that line, *last* the character before the break. A line that opens a new identifier,
    a label, a name, or the number of the next reference is a new thing, however the URL
    or DOI above it ended. A number after a digit continues that number.
    """
    if _NEW_IDENTIFIER_RE.match(right) or _LABEL_TOKEN_RE.match(right):
        return True
    if _CITATION_WORD_RE.match(right):
        return True
    if _NAME_TOKEN_RE.match(right):
        return True
    if _ACRONYM_TOKEN_RE.match(right) and following[:1] in (":", *"0123456789"):
        return True  # "ISSN 2469-6501", "ISBN: 978"
    return bool(_NUMBER_TOKEN_RE.match(right) and following[:1].isupper() and not last.isdigit())


def _slug_parts(right: str) -> list[str] | None:
    """The words of a lower-case hyphenated slug, or None for any other word."""
    if not _URL_SLUG_RE.match(right):
        return None  # a label ("E-ISSN"), a code ("COVID-19") or a word without a hyphen
    return _HYPHEN_SPLIT_RE.split(re.sub(r"[.,;:)/]+$", "", right))


def _is_link_text(right: str) -> bool:
    """Whether the first word of the next line is made like the text of a link.

    It holds a path or query character, a file name, an identifier or a number of several
    parts, or it is a slug of three words or more. A word on its own, a code with a digit,
    a word with one hyphen and an address are not.
    """
    if _EMAIL_RE.search(right):
        return False
    if (
        _URL_PATH_TOKEN_RE.search(right)
        or _URL_FILE_TOKEN_RE.search(right)
        or _URL_ID_TOKEN_RE.search(right)
    ):
        return True
    parts = _slug_parts(right)
    return (parts is not None and len(parts) >= 3) or bool(_URL_LONG_SLUG_RE.match(right))


# Labels of the export links that a publisher prints under a DOI; never the rest of it.
_EXPORT_LABEL_RE = re.compile(r"(?:BIBTEX|PLAINTEXT|ENDNOTE|REFWORKS|MENDELEY|ZOTERO)\b", re.I)


def _stands_alone(
    right: str, following: str, *, name_ok: bool = False, caps_ok: bool = False
) -> bool:
    """Whether the first word of the next line ends the link: nothing or no prose follows it.

    Used after a character that no link ends on ("/", "=", "&", "_"). A word that is alone
    on its line, that has a citation after it ("accessed 2024") or that has no word of prose
    after it ("cloud-gaming, 2022.") is the rest of the link. A word with prose after it
    ("results show"), one that opens a bracket and a capitalized word with anything after it
    are not. *name_ok* lets a capitalized word that is alone on its line count ("q=Smith"),
    *caps_ok* a word in capitals of four letters or more, the end of a link that has capitals
    in it ("10.7910/DVN/" + "GKULIW"), unless it names an export format.
    """
    core = right.rstrip(".,;:)>]")
    if (len(core) < 2 and not core.isdigit()) or not right[0].isalnum():
        return False
    # a word that opens in lower case or with a digit, or an identifier that holds a digit
    # ("NCT02319005", "Table2020"); not a capitalized word or a word in capitals
    lower_or_digit = right[0].islower() or right[0].isdigit() or any(c.isdigit() for c in core)
    caps = (
        caps_ok
        and len(core) >= 4
        and core.isalpha()
        and core.isupper()
        and not _EXPORT_LABEL_RE.match(core)
    )
    if not following or _CITATION_AFTER_RE.match(following):
        return lower_or_digit or name_ok or caps
    if core.isdigit() and len(core) < 5:
        return False  # a year, a page or the number of a reference, with text after it
    return lower_or_digit and not _PROSE_WORD_RE.match(following)


# The lexicons that tell whether a wrap split a word of a link. The text of a link names no
# language, and the words of a path are in the language of the site.
_SPLIT_WORD_LANGS = ("en", "es", "fr", "de", "pt", "it", "nl")


def _splits_a_word(right: str, left: str) -> bool:
    """Whether the word on the next line holds the end of a word that the wrap split."""
    core = right.rstrip(".,;:)>]")
    run = re.search(r"[A-Za-z]+$", left)
    return bool(
        run
        and core.isalpha()
        and core.islower()
        and _max_zipf((run.group() + core).lower(), _SPLIT_WORD_LANGS) >= _ZIPF_VALID_WORD
        # the end is no word, or the start is none: "do" + "wnload", "ecuad" + "or"
        and (
            _max_zipf(core, _SPLIT_WORD_LANGS) < _ZIPF_VALID_WORD
            or _max_zipf(run.group().lower(), _SPLIT_WORD_LANGS) < _ZIPF_VALID_WORD
        )
    )


def _continues_path_word(right: str, left: str = "") -> bool:
    """Whether the word a wrap leads to reads as the rest of a word of a URL path."""
    if _is_link_text(right):
        return True
    if _EMAIL_RE.search(right):
        return False
    parts = _slug_parts(right)
    if parts is None:
        return _splits_a_word(right, left)  # "do" + "wnload."
    run = re.search(r"[A-Za-z]+$", left)
    if run and _max_zipf((run.group() + parts[0]).lower(), ("en",)) >= _ZIPF_VALID_WORD:
        return not right.endswith(":")  # "cys" + "tic-fibrosis"; not "e-mail:" under a link
    if len(parts) < 3:
        return False  # "co-authored", "well-being": one hyphen, no split word
    if any(part.isdigit() for part in parts):
        return True
    # "state-of-the-art" is prose; "ducing-recaptcha-v3-new-way-to" is not.
    return not all(len(part) > 1 and _max_zipf(part, ("en",)) >= _ZIPF_VALID_WORD for part in parts)


def _doi_suffix_wrap(last: str, right: str, following: str) -> bool:
    """Whether a line break inside a DOI suffix falls where a wrapped DOI continues."""
    if _is_prose_after(right, following, last) or _PROSE_WORD_RE.match(following):
        return False
    if last in ")]" and _DOI_BRACKET_TAIL_RE.match(right):
        return True
    if right[0] == "(" and last.isdigit() and _DOI_BRACKET_OPEN_RE.match(right):
        return True
    if last == "_":
        return True  # a DOI never ends on an underscore
    return (
        last == "."
        and bool(_DOI_DOT_UPPER_RE.match(right))
        and not _NEW_TEXT_RE.match(following)  # "COVID-19 Genomics", "A2 Smith, J."
    )


def _wrap_joins(left: str, right: str, following: str, reference: bool = True) -> bool:
    """Decide whether a line break sits inside a URL/DOI token and should be joined.

    *left* is the token before the break (the last ``_URL_TOKEN_WINDOW`` characters of it),
    *right* the first word of the next line, *following* the word after that on its line.
    *reference* is False for text that is not a reference entry: the shapes that repair
    the suffix of a DOI after a bracket, an underscore or a dot are then left out.

    A wrap joins where the text before it is a link that cannot end there and the word
    after it is made like the rest of that link, or stands alone on its line. A word of
    prose ("results", "well-being"), a code ("COVID-19", "H1N1"), a name ("Smith J.") or
    the number of the next reference stays on its own line.
    """
    if not left or not right:
        return False
    # URL/DOI context on the JOINED form (handles scheme-splits like https:/ //doi.org).
    joined_tail = right[:8]
    if not DOI_URL_CONTEXT_RE.search(left + joined_tail):
        return False
    last, first = left[-1], right[0]
    # A token whose scheme or registrant lies beyond the last 40 characters is a long
    # URL's; the earlier releases never joined it. It joins where the line that follows
    # is no new thing.
    long_token = not DOI_URL_CONTEXT_RE.search(left[-_URL_NARROW_WINDOW:] + joined_tail)
    if long_token and _is_prose_after(right, following, last):
        return False
    # A DOI never ends at its registrant: "10." is followed by the registrant number and
    # its slash, "10.NNNN/" by the rest of the identifier, whatever its case
    # ("10.1038/\nNmat2630"), never by a name or the numbers of a page range.
    if _DOI_REGISTRANT_END_RE.search(left[-20:]):
        if left.endswith("/"):
            if (
                _DOI_SUFFIX_START_RE.match(right)
                and not _YEAR_IN_BRACKETS_RE.match(right)
                and not _is_prose_after(right, following, last)
                and not _PROSE_WORD_RE.match(following)
            ):
                return True
        elif _DOI_REGISTRANT_NUMBER_RE.match(right):
            return True
    in_doi_body = bool(_DOI_BODY_RE.search(left))
    # A URL query does not end on "=" or "&": the wrap continues it, whatever the value,
    # when the value is made like link text or stands alone ("package=" + "BayesFactor").
    if (
        last in "=&"
        and not in_doi_body
        and not _is_prose_after(right, following, last)
        and (_is_link_text(right) or _stands_alone(right, following, name_ok=True))
    ):
        return True
    # Veto prose: a capitalized word continuation is a name/sentence, not a URL. A token
    # that carries path or query characters ("Enrollment-File/", "Salzburg_Bakom.pdf") is
    # no name, nor is a file name ("Register.aspx") after a slash or an underscore. After a
    # letter, a capitalized word with a dot is the name of a site ("YouTube.com").
    if (
        _CAP_WORD_RE.match(right)
        and not _URL_PATH_TOKEN_RE.search(right)
        and not _URL_ID_TOKEN_RE.search(right)
        and not _URL_LONG_SLUG_RE.match(right)
        and not (last in "_/" and _URL_FILE_TOKEN_RE.search(right))
    ):
        return False
    # A URL path does not end on "_" either.
    if (
        last == "_"
        and not in_doi_body
        and not _is_prose_after(right, following, last)
        and (_is_link_text(right) or _stands_alone(right, following))
    ):
        return True
    # Never join a 10.NNNN/ DOI-body letter->letter wrap: that is
    # _bridge_doi_midword_wraps's domain (incl. the annurev structural-hyphen
    # guard). Make the invariant explicit rather than relying on fall-through.
    if in_doi_body and last.isalpha() and first.isalpha():
        return False
    # Rules for the suffix of a DOI (the text after "10.NNNN/"): a wrap there continues
    # the identifier when what follows has the shape of the rest of a DOI, never a word.
    if reference and in_doi_body and _doi_suffix_wrap(last, right, following):
        return True
    # Rule 1: digit -> digit (mid-number / registrant-internal). A complete DOI
    # ending ".x" has a letter before the wrap, so the page-range guard is free.
    if last.isdigit() and first.isdigit():
        return True
    # Rule 2: continuation starts with a URL-structural char.
    if first in "./:)":
        return True
    # Rule 2b: left ends with a path/scheme separator, and the next word is made like the
    # rest of the path or ends the line. A link does not end on its scheme.
    if left.endswith("://"):
        return True
    if last in "/:":
        # the capitals of a DOI or of the path before it say that a word in capitals is its end
        caps_ok = in_doi_body or any(c.isupper() for c in left.rsplit("://", 1)[-1])
        return _is_link_text(right) or _stands_alone(right, following, caps_ok=caps_ok)
    # Rule 2c: host-separator wrap — the previous line ends with a bare "."
    # that is a domain-label separator ("blog." in "https://blog.\nopenai.com"),
    # not sentence punctuation. Gated tightly: OUTSIDE a 10.NNNN/ DOI body
    # (sentence-final periods after complete DOIs are excluded via
    # in_doi_body), the token must still be inside the URL's host — i.e. no
    # path "/" has appeared yet after the scheme/www — since a period after
    # the URL has already reached its path is prose punctuation, not a host
    # separator. The next line must start with a domain/path-ish char.
    if last == "." and not in_doi_body and _DOMAINISH_START_RE.match(first):
        # the last scheme or "www." counts: a link may hold another ("web/2012/http://www.")
        schemes = list(_HOST_SCHEME_RE.finditer(left))
        scheme_m = schemes[-1] if schemes else None
        if scheme_m and "/" not in left[scheme_m.end() :]:
            return True
    # Rule 2d: a wrap after the dot before a file extension ("index." + "php/...").
    if last == "." and not in_doi_body and _URL_EXTENSION_RE.match(right):
        return True
    # Rule 3: host continuation (letter->letter) OUTSIDE a 10.NNNN/ DOI body. The
    # continuation carries a host "." or "/", a query or path character, or three words
    # of a slug ("sults-pbr-arrangements"), or holds the rest of a split word. A word of
    # two dictionary words with one hyphen ("co-authored") and a label ("E-ISSN") are prose.
    return (
        not in_doi_body and last.isalpha() and first.isalpha() and _continues_path_word(right, left)
    )


def _bridge_doi_url_crlf_wraps(text: str, reference: bool = True) -> str:
    """Re-join CRLF wraps inside a URL/DOI token at positions the mid-word DOI
    bridge does not cover. Additive to :func:`_bridge_doi_midword_wraps`; never
    joins a 10.NNNN/ DOI-body letter->letter wrap (that function's domain, incl.
    the annurev structural-hyphen guard).

    One pass carries the token before each wrap (back to the last white space of the text as
    bridged so far, at most ``_URL_TOKEN_WINDOW`` characters) from wrap to wrap, so a URL
    printed over many lines joins in one pass at a cost that grows with the number of
    wraps, not with its square. A further pass runs only where the first changed something,
    and no more than ``_URL_BRIDGE_PASSES`` run.
    """
    if "\n" not in text or not (
        DOI_URL_CONTEXT_RE.search(text) or _DOI_SPLIT_START_RE.search(text)
    ):
        return text
    for _ in range(_URL_BRIDGE_PASSES):
        joined = _bridge_url_crlf_pass(text, reference)
        if joined == text:
            break
        text = joined
    return text


def _bridge_url_crlf_pass(text: str, reference: bool) -> str:
    parts: list[str] = []
    pos = 0
    token = ""
    for m in _URL_CRLF_WRAP_RE.finditer(text):
        start, end = m.span()
        piece = text[pos:start]
        space = max(map(piece.rfind, _WORD_END))
        token = (token + piece if space < 0 else piece[space + 1 :])[-_URL_TOKEN_WINDOW:]
        parts.append(piece)
        pos = end
        rend = end
        while rend < len(text) and text[rend] not in _WORD_END:
            rend += 1
        right = text[end:rend]
        if _wrap_joins(token, right, _next_token(text, rend), reference):
            continue  # the token goes on through the wrap
        parts.append(m.group(0))
        token = ""
    parts.append(text[pos:])
    return "".join(parts)


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
# The same for text a parser read from the document, where most dollar signs
# are literal: currency ("US$ 5", "35 MJ/$, while"), R's ``df$age_group``,
# shell ``file:$ADAPTER``. There neither delimiter may face a space (Pandoc's
# rule), a word character may touch neither from outside, and the opening one
# may not be followed by closing punctuation. JATS tex-math
# (``}$\textbf{q}$\end``) still matches; DOCX inline equations need no rule
# (see ``clean_text_content_late``'s ``inline_math``). The span never crosses
# an unescaped ``$``: one that cannot close it ends the attempt, so a sentence
# full of currency amounts is scanned once, not once per dollar sign.
_STRICT_INLINE_MATH_RE = re.compile(
    r"(?<![\w$\\])\$(?![\s$,;:!?)\]}])((?:[^$\\\n]|\\.)+?)(?<![\s$\\])\$(?![\w$])"
)
# Paddle-OCR-VL writes inline math in text regions as ``\( ... \)``. The span
# stops at the next ``\(`` for the same reason.
_PAREN_INLINE_MATH_RE = re.compile(r"\\\(((?:[^\\\n]|\\[^()\n])+?)\\\)")

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


def fix_ocr_artifacts(text: str, *, reference: bool | None = None) -> str:
    """Replace common OCR artifacts with correct characters.

    Handles ligatures (ff, fi, fl, ffi, ffl), soft hyphens, and the
    full-width punctuation GLM-OCR occasionally substitutes into Latin text.

    Args:
        text: Input text potentially containing OCR artifacts.
        reference: Whether the text is a reference entry or list (True), any other
            region (False), or not known (None, the default). The shapes that rejoin
            a DOI broken after a bracket, an underscore or a dot, or inside a word
            that opens upper-case, run on reference text and on text of unknown
            kind, not on a body paragraph, a caption or a title.

    Returns:
        Cleaned text.
    """
    ref_shapes = reference is not False
    text = normalize_unicode(text)
    text = _resolve_stx_marks(text)
    text = CONTROL_CHAR_RE.sub("", text)
    for artifact, replacement in _OCR_ARTIFACT_MAP.items():
        text = text.replace(artifact, replacement)
    text = _bridge_url_linewraps(text)
    text = _bridge_doi_url_crlf_wraps(text, ref_shapes)
    # Wraps bridged above can bring a hyphen wrap into a URL that began on an earlier line.
    text = _bridge_url_linewraps(text)
    text = _bridge_doi_midword_wraps(text, ref_shapes)
    text = _bridge_doi_dot_wraps(text, ref_shapes)
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
    stripped = text.strip()
    m = _LATEX_TEXT_WRAP_RE.match(stripped)
    if m:
        parts = _iter_latex_text_inners(m.group(1))
    else:
        parts = _balanced_wrapped_text_inners(stripped)
        if parts is None:
            return text
    return " ".join(p.strip() for p in parts if p.strip())


def _balanced_wrapped_text_inners(text: str) -> list[str] | None:
    r"""The ``\text{...}`` arguments of a ``$$\text{...}$$`` whose arguments
    nest braces (``[^}]*`` in :data:`_LATEX_TEXT_WRAP_RE` stops at the first
    ``}``), or ``None`` when *text* is not such a wrapper."""
    if not (text.startswith("$$") and text.endswith("$$")):
        return None
    body, parts, pos = text[:-2], [], 2
    while True:
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos == len(body):
            return parts or None
        m = _LATEX_TEXT_PREFIX_RE.match(body, pos)
        if m is None:
            return None
        depth, i = 1, m.end()
        while i < len(body) and depth:
            depth += (body[i] == "{") - (body[i] == "}")
            i += 1
        if depth:
            return None
        parts.append(body[m.end() : i - 1])
        pos = i


def _inline_math_spans(text: str, *, strict: bool = False) -> Iterator[re.Match[str]]:
    """Yield the ``$ ... $`` spans :func:`strip_inline_math` unwraps.

    A ``$`` immediately followed by a digit is a currency amount, not a math
    delimiter — otherwise two amounts in one sentence (``$5 … $10``) would pair
    up as a math span and the text between them would be fused.  When a
    currency ``$`` is skipped, scanning resumes right after it so a later real
    math span can still match.  ``strict`` applies the tighter delimiter rule
    of :data:`_STRICT_INLINE_MATH_RE`, for text that is not OCR output.
    """
    pattern = _STRICT_INLINE_MATH_RE if strict else _INLINE_MATH_RE
    pos = 0
    while (m := pattern.search(text, pos)) is not None:
        if text[m.start() + 1].isdigit():
            # Currency, not math: let its closing ``$`` candidate re-open the
            # next search.
            pos = m.start() + 1
            continue
        yield m
        pos = m.end()


def strip_inline_math(text: str) -> str:
    r"""Strip inline math delimiters ``$ ... $`` from body text.

    The OCR engine wraps inline statistics like ``t(97.7)=2.9`` in single-dollar
    math delimiters: ``$ t(97.7)=2.9 $``.  This strips the delimiters and keeps
    the content as plain text.  Double-dollar display math (``$$...$$``) is not
    affected, and neither is a currency ``$`` (see :func:`_inline_math_spans`).
    """
    out: list[str] = []
    pos = 0
    for m in _inline_math_spans(text):
        out.append(text[pos : m.start()])
        out.append(m.group(1))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def strip_affiliation_markers(text: str) -> str:
    r"""Strip LaTeX superscript affiliation markers from text.

    Removes patterns like ``^{1}``, ``^{1,2}``, ``$ ^{1} $``, ``^{†}``
    that OCR engines produce for author–affiliation superscripts.
    Collapses any resulting extra whitespace.
    """
    cleaned = _AFFILIATION_MARKER_RE.sub("", text)
    return " ".join(cleaned.split())


def strip_latex_commands(text: str) -> str:
    r"""Strip residual LaTeX commands from math text.

    Handles the common patterns in OCR and inline-equation math:

    * ``\mathrm{...}``, ``\text{...}``, ``\textit{...}``, ``\textbf{...}``
      -- unwrapped to their content.
    * ``\Delta``, ``\alpha``, ``\varDelta``, etc. -- converted to Unicode.
    * ``_{...}`` and ``^{...}`` -- flattened to just the content inside braces,
      and so are the single-character forms ``_x`` and ``^x``.
    * Any remaining ``\command`` -- the backslash prefix is removed.

    Whitespace runs collapse to one space. Meant for math: in prose the
    single-character rule deletes the underscore of ``age_group`` and turns
    ``10^6`` into ``106`` (see :func:`clean_text_content_late`).
    """
    return " ".join(_flatten_latex(text, single_char=True).split())


def _flatten_latex(text: str, *, single_char: bool) -> str:
    """The rewrites of :func:`strip_latex_commands`, leaving whitespace alone.

    ``single_char=False`` keeps bare ``_x`` / ``^x``, which outside a math
    span are far more often an identifier, a file name or an exponent than
    LaTeX.
    """
    result = text

    # 1. Unwrap text-mode commands: \mathrm{n=5} -> n=5
    result = _LATEX_TEXT_CMD_RE.sub(r"\1", result)

    # 2. Flatten subscripts and superscripts: _{c} -> c, ^{2} -> 2
    result = _LATEX_SUB_RE.sub(r"\1", result)
    result = _LATEX_SUP_RE.sub(r"\1", result)
    if single_char:
        result = _LATEX_SUB_SINGLE_RE.sub(r"\1", result)
        result = _LATEX_SUP_SINGLE_RE.sub(r"\1", result)

    # 3. Replace known LaTeX symbol commands with Unicode equivalents.
    result = _LATEX_SYMBOL_RE.sub(lambda m: _LATEX_SYMBOL_MAP[m.group()], result)

    # 4. Strip any remaining \command sequences (remove backslash + command name).
    result = _LATEX_BACKSLASH_CMD_RE.sub(r"\1", result)

    # 5. Clean up leftover empty brace pairs
    result = result.replace("{}", "")
    return result.replace("{ }", "")


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


# Email addresses: masked like URLs, since the local part is exactly where
# identifiers put underscores (``john_smith@uni.edu``). The local part starts
# where a run of its characters starts: unanchored, a long token without an
# ``@`` was rescanned from every position, quadratic in its length.
_EMAIL_PATTERN = r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"
_PROTECTED_SPAN_RE = re.compile(f"{URL_RE.pattern}|{_EMAIL_PATTERN}", re.IGNORECASE)

# Mask placeholders: NUL-delimited indices survive every late-phase transform
# (no whitespace, no LaTeX metacharacters).
_PROTECTED_MASK_RE = re.compile("\x00(\\d+)\x00")


def _mask_protected_spans(text: str, math: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Replace URL and email spans with ``\\x00<i>\\x00`` placeholders.

    Each ``$…$`` span in *math* is replaced by a placeholder for its flattened
    content (see ``clean_text_content_late``'s ``inline_math``).
    """
    spans: list[str] = []

    def _hold(value: str) -> str:
        spans.append(value)
        return f"\x00{len(spans) - 1}\x00"

    masked = _PROTECTED_SPAN_RE.sub(lambda m: _hold(m.group(0)), text)
    wanted = sorted({span for span in math if len(span) > 1}, key=len, reverse=True)
    if wanted:
        # One left-to-right pass, longest span first at each position: two
        # equations around a plain letter ("$a$x$b$") never pair up as "$x$".
        spans_re = re.compile("|".join(map(re.escape, wanted)))
        masked = spans_re.sub(lambda m: _hold(strip_latex_commands(m.group(0)[1:-1])), masked)
    return masked, spans


def _restore_protected_spans(text: str, spans: list[str]) -> str:
    if not spans:
        return text
    return _PROTECTED_MASK_RE.sub(lambda m: spans[int(m.group(1))], text)


def _flatten_ocr_prose(text: str) -> str:
    r"""Repair the LaTeX OCR leaks into prose outside ``$…$`` spans.

    ``^{3}``, ``\alpha`` and ``\mathrm{…}`` are flattened, but a bare ``_x``
    or ``^x`` is kept: in prose it is an identifier, a file name or an
    exponent. Inside the ``\(…\)`` inline math Paddle-OCR-VL writes it is
    math, so that span is flattened in full, as ``$…$`` is; its delimiters stay.
    """
    parts: list[str] = []
    pos = 0
    for m in _PAREN_INLINE_MATH_RE.finditer(text):
        parts.append(_flatten_latex(text[pos : m.start()], single_char=False))
        parts.append(f"\\({_flatten_latex(m.group(1), single_char=True)}\\)")
        pos = m.end()
    parts.append(_flatten_latex(text[pos:], single_char=False))
    return "".join(parts)


def clean_text_content_late(
    text: str, *, from_ocr: bool = True, inline_math: Iterable[str] = ()
) -> str:
    """Late-phase cleaning — strips inline math delimiters and LaTeX commands.

    Must run **after** citation linking, equation extraction, and xref
    detection, because those stages depend on patterns like ``^{3}`` and
    ``$...$`` that this function removes.

    Inline math spans are unwrapped and their LaTeX flattened in full
    (:func:`strip_latex_commands`). In OCR text these are the ``$...$`` spans
    :func:`strip_inline_math` finds. Text a parser read from the document
    itself (``from_ocr=False``: DOCX, JATS, HTML, ePub, the PDF text layer)
    uses literal dollar signs, so only a tightly delimited span counts there
    (:data:`_STRICT_INLINE_MATH_RE`): JATS tex-math is written that way, while
    ``df$age_group`` and "US$ 5" are left alone. A pair of literal dollars
    that happens to meet that rule ("$/cap/yr … MJ/$") is still unwrapped.
    ``inline_math`` lists the ``$…$`` spans the parser itself wrote into the
    text (DOCX inline equations, ``PaperSentence.inline_math``); these are
    unwrapped wherever they sit, glued to a word included ("the $n$th" ->
    "the nth").

    Outside those spans only OCR text is touched: OCR leaks ``^{3}``,
    ``\\alpha`` and ``\\mathrm{…}`` into prose and spaces out characters
    (``1 7. 9 0 6``); see :func:`_flatten_ocr_prose`. Text from the document
    keeps its prose as written — "a 2 x 2 design", ``age_group``, ``10^6``.
    The default treats unknown text as OCR, so the repairs still reach it.
    Whitespace runs collapse to one space for every source.

    URL and email spans are masked during cleaning — LaTeX stripping would
    otherwise corrupt them (the single-character subscript rule flattens
    ``…/to_err_is_human`` to ``…/toerrishuman``).
    """
    masked, protected = _mask_protected_spans(text, inline_math)
    parts: list[str] = []
    pos = 0
    for m in _inline_math_spans(masked, strict=not from_ocr):
        outside = masked[pos : m.start()]
        parts.append(_flatten_ocr_prose(outside) if from_ocr else outside)
        parts.append(strip_latex_commands(m.group(1)))
        pos = m.end()
    outside = masked[pos:]
    parts.append(_flatten_ocr_prose(outside) if from_ocr else outside)
    result = " ".join("".join(parts).split())
    if from_ocr:
        # Collapse OCR character-spacing in regular text (e.g. "9 0 6" → "906").
        # This handles spaced text that came from formula OCR regions merged
        # into body text, or from the text OCR prompt on dense statistical
        # notation.
        result = _collapse_bare_spaced_runs(result)
    return _restore_protected_spans(result, protected)


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


_MATH_OPERATORS = frozenset("=+<>~≤≥≠±∓×")


def _has_math_operator(run: list[str]) -> bool:
    """Whether a run of single-character tokens contains an operator.

    A letter ``x`` between two digits is a multiplication sign ("a 2 x 2
    design"), not a spaced-out variable name.
    """
    for i, char in enumerate(run):
        if char in _MATH_OPERATORS:
            return True
        if char in "xX" and 0 < i < len(run) - 1 and run[i - 1].isdigit() and run[i + 1].isdigit():
            return True
    return False


def _collapse_bare_spaced_runs(text: str) -> str:
    """Collapse bare character-spaced sequences from OCR formula recognition.

    OCR engines sometimes space out every character in statistical notation,
    producing ``F ( 1, 2 6 )=1 7. 9 0 6`` instead of ``F(1,26)=17.906``.

    Strategy: split on spaces, find maximal runs of single-character tokens
    (length == 1), and collapse runs of 3+ that contain at least one digit
    and no mathematical operators.  The operator check preserves intentional
    spacing in expressions like ``x = 5``, ``a + b`` or ``2 x 2``.  Prose has
    such runs too ("Items 1 2 3 and 4"), so only OCR text may be collapsed.
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
            if len(run) >= 3 and has_digit and not _has_math_operator(run):
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
