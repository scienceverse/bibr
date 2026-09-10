import re
import unicodedata

# Raw PDF/native extraction and OCR can occasionally emit C0 control bytes in
# readable text. Keep whitespace controls intact; strip the rest before text is
# used for lookup/classification.
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# C0/DEL controls that signal *corrupted* OCR output (the vllm-mlx --mllm NUL
# failure mode). Excludes whitespace controls and STX (\x02), which GLM-OCR
# emits deliberately as a line-end soft-hyphen marker (resolved contextually
# by consolidate_text._resolve_stx_marks).
OCR_CORRUPTION_CHAR_RE = re.compile(r"[\x00\x01\x03-\x08\x0b\x0c\x0e-\x1f\x7f]")

# A single stray control char happens in healthy text (broken ToUnicode CMaps
# leak lone C0 glyphs); two or more in one region means the OCR output itself
# is corrupt.
OCR_CORRUPTION_MIN_CHARS = 2


def ocr_corruption_count(text: str) -> int:
    """Number of control chars in *text* that indicate corrupted OCR output."""
    if not text:
        return 0
    return len(OCR_CORRUPTION_CHAR_RE.findall(text))


_SURROGATE_LO = 0xD800
_SURROGATE_HI = 0xDFFF


def strip_lone_surrogates(text: str) -> str:
    """Replace UTF-16 surrogate code points with U+FFFD.

    OCR backends can emit lone surrogates: valid Python ``str`` code points
    that have no UTF-8 encoding. Hugging Face fast tokenizers reject a batch
    containing one with ``TextEncodeInput``. Replacement preserves string
    length, so tokenizer offset mappings remain aligned.
    """
    if not text:
        return text
    if not any(_SURROGATE_LO <= ord(ch) <= _SURROGATE_HI for ch in text):
        return text
    return "".join("\ufffd" if _SURROGATE_LO <= ord(ch) <= _SURROGATE_HI else ch for ch in text)


# Regex for detecting plain-text URLs (replaces spaCy's token.like_url).
# ``@`` is excluded so that email-like fragments (e.g. ``www.x.com@attacker``)
# don't match as a single URL.
# Lives here (leaf module) so both bibr.structure and bibr.input can use it
# without an import cycle; bibr.structure.xref_utils re-exports it.
URL_RE = re.compile(
    r"(?:https?://[^\s<>\"'\]},;@]+|www\.[^\s<>\"'\]},;@]+)",
    re.IGNORECASE,
)


def clean_extracted_url(url: str) -> str:
    """Trim trailing chars a regex over prose wrongly captures: sentence
    punctuation and an UNBALANCED trailing ')'. Keeps balanced DOI parens
    (e.g. '…S0140-6736(16)00427-X') and a trailing '/'."""
    while url:
        last = url[-1]
        if last in ".,;:!?" or last == ")" and url.count(")") > url.count("("):
            url = url[:-1]
        else:
            break
    return url


# DOI URL prefixes to strip (order matters — longest first)
_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "://doi.org/",
    "doi.org/",
    "doi:",
)

# DOI body: the "10.<registrant>/" prefix that opens every DOI. Shared atom for
# the bare-DOI validator (here), OCR wrap-bridging (consolidate_text), and
# reference locators (segment_filter).
DOI_BODY = r"10\.\d{4,9}/"

# Bare DOI pattern: registrant body then a non-whitespace suffix that does not
# end on a hyphen. A DOI printed across a line wrap breaks at one of its own
# hyphens ("10.1037/0033-" / "2909.115.1.102") and the leading fragment would
# otherwise validate — a suffix is a suffix. That costs twice over: the stub is
# exported as a DOI that resolves to nothing, and it suppresses the wrap rescue
# in ref_extractor, which is reached only when this returns None. Rejecting is
# safe and stripping is not: no DOI ends on a hyphen (0 of the 4787 distinct
# DOIs printed across the gold corpus), whereas "10.1037/0033" is a well-formed
# DOI that simply is not the one on the page.
_BARE_DOI_RE = re.compile(r"^" + DOI_BODY + r"[^\s]*[^\s-]$")

# Path segments that traverse out of a ``/works/{doi}`` URL if left in the path.
_DOI_TRAVERSAL_SEGMENTS = frozenset({"", ".", ".."})


def is_url_safe_doi(doi: str | None) -> bool:
    """True if *doi* can be interpolated into a ``/works/{doi}`` URL path safely.

    Guards the resolver (``clients/resolver.py``) and Crossref
    (``clients/crossref.py``) DOI-lookup paths against SSRF / path traversal: a
    DOI extracted from a malicious PDF bibliography such as ``10.1234/../jobs`` or
    ``../admin/search`` survives ``quote(doi, safe="/")`` (which preserves ``/``
    and ``.``) and httpx forwards ``..`` unnormalized, so it can escape
    ``/works/*`` and reach internal endpoints on the operator's resolver host.

    A DOI is safe only when it opens with the ``10.`` registrant prefix, contains
    a suffix slash, and has no ``.``/``..``/empty path segment. We deliberately
    anchor on ``10.`` rather than the stricter ``^10\\.\\d{4,9}/`` so
    short-registrant DOIs are not silently dropped from enrichment; the
    traversal-segment check is what actually closes the hole. Other unusual
    characters (spaces, ``#``, ``<``) are left to the caller's ``quote(...)`` to
    percent-encode — they cannot traverse and are not a URL-path escape.
    """
    if not doi or not isinstance(doi, str):
        return False
    if not doi.startswith("10.") or "/" not in doi:
        return False
    return not any(seg in _DOI_TRAVERSAL_SEGMENTS for seg in doi.split("/"))


# Trailing punctuation that shouldn't be part of a DOI
_DOI_TRAILING_JUNK_RE = re.compile(r"[.,;:)\]}>]+$")

# URL-like patterns that get concatenated onto DOIs by OCR (no whitespace between
# the DOI and the following URL on the page).  Truncate at the first match.
_DOI_EMBEDDED_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    """Normalize a DOI string to its bare form (e.g. ``10.1016/j.foo.2024``).

    Strips URL prefixes (https://doi.org/..., doi:, etc.), trailing punctuation,
    embedded URL suffixes (from OCR concatenation), and validates the result looks
    like a real DOI (starts with ``10.NNNN/``).

    Returns None if the input is empty or not a valid DOI after cleanup.
    """
    if not doi or not isinstance(doi, str):
        return None

    cleaned = doi.strip()

    # Strip known URL prefixes (case-insensitive)
    lower = cleaned.lower()
    for prefix in _DOI_URL_PREFIXES:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    # Strip trailing punctuation
    cleaned = _DOI_TRAILING_JUNK_RE.sub("", cleaned)

    # Strip embedded URL suffixes (OCR often concatenates the next URL on the page
    # directly onto the DOI, e.g. "10.1177/0956797618796480www.example.org/PS").
    # Search from after the first "/" to avoid matching the DOI prefix itself.
    slash_pos = cleaned.find("/")
    if slash_pos >= 0:
        suffix = cleaned[slash_pos + 1 :]
        m = _DOI_EMBEDDED_URL_RE.search(suffix)
        if m:
            cleaned = cleaned[: slash_pos + 1 + m.start()]

    # Strip trailing punctuation again (embedded URL removal may expose new junk)
    cleaned = _DOI_TRAILING_JUNK_RE.sub("", cleaned)

    # Collapse repeated slashes (OCR artifact: "10.1037//0096..." → "10.1037/0096...").
    # Done after URL stripping so "https://" etc. aren't accidentally normalized first.
    cleaned = re.sub(r"/+", "/", cleaned)

    # Validate: must match bare DOI pattern
    if not _BARE_DOI_RE.match(cleaned):
        return None

    return cleaned


# Runs of any whitespace — spaces, tabs, newlines, CR, NBSP. ``\s`` (str mode)
# already matches the literal space and U+00A0, so this is exactly the
# ``[\s ]+`` / ``\s+`` patterns the call sites used to reimplement.
_WS_RUN_RE = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    """Collapse every run of whitespace to a single space and strip the ends.

    Shared home for the identical whitespace normalisation that xref/citation
    text storage and reference-row joining each used to reimplement (line
    wraps, NBSPs, and multiple spaces leak in via OCR region joining).
    """
    return _WS_RUN_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalize_unicode(text: str) -> str:
    """Compose canonically equivalent text into Unicode NFC."""
    return unicodedata.normalize("NFC", text)


# Year-like token: a plausible publication year (1600-2099, optional
# disambiguation suffix), "n.d.", "in press", or "forthcoming". Shared by
# reference-entry detection (extractor) and short-segment filtering
# (segment_filter); byte-identical to the per-module regexes it replaces.
YEARISH_RE = re.compile(
    r"\b(?:1[6-9]|20)\d{2}[a-z]?\b|\(?\bn\.\s?d\.\)?|\bin press\b|\bforthcoming\b",
    re.IGNORECASE,
)


# Apostrophe variants: straight (U+0027), left-curly (U+2018), right-curly
# (U+2019). Built via chr() — mirroring citation_matcher._APOS — so an editor
# can't silently normalize the curly glyphs to a plain ASCII apostrophe.
_NAME_APOS = chr(0x27) + chr(0x2018) + chr(0x2019)
# Latin name characters for in-text author-year patterns: ASCII letters,
# Latin-1/Extended-A/B accented letters (À-ɏ == U+00C0–U+024F), the three
# apostrophe variants above, and the hyphen. A *superset* of the per-module
# classes it consolidates, so a surname matched in one detector is matched in
# every detector ("Müller", "O'Brien", "O'Brien").
NAME_CHAR_CLS = "[a-zA-ZÀ-ɏ" + _NAME_APOS + r"\-]"


# In-text "in press" / "forthcoming" citation label, shared by the two in-text
# citation detectors (citation_linker, citation_matcher). The extractor's
# reference-FIELD vocabulary is intentionally richer ("advance online
# publication", "manuscript submitted", "epub ahead") and stays separate —
# those forms appear in printed bib entries, not in body-text citations.
IN_PRESS_LABEL = r"in\s+press|forthcoming"


def normalize_text(text: str) -> str:
    """
    Removes digits, periods, and extra whitespace, then converts to lowercase.
    Also collapses spaced-out letters (OCR artifact from tracking/kerning),
    e.g. "A B S T R A C T" -> "abstract".
    Example: "1. Introduction..." -> "introduction"
    """
    if not text:
        return ""
    # Standardize normalization logic in one place
    clean = CONTROL_CHAR_RE.sub("", text)
    clean = re.sub(r"[\d\.]+", "", clean)
    clean = clean.strip().lower()
    # Collapse spaced-out single characters: "a b s t r a c t" -> "abstract"
    # Matches when the text consists of single chars separated by spaces.
    tokens = clean.split()
    if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
        clean = "".join(tokens)
    return clean
