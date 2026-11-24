import re
import unicodedata
from typing import NamedTuple

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


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def strip_lone_surrogates(text: str) -> str:
    """Replace UTF-16 surrogate code points with U+FFFD.

    OCR backends can emit lone surrogates: valid Python ``str`` code points
    that have no UTF-8 encoding. Hugging Face fast tokenizers reject a batch
    containing one with ``TextEncodeInput``. Replacement preserves string
    length, so tokenizer offset mappings remain aligned.
    """
    if not text or text.isascii():
        return text
    if _SURROGATE_RE.search(text) is None:
        return text
    return _SURROGATE_RE.sub("\ufffd", text)


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
    # The parens are counted once and the end walked back: slicing and counting
    # again per character is quadratic in a long trailing run.
    opens, closes = url.count("("), url.count(")")
    end = len(url)
    while end:
        last = url[end - 1]
        if last in ".,;:!?":
            end -= 1
        elif last == ")" and closes > opens:
            closes -= 1
            end -= 1
        else:
            break
    return url[:end]


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

# A DOI as printed in running text, for the paper-DOI extractors
# (doi_identity, front_matter). The suffix class stops at whitespace and prose
# punctuation. Wiley's 1996–2004 SICI DOIs
# ("10.1002/(SICI)1097-4679(199901)55:1<1::AID-JCLP1>3.0.CO;2-K") also carry an
# angle-bracketed ``item::AID-code`` segment. Brackets are accepted only in that
# ``::`` shape, so an inline tag such as ``<sup>`` never becomes part of a DOI.
# Both parts are bounded and the first excludes ``:``, so a long unclosed
# ``<:::`` run (OCR separator lines) cannot make the match backtrack.
DOI_CANDIDATE_RE = re.compile(
    r"\b"
    + DOI_BODY
    + r"(?:[-._;()/:A-Za-z0-9]|<[-._;()/A-Za-z0-9]{0,40}::[-._;()/:A-Za-z0-9]{0,60}>)+",
    re.IGNORECASE,
)

# Typeset hyphens (U+2010 and kin) stop DOI_CANDIDATE_RE early, which cuts a DOI
# down to its first hyphen-free stretch; read them as ASCII hyphens, and drop
# soft hyphens.
DOI_HYPHENS = str.maketrans(
    {"\u00ad": None, **dict.fromkeys("\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d", "-")}
)

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


# Trailing punctuation that shouldn't be part of a DOI. The lookbehind starts a
# match only where a run of it starts: tried from inside a long run that does not
# end the string, ``[...]+$`` backtracks quadratically. (``rstrip`` would differ
# where ``$`` matches before a final newline.)
_DOI_TRAILING_JUNK_RE = re.compile(r"(?<![.,;:)\]}>])[.,;:)\]}>]+$")

# URL-like patterns that get concatenated onto DOIs by OCR (no whitespace between
# the DOI and the following URL on the page).  Truncate at the first match.
_DOI_EMBEDDED_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)

# HighWire-hosted journals (PNAS, Genetics, J Neurosci, ...) link supplementary
# material as ``…/lookup/suppl/doi:10.1073/pnas.2501823122/-/DCSupplemental``
# (or ``/-/DC1``); the candidate regex reads the path tail as DOI suffix. No DOI
# has a ``/-/`` segment, so truncate there.
_DOI_SUPPLEMENT_PATH_RE = re.compile(r"/-/DC(?:Supplemental|\d+)\b.*$", re.IGNORECASE)

# A publisher link that carries a DOI often adds a query to it ("…/10.12968/bjca.2019.0072
# ?journalCode=bjca", "?via%3Dihub", "?download=true"); a DOI holds no "?name=", so the
# DOI ends before it.
_DOI_URL_QUERY_RE = re.compile(r"\?[A-Za-z_][\w.%-]*(?:=|%3D).*$", re.DOTALL)


def normalize_doi(doi: str | None) -> str | None:
    """Normalize a DOI string to its bare form (e.g. ``10.1016/j.foo.2024``).

    Strips URL prefixes (https://doi.org/..., doi:, etc.), trailing punctuation,
    embedded URL suffixes (from OCR concatenation), HighWire supplement paths
    (``/-/DCSupplemental``), a link's query (``?journalCode=…``), and validates the result looks
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

    # Only the last line can hold it (".*$" stops at a newline); trying the
    # pattern from every "/-/DC1" before a newline is quadratic.
    last_line = cleaned.rfind("\n", 0, len(cleaned) - 1) + 1
    cleaned = cleaned[:last_line] + _DOI_SUPPLEMENT_PATH_RE.sub("", cleaned[last_line:])

    cleaned = _DOI_URL_QUERY_RE.sub("", cleaned)

    # Strip trailing punctuation again (embedded URL removal may expose new junk)
    cleaned = _DOI_TRAILING_JUNK_RE.sub("", cleaned)

    # Collapse repeated slashes (OCR artifact: "10.1037//0096..." → "10.1037/0096...").
    # Done after URL stripping so "https://" etc. aren't accidentally normalized first.
    cleaned = re.sub(r"/+", "/", cleaned)

    # Validate: must match bare DOI pattern
    if not _BARE_DOI_RE.match(cleaned):
        return None

    return cleaned


class TextLink(NamedTuple):
    """A link printed in running text: where it is and what it points to."""

    start: int
    end: int
    # The link target. A URL printed without a scheme gets "https://"; a DOI
    # printed bare ("doi:10.5281/zenodo.1234") becomes its doi.org URL.
    url: str
    # The printed form when it is not the target itself, else None.
    link_text: str | None


# A second URL printed straight after the first with no space between them
# ("…/CompartmentShiny):https://doi.org/10.5281/…"). A scheme after "/" or "="
# belongs to the first URL: an archived copy ("web.archive.org/web/2020/https://…")
# or a query value ("?url=https://…", ";origin=https://…").
_GLUED_URL_RE = re.compile(r"(?<![/=])https?://", re.IGNORECASE)

# "host/path" printed without a scheme. What may not stand before it: a letter
# or digit (the match would start inside a word), "@" (an email address), "/"
# and "." (a file path or the rest of a longer host), the characters that join
# the parts of a URL, and the invisible characters (zero-width space and joiners,
# word joiner, BOM, soft hyphen) a typesetter puts inside a long URL to let it
# wrap. The path is required: a bare "name.tld" is as often a company, a file
# or two sentences with no space between them as it is a link.
_SCHEMELESS_URL_RE = re.compile(
    r"(?<![\w@/.:=~%+#&?\-\u200b\u200c\u200d\u2060\ufeff\u00ad])"
    r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,6}([a-z]{2,24}))"
    r"(/[^\s<>\"'\]},;@]+)",
    re.IGNORECASE,
)

# Hosts papers print without a scheme, whatever their top-level domain: DOI and
# short-link resolvers, and the repositories data and code are deposited in.
# A subdomain of one counts too ("gist.github.com", "www.osf.io").
_SCHEMELESS_KNOWN_HOSTS = (
    "arxiv.org",
    "aspredicted.org",
    "bit.ly",
    "bitbucket.org",
    "doi.org",
    "figshare.com",
    "github.com",
    "gitlab.com",
    "hdl.handle.net",
    "huggingface.co",
    "openneuro.org",
    "osf.io",
    "psyarxiv.com",
    "researchbox.org",
    "zenodo.org",
)

# Top-level domains read as a link on any other host. Left out on purpose are
# those that are also a file extension, an English word or a statistic's name
# ("analysis.py/", "README.md/", "beta.se/", "is.it/", "train.dev/test"): a
# missed link is cheaper than a made-up one.
_SCHEMELESS_TLDS = frozenset(
    {
        "com",
        "org",
        "net",
        "edu",
        "gov",
        "mil",
        "int",
        "io",
        "info",
        "eu",
        "uk",
        "de",
        "fr",
        "nl",
        "ch",
        "ca",
        "au",
        "jp",
        "cn",
        "dk",
        "fi",
        "es",
        "pt",
        "br",
        "kr",
        "nz",
        "ie",
    }
)

# A DOI printed without the doi.org URL, with or without a "doi:" label. The
# characters that may not stand before it keep the match out of a longer number
# or name; "/" may, because funders print their DOI after their name
# ("MICIU/AEI/10.13039/501100011033"). The suffix class is DOI_CANDIDATE_RE's
# plus the typeset hyphens DOI_HYPHENS reads as "-".
_BARE_DOI_IN_TEXT_RE = re.compile(
    r"(?<![\w.\-])(?:doi\s?:\s?)?(" + DOI_BODY + r"[-._;()/:A-Za-z0-9‐‑‒–−﹣－]+)",
    re.IGNORECASE,
)

# A funder's DOI in the Open Funder Registry is its prefix and a number, nothing
# more. Funders print it inside a longer string cut up by slashes
# ("AEI/10.13039/501100011033/FEDER, UE"), so it ends where the digits do.
_FUNDER_DOI_RE = re.compile(r"10\.13039/\d+")


def _is_linkable_host(host: str, tld: str) -> bool:
    """Whether a scheme-less *host* is read as a link (see the two lists above)."""
    host = host.lower()
    if any(host == known or host.endswith("." + known) for known in _SCHEMELESS_KNOWN_HOSTS):
        return True
    if tld.lower() not in _SCHEMELESS_TLDS:
        return False
    # The name before the top-level domain must hold a letter: "1.2.com" is not a host.
    return any(ch.isalpha() for ch in host[: -len(tld) - 1])


def find_text_links(text: str, *, bare: bool = True) -> list[TextLink]:
    """Links printed in *text*, in reading order.

    Always found: a URL with a scheme or a leading "www." (``URL_RE``), trimmed by
    :func:`clean_extracted_url`. Two URLs printed with no space between them are
    two links.

    With *bare* (the default) also found, where no such URL already covers them:

    - a URL printed without a scheme ("osf.io/s6bjp", "github.com/org/repo"), on a
      host papers are known to print that way or under a common top-level domain,
      and only with a path;
    - a DOI printed without one ("doi:10.6084/m9.figshare.1", "(DOI: 10.15154/z563)").

    Their ``url`` is the full target and ``link_text`` the printed form. Pass
    ``bare=False`` for text where a DOI is not a link the author put in the prose
    — a reference list, whose entries carry their DOI as a field of their own.
    """
    links: list[TextLink] = []
    raw_ends: set[int] = set()
    for match in URL_RE.finditer(text):
        raw = match.group(0)
        raw_ends.add(match.end())
        cuts = [m.start() for m in _GLUED_URL_RE.finditer(raw) if m.start() > 0]
        for lo, hi in zip([0, *cuts], [*cuts, len(raw)], strict=True):
            # Trim trailing prose delimiters the regex over-captures (sentence
            # punctuation, an unbalanced ')'); keeps balanced DOI parens.
            url = clean_extracted_url(raw[lo:hi])
            if url:
                start = match.start() + lo
                links.append(TextLink(start, start + len(url), url, None))
    if not bare:
        return links

    def overlaps(start: int, end: int) -> bool:
        return any(start < link.end and link.start < end for link in links)

    schemed = len(links)
    for match in _SCHEMELESS_URL_RE.finditer(text):
        start = match.start()
        if not _is_linkable_host(match.group(1), match.group(2)):
            continue
        # Straight after a URL that a line wrap cut short ("https://www.exam-
        # ple.com/data"), this is the rest of that URL and not a second link.
        if start and text[start - 1].isspace() and start - 1 in raw_ends:
            continue
        printed = clean_extracted_url(match.group(0))
        path = printed[len(match.group(1)) :]
        if not any(ch.isalnum() for ch in path) or overlaps(start, start + len(printed)):
            continue
        links.append(TextLink(start, start + len(printed), "https://" + printed, printed))

    for match in _BARE_DOI_IN_TEXT_RE.finditer(text):
        start = match.start(1)
        funder = _FUNDER_DOI_RE.match(match.group(1))
        if funder:
            printed = funder.group(0)
        else:
            # No DOI ends on "/": one printed there separates it from what follows.
            printed = clean_extracted_url(match.group(1).rstrip("/")).rstrip("/")
        doi = normalize_doi(printed.translate(DOI_HYPHENS))
        if doi is None or overlaps(start, start + len(printed)):
            continue
        links.append(TextLink(start, start + len(printed), "https://doi.org/" + doi, printed))

    if len(links) > schemed:
        links.sort(key=lambda link: link.start)
    return links


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


class CollapsedLength:
    """``len(collapse_ws(text))`` of a text that grows at its end, in linear time.

    Collapsing all of it again after every addition is quadratic. The text is
    normalized a stretch at a time instead, cut before a starter NFC does not
    join to the character before it, so that the stretches normalize as the
    whole text does; a whitespace run across a cut counts once. Only the
    stretch after the last cut is redone.
    """

    # The open stretch is closed at this length even with no cut in it (a run
    # of combining marks, or of starters that compose), so an addition or a
    # length() redoes a bounded amount of text. That is exact unless what
    # follows composes with the stretch's end.
    _MAX_OPEN = 1024
    # The starters tried as a cut, from the last: more than any run of starters
    # NFC joins into one (Hangul LVT is three).
    _CUT_TRIES = 4

    def __init__(self) -> None:
        self.last = ""  # the last character added
        self._open = ""
        # The text before the open stretch, normalized and collapsed but not
        # stripped: its length, whether it starts with a space (None while it
        # is empty) and whether it ends with one.
        self._closed: tuple[int, bool | None, bool] = (0, None, False)

    def add(self, text: str) -> None:
        """Append *text*."""
        if not text:
            return
        self.last = text[-1]
        stretch = self._open + text
        # Cut before the last character of *text* that decomposes to a starter
        # (nothing after a starter reorders or composes across it) and does not
        # compose with the character before it.
        cut = len(stretch)
        floor = max(len(self._open), 1)
        for _ in range(self._CUT_TRIES):
            cut -= 1
            while cut >= floor and unicodedata.combining(
                unicodedata.normalize("NFD", stretch[cut])[0]
            ):
                cut -= 1
            if cut < floor:
                break
            head = unicodedata.normalize("NFC", stretch[:cut])
            before, starter = head[-1], stretch[cut]
            nfc_starter = unicodedata.normalize("NFC", starter)
            if unicodedata.normalize("NFC", before + starter) == before + nfc_starter:
                self._closed = _collapsed_after(self._closed, head)
                stretch = stretch[cut:]
                break
        if len(stretch) > self._MAX_OPEN:
            self._closed = _collapsed_after(self._closed, unicodedata.normalize("NFC", stretch))
            stretch = ""
        self._open = stretch

    def length(self, tail: str = "") -> int:
        """The collapsed length of the text added, followed by *tail*."""
        size, starts_space, ends_space = _collapsed_after(
            self._closed, unicodedata.normalize("NFC", self._open + tail)
        )
        return max(size - bool(starts_space) - ends_space, 0)


def _collapsed_after(
    closed: tuple[int, bool | None, bool], normalized: str
) -> tuple[int, bool | None, bool]:
    """*closed* (see :class:`CollapsedLength`) with *normalized* collapsed after it."""
    if not normalized:
        return closed
    size, starts_space, ends_space = closed
    done = _WS_RUN_RE.sub(" ", normalized)
    if starts_space is None:
        starts_space = done[0] == " "
    elif ends_space and done[0] == " ":
        size -= 1  # one whitespace run across the cut
    return size + len(done), starts_space, done[-1] == " "


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
