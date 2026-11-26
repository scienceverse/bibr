"""Corresponding-author email harvester.

Bridges parsed PaperContents to LLM-extracted authors: finds explicit
"Corresponding author:" anchors in the OCR'd text, promotes the matching
author's ``corresponding`` flag, and attaches the anchored email. The footnote
"E-mail address: x@y (Initials Surname)" is read from the whole paper before the
proximity rules.

Previously inline methods on MetadataExtractor; moved here as a
self-contained class so the harvesting logic can be tested without
spinning up the full extractor.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.models import PaperAuthor
    from bibr.paper_contents import PaperContents

logger = logging.getLogger(__name__)

# Email pattern — covers most ASCII addresses; rejects bare-domain mentions.
_EMAIL_RE = re.compile(r"\b([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b")
# One domain shared by several local parts, as LaTeX bylines print it: "{alice, bob}@uni.edu".
_BRACED_EMAILS_RE = re.compile(
    r"\{\s*([\w.+-]+(?:\s*[,;|]\s*[\w.+-]+)*)\s*\}\s*@\s*([\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,})\b"
)


_KNOWN_WRAPPED_TLDS = frozenset(
    {
        "ac",
        "at",
        "au",
        "be",
        "br",
        "ca",
        "ch",
        "cn",
        "co",
        "com",
        "cz",
        "de",
        "dk",
        "edu",
        "es",
        "eu",
        "fi",
        "fr",
        "gov",
        "gr",
        "hk",
        "ie",
        "il",
        "in",
        "int",
        "ir",
        "it",
        "jp",
        "kr",
        "mil",
        "mo",
        "mx",
        "my",
        "net",
        "nl",
        "no",
        "nz",
        "org",
        "pl",
        "pt",
        "ro",
        "ru",
        "se",
        "sg",
        "th",
        "tr",
        "tw",
        "uk",
        "us",
        "vn",
        "za",
    }
)
_PROSE_AMBIGUOUS_TLDS = frozenset({"at", "be", "in", "it", "my", "no", "us"})
_SPACE_WRAPPED_TLD_RE = re.compile(r"([\w.+-]+@[\w.-]+\.)[ \t]+([a-z]{2,3})(?=$|[\s,;)\]}])")
_STANDALONE_EMAIL_LINE_RE = re.compile(r"^\s*[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\s*$")
_NON_CORR_STAR_NOTE_RE = re.compile(
    r"[*\u2217\u204e]\s*(?:these authors|equal(?:ly)?\s+contribut|co-?first|"
    r"deceased|present address|current address)",
    re.IGNORECASE,
)


def _repair_wrapped_emails(text: str) -> str:
    """Rejoin emails split across a line wrap, space-after-dot TLD wrap, or soft-hyphen."""
    if "@" not in text:
        return text
    if "\ufffe" in text:
        text = re.sub(r"([\w.+-]+@[\w.-]*)\ufffe([\w.-]+)", r"\1-\2", text)
    if "\n" in text or "\r" in text:
        text = re.sub(
            r"([\w.+-]+@)[ \t]*\r?\n[ \t]*([\w.-]+\.[A-Za-z]{2,})\b",
            r"\1\2",
            text,
        )
        text = re.sub(
            r"([\w.+-]+@[\w.-]*[.-])[ \t]*\r?\n[ \t]*([A-Za-z0-9.-]+\.[A-Za-z]{2,}|[A-Za-z]{2,})\b",
            r"\1\2",
            text,
        )

    def _join_tld(m: re.Match[str]) -> str:
        tld = m.group(2)
        if tld not in _KNOWN_WRAPPED_TLDS:
            return m.group(0)
        if tld in _PROSE_AMBIGUOUS_TLDS:
            after = text[m.end() :]
            # Do not join an English word like ". in our study" unless terminated by punctuation/end.
            if re.match(r"^[ \t]+[a-z]", after):
                return m.group(0)
        return f"{m.group(1)}{tld}"

    if ". " in text or ".\t" in text:
        text = _SPACE_WRAPPED_TLD_RE.sub(_join_tld, text)
    return text


def _expand_braced_emails(sentences: list) -> list:
    """*sentences*, with wrapped emails rejoined and braced address lists expanded."""

    def expand(m: re.Match) -> str:
        return ", ".join(f"{local}@{m.group(2)}" for local in re.split(r"\s*[,;|]\s*", m.group(1)))

    out = []
    for sent in sentences:
        text = sent.text or ""
        repaired = _repair_wrapped_emails(text)
        if "}" in repaired and "@" in repaired:
            repaired = _BRACED_EMAILS_RE.sub(expand, repaired)
        if repaired != text:
            sent = SimpleNamespace(
                text=repaired,
                section_id=sent.section_id,
                page_number=getattr(sent, "page_number", None),
            )
        out.append(sent)
    return out


# Anchors that indicate a "Corresponding Author" footer (vs. a plain byline
# listing every author's email).  Used to gate the corresponding=True
# promotion — without this gate, papers like "Attention Is All You Need"
# (8 authors all listing name@google.com in the byline) would have every
# author flagged as corresponding.
_CORRESPONDING_MARKER_RE = re.compile(
    r"\bcorrespond"  # corresponding, correspondence
    r"|✉"  # envelope glyph
    # Taylor & Francis page-1 correspondence footer: "CONTACT Jane Doe jane@uni.edu ..."
    # Kept strictly uppercase and line-anchored so prose ("contact us") never fires.
    r"|(?:^|[\r\n])\s*[\W\d_]*(?-i:CONTACT)\b"
    # GLM-OCR misread of envelope at line start; the capital stays case-sensitive so
    # a unit ("10 pg ml-1") is not read as a marker.
    r"|\b(?i:pg)\s+(?-i:(?=[A-Z]))",
    re.IGNORECASE,
)


def has_correspondence_cue(text: str | None) -> bool:
    """Whether *text* carries an explicit correspondence or ``CONTACT`` anchor."""
    return bool(text and _CORRESPONDING_MARKER_RE.search(text))


# Window of adjacent sentences to consider when matching an email to an
# author name — the "Corresponding Author: <Name> ... E-mail: <addr>" line
# is often emitted as two separate sentences by the segmenter.
_EMAIL_NAME_WINDOW = 3


def _drop_filled_author(
    authors_by_family: dict[str, list[PaperAuthor]],
    family: str,
    author: PaperAuthor,
) -> None:
    """Retire one filled author, keeping any same-surname co-authors eligible."""
    group = authors_by_family.get(family)
    if group is None:
        return
    remaining = [a for a in group if a is not author]
    if remaining:
        authors_by_family[family] = remaining
    else:
        del authors_by_family[family]


def _given_name_affinity(given: str | None, local_compact: str) -> int:
    """Evidence that an email local part belongs to *this* given name.

    Used only to separate co-authors who share a surname, where the surname
    itself cannot discriminate. A full given name inside the local part
    ("jane.smith") is strong; a leading initial ("jsmith") is weak.
    """
    tokens = [
        token
        for spelling in _letter_spellings(given)
        for token in re.findall(r"[a-z0-9]+", spelling)
    ]
    if not tokens or not local_compact:
        return 0
    if any(len(token) >= 3 and token in local_compact for token in tokens):
        return 2
    return 1 if local_compact.startswith(tokens[0][0]) else 0


def _email_name_affinity(given: str | None, family: str | None, local_compact: str) -> bool:
    """Whether an email local part plausibly names the author.

    Gates the email assignment itself (not just the corresponding flag): a
    surname or given-name token (3+ chars) inside the local part, or the
    local part inside such a token. Covers 'jane.doe' for Doe, 'yxiangmind'
    for Xiang-Min Yang, and diminutives like 'bathri' for Bathrinath.
    Without this gate any nearby unrelated address (editorial office, lab,
    journal) is handed to whoever is still emailless, and the author's real
    marker-anchored address is then blocked.
    """
    if not local_compact or len(local_compact) < 3:
        return False
    for name in (given or "", family or ""):
        # Accents folded first: the bare letters of "Šimić" are "imi", which
        # turns up in unrelated addresses.
        for spelling in _letter_spellings(name):
            for token in re.findall(r"[a-z0-9]+", spelling):
                if len(token) >= 3 and (token in local_compact or local_compact in token):
                    return True
    # A 2-letter family name (Li, Wu, He) cannot match by containment — it
    # would hit any address — but as the address's leading letters ('lixh@'
    # for Xiaohong Li) it still names the author. Given names stay at 3+
    # chars: a 2-letter given token ('Yu' in Yu-Zhong) prefix-matches far too
    # often and breaks same-surname disambiguation.
    return any(
        len(family_compact) == 2 and local_compact.startswith(family_compact)
        for family_compact in _folds(family)
    )


# --- name-paired e-mails -------------------------------------------------------
# Elsevier prints the corresponding author's address in a page footnote, paired
# with the name in parentheses: "E-mail address: x@y.org (J.K. Okafor)." or
# "E-mail addresses: a@y.org (A. Lindqvist), b@z.org (B. Okafor)." The pairing is explicit,
# so it is matched against the author list by surname and initials, which makes
# the footnote safe to read wherever it landed (it usually sits outside the
# front-matter block and is often typed as a footnote or even a references section).
_EMAIL_LABEL_RE = re.compile(r"\be-?\s?mail(?:\s+address(?:es)?)?\s*:", re.IGNORECASE)
_NAMED_EMAIL_RE = re.compile(_EMAIL_RE.pattern + r"\s*\(\s*([^()@;:]{3,70}?)\s*\)")
# A "Corresponding author(s)" / "Correspondence" / envelope phrase that OPENS a line or
# sentence, as a footnote marker does ("* Corresponding author at: ..."). Anchoring it keeps
# body prose ("multiple correspondence analysis ... (e-mail: x@y.org (H.K. Tan))") out, and
# "Correspondence" counts only as a label: followed by ":" or "*", or by "to", "concerning"
# or "should" ("Correspondence between the two raters was high" is prose).
_CORRESPONDENCE_PHRASE_RE = re.compile(
    r"^[\W\d_]*(?:(?:co-?)?corresponding\s+authors?\b"
    r"|(?-i:CONTACT)\b"
    r"|(?:author\s+for\s+|for\s+|address\s+)?correspondence(?=\s*[:*]|\s+(?:to|concerning|should)\b)|✉)",
    re.IGNORECASE | re.MULTILINE,
)
# Explicit correspondence cue prefixed with a star ("*Correspondence:",
# "*CORRESPONDENCE", "*For correspondence:") — ties byline stars to
# corresponding authors even when the footnote email has no surname match.
_STAR_CORRESPONDENCE_CUE_RE = re.compile(
    r"^[\s]*(?:\\\(\s*\^\s*\{\s*\*\s*\}\s*\\\)|[*∗⁎])\s*"
    r"(?:(?:co-?)?corresponding\s+authors?\b|(?:for\s+)?correspondence\b)",
    re.IGNORECASE | re.MULTILINE,
)
# Parenthetical author initials directly after an email, as PLOS and eLife print:
# "jndoe@uni.es (JND)" or "qdu@uni.edu.cn (QD)".
_INITIALS_AFTER_EMAIL_RE = re.compile(r"^\s*\(\s*([A-Z](?:\.?[\s\-]?[A-Z]){1,4}\.?)\s*\)")
# Back-matter correspondence statement (Nature Communications & similar):
# "Correspondence and requests for materials should be addressed to Anne C. Ferguson-Smith."
_BACK_MATTER_ADDRESSED_RE = re.compile(
    r"\b(?:correspondence(?:\s+and\s+requests\s+for\s+materials?)?\s+(?:should|may)\s+be\s+addressed\s+to"
    r"|address\s+correspondence\s+to)\s+(.+)",
    re.IGNORECASE,
)
_RECIPIENT_SPLIT_RE = re.compile(
    r"\s*(?:,\s*(?:and|&)\s*|\s+(?:and|&)\s*|,\s*|;\s*|\|\s*)", re.IGNORECASE
)
_CORRESPONDENCE_WORD_FOLLOW_RE = re.compile(r"correspond", re.IGNORECASE)
_PLURAL_CORRESPONDENCE_RE = re.compile(
    r"^[\W\d_]*(?:co-?)?corresponding\s+authors\b", re.IGNORECASE | re.MULTILINE
)
_NAME_TITLES = frozenset(
    {"prof", "professor", "dr", "mr", "mrs", "ms", "miss", "assoc", "assist", "phd", "md"}
)
# Lower-case surname particles; a printed "H. de Vries" must not read as initials H, D.
_NAME_PARTICLES = frozenset(
    {"de", "den", "der", "des", "di", "da", "del", "della", "dos", "das", "du", "la", "le"}
    | {"ter", "ten", "van", "von"}
)
# Letters NFKD leaves whole, spelled as addresses spell them ("Bræin" -> "braein").
_FOLD_EXTRA = str.maketrans(
    {"ł": "l", "ı": "i", "ø": "o", "đ": "d", "ð": "d", "æ": "ae", "œ": "oe", "þ": "th"}
)


def _fold_letters(text: str | None) -> str:
    """Lower-case and accent-free ("Šimić-Dvořák" -> "simic-dvorak")."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return base.casefold().translate(_FOLD_EXTRA)


def _fold(text: str | None) -> str:
    """Lower-case, accent-free, letters and digits only ("Al-Tammemi" -> "altammemi")."""
    return re.sub(r"[\W_]+", "", _fold_letters(text))


# German, Danish and Norwegian addresses often spell these letters out
# ("Müller" -> "mueller", "Kierkegård" -> "kierkegaard"); folding alone gives
# "muller". A name matches an address in either spelling.
_SPELLED_OUT = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "å": "aa", "ø": "oe", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}
    | {"Å": "Aa", "Ø": "Oe"}
)


def _letter_spellings(text: str | None) -> tuple[str, ...]:
    """:func:`_fold_letters` of *text*, and of it with umlauts and ø/å spelled out."""
    folded = _fold_letters(text)
    spelled = _fold_letters(unicodedata.normalize("NFC", text or "").translate(_SPELLED_OUT))
    return (folded,) if spelled == folded else (folded, spelled)


def _folds(text: str | None) -> tuple[str, ...]:
    """:func:`_fold` of each of *text*'s :func:`_letter_spellings`."""
    return tuple(re.sub(r"[\W_]+", "", spelling) for spelling in _letter_spellings(text))


def _initials(tokens: list[str]) -> list[str]:
    """Initials of given-name tokens: "H.-K." -> H, K; "HK" -> H, K; "Hong-Kwan" -> H, K."""
    out: list[str] = []
    for token in tokens:
        for raw in re.split(r"[\s.\-\u2010-\u2015]+", token):
            part = _fold(raw)
            if not part or raw in _NAME_PARTICLES:
                continue
            # "HK" is two initials, "Hong" is one given name.
            out.extend(part if raw.isupper() and len(part) <= 3 else part[0])
    return out


def _initials_compatible(left: list[str], right: list[str]) -> bool:
    """The first initial must agree; later ones are compared only when both sides have them."""
    if not left or not right or left[0] != right[0]:
        return False
    if len(left) >= 2 and len(right) >= 2:
        return left[: min(len(left), len(right))] == right[: min(len(left), len(right))]
    return True


def _given_tokens(given: str | None) -> list[str]:
    return [t for t in (given or "").split() if _fold(t)]


def _strip_titles(tokens: list[str]) -> list[str]:
    return [t for t in tokens if _fold(t) not in _NAME_TITLES]


def _index_authors_by_family(authors: list[PaperAuthor]) -> dict[str, list]:
    """Folded family name -> [(author, given-name initials)], built once per paper."""
    index: dict[str, list] = {}
    for author in authors:
        family = _fold(author.family)
        if len(family) >= 2:
            index.setdefault(family, []).append((author, _initials(_given_tokens(author.given))))
    return index


def _authors_named(name: str, index: dict[str, list]) -> list[PaperAuthor]:
    """The authors a printed "Initials Surname" (or "Given Surname") names.

    The surname is the trailing run of tokens whose folded text equals an author's
    family name (so "van der Berg" works whether or not the extractor kept the
    particle in the family name); everything before it must be initial-compatible
    with that author's given name.
    """
    tokens = _strip_titles(name.replace("\r", " ").replace("\n", " ").split())
    fits: list[PaperAuthor] = []
    for k in range(1, len(tokens)):
        entries = index.get(_fold("".join(tokens[k:])))
        if not entries:
            continue
        printed = _initials(tokens[:k])
        for author, initials in entries:
            if _initials_compatible(printed, initials) and all(author is not f for f in fits):
                fits.append(author)
    return fits


# --- a bare corresponding marker next to an address -----------------------------
# Many journals mark the corresponding author with a glyph and no word: "* E-mail: a@x.org;
# c@y.org" (the star stands for "corresponding"). The reading only flags an author who already
# holds the address (see _flag_from_marker_glyphs); it never attaches or moves an address.
_GLYPH = r"(?:\\\(\s*\^\s*\{\s*\*\s*\}\s*\\\)|[*∗⁎✉])"  # * ∗ ⁎ ✉ and LaTeX ^{*}
# An address, optionally followed by the initials PLOS prints after it: "a@x.org (AB)".
_ITEM = _EMAIL_RE.pattern + r"(?:\s*\(\s*(?:[A-Z]\.?[\s\-]?){2,4}\s*\))?"
_ITEM_RE = re.compile(_ITEM)
# The glyph directly followed by an address list (an "E-mail:" label may sit between).
_BARE_LIST_RE = re.compile(
    r"(?:^|(?<=[\s.;:,]))"
    + _GLYPH
    + r"\s*(?i:e-?\s?mail(?:\s+address(?:es)?)?\s*:?\s*)?"
    + _ITEM
    + r"(?:\s*(?:[;,]|\band\b)\s*"
    + _ITEM
    + r")*"
)
# A footnote symbol that follows the star in a series (* † ‡ § ¶ ‖) directly before an address:
# the star is then only the first of the series, not "corresponding".
_SERIES_ADDRESS_RE = re.compile(
    r"(?:^|(?<=[\s.;:,]))(?:[†‡§¶‖]|\\\(\s*\^\s*\{\s*\\d?dagger\s*\}\s*\\\))"
    r"\s*(?i:e-?\s?mail(?:\s+address(?:es)?)?\s*:?\s*)?" + _EMAIL_RE.pattern
)
_STAR_RE = re.compile(r"[*∗⁎]")
_NON_LETTERS_RE = re.compile(r"[\W\d_]*")
_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
# The ORCID badge text a PLOS byline prints glued to the surname: "SmithID1" -> "Smith1".
_ORCID_BADGE_RE = re.compile(r"(?<=[^\W\d_])ID(?=\d)")


def _names_another_author(owner: PaperAuthor, email: str, authors: list[PaperAuthor]) -> bool:
    """Whether the address's local part spells another author but not its holder."""
    local_compact = re.sub(r"[^a-z0-9]", "", email.split("@", 1)[0].lower())
    if _email_name_affinity(owner.given, owner.family, local_compact):
        return False
    return any(
        a is not owner and _email_name_affinity(a.given, a.family, local_compact) for a in authors
    )


def _author_initial_codes(author: PaperAuthor) -> set[str]:
    """Plausible uppercase initialisms for *author* (e.g. Jorge N Domínguez -> JND, JD)."""
    given_inits = _initials(_given_tokens(author.given))
    fam_tokens = [t for t in re.split(r"[\s.\-\u2010-\u2015]+", author.family or "") if _fold(t)]
    if not given_inits or not fam_tokens:
        return set()
    fam_inits = [_fold(t)[0] for t in fam_tokens if _fold(t)]
    if not fam_inits:
        return set()
    codes = {
        "".join(given_inits + fam_inits),
        given_inits[0] + "".join(fam_inits),
        "".join(given_inits) + fam_inits[-1],
        given_inits[0] + fam_inits[-1],
    }
    return {c.upper() for c in codes if len(c) >= 2}


def _match_unique_by_initials(code_raw: str, authors: list[PaperAuthor]) -> PaperAuthor | None:
    """Return the unique author whose initialisms contain *code_raw*, else None."""
    code = re.sub(r"[^A-Z]", "", code_raw.upper())
    if len(code) < 2:
        return None
    matches = [a for a in authors if code in _author_initial_codes(a)]
    return matches[0] if len(matches) == 1 else None


def _starred_byline_authors(sentences: list, index: dict[str, list]) -> list[PaperAuthor]:
    """Authors marked with a ``*`` after their surname in address-free byline sentences.

    Unlike :func:`_byline_stars` (used as a negative guard in ``_flag_from_marker_glyphs``),
    this positively identifies starred authors, disambiguating shared surnames (e.g.
    ``Xinyi Li1, ..., Yifan Li6*``) from the given name immediately preceding the surname,
    and ignores a ``*`` that opens a following ``*Correspondence:`` footnote.
    """
    chosen: list[PaperAuthor] = []
    seen_ids: set[int] = set()
    for sent in sentences:
        text = sent.text or ""
        if "@" in text or not _STAR_RE.search(text):
            continue
        cleaned = _ORCID_BADGE_RE.sub("", text)
        cue_m = _CORRESPONDENCE_PHRASE_RE.search(cleaned)
        scan_limit = cue_m.start() if cue_m is not None else len(cleaned)
        if scan_limit == 0:
            continue
        for word in _WORD_RE.finditer(cleaned):
            if word.start() >= scan_limit:
                break
            entries = index.get(_fold(word.group()), [])
            if not entries:
                continue
            after_match = _NON_LETTERS_RE.match(cleaned, word.end())
            after = after_match.group() if after_match else ""
            if not _STAR_RE.search(after):
                continue
            # Do not mistake a following "*Correspondence:" footnote glued after
            # the last author's affiliation digits for a star on that author.
            tail = cleaned[after_match.end() :] if after_match else ""
            if _CORRESPONDENCE_WORD_FOLLOW_RE.match(tail):
                # Only counts if the star is glued before whitespace/newline, not right before "Correspondence".
                star_idx = _STAR_RE.search(after).start()  # type: ignore[union-attr]
                if not after[star_idx + 1 :].strip() and any(
                    ch in after[:star_idx] for ch in ("\n", "\r", " ")
                ):
                    continue
            if len(entries) == 1:
                author = entries[0][0]
            else:
                # Disambiguate shared surnames by the given-name segment
                # immediately preceding this surname (do not cross comma/digit
                # boundaries into an earlier author's name).
                segment = re.split(
                    r"[,;\n\r\d]+|\band\b|&",
                    cleaned[: word.start()],
                    flags=re.IGNORECASE,
                )[-1]
                before_words = _WORD_RE.findall(segment)[-3:]
                if not before_words:
                    continue
                before_folds = {_fold(w) for w in before_words if _fold(w)}
                before_inits = _initials(before_words)
                fits: list[PaperAuthor] = []
                for cand, cand_inits in entries:
                    g_folds = {_fold(t) for t in _given_tokens(cand.given) if len(_fold(t)) >= 2}
                    if g_folds & before_folds or (
                        before_inits
                        and cand_inits
                        and _initials_compatible(before_inits[-len(cand_inits) :], cand_inits)
                    ):
                        fits.append(cand)
                if len(fits) != 1:
                    continue
                author = fits[0]
            if id(author) not in seen_ids:
                seen_ids.add(id(author))
                chosen.append(author)
    return chosen


def _authors_named_in_text(text: str, index: dict[str, list]) -> list[PaperAuthor]:
    """Find every author explicitly named (given/initials + surname) inside *text*."""
    cleaned = _EMAIL_RE.sub(" ", text or "")
    cleaned = _ORCID_BADGE_RE.sub("", cleaned)
    word_matches = list(_WORD_RE.finditer(cleaned))
    if not word_matches:
        return []
    found: list[PaperAuthor] = []
    seen_ids: set[int] = set()
    n = len(word_matches)
    for i in range(n):
        # Try 1..3-word surnames (covers "Ferguson-Smith", "van der Berg").
        for span_len in (3, 2, 1):
            if i + span_len > n:
                continue
            fam_key = _fold("".join(word_matches[i + k].group() for k in range(span_len)))
            entries = index.get(fam_key)
            if not entries:
                continue
            # Require at least one preceding given-name/initial token matching the author.
            prev_words = _strip_titles([word_matches[j].group() for j in range(max(0, i - 4), i)])
            if not prev_words:
                continue
            prev_folds = {_fold(w) for w in prev_words if len(_fold(w)) >= 2}
            prev_inits = _initials(prev_words)
            fits: list[PaperAuthor] = []
            for cand, cand_inits in entries:
                g_tokens = [_fold(t) for t in _given_tokens(cand.given) if _fold(t)]
                full_match = any(len(gt) >= 2 and gt in prev_folds for gt in g_tokens)
                init_match = bool(
                    prev_inits
                    and cand_inits
                    and (
                        _initials_compatible(prev_inits[-len(cand_inits) :], cand_inits)
                        or _initials_compatible(prev_inits[-1:], cand_inits)
                    )
                )
                if full_match or (init_match and len(entries) == 1):
                    fits.append(cand)
            if len(fits) == 1 and id(fits[0]) not in seen_ids:
                seen_ids.add(id(fits[0]))
                found.append(fits[0])
                break
    return found


def _byline_stars(sentences: list, index: dict[str, list]) -> tuple[set[int], set[int]]:
    """(ids of authors a star follows, ids of authors named without one) in address-free sentences.

    An author is read from a word that is exactly their surname; a name the extraction ran
    together ("AliceLee") or a surname shared by two authors is skipped, so the caller can
    only act on authors it positively saw without a star.
    """
    starred: set[int] = set()
    seen: set[int] = set()
    for sent in sentences:
        text = sent.text or ""
        if "@" in text or not _STAR_RE.search(text):
            continue
        text = _ORCID_BADGE_RE.sub("", text)
        for word in _WORD_RE.finditer(text):
            entries = index.get(_fold(word.group()), [])
            if len(entries) != 1:
                continue
            author = entries[0][0]
            seen.add(id(author))
            # the run of non-letters right after the surname: "1,*," or "\\(^{1,*}\\)" or " "
            after = _NON_LETTERS_RE.match(text, word.end()).group()
            if _STAR_RE.search(after):
                starred.add(id(author))
    return starred, seen - starred


class AuthorEmailHarvester:
    """Promote corresponding authors and attach their emails.

    Operates on a ``PaperContents`` (read-only) and a mutable list of
    ``PaperAuthor`` (modified in place).
    """

    def __init__(self, contents: PaperContents, document: PaperContents | None = None):
        """``contents`` is what the extractor scoped to the selected front-matter block.

        ``document`` is the whole paper. Only the name-paired footnote pass reads it,
        because the footnote carrying a corresponding author's address sits outside
        the front-matter block on most layouts. Everything else stays scoped.
        """
        self.contents = contents
        self._document = document if document is not None else contents

    @staticmethod
    def demote_implausible_flags(authors: list[PaperAuthor]) -> None:
        """If every author in a multi-author paper is flagged corresponding,
        treat the LLM's response as untrustworthy and demote them all.

        This is the backstop for papers where every author lists their email
        in the byline (e.g. arXiv-style "Attention Is All You Need" with 8
        @google.com addresses). Without this, every author gets promoted to
        corresponding=True. We pick a 3-author threshold because 1-2 author
        papers commonly do have all authors listed as co-corresponding.

        Mutates ``authors`` in place. ``harvest`` runs after this and will
        re-promote any author with an explicit anchor in the body text.
        """
        if len(authors) < 3:
            return
        if not all(a.corresponding for a in authors):
            return
        for a in authors:
            a.corresponding = False
        logger.info(
            "Demoted %d implausibly all-corresponding author flags from LLM extraction",
            len(authors),
        )

    @staticmethod
    def _marker_sentence_emails(sentences: list) -> set[str]:
        """Emails printed on a sentence that itself carries the corresponding marker.

        When such explicit pairings exist ('* Correspondence: x@y.org' — the MDPI
        house style), they exhaustively name the corresponding authors. The other
        byline emails sit within the proximity window of that line and must NOT be
        promoted from mere closeness.
        """
        out: set[str] = set()
        for sent in sentences:
            if _CORRESPONDING_MARKER_RE.search(sent.text):
                out.update(m.group(1).lower() for m in _EMAIL_RE.finditer(sent.text))
        return out

    @staticmethod
    def _promote_sole_author(authors: list[PaperAuthor]) -> None:
        """A sole author with a contact email is the de-facto corresponding author."""
        if len(authors) == 1 and authors[0].email and not authors[0].corresponding:
            authors[0].corresponding = True
            logger.info("Promoted sole author to corresponding=True (has contact email)")

    def harvest(self, authors: list[PaperAuthor]) -> None:
        """Backfill ``email`` / ``corresponding`` from explicit author footers.

        Many psych-journal PDFs print a "Corresponding Author:" footer that
        names the author and gives their email — but the LLM metadata pass
        rarely surfaces those emails into ``PaperAuthor.email``.  This pass
        scans body sentences for email patterns and assigns each email to
        the best-matching author within a small window of surrounding
        sentences.

        Selection algorithm per email:
          1. Skip if the email is already claimed by another author.
          2. Rank candidates by (sentence distance, -surname-affinity), then
             assign only an explicitly marker-paired address, one whose local
             part names the author, or one printed in the same sentence as
             their surname — never on window proximity alone.
          3. Final fallback: a single still-corresponding author missing an
             email gets it by elimination, only with a marker-paired address,
             a correspondence marker in the window (only when no sentence
             pairs the marker with an explicit address), a '* E-mail:'
             footnote line, or a name match.
             Anything else is skipped, leaving the author emailless for
             their real address instead of blocking it with someone else's.

        Name-paired footnotes ("E-mail address(es): x@y (Initials Surname)") are read
        last, from the whole paper rather than the scoped block, and only for authors
        still without an address after everything above. Each pair is attached to the
        one author whose surname and initials fit (none or two fit: nothing), whether or
        not the footnote says "Corresponding author". The author is also
        flagged corresponding, after the steps above, only when a "Corresponding
        author(s)" / "Correspondence" phrase opens the sentence or one of the three
        before it, the phrase is plural or the window holds a single pair, and no other
        author is already flagged (by the LLM, a marker sentence or the envelope glyph).

        A bare corresponding marker is read last (:meth:`_flag_from_marker_glyphs`): a star
        directly before an address flags the author who already holds it. That pass only
        flags; it never attaches or moves an address.

        Mutates ``authors`` in place.  Skips silently when no authors or no
        email-bearing text are available.
        """
        if not authors:
            return

        from bibr.paper_contents import CanonicalSection

        # Skip references section so reference URLs/DOIs/emails don't bleed in.
        section_type_by_id = {s.section_id: s.section_type for s in self.contents.sections}
        sentences = _expand_braced_emails(
            [
                s
                for s in self.contents.sentences
                if section_type_by_id.get(s.section_id) != CanonicalSection.REFERENCES
            ]
        )

        # Only fill gaps — never overwrite an LLM-extracted email.
        # Surname → every still-emailless author carrying it. Keeping a single
        # author per surname let same-surname co-authors clobber each other:
        # the last one won, so the earlier ones could never be filled and the
        # survivor could be handed an address that was not theirs.
        authors_by_family: dict[str, list[PaperAuthor]] = {}
        for author in authors:
            if author.family and not author.email:
                authors_by_family.setdefault(author.family.lower().strip(), []).append(author)
        if not authors_by_family:
            self._finish(authors, sentences)
            return

        strict_emails = self._marker_sentence_emails(sentences)

        claimed_emails: set[str] = {a.email.lower() for a in authors if a.email}
        all_authors_by_family: dict[str, list[PaperAuthor]] = {}
        for author in authors:
            if author.family:
                all_authors_by_family.setdefault(author.family.lower().strip(), []).append(author)
        shared_families = {fam for fam, group in all_authors_by_family.items() if len(group) > 1}
        family_pats: dict[str, re.Pattern[str]] = {
            family: re.compile(r"\b" + re.escape(family) + r"\b", re.IGNORECASE)
            for family in authors_by_family
            if family
        }
        family_folds: dict[str, tuple[str, ...]] = {
            family: _folds(family) for family in authors_by_family if family
        }

        def _sentence_mentions_shared_author(
            text: str, family: str, family_pat: re.Pattern[str], author: PaperAuthor
        ) -> bool:
            """For a shared surname, avoid matching co-author A when *text* only names co-author B."""
            matches = list(family_pat.finditer(text))
            if not matches:
                return False
            peers = all_authors_by_family.get(family, ())
            author_given_folds = {
                _fold(tok) for tok in _given_tokens(author.given) if len(_fold(tok)) >= 2
            }
            author_inits = _initials(_given_tokens(author.given))
            any_unqualified = False
            for m in matches:
                seg = re.split(
                    r"[,;\n\r\d:]+|\band\b|&",
                    text[: m.start()],
                    flags=re.IGNORECASE,
                )[-1]
                before_words = _WORD_RE.findall(seg)[-3:]
                before_folds = {_fold(w) for w in before_words if _fold(w)}
                before_inits = _initials(before_words)
                if author_given_folds & before_folds or (
                    author_inits
                    and before_inits
                    and _initials_compatible(before_inits[-len(author_inits) :], author_inits)
                ):
                    return True
                other_named = any(
                    peer is not author
                    and (
                        (
                            {_fold(t) for t in _given_tokens(peer.given) if len(_fold(t)) >= 2}
                            & before_folds
                        )
                        or (
                            (p_inits := _initials(_given_tokens(peer.given)))
                            and before_inits
                            and _initials_compatible(before_inits[-len(p_inits) :], p_inits)
                        )
                    )
                    for peer in peers
                )
                if not other_named:
                    any_unqualified = True
            return any_unqualified

        harvested = 0
        for i, sent in enumerate(sentences):
            if "@" not in sent.text:
                continue
            window_indices = [
                j
                for j in range(
                    max(0, i - _EMAIL_NAME_WINDOW),
                    min(len(sentences), i + _EMAIL_NAME_WINDOW + 1),
                )
                if sentences[j].section_id == sent.section_id
            ]
            joined_window = " ".join(sentences[j].text for j in window_indices)
            window_has_marker = bool(_CORRESPONDING_MARKER_RE.search(joined_window))

            for email_match in _EMAIL_RE.finditer(sent.text):
                email = email_match.group(1)
                email_low = email.lower()
                if email_low in claimed_emails:
                    continue

                # If an already-filled author holds a truncated TLD prefix of
                # *email* (e.g. "yltang@uni.edu" vs repaired "yltang@uni.edu.mo"),
                # upgrade their address instead of handing it to a neighbor.
                upgraded_existing = False
                for existing_author in authors:
                    if not existing_author.email:
                        continue
                    ex_low = existing_author.email.lower()
                    if email_low.startswith(ex_low + "."):
                        existing_author.email = email
                        claimed_emails.add(email_low)
                        if email_low in strict_emails:
                            existing_author.corresponding = True
                        upgraded_existing = True
                        break
                    if ex_low.startswith(email_low + "."):
                        claimed_emails.add(email_low)
                        upgraded_existing = True
                        break
                if upgraded_existing:
                    continue

                local = email.split("@", 1)[0].lower()
                local_compact = re.sub(r"[^a-z0-9]", "", local)

                candidates: list[tuple[int, int, int, str, PaperAuthor]] = []
                order_idx = 0
                for family, family_authors in authors_by_family.items():
                    if not family:
                        order_idx += len(family_authors)
                        continue
                    family_pat = family_pats[family]
                    affinity = int(any(f and f in local_compact for f in family_folds[family]))
                    if family in shared_families:
                        for author in family_authors:
                            best_dist: int | None = None
                            for j in window_indices:
                                if _sentence_mentions_shared_author(
                                    sentences[j].text, family, family_pat, author
                                ):
                                    dist = abs(j - i)
                                    if best_dist is None or dist < best_dist:
                                        best_dist = dist
                            if best_dist is not None:
                                given_bonus = _given_name_affinity(author.given, local_compact)
                                candidates.append(
                                    (
                                        best_dist,
                                        -(affinity + given_bonus),
                                        order_idx,
                                        family,
                                        author,
                                    )
                                )
                            order_idx += 1
                    else:
                        best_dist = None
                        for j in window_indices:
                            if family_pat.search(sentences[j].text):
                                dist = abs(j - i)
                                if best_dist is None or dist < best_dist:
                                    best_dist = dist
                        if best_dist is None:
                            order_idx += len(family_authors)
                            continue
                        for author in family_authors:
                            given_bonus = (
                                _given_name_affinity(author.given, local_compact)
                                if len(family_authors) > 1
                                else 0
                            )
                            candidates.append(
                                (best_dist, -(affinity + given_bonus), order_idx, family, author)
                            )
                            order_idx += 1

                if candidates:
                    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
                    _passing = [
                        c
                        for c in candidates
                        if (
                            email.lower() in strict_emails
                            or c[0] == 0
                            or _email_name_affinity(c[4].given, c[4].family, local_compact)
                        )
                    ]
                    _, _, _, family, author = (_passing or candidates)[0]
                    candidates = _passing or candidates
                    # The surname window alone must not license the address:
                    # the window routinely covers an unrelated earlier email
                    # (editorial office, lab, journal) next to the byline, and
                    # a correspondence marker elsewhere in it may belong to the
                    # author's real address printed lines below. Assign an
                    # explicitly paired address (strict), one that names the
                    # author, or one printed in the same sentence as their
                    # surname; otherwise leave them emailless for their real
                    # one instead of blocking it with someone else's.
                    best_dist = candidates[0][0]
                    if (
                        email.lower() in strict_emails
                        or best_dist == 0
                        or _email_name_affinity(author.given, author.family, local_compact)
                    ):
                        author.email = email
                        claimed_emails.add(email.lower())
                        # Strict mode: when some sentence pairs the marker with
                        # explicit email(s), only those emails are corresponding.
                        if email.lower() in strict_emails if strict_emails else window_has_marker:
                            author.corresponding = True
                        _drop_filled_author(authors_by_family, family, author)
                        harvested += 1
                else:
                    # Surname wasn't within the 3-sentence same-section window.
                    # On explicit correspondence/contact or bare-star email lines
                    # ("* rvale@uni.ca", "*For correspondence: x@y (JND)"),
                    # match uniquely by trailing parenthetical initials or email
                    # local-part affinity before falling back to single-corresponding
                    # elimination. Only promote ``corresponding`` when an explicit
                    # marker (not a bare star alone) authorizes it.
                    sent_has_marker = bool(_CORRESPONDING_MARKER_RE.search(sent.text))
                    is_bare_star_line = bool(_BARE_LIST_RE.search(sent.text))
                    if email.lower() in strict_emails or sent_has_marker or is_bare_star_line:
                        remaining_authors = [
                            a for group in authors_by_family.values() for a in group
                        ]
                        matched_author: PaperAuthor | None = None
                        init_m = _INITIALS_AFTER_EMAIL_RE.match(sent.text[email_match.end() :])
                        if init_m:
                            matched_author = _match_unique_by_initials(
                                init_m.group(1), remaining_authors
                            )
                        if matched_author is None:
                            aff_hits = [
                                a
                                for a in remaining_authors
                                if _email_name_affinity(a.given, a.family, local_compact)
                            ]
                            if len(aff_hits) == 1:
                                matched_author = aff_hits[0]
                        if matched_author is not None:
                            matched_author.email = email
                            claimed_emails.add(email.lower())
                            if email.lower() in strict_emails if strict_emails else sent_has_marker:
                                matched_author.corresponding = True
                            _drop_filled_author(
                                authors_by_family,
                                matched_author.family.lower().strip(),
                                matched_author,
                            )
                            harvested += 1
                            continue

                    fallback = [
                        a for group in authors_by_family.values() for a in group if a.corresponding
                    ]
                    if len(fallback) == 1:
                        # Elimination with no surname evidence at all: only an
                        # explicitly paired address, a correspondence marker in
                        # the email's own window, a PLOS '* E-mail:' line, or a
                        # name-matching address. When some sentence pairs the
                        # marker with explicit email(s), those pairings are
                        # exhaustive and a bare window marker no longer counts.
                        candidate = fallback[0]
                        if (
                            email.lower() in strict_emails
                            or (
                                not strict_emails
                                and (
                                    window_has_marker
                                    or re.match(r"\s*\*\s*e-?mail", sent.text, re.I)
                                )
                            )
                            or _email_name_affinity(
                                candidate.given, candidate.family, local_compact
                            )
                        ):
                            candidate.email = email
                            claimed_emails.add(email.lower())
                            _drop_filled_author(
                                authors_by_family,
                                candidate.family.lower().strip(),
                                candidate,
                            )
                            harvested += 1

                if not authors_by_family:
                    break
            if not authors_by_family:
                break

        if harvested:
            logger.info(
                "Harvested %d corresponding-author email(s) from body text",
                harvested,
            )

        self._finish(authors, sentences)

    def _finish(self, authors: list[PaperAuthor], sentences: list) -> None:
        """Name-paired footnote addresses, then the promotions, in order.

        Footnote pairs are attached only now, after every in-scope rule above had its
        turn, and only to authors still without an address, so they can never displace
        an address those rules assigned. The anchor promotion then sees them (an in-scope
        "Correspondence to: x@y" line that names no one flags the author a footnote pairs
        with x@y), and the footnote flags go after it so they can see who is flagged.
        """
        footnote_groups = self._harvest_named_footnote_emails(authors)
        self._promote_corresponding_from_anchors(authors, sentences)
        self._flag_named_footnote_authors(authors, footnote_groups)
        self._flag_from_marker_glyphs(authors, sentences)
        self._flag_from_correspondence_blocks(authors, sentences)
        self._flag_from_byline_stars(authors, sentences)
        self._flag_from_back_matter_correspondence(authors)
        self._promote_sole_author(authors)

    def _apply_corresponding_group(
        self,
        authors: list[PaperAuthor],
        chosen: list[PaperAuthor],
        reason: str,
    ) -> None:
        """Mark *chosen* as corresponding if compatible with existing flags and cardinality."""
        if not chosen:
            return
        if len(authors) >= 3 and len({id(a) for a in chosen}) == len(authors):
            return
        flagged = {id(a) for a in authors if a.corresponding}
        chosen_ids = {id(a) for a in chosen}
        if flagged - chosen_ids:
            return
        added = [a for a in chosen if not a.corresponding]
        for author in added:
            author.corresponding = True
        if len(authors) >= 3 and all(a.corresponding for a in authors):
            for author in added:
                author.corresponding = False
            return
        if added:
            logger.info("Marked %d author(s) corresponding from %s", len(added), reason)

    def _flag_from_correspondence_blocks(self, authors: list[PaperAuthor], sentences: list) -> None:
        """Flag authors explicitly named in a page-1 correspondence/CONTACT block.

        Covers layouts whose correspondence block prints the author's full name
        next to an opaque or separate-line email (BMC ``*Correspondence:``,
        Taylor & Francis ``CONTACT <Name> <email>``, Psychological Medicine
        ``Corresponding authors: <Name>``, Frontiers ``*CORRESPONDENCE``,
        PeerJ ``Corresponding author``).
        """
        if not authors or not sentences:
            return
        index = _index_authors_by_family(authors)
        claimed = {a.email.lower() for a in authors if a.email}
        chosen: list[PaperAuthor] = []
        chosen_ids: set[int] = set()

        for i, sent in enumerate(sentences):
            text = sent.text or ""
            cue_m = _CORRESPONDENCE_PHRASE_RE.search(text)
            if not cue_m:
                continue
            # Collect from the correspondence cue onward. Only absorb immediate
            # continuation sentences when the cue sentence itself is a bare
            # heading without an author name or email ("*Correspondence:",
            # "Corresponding authors:"), so a self-contained line like
            # "✉ Jane Doe, jane@uni.edu" never pulls in the next author's line.
            head_text = text[cue_m.start() :]
            block_parts = [head_text]
            if "@" not in head_text and not _authors_named_in_text(head_text, index):
                for j in range(i + 1, min(len(sentences), i + 4)):
                    nxt = sentences[j]
                    if nxt.section_id != sent.section_id:
                        break
                    nxt_text = (nxt.text or "").strip()
                    if not nxt_text or len(nxt_text) > 260:
                        break
                    if _CORRESPONDENCE_PHRASE_RE.search(nxt_text):
                        break
                    block_parts.append(nxt_text)
                    if "@" in nxt_text and _authors_named_in_text("\n".join(block_parts), index):
                        break
            block_text = "\n".join(block_parts)
            # Guard against a merged paragraph containing the entire byline.
            named = _authors_named_in_text(block_text, index)
            if not named:
                continue
            if len(authors) >= 3 and len(named) == len(authors):
                continue
            # If the block pairs one named author with one unclaimed email (or
            # equal counts of named authors and emails in order), backfill missing emails.
            block_emails = [m.group(1) for m in _EMAIL_RE.finditer(block_text)]
            if len(named) == 1 and len(block_emails) == 1:
                self._attach_email(named[0], block_emails[0], claimed)
            elif len(named) == len(block_emails) and len(named) > 1:
                for auth, addr in zip(named, block_emails, strict=True):
                    self._attach_email(auth, addr, claimed)
            for auth in named:
                if id(auth) not in chosen_ids:
                    chosen_ids.add(id(auth))
                    chosen.append(auth)

        self._apply_corresponding_group(authors, chosen, "correspondence blocks")

    def _flag_from_byline_stars(self, authors: list[PaperAuthor], sentences: list) -> None:
        """Flag authors marked with ``*`` in the byline when a ``*`` correspondence cue exists.

        Fires after :meth:`_flag_from_marker_glyphs` and :meth:`_flag_from_correspondence_blocks`
        so it never contradicts an already-resolved email/name match, while rescuing
        papers whose correspondence footnote has an opaque email (e.g. PLOS
        ``* evergreen88@example.com`` + ``Liwei Sun*``) or BMC/eLife/Frontiers
        co-corresponding stars.
        """
        if not authors or not sentences:
            return
        has_star_word_cue = any(_STAR_CORRESPONDENCE_CUE_RE.search(s.text or "") for s in sentences)
        bare_star_sents = []
        standalone_email_sents = []
        if not has_star_word_cue:
            printing = [s for s in sentences if "@" in (s.text or "")]
            in_series = any(_SERIES_ADDRESS_RE.search(s.text) for s in printing)
            if not in_series:
                bare_star_sents = [s for s in printing if _BARE_LIST_RE.search(s.text)]
                if not bare_star_sents and not any(
                    _NON_CORR_STAR_NOTE_RE.search(s.text or "") for s in sentences
                ):
                    standalone_email_sents = [
                        s for s in printing if _STANDALONE_EMAIL_LINE_RE.match(s.text or "")
                    ]
        if not (has_star_word_cue or bare_star_sents or standalone_email_sents):
            return
        index = _index_authors_by_family(authors)
        starred = _starred_byline_authors(sentences, index)
        if not starred:
            return
        if not has_star_word_cue and bare_star_sents:
            # For bare "* email" lines without an explicit "*Correspondence" word,
            # only fall back to byline stars when an email on the "*" line is
            # unattached (opaque, e.g. "* evergreen88@example.com") AND no email on the
            # "*" line already belongs to a non-starred author.
            by_email = {a.email.lower(): a for a in authors if a.email}
            starred_ids = {id(a) for a in starred}
            opaque_emails: list[str] = []
            for s in bare_star_sents:
                for m in _EMAIL_RE.finditer(s.text or ""):
                    addr = m.group(1)
                    owner = by_email.get(addr.lower())
                    if owner is not None:
                        if id(owner) not in starred_ids:
                            return
                    else:
                        opaque_emails.append(addr)
            if not opaque_emails:
                return
            if len(starred) == 1 and len(opaque_emails) == 1:
                self._attach_email(starred[0], opaque_emails[0], set(by_email))
        elif not has_star_word_cue and standalone_email_sents:
            # PLOS layout where OCR clipped the tiny leading "*" off the
            # standalone email line ("Han.Liu@examplehealth.org"): only fire
            # when the starred byline author(s) are the exact owners of all
            # harvested emails and at least one of those emails sits on a
            # standalone line.
            if len(starred) >= len(authors):
                return
            by_email = {a.email.lower(): a for a in authors if a.email}
            starred_ids = {id(a) for a in starred}
            all_email_owner_ids = {id(a) for a in authors if a.email}
            standalone_owner_ids = {
                id(owner)
                for s in standalone_email_sents
                for m in _EMAIL_RE.finditer(s.text or "")
                if (owner := by_email.get(m.group(1).lower())) is not None
            }
            if not standalone_owner_ids or all_email_owner_ids != starred_ids:
                return
        self._apply_corresponding_group(authors, starred, "byline stars")

    def _flag_from_back_matter_correspondence(self, authors: list[PaperAuthor]) -> None:
        """Flag authors named in back-matter correspondence statements.

        Handles journals (e.g. Nature Communications) whose page-1 envelope icon
        is a vector graphic absent from the text layer, and which state explicitly
        in the back matter: ``Correspondence and requests for materials should be
        addressed to <Author(s)>.``
        """
        from bibr.paper_contents import CanonicalSection

        raw_sentences = getattr(self._document, "sentences", None)
        doc_sentences = list(raw_sentences) if isinstance(raw_sentences, (list, tuple)) else []
        if not authors or not doc_sentences:
            return
        raw_sections = getattr(self._document, "sections", None)
        doc_sections = list(raw_sections) if isinstance(raw_sections, (list, tuple)) else []
        ref_sids = {
            s.section_id for s in doc_sections if s.section_type == CanonicalSection.REFERENCES
        }
        index = _index_authors_by_family(authors)
        chosen: list[PaperAuthor] = []
        chosen_ids: set[int] = set()

        for sent in doc_sentences:
            if sent.section_id in ref_sids:
                continue
            text = sent.text or ""
            if "correspondence" not in text.lower():
                continue
            m = _BACK_MATTER_ADDRESSED_RE.search(text)
            if not m:
                continue
            clause = m.group(1).strip()
            # Cut at a sentence-ending period (not an initial like "C." or title like "Dr.").
            end_m = re.search(r"(?<![A-Z])(?<!\bDr)(?<!\bProf)\.(?:\s|$)", clause)
            if end_m:
                clause = clause[: end_m.start()]
            # First split recipient list and try exact _authors_named on each item.
            matched: list[PaperAuthor] = []
            for part in _RECIPIENT_SPLIT_RE.split(clause):
                part = re.sub(r"\([^)]*\)", "", part).strip(" .,:;")
                if not part:
                    continue
                fits = _authors_named(part, index)
                if len(fits) == 1:
                    matched.append(fits[0])
            if not matched:
                matched = _authors_named_in_text(clause, index)
            for auth in matched:
                if id(auth) not in chosen_ids:
                    chosen_ids.add(id(auth))
                    chosen.append(auth)

        self._apply_corresponding_group(authors, chosen, "back-matter correspondence")

    @staticmethod
    def _attach_email(author: PaperAuthor, email: str, claimed: set[str]) -> bool:
        """Attach *email* to *author*; True when the author holds it afterwards.

        Never overwrites a different address and never takes one another author holds.
        """
        wanted = email.lower()
        current = (author.email or "").lower()
        if current:
            return current == wanted
        if wanted in claimed:
            return False
        author.email = email
        claimed.add(wanted)
        return True

    def _harvest_named_footnote_emails(self, authors: list[PaperAuthor]) -> list[list[PaperAuthor]]:
        """Attach "E-mail address(es): x@y (Initials Surname), ..." footnote pairs to authors.

        Reads the whole paper, not the front-matter scope. Each pair goes to the one
        author whose surname and initials fit the printed name (no fit or two fits:
        nothing). Whether the footnote also marks them corresponding is decided
        later (:meth:`_flag_named_footnote_authors`); this returns the groups of
        authors that earned a flag: a "Corresponding author(s)"/"Correspondence"
        phrase opens the sentence or one of the three before it (same section), and
        either that phrase is plural or the window holds exactly one pair.
        """
        sentences = list(getattr(self._document, "sentences", None) or [])
        if not authors or not sentences:
            return []
        index: dict[str, list] | None = None
        claimed = {a.email.lower() for a in authors if a.email}
        groups: list[list[PaperAuthor]] = []
        attached = 0
        for i, sent in enumerate(sentences):
            text = sent.text or ""
            if "@" not in text:
                continue
            label = _EMAIL_LABEL_RE.search(text)
            if label is None:
                continue
            pairs = list(_NAMED_EMAIL_RE.finditer(text, label.end()))
            if not pairs:
                continue
            if index is None:
                index = _index_authors_by_family(authors)
            window = [
                sentences[j].text or ""
                for j in range(max(0, i - _EMAIL_NAME_WINDOW), i)
                if sentences[j].section_id == sent.section_id
            ]
            openers = [*window, text[: label.start()]]
            holders: list[PaperAuthor] = []
            for pair in pairs:
                fits = _authors_named(pair.group(2), index)
                if len(fits) != 1:
                    continue
                had_email = bool(fits[0].email)
                if self._attach_email(fits[0], pair.group(1), claimed):
                    holders.append(fits[0])
                    attached += not had_email
            if not holders or not any(_CORRESPONDENCE_PHRASE_RE.search(t) for t in openers):
                continue
            plural = any(_PLURAL_CORRESPONDENCE_RE.search(t) for t in openers)
            pairs_in_window = len(pairs) + sum(
                len(_NAMED_EMAIL_RE.findall(t)) for t in window if "@" in t
            )
            if plural or pairs_in_window == 1:
                groups.append(holders)
        if attached:
            logger.info("Attached %d e-mail(s) from name-paired footnotes", attached)
        return groups

    @staticmethod
    def _flag_named_footnote_authors(
        authors: list[PaperAuthor], groups: list[list[PaperAuthor]]
    ) -> None:
        """Mark the authors a name-paired footnote names as corresponding.

        Never adds a flag next to a different flagged author, whether that flag came from
        the LLM, a marker sentence or the envelope glyph: a group is applied only when
        every currently flagged author belongs to it.
        """
        if not groups:
            return
        flagged = {id(a) for a in authors if a.corresponding}
        promoted = 0
        for group in groups:
            if flagged - {id(a) for a in group}:
                continue
            for author in group:
                if not author.corresponding:
                    author.corresponding = True
                    promoted += 1
        if promoted:
            logger.info("Marked %d author(s) corresponding from name-paired footnotes", promoted)

    def _flag_from_marker_glyphs(self, authors: list[PaperAuthor], sentences: list) -> None:
        """Flag the authors who hold an address a bare corresponding glyph points to.

        ``* E-mail: a@x.org (AB); c@y.org`` over the scoped front-matter sentences that print an
        address flags the authors who already own the listed addresses (the star stands for
        "corresponding"). The glyph must be directly followed by the address or an "E-mail:"
        label, so "* These authors contributed equally" never counts. Several owners are taken
        only from one list; separate glyph lines for different owners are left alone.

        A star is not read when it is only the first of a footnote series (another of
        ``† ‡ § ¶ ‖`` directly before an address in scope), when another address follows the
        matched list in the same sentence (a list cut short by a parenthetical that is not
        initials), or for an owner whose address spells a different author's name and not
        their own (an address attached to the wrong author upstream).

        This pass only sets ``corresponding``. It never attaches, moves or replaces an address,
        so an address an author holds (or lacks) is the same before and after.

        Guards: a star that follows author names in the byline must follow this author too;
        an owner is never added next to a different author who is already flagged (by the LLM or
        another marker); and a 3+ author paper never ends with every author flagged.
        """
        if not authors:
            return
        printing = [s for s in sentences if "@" in (s.text or "")]
        if not printing:
            return
        owner_by_email = {a.email.lower(): a for a in authors if a.email}
        if not owner_by_email:
            return

        in_series = any(_SERIES_ADDRESS_RE.search(s.text) for s in printing)
        groups: list[list[PaperAuthor]] = []
        for sent in printing:
            text = sent.text
            for m in _BARE_LIST_RE.finditer(text):
                if in_series and not m.group(0).startswith("✉"):
                    continue  # the star is the first of a * † ‡ series, not "corresponding"
                if _EMAIL_RE.search(text, m.end()):
                    continue  # the list was cut short (a parenthetical that is not initials)
                owners: list[PaperAuthor] = []
                for item in _ITEM_RE.finditer(m.group(0)):
                    email = item.group(1)
                    owner = owner_by_email.get(email.lower())
                    if owner is None or any(owner is o for o in owners):
                        continue
                    if _names_another_author(owner, email, authors):
                        continue  # the address spells someone else: probably attached wrongly
                    owners.append(owner)
                if owners:
                    groups.append(owners)
        if not groups:
            return

        starred, unstarred = _byline_stars(sentences, _index_authors_by_family(authors))
        if starred:
            # a star in the byline marks other authors, and this owner is printed without one
            blocked = unstarred - starred
            groups = [[a for a in g if id(a) not in blocked] for g in groups]
            groups = [g for g in groups if g]
        owner_ids = {id(a) for g in groups for a in g}
        if len(owner_ids) >= 2 and not any(
            len({id(a) for a in g}) == len(owner_ids) for g in groups
        ):
            groups = []  # owners spread over separate lines: no single co-corresponding list

        chosen: list[PaperAuthor] = []
        for group in groups:
            chosen.extend(a for a in group if all(a is not c for c in chosen))
        if not chosen:
            return
        flagged = {id(a) for a in authors if a.corresponding}
        if flagged - {id(a) for a in chosen}:
            return
        added = [a for a in chosen if not a.corresponding]
        for author in added:
            author.corresponding = True
        if len(authors) >= 3 and all(a.corresponding for a in authors):
            for author in added:
                author.corresponding = False
            return
        if added:
            logger.info("Marked %d author(s) corresponding from bare marker glyphs", len(added))

    def _promote_corresponding_from_anchors(
        self, authors: list[PaperAuthor], sentences: list
    ) -> None:
        """Set ``corresponding=True`` on any author whose (already-attached)
        email appears next to an explicit corresponding-author anchor.

        When a marker-bearing sentence itself names email(s), those pairings are
        exhaustive (strict mode) and window/header/footnote proximity is ignored.
        """
        if not authors:
            return
        email_to_author = {a.email.lower(): a for a in authors if a.email and not a.corresponding}
        if not email_to_author:
            return

        strict_emails = self._marker_sentence_emails(sentences)

        section_meta = {s.section_id: s for s in self.contents.sections}

        promoted = 0
        for i, sent in enumerate(sentences):
            if "@" not in sent.text:
                continue
            window_indices = [
                j
                for j in range(
                    max(0, i - _EMAIL_NAME_WINDOW),
                    min(len(sentences), i + _EMAIL_NAME_WINDOW + 1),
                )
                if sentences[j].section_id == sent.section_id
            ]
            joined_window = " ".join(sentences[j].text for j in window_indices)
            section = section_meta.get(sent.section_id)
            section_header = (section.header or "") if section else ""
            section_type = (
                section.section_type.value
                if section and hasattr(section.section_type, "value")
                else (section.section_type if section else "")
            )

            anchor_in_window = bool(_CORRESPONDING_MARKER_RE.search(joined_window))
            anchor_in_header = bool(_CORRESPONDING_MARKER_RE.search(section_header))

            for m in _EMAIL_RE.finditer(sent.text):
                email = m.group(1).lower()
                author = email_to_author.get(email)
                if author is None:
                    continue

                footnote_affinity = False
                if section_type == "footnote" and author.family:
                    family_pat = re.compile(r"\b" + re.escape(author.family) + r"\b", re.IGNORECASE)
                    if family_pat.search(joined_window):
                        footnote_affinity = True
                    else:
                        local_compact = re.sub(r"[^a-z0-9]", "", email.split("@", 1)[0])
                        if any(f and f in local_compact for f in _folds(author.family)):
                            footnote_affinity = True

                if strict_emails:
                    promote = email in strict_emails
                else:
                    promote = anchor_in_window or anchor_in_header or footnote_affinity
                if promote:
                    author.corresponding = True
                    email_to_author.pop(email, None)
                    promoted += 1
            if not email_to_author:
                break

        if promoted:
            logger.info(
                "Promoted %d author(s) to corresponding=True from anchor proximity",
                promoted,
            )
