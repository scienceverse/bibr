"""Corresponding-author email harvester.

Bridges parsed PaperContents to LLM-extracted authors: finds explicit
"Corresponding author:" anchors in the OCR'd text, promotes the matching
author's ``corresponding`` flag, and attaches the anchored email. Two
name-paired forms are read before the proximity rules: the footnote
"E-mail address: x@y (Initials Surname)" (read from the whole paper) and a
labelled "E-mail:" inside an affiliation block that names the author.

Previously inline methods on MetadataExtractor; moved here as a
self-contained class so the harvesting logic can be tested without
spinning up the full extractor.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.models import PaperAuthor
    from bibr.paper_contents import PaperContents

logger = logging.getLogger(__name__)

# Email pattern — covers most ASCII addresses; rejects bare-domain mentions.
_EMAIL_RE = re.compile(r"\b([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b")
# GLM-OCR emits "pg " in place of an envelope glyph (✉) at the start of
# corresponding-author lines.  Strip when it precedes an uppercase letter.
_PG_PREFIX_RE = re.compile(r"^\s*pg\s+(?=[A-Z])")
# Anchors that indicate a "Corresponding Author" footer (vs. a plain byline
# listing every author's email).  Used to gate the corresponding=True
# promotion — without this gate, papers like "Attention Is All You Need"
# (8 authors all listing name@google.com in the byline) would have every
# author flagged as corresponding.
_CORRESPONDING_MARKER_RE = re.compile(
    r"\bcorrespond"  # corresponding, correspondence
    r"|✉"  # envelope glyph
    r"|\bpg\s+(?=[A-Z])",  # GLM-OCR misread of envelope at line start
    re.IGNORECASE,
)

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
    token = re.sub(r"[^a-z0-9]", "", (given or "").lower())
    if not token or not local_compact:
        return 0
    if len(token) >= 3 and token in local_compact:
        return 2
    return 1 if local_compact.startswith(token[0]) else 0


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
        for token in re.findall(r"[a-z0-9]+", name.lower()):
            if len(token) >= 3 and (token in local_compact or local_compact in token):
                return True
    # A 2-letter family name (Li, Wu, He) cannot match by containment — it
    # would hit any address — but as the address's leading letters ('lixh@'
    # for Xiaohong Li) it still names the author. Given names stay at 3+
    # chars: a 2-letter given token ('Yu' in Yu-Zhong) prefix-matches far too
    # often and breaks same-surname disambiguation.
    family_compact = re.sub(r"[^a-z0-9]", "", (family or "").lower())
    return len(family_compact) == 2 and local_compact.startswith(family_compact)


# --- name-paired e-mails -------------------------------------------------------
# Elsevier prints the corresponding author's address in a page footnote, paired
# with the name in parentheses: "E-mail address: x@y.org (J.K. Okafor)." or
# "E-mail addresses: a@y.org (A. Lindqvist), b@z.org (B. Okafor)." The pairing is explicit,
# so it is matched against the author list by surname and initials, which makes
# the footnote safe to read wherever it landed (it usually sits outside the
# front-matter block and is often typed as a footnote or even a references section).
_EMAIL_LABEL_RE = re.compile(r"\be-?\s?mail(?:\s+address(?:es)?)?\s*:", re.IGNORECASE)
_NAMED_EMAIL_RE = re.compile(_EMAIL_RE.pattern + r"\s*\(\s*([^()@;:]{3,70}?)\s*\)")
# A real "Corresponding author(s)" / "Correspondence" phrase. Unlike the broad
# marker above it does not match "corresponds to" or "correspondingly" in body text.
_CORRESPONDENCE_PHRASE_RE = re.compile(r"\bcorrespond(?:ing\s+authors?|ence)\b|✉", re.IGNORECASE)
_NAME_TITLES = frozenset(
    {"prof", "professor", "dr", "mr", "mrs", "ms", "miss", "assoc", "assist", "phd", "md"}
)
# A Wiley-style affiliation block can legitimately print one or two addresses;
# more than this many labelled addresses looks like an every-author layout.
_MAX_LABELLED_AFFILIATION_EMAILS = 3


def _fold(text: str | None) -> str:
    """Lower-case, accent-free, letters and digits only ("Al-Tammemi" -> "altammemi")."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[\W_]+", "", base.casefold())


def _initials(tokens: list[str]) -> list[str]:
    """Initials of given-name tokens: "H.-K." -> H, K; "HK" -> H, K; "Hong-Kwan" -> H, K."""
    out: list[str] = []
    for token in tokens:
        for raw in re.split(r"[\s.\-\u2010-\u2015]+", token):
            part = _fold(raw)
            if not part:
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


def _printed_name_matches(name: str, author: PaperAuthor) -> bool:
    """Whether a printed "Initials Surname" (or "Given Surname") names this author.

    The surname is the trailing run of tokens whose folded text equals the
    author's family name (so "van der Berg" works whether or not the extractor
    kept the particle in the family name); everything before it must be
    initial-compatible with the author's given name.
    """
    family = _fold(author.family)
    if len(family) < 2:
        return False
    tokens = _strip_titles(name.replace("\r", " ").replace("\n", " ").split())
    author_initials = _initials(_given_tokens(author.given))
    for k in range(1, len(tokens)):
        if _fold("".join(tokens[k:])) == family and _initials_compatible(
            _initials(tokens[:k]), author_initials
        ):
            return True
    return False


def _named_in_text(text: str, author: PaperAuthor) -> bool:
    """Whether *text* prints this author as "[Title] Initials Surname" somewhere."""
    family = _fold(author.family)
    if len(family) < 2:
        return False
    tokens = text.replace("\r", " ").replace("\n", " ").split()
    author_initials = _initials(_given_tokens(author.given))
    for start in range(len(tokens)):
        for span in range(1, 5):
            if start + span > len(tokens):
                break
            if _fold("".join(tokens[start : start + span])) != family:
                continue
            before: list[str] = []
            for tok in reversed(tokens[max(0, start - 4) : start]):
                bare = tok.strip(",;:()")
                if not bare or not bare[0].isupper() or _fold(bare) in _NAME_TITLES:
                    break
                before.insert(0, bare)
                if _initials_compatible(_initials(before), author_initials):
                    return True
    return False


def _surname_in_local_part(family: str | None, local_compact: str) -> bool:
    """A 3+ letter surname inside the address, or a 2-letter one leading it ('lixh@' for Li)."""
    family_compact = _fold(family)
    if len(family_compact) >= 3:
        return family_compact in local_compact
    return len(family_compact) == 2 and local_compact.startswith(family_compact)


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
            cleaned = _PG_PREFIX_RE.sub("", sent.text)
            if _CORRESPONDING_MARKER_RE.search(cleaned):
                out.update(m.group(1).lower() for m in _EMAIL_RE.finditer(cleaned))
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
        scans body sentences for email patterns, strips the OCR ``pg ``
        artifact (a misread of the envelope glyph), and assigns each email to
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

        Mutates ``authors`` in place.  Skips silently when no authors or no
        email-bearing text are available.
        """
        if not authors:
            return

        from bibr.paper_contents import CanonicalSection

        # Skip references section so reference URLs/DOIs/emails don't bleed in.
        section_type_by_id = {s.section_id: s.section_type for s in self.contents.sections}
        sentences = [
            s
            for s in self.contents.sentences
            if section_type_by_id.get(s.section_id) != CanonicalSection.REFERENCES
        ]

        # Name-paired addresses first: they carry the strongest evidence (an explicit
        # "x@y (Initials Surname)" pairing or a named affiliation block), and an author
        # who gets an address here drops out of the proximity matching below.
        self._harvest_named_footnote_emails(authors)
        self._harvest_affiliation_emails(authors, sentences)

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
            self._promote_corresponding_from_anchors(authors, sentences)
            self._promote_sole_author(authors)
            return

        strict_emails = self._marker_sentence_emails(sentences)

        claimed_emails: set[str] = {a.email.lower() for a in authors if a.email}

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

            for email_match in _EMAIL_RE.finditer(_PG_PREFIX_RE.sub("", sent.text)):
                email = email_match.group(1)
                if email.lower() in claimed_emails:
                    continue

                local = email.split("@", 1)[0].lower()
                local_compact = re.sub(r"[^a-z0-9]", "", local)

                candidates: list[tuple[int, int, int, str, PaperAuthor]] = []
                order_idx = 0
                for family, family_authors in authors_by_family.items():
                    if not family:
                        order_idx += len(family_authors)
                        continue
                    family_pat = re.compile(r"\b" + re.escape(family) + r"\b", re.IGNORECASE)
                    best_dist: int | None = None
                    for j in window_indices:
                        if family_pat.search(_PG_PREFIX_RE.sub("", sentences[j].text)):
                            dist = abs(j - i)
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                    if best_dist is None:
                        order_idx += len(family_authors)
                        continue
                    family_compact = re.sub(r"[^a-z0-9]", "", family.lower())
                    affinity = 1 if family_compact and family_compact in local_compact else 0
                    for author in family_authors:
                        # Only disambiguate on the given name when the surname
                        # is genuinely shared, so ranking between distinct
                        # surnames stays exactly as before.
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

        self._promote_corresponding_from_anchors(authors, sentences)
        self._promote_sole_author(authors)

    @staticmethod
    def _assign_corresponding_email(
        author: PaperAuthor, email: str, claimed_emails: set[str]
    ) -> bool:
        """Attach *email* to *author* and mark them corresponding; True if anything changed.

        Never overwrites a different address and never takes one another author holds.
        """
        wanted = email.lower()
        current = (author.email or "").lower()
        if current and current != wanted:
            return False
        if not current and wanted in claimed_emails:
            return False
        changed = False
        if not current:
            author.email = email
            claimed_emails.add(wanted)
            changed = True
        if not author.corresponding:
            author.corresponding = True
            changed = True
        return changed

    def _harvest_named_footnote_emails(self, authors: list[PaperAuthor]) -> None:
        """Pair "E-mail address(es): x@y (Initials Surname), ..." footnotes with authors.

        Reads the whole paper, not the front-matter scope. A pair is used only when
        a "Corresponding author"/"Correspondence" phrase sits in the same sentence or
        among the three sentences before it (same section), and exactly one author
        fits the printed name by surname and initials; anything else is left alone.
        """
        sentences = list(getattr(self._document, "sentences", None) or [])
        if not authors or not sentences:
            return
        claimed = {a.email.lower() for a in authors if a.email}
        changed = 0
        for i, sent in enumerate(sentences):
            text = sent.text or ""
            if "@" not in text:
                continue
            label = _EMAIL_LABEL_RE.search(text)
            if label is None:
                continue
            preceding = " ".join(
                sentences[j].text or ""
                for j in range(max(0, i - _EMAIL_NAME_WINDOW), i)
                if sentences[j].section_id == sent.section_id
            )
            for pair in _NAMED_EMAIL_RE.finditer(text, label.end()):
                email, name = pair.group(1), pair.group(2)
                if not _CORRESPONDENCE_PHRASE_RE.search(preceding + " " + text[: pair.start()]):
                    continue
                fits = [a for a in authors if _printed_name_matches(name, a)]
                if len(fits) == 1 and self._assign_corresponding_email(fits[0], email, claimed):
                    changed += 1
        if changed:
            logger.info(
                "Paired %d corresponding-author e-mail(s) from name-paired footnotes", changed
            )

    def _harvest_affiliation_emails(self, authors: list[PaperAuthor], sentences: list) -> None:
        """Pair a labelled "E-mail:" in an affiliation block with the author it names.

        Wiley prints the corresponding author's address inside the affiliation block,
        after the block's author names ("Prof. M. Lindqvist ... E-mail: m.lindqvist@x.edu"),
        with no "Corresponding author" wording. The address is used when the block prints
        exactly one author whose surname also appears in the address. A layout that
        labels many addresses (every author listed) is ignored.
        """
        found: list[tuple[PaperAuthor, str]] = []
        labelled: set[str] = set()
        for i, sent in enumerate(sentences):
            text = _PG_PREFIX_RE.sub("", sent.text or "")
            label = _EMAIL_LABEL_RE.search(text)
            if label is None:
                continue
            emails = [m.group(1) for m in _EMAIL_RE.finditer(text, label.end())]
            if not emails:
                continue
            labelled.update(e.lower() for e in emails)
            block = text[: label.start()]
            if len(block.split()) < 3 and i > 0 and sentences[i - 1].section_id == sent.section_id:
                block = (sentences[i - 1].text or "") + " " + block
            named = [a for a in authors if _named_in_text(block, a)]
            for email in emails:
                local_compact = re.sub(r"[^a-z0-9]", "", email.split("@", 1)[0].lower())
                fits = [a for a in named if _surname_in_local_part(a.family, local_compact)]
                if len(fits) == 1:
                    found.append((fits[0], email))
        if not found or len(labelled) > _MAX_LABELLED_AFFILIATION_EMAILS:
            return
        claimed = {a.email.lower() for a in authors if a.email}
        changed = sum(self._assign_corresponding_email(a, e, claimed) for a, e in found)
        if changed:
            logger.info("Paired %d corresponding-author e-mail(s) from affiliation blocks", changed)

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

            for m in _EMAIL_RE.finditer(_PG_PREFIX_RE.sub("", sent.text)):
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
                        family_compact = re.sub(r"[^a-z0-9]", "", author.family.lower())
                        local_compact = re.sub(r"[^a-z0-9]", "", email.split("@", 1)[0])
                        if family_compact and family_compact in local_compact:
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
