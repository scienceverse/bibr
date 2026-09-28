"""A reference list from the citations in a paper's notes.

Law, history and much of the humanities cite in footnotes or endnotes and
often print no reference list at all. When the reference locator finds no
list, or one that yields at most two references, while many notes read as
full citations, the notes become the reference list: every note is split into
the citations it carries, repeats (ibid., op. cit., supra, id., "ref. 5") are
dropped, and what is left goes through the configured reference parser like
the entries of a printed list. :func:`collapse_repeats` then folds the short
form of an earlier citation ("Stoczkowski, “Le peintre”, p. 237") into that
work's first full citation, so each cited work is one reference, and each
reference keeps the ``text_id`` of the note that first cites it.

The functions here never parse fields themselves; they decide which note text
is a citation and which parsed references are one work.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.utils.text import collapse_ws

# At least this many notes must carry a full citation before the notes stand
# in for a missing reference list. Below it, a stray "see Smith (2001)" in an
# acknowledgement note is not a citation apparatus.
MIN_CITATION_NOTES = 5
# A located reference list with at most this many references can be
# supplemented from the notes; a longer one is the paper's reference list and
# is never touched.
MAX_LISTED_REFERENCES = 2
# ... and only when the notes cite this many times more works than the list
# holds: a two-entry "Legal acts" list beside five citing notes stays as it is.
LISTED_DOMINANCE = 15
# A note longer than this is not scanned for citations. The longest real note
# seen runs to under 2,000 characters; the split scan grows faster than the
# note (about 0.6 s for one note of 25,000 characters), and it runs on the
# event loop.
MAX_NOTE_CHARS = 8000

_EXTENDED = [chr(code) for code in range(0x100, 0x250)]
_UP = "A-ZÀ-ÖØ-Þ" + "".join(c for c in _EXTENDED if c.isupper())
_LOW = "a-zß-öø-ÿ" + "".join(c for c in _EXTENDED if c.islower())
_YEAR = r"(?:1[5-9]\d\d|20[0-2]\d)"

# Notes printed under a table or figure rather than in the note apparatus:
# "Note. All survey data ...", "Source: ...".
_TABLE_NOTE_RE = re.compile(
    r"^\s*(?i:notes?|sources?|fuentes?|quellen?|nota|remarques?)\s*[.:]",
)
# The printed mark opening a note: "12. ", "12) ", "(12) ", "12 " or a mark on
# a line of its own. One to three digits, so a year opening a note stays.
_NOTE_MARK_RE = re.compile(r"^\s*\(?\s*\d{1,3}\s*\)?(?:[.)](?=\s)|(?=\s))\s*")
# A later note run into the same text row: a mark at the start of a line.
_INNER_NOTE_RE = re.compile(r"\n\s*(?=\d{1,3}[.)]\s+\S)")

# Words that introduce a citation ("See", "Cf.", "Voir aussi", "Véase").
_LEAD_WORDS = (
    r"see\s+also|see\s+e\.\s?g\.|see|cf\.|compare|for\s+example|for\s+instance|e\.\s?g\.|"
    r"voir\s+aussi|voir\s+notamment|voir|notamment|par\s+exemple|"
    r"véase\s+también|véase|vease|vid\.|por\s+ejemplo|también|según|selon|"
    r"vgl\.|siehe|z\.\s?b\.|zob\.|also|aussi"
)
# Conjunctions that open a second citation ("and W. Smith, ...", "Y también").
_CONJUNCTIONS = r"and|et|und|y|e|oraz|or|ou"
_LEAD_IN_RE = re.compile(
    rf"^(?:(?:(?i:{_CONJUNCTIONS})\s+)?(?i:{_LEAD_WORDS})(?:\s*[,:]\s*|\s+)"
    rf"|(?i:{_CONJUNCTIONS})\s+(?=[{_UP}“«‘\"„])"
    r"|[\s,:;(]+)+"
)

# The name that opens a citation: "W. Stoczkowski", "Glenn B. Canner",
# "LIEVEN, Dominic", "CURRAN Cynthia", "Angelo PASSOLUNGHI", "Smith, J.".
_NAME = rf"(?:Mc|Mac|O['’])?[{_UP}][{_LOW}'’]+(?:-[{_UP}][{_LOW}'’]+)?"
_INITIALS = rf"(?:[{_UP}][{_LOW}]?(?:-[{_UP}])?\.\s?)+"
_UPPER_SURNAME = rf"[{_UP}]{{2,}}(?:[-\s][{_UP}]{{2,}})*"
# Two capitalised words that are not a name: "The Book", "La nature".
_NOT_FIRST_NAME = (
    r"(?!(?:The|An?|La|Le|Les|L['’]|El|Los|Las|Lo|Il|Der|Die|Das|Den|Un|Une|Uno|Una|Ein|Eine|"
    r"In|On|At|For|From|With|And|Of|Du|De|Des|Del|Au|Aux|Sur|This|That)\b)"
)
_INITIALS_NAME = rf"{_INITIALS}(?:{_NAME}\s)?{_NAME}(?:\s{_NAME})?"
_MARKED_NAME = (
    rf"(?:{_INITIALS_NAME}"
    rf"|{_UPPER_SURNAME}[,.]?\s(?:{_NAME}(?:\s{_NAME})?(?:\s{_INITIALS})?|{_INITIALS})"
    rf"|{_NOT_FIRST_NAME}{_NAME}\s{_UPPER_SURNAME}"
    rf"|{_NAME},\s{_INITIALS}(?!\s?[{_UP}][{_LOW}]))"
)
# "Glenn B. Canner", "François Roustang": two capitalised words read as a
# name only before a comma, a bracket or a second author, never before a
# period ("A New History." is a title).
_FULL_NAME = rf"{_NOT_FIRST_NAME}{_NAME}\s(?:{_INITIALS})?{_NAME}(?:\s{_NAME})?"
_CO_AUTHOR = r"\s(?:y|et|and|und|e|&|u)\s"
# A name ends where the byline or the title begins: at punctuation, a
# bracket ("(ed.)"), a dash or a conjunction joining a second author.
_ONSET = (
    rf"(?:{_MARKED_NAME}(?=[,.:;]|\s\(|\s?[–—]|(?<=\.)\s|{_CO_AUTHOR})"
    rf"|{_FULL_NAME}(?=[,;]|\s\(|\s?[–—]|{_CO_AUTHOR}))"
)
_ONSET_BODY = rf"(?:{_MARKED_NAME}|{_FULL_NAME})"
_ONSET_RE = re.compile(_ONSET)
# A name closed by punctuation or joined to a second author: "W.
# Stoczkowski,", "M. Bost-Fiévet y S. Provini", but not "John Milius (1982)".
_STRICT_ONSET_RE = re.compile(
    rf"(?:{_MARKED_NAME}(?=[,.:;]|(?<=\.)\s|{_CO_AUTHOR}[{_UP}])"
    rf"|{_FULL_NAME}(?=[,;]|{_CO_AUTHOR}[{_UP}]))"
)
# A classical work cited by book and chapter: "Plinio, Historia Natural, V,
# 45", "Diodoro de Sicilia, Biblioteca histórica, III, 32, 4".
_CLASSICAL = (
    rf"{_NAME}(?:\s(?:de|of|von|d['’])\s?{_NAME})?,\s[^,;]{{3,80}},\s"
    r"(?:[IVXLCDM]{1,6}|\d{1,3})\b(?![.]\d)"
)
_CLASSICAL_RE = re.compile(_CLASSICAL)
# Where commentary hands over to a citation: a lead-in word, a preposition
# naming the author ("by", "par", "por", "de", "von") or an opening bracket
# before a name, or a comma or a plain word before initials and a surname
# ("... como bien ha demostrado J. Kany-Turpin, “Notre passé ...").
_HANDOVER_RE = re.compile(
    rf"(?:(?<![\w.])(?P<lead>(?i:{_LEAD_WORDS}))\.?(?:\s*[,:]\s*|\s+)"
    rf"|(?<![\w.])(?i:by|par|por|de|von|chez|bei|przez)\s+"
    rf"|\(\s*)"
    rf"(?=(?:{_ONSET}|{_CLASSICAL}))"
    rf"|(?:[,:]\s*|(?<=[{_LOW}])\s+)"
    rf"(?=(?:{_INITIALS_NAME}|{_UPPER_SURNAME},\s{_NAME}(?:\s{_NAME})?)[,:.])"
)
# A name in running text: "John Milius (1982) con los Hércules ...".
_NAME_IN_PROSE_RE = re.compile(rf"{_ONSET_BODY}\s\(\s*{_YEAR}\s*\),?\s+[{_LOW}]")
# A lowercase word of four letters or more: commentary, not a byline.
_PROSE_WORD_RE = re.compile(rf"(?<![\w{_UP}])[{_LOW}]{{4,}}\b")
# A sentence break inside a note; what follows may be a new citation.
_SENTENCE_BREAK_RE = re.compile(rf"(?<=[{_LOW}\d)»”ˮ’])\.\s+(?=[{_UP}“«‘\"„])")
# Double quotes around a title: “…”, «…», „…“ and the “…ˮ of some typesetters.
# A period inside them ("“Representing archaeological Knowledge in museums.
# Exhibiting human origins ...ˮ") ends no sentence. An opening quote still
# unclosed this many characters on stops counting (a lost closing quote).
_OPENING_QUOTES = "“«„"
_CLOSING_QUOTES = "”»ˮ“"
_MAX_QUOTED = 250
# A conjunction between two citations: "... ISSN 1477-4747 or BRADBURY, ...".
_CONJUNCTION_BREAK_RE = re.compile(rf"\s(?i:or|and|ou|et|y|o)\s(?={_UPPER_SURNAME},\s)")
# The word before a period that makes it an abbreviation, not a sentence end.
_ABBREVIATION_END_RE = re.compile(
    rf"(?:^|[\s(])(?:[{_LOW}]{{1,4}}|(?i:coord|trans|transl|trad|hrsg|comp|eds|éds|ed|éd))$"
)

# A clause ending on a place of publication: "... Montreal. Vancouver".
_ENDS_ON_PLACE_RE = re.compile(rf"[{_UP}][{_LOW}]+(?:[\s-][{_UP}][{_LOW}]+)?\s*$")
# An ISO 690 byline, "BLANCHARD, Rae." or "COLLALTO, Marie Therese.".
_ISO_BYLINE_RE = re.compile(rf"{_UPPER_SURNAME},\s{_NAME}(?:\s{_NAME})*\.\s")

# Repeat citations: they point back at a work cited in full before. "supra"
# and "infra" count as words of their own, never inside a hyphenated one
# ("Supra-national Law ..." is a title).
_REPEAT_START_RE = re.compile(
    r"^(?i:ibid|ibidem|ibíd|ibídem|id|idem|eadem|ead|ebd|ebenda|tamže|tamtéž|tamze)\b\.?"
)
_REPEAT_ANY_RE = re.compile(
    r"(?i:\bop\.?\s?cit\b|\bloc\.?\s?cit\b|\bart\.?\s?cit\b|\bob\.?\s?cit\b|\bopus\s+citatum\b"
    r"|\ba\.\s?a\.\s?O\b|(?<![\w-])(?:supra|infra)(?![\w-])|\bpr[ée]cit[ée]e?s?\b"
    r"|\bcit\.\s*(?:supra|n\.|note)"
    r"|\bcit[ée]e?s?,?\s+(?:à\s+la\s+)?note\s+\d|\bref\.\s?\d|\bcit\.\s?d\.|\bdz\.\s?cyt)"
)

# Bibliographic cues.
# A year, not a decade ("the 1970s") or part of a date or number.
_YEAR_RE = re.compile(rf"(?<![\d/.])({_YEAR})(?![\d/]|s\b)")
_LIFESPAN_RE = re.compile(rf"\(\s*{_YEAR}\s*[-–—]\s*{_YEAR}\s*\)")
_DATE_STAMP_RE = re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{4}\b")
# A quoted title: the quote closes on a comma or period (‘‘Title,’’) or is
# followed by one or by "in" (« Title », in ...). A quoted phrase inside a
# sentence (the funds used for ‘‘home improvement’’ may) is no title.
_OPEN_QUOTE = r"(?:[“«„‘]{1,2}|''|\")"
_CLOSE_QUOTE = r"(?:[”»’ˮ\"“]{1,2}|'')"
_QUOTED_RE = re.compile(
    rf"{_OPEN_QUOTE}\s*\S[^”»“«\"]{{4,}}?(?:[,.]\s*{_CLOSE_QUOTE}"
    rf"|{_CLOSE_QUOTE}(?=\s*[,.;:()]|\s+(?i:in|en|dans|im)\b))"
)
_PLACE_PUBLISHER_RE = re.compile(rf"\b[{_UP}][{_LOW}]+(?:[\s-][{_UP}][{_LOW}]+)?\s?:\s?[{_UP}]")
_CONTAINER_RE = re.compile(
    rf"(?:^|[,.”’»ˮ\"]\s*)(?:[Ii]n|[Ee]n|[Dd]ans|[Ii]m|[Ww])\s?:?\s[{_UP}“«\"‘„]"
    r"|\((?i:eds?|éds?|dir|coord|hrsg|red|comp)\.?\)"
    r"|\b(?i:vol|no|nr|tome|bd|heft|n[°º]|year|roč)\.?\s?\d"
    r"|\b(?i:pp?)\.\s?\d|\bS\.\s?\d|(?i:\bisbn\b|\bissn\b|\bdoi\b)"
    r"|\b(?i:press|verlag|presses|éditions?|editions?|editorial|edizioni|publishers?|"
    r"publications|university|universidad|université|universitätsverlag|wydawnictwo)\b"
)
# "Title, Publisher, 1995." closes on its year.
_TRAILING_YEAR_RE = re.compile(rf",\s*{_YEAR}[a-z]?\s*[.;]?\s*$")
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class NoteCitation:
    """One citation read out of a note, ready for the reference parser."""

    text: str
    text_id: int
    # False for a citation with no year (a classical work, a short form):
    # it joins the list but never counts toward the trigger.
    full: bool = True


@dataclass(frozen=True)
class NoteCitations:
    """The citations of a paper's notes and the evidence behind them."""

    citations: tuple[NoteCitation, ...]
    notes: int  # notes read
    citing_notes: int  # notes carrying at least one full citation
    repeats: int  # ibid./op. cit./supra citations dropped


def _note_rows(contents: PaperContents) -> list[tuple[int, str]]:
    """``(text_id, text)`` of every note row, in document order.

    A note the page break cuts in two is read as one row with the text id of
    its first part (see :func:`_carries_over`).
    """
    note_sections = {
        section.section_id: section
        for section in contents.sections
        if section.section_type == CanonicalSection.FOOTNOTE
    }
    if not note_sections:
        return []
    rows: list[tuple[int, str]] = []
    previous: tuple[PaperSection, PaperSentence] | None = None
    for sentence in contents.sentences:
        section = note_sections.get(sentence.section_id)
        if section is None or sentence.is_display_formula or not (sentence.text or "").strip():
            continue
        if previous is not None and _carries_over(section, sentence, *previous):
            rows[-1] = (rows[-1][0], f"{rows[-1][1]} {sentence.text}")
        else:
            rows.append((sentence.text_id, sentence.text))
        previous = (section, sentence)
    return rows


def _carries_over(
    section: PaperSection,
    sentence: PaperSentence,
    previous_section: PaperSection,
    previous: PaperSentence,
) -> bool:
    """Whether *sentence* is the rest of the note before it, carried over to the next page.

    The parser gives the carried-over part a note section of its own with no
    printed mark: "... Vdovy hospodařící na venkovských usedlostech v první
    polovině 19. století [Confident or Desperate? Widows Farming on Rural" on
    one page, "Estates in the First Half of the 19th Century]. In VOJÁČEK,
    Milan (ed.). ..." on the next. It continues the note before it when that
    note, on an earlier page, breaks off inside a citation. A note that breaks
    off in commentary ("... describe e ilustra este « mundo hostil » de") is
    left apart: its commentary would hide the citation the next part opens.
    """
    return (
        section is not previous_section
        and section.synthetic_kind == "footnote"
        and previous_section.synthetic_kind == "footnote"
        and section.footnote_label is None
        and _NOTE_MARK_RE.match(sentence.text) is None
        and sentence.page_number is not None
        and previous.page_number is not None
        and sentence.page_number > previous.page_number
        and _breaks_off(previous.text)
    )


def _breaks_off(text: str) -> bool:
    """Whether *text* stops inside a citation: in an open bracket or quote, or
    on a comma or colon."""
    text = text.rstrip()
    return (
        text.endswith((",", ":"))
        or text.count("[") > text.count("]")
        or text.count("(") > text.count(")")
        or _quote_left_open(text)
    )


def _split_notes(text: str) -> list[str]:
    """One text row may hold several notes ("2. ... 2019.\\n3. Ibid.")."""
    return [part for part in _INNER_NOTE_RE.split(text) if part.strip()]


def _strip_lead_in(clause: str) -> str:
    return _LEAD_IN_RE.sub("", clause, count=1).strip()


def _starts_work(text: str) -> bool:
    return bool(_ONSET_RE.match(text) or _CLASSICAL_RE.match(text))


def _citation_start(clause: str) -> int:
    """Where the citation in *clause* starts, past any commentary before it.

    "... ejemplos recogidos por W. Stoczkowski, Aux origines ..." starts at
    "W. Stoczkowski". Commentary is a run of words holding a lowercase word
    of four letters or more, so a corporate author ("U.S. Department of
    Housing and Urban Development, U.S. Housing Market Conditions") is never
    cut. The first hand-over that leads to a citation wins, passing over a
    name in running text ("... de John Milius (1982) con los ..."). A clause
    opening with a name (unless such running text follows it), a classical
    work or a quoted title starts at 0, and so does one where no hand-over
    leads to a citation.
    """
    if clause[:1] in '“«„‘"' or _CLASSICAL_RE.match(clause):
        return 0
    if _ONSET_RE.match(clause) and not _NAME_IN_PROSE_RE.match(clause):
        return 0
    for match in _HANDOVER_RE.finditer(clause):
        rest = clause[match.end() :].strip()
        # A lead-in word marks what precedes it as commentary; other
        # hand-overs need a prose word before them.
        if match.group("lead") is None and not _PROSE_WORD_RE.search(clause[: match.start()]):
            continue
        if _NAME_IN_PROSE_RE.match(rest) or not looks_like_citation(rest):
            continue
        return match.end()
    return 0


def _cut_commentary(clause: str) -> str:
    return clause[_citation_start(clause) :].strip()


def _clean(piece: str) -> str:
    return _cut_commentary(_strip_lead_in(piece)).strip(" ,;:")


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """``(open, close)`` offsets of the double-quoted spans in *text*."""
    spans: list[tuple[int, int]] = []
    opened: int | None = None
    for index, char in enumerate(text):
        if opened is None:
            if char in _OPENING_QUOTES:
                opened = index
        elif char in _CLOSING_QUOTES:
            spans.append((opened, index))
            opened = None
        elif index - opened > _MAX_QUOTED:
            opened = None
    return spans


def _quote_left_open(text: str) -> bool:
    """Whether *text* ends inside a double-quoted span."""
    spans = _quoted_spans(text)
    closed_at = spans[-1][1] if spans else -1
    tail = text[closed_at + 1 :]
    opened = next((i for i, char in enumerate(tail) if char in _OPENING_QUOTES), None)
    return opened is not None and len(tail) - opened <= _MAX_QUOTED


def _sentence_breaks(text: str) -> list[re.Match[str]]:
    """The sentence breaks of *text* that fall outside quoted titles."""
    spans = _quoted_spans(text)
    return [
        match
        for match in _SENTENCE_BREAK_RE.finditer(text)
        if not any(opened < match.start() < closed for opened, closed in spans)
    ]


def _next_piece(text: str) -> str:
    """*text* up to its next sentence break: the right side of a candidate split."""
    breaks = _sentence_breaks(text)
    return text[: breaks[0].start()] if breaks else text


def _soft_split(clause: str) -> list[str]:
    """Split *clause* where a sentence or a conjunction hands over to a new citation.

    A break is taken when a lead-in word and a name follow it ("... savante.
    Y también, W. Stoczkowski, ..."), or when a name follows it and both
    sides read as citations ("Plinio, Historia Natural, V, 45. Esquilo,
    Prometeo encadenado, 454"), or when a lead-in word follows a citation
    or a repeat and a citation comes after it, whatever its byline looks like
    ("L. M c Le o d , op. cit., supra, note 8, p. 3. Voir aussi Rapport
    fédéral-provincial ..."). A period after an abbreviation ("tr. fr.
    Olivier Putois") is no break. A break is also taken when commentary
    follows a citation and then hands over to a new one ("..., 2005, p.
    175-187. También fue representado ..., S. Moser, “Archaelogical ..."),
    and commentary after a dated citation that leads to no new one ("...,
    1987, p. 180-193. Estos autores han demostrado ...") is cut off.
    """
    breaks = sorted(
        [(m.start(), m.end(), True) for m in _sentence_breaks(clause)]
        + [(m.start(), m.end(), False) for m in _CONJUNCTION_BREAK_RE.finditer(clause)]
    )
    pieces: list[str] = []
    start = 0
    for begin, end, sentence in breaks:
        if begin < start:
            continue
        left, right = clause[start:begin], clause[end:]
        if sentence and _ABBREVIATION_END_RE.search(left):
            continue
        led = sentence and _LEAD_IN_RE.match(right) is not None
        left_kind = _citation_kind(_clean(left))
        # A repeat ("ABRAMS, ref. 6.") closes a citation as a full one does.
        left_cites = left_kind is not None or is_repeat_citation(_clean(left))
        cleaned = _clean(right)
        # Commentary that hands over to a citation within its first sentence.
        stripped = _strip_lead_in(right)
        cited_from = _citation_start(stripped)
        handed_over = 0 < cited_from <= len(_next_piece(stripped))
        # After a sentence break the first sentence must read as a citation;
        # "or SURNAME, Name" opens a work whose byline is a sentence of its own.
        opened = right if not sentence else _next_piece(right)
        if (
            (led and _starts_work(stripped))
            or (led and left_cites and looks_like_citation(cleaned))
            or (left_cites and _starts_work(right) and _citation_kind(_clean(opened)))
            or (left_cites and handed_over and looks_like_citation(cleaned))
        ):
            pieces.append(left)
            start = end
        elif sentence and left_kind == "full" and not looks_like_citation(cleaned):
            pieces.append(left)
            return pieces
    pieces.append(clause[start:])
    return pieces


def _split_citations(note: str) -> list[str]:
    """The citation clauses of one note, lead-ins and commentary removed."""
    text = collapse_ws(_NOTE_MARK_RE.sub("", note, count=1))
    clauses: list[str] = []
    for piece in re.split(r"\s*;\s*", text):
        # "London; New York: Routledge" is one imprint, not two citations; a
        # clause after a locator ("..., p. 7; Rapport : Colloques ...") is.
        if (
            clauses
            and _PLACE_PUBLISHER_RE.match(piece)
            and len(piece.split(":", 1)[0]) <= 30
            and _ENDS_ON_PLACE_RE.search(clauses[-1])
        ):
            clauses[-1] = f"{clauses[-1]}; {piece}"
            continue
        clauses.append(piece)
    out: list[str] = []
    for clause in clauses:
        for piece in _soft_split(clause):
            piece = _clean(piece)
            if piece:
                out.append(piece)
    return out


def is_repeat_citation(clause: str) -> bool:
    """True for "Ibid.", "Id., p. 75", "L. McLeod, op. cit., supra, note 8", "BAY, ref. 27"."""
    return bool(_REPEAT_START_RE.match(clause) or _REPEAT_ANY_RE.search(clause))


def _citation_kind(clause: str) -> str | None:
    """ "full" for a dated citation, "short" for an undated one, None for commentary.

    A full citation needs a publication year (a lifespan such as
    "(1880-1943)" or a date stamp does not count) and one bibliographic cue:
    a name opening the clause, a quoted title, a "Place: Publisher" imprint, a
    container, edition or locator marker ("In", "(ed.)", "vol. 3", "pp. 12-19",
    "ISBN"), a publisher word, or the year closing the clause. An undated
    citation needs three of those cues, the book-and-chapter shape of a
    classical work ("Estrabón, Geografía, XVI, 17"), or an ISO 690 byline
    ("COLLALTO, Marie Therese.") and a container or locator.
    """
    if len(_WORD_RE.findall(clause)) < 4 or _TABLE_NOTE_RE.match(clause):
        return None
    probe = _DATE_STAMP_RE.sub(" ", _LIFESPAN_RE.sub(" ", clause))
    cues = sum(
        bool(found)
        for found in (
            _STRICT_ONSET_RE.match(clause),
            _QUOTED_RE.search(clause),
            _PLACE_PUBLISHER_RE.search(clause),
            _CONTAINER_RE.search(clause),
        )
    )
    if _YEAR_RE.search(probe):
        if cues or (_TRAILING_YEAR_RE.search(probe) and probe.count(",") >= 2):
            return "full"
        return None
    if cues >= 3 or _CLASSICAL_RE.match(clause):
        return "short"
    # "BLANCHARD, Rae. Richard Steele and the Status of Women. In Studies in
    # Philology, year 26, no. 3, p. 322–355.": an ISO 690 byline and a
    # container or locator.
    if _ISO_BYLINE_RE.match(clause) and _CONTAINER_RE.search(clause):
        return "short"
    return None


def looks_like_citation(clause: str) -> bool:
    """True when *clause* reads as a citation of a work, dated or not."""
    return _citation_kind(clause) is not None


def note_citations(contents: PaperContents) -> NoteCitations:
    """Every citation the paper's notes carry, in note order."""
    citations: list[NoteCitation] = []
    notes = citing = repeats = 0
    for text_id, row in _note_rows(contents):
        if _TABLE_NOTE_RE.match(row):
            continue
        for note in _split_notes(row):
            notes += 1
            if len(note) > MAX_NOTE_CHARS:
                continue
            full_found = False
            for clause in _split_citations(note):
                if is_repeat_citation(clause):
                    repeats += 1
                    continue
                kind = _citation_kind(clause)
                if kind is None:
                    continue
                citations.append(NoteCitation(clause, text_id, full=kind == "full"))
                full_found = full_found or kind == "full"
            citing += full_found
    return NoteCitations(tuple(citations), notes, citing, repeats)


def notes_replace_list(found: NoteCitations, listed: int) -> bool:
    """Whether the notes' citations should stand in for the reference list.

    Only a missing list, or one with at most :data:`MAX_LISTED_REFERENCES`
    references, qualifies, and only when at least :data:`MIN_CITATION_NOTES`
    notes carry a full citation and they outnumber the list's references
    :data:`LISTED_DOMINANCE` times.
    """
    if listed > MAX_LISTED_REFERENCES:
        return False
    return found.citing_notes >= max(MIN_CITATION_NOTES, LISTED_DOMINANCE * listed)


# ---------------------------------------------------------------------------
# One reference per cited work
# ---------------------------------------------------------------------------


def _fold(text: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_WORD_RE.findall(ascii_only.casefold()))


def _surnames(authors: str | None) -> set[str]:
    """Name-like tokens of an author string: words of three letters or more."""
    return {token for token in _fold(authors).split() if len(token) >= 3}


def _same_work(title: str, other: str) -> bool:
    """Two folded titles name one work: equal, or the shorter (two words or
    more) opens the longer."""
    if not title or not other:
        return False
    if title == other:
        return True
    short, long_ = sorted((title, other), key=len)
    return len(short.split()) >= 2 and long_.startswith(short + " ")


def _filled(ref) -> int:
    return sum(
        bool(getattr(ref, name, None))
        for name in ("authors", "year", "container", "publisher", "volume", "first_page", "doi")
    )


_WorkKey = tuple[str, set[str], int | None]


def _work_key(ref) -> _WorkKey:
    """Folded title, author names and year: what tells two citations' works apart."""
    return _fold(ref.title), _surnames(ref.authors), ref.year


def collapse_repeats(refs: list, existing: list | None = None) -> list:
    """Keep one reference per cited work, in the order works are first cited.

    A later citation of a work already kept (the same title, or a title the
    other opens, by an overlapping author or with no author on one side, and
    not dated to another year) is its short form or a repeat: the fuller of the two parses stands in the
    first citation's place and keeps its ``text_id``. A note citation of a
    reference in *existing* (a located list) is dropped; those are never
    replaced.
    """
    kept: list = []
    keys: list[_WorkKey] = []
    existing_keys = [_work_key(ref) for ref in existing or ()]

    def matches(key: _WorkKey, other: _WorkKey) -> bool:
        (title, names, year), (other_title, other_names, other_year) = key, other
        if not _same_work(title, other_title):
            return False
        if year and other_year and year != other_year:
            return False
        return not names or not other_names or bool(names & other_names)

    for ref in refs:
        key = _work_key(ref)
        if any(matches(key, other) for other in existing_keys):
            continue
        for index, other in enumerate(keys):
            if matches(key, other):
                if _filled(ref) > _filled(kept[index]):
                    ref.text_id = kept[index].text_id
                    kept[index] = ref
                    keys[index] = key
                break
        else:
            kept.append(ref)
            keys.append(key)
    return kept


def quoted_title(citation: str) -> tuple[str, int | None] | None:
    """Title and year of a citation that opens on a quoted title, else None.

    The reference tagger finds nothing in "« Au commencement était
    l’hypnose », Conférences et débats, septembre 2005."; the quotes mark
    the title, and the last year after them dates the work.
    """
    text = _NOTE_MARK_RE.sub("", citation, count=1).lstrip()
    spans = _quoted_spans(text)
    if not spans or spans[0][0] != 0:
        return None
    closed = spans[0][1]
    title = text[1:closed].strip(" ,;:")
    if len(re.sub(r"\W", "", title)) < 3:
        return None
    years = _YEAR_RE.findall(text[closed + 1 :])
    return title, int(years[-1]) if years else None


def usable_reference(ref) -> bool:
    """A parsed note citation worth keeping: a title of three characters or
    more, or authors and a year."""
    title = re.sub(r"\W", "", ref.title or "")
    return len(title) >= 3 or bool(ref.authors and ref.year)


def assign_note_text_ids(refs: list, citations: tuple[NoteCitation, ...]) -> None:
    """Give each reference the ``text_id`` of the note citation it was parsed from.

    For parse strategies that do not keep one output per input (the LLM
    parsers): a reference takes the first citation carrying its title, else
    the citation at its position when the counts agree, else none.
    """
    folded = [_fold(citation.text) for citation in citations]
    aligned = len(refs) == len(citations)
    for index, ref in enumerate(refs):
        title = _fold(ref.title)
        text_id = next(
            (
                c.text_id
                for c, text in zip(citations, folded, strict=True)
                if title and title in text
            ),
            None,
        )
        if text_id is None and aligned:
            text_id = citations[index].text_id
        ref.text_id = text_id
