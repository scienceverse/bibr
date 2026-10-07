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

# Words that introduce a citation ("See", "Cf.", "Voir aussi", "Lire", "Véase").
_LEAD_WORDS = (
    r"see\s+also|see\s+e\.\s?g\.|see|cf\.|compare|for\s+example|for\s+instance|e\.\s?g\.|"
    r"voir\s+aussi|voir\s+notamment|voir|lire\s+aussi|lire|notamment|par\s+exemple|"
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
# Initials may join with a hyphen, with or without a period before it:
# "J-M. Martin", "J.-M. Bruguière", "P.-B. Hugenholtz".
_NAME = rf"(?:Mc|Mac|O['’])?[{_UP}][{_LOW}'’]+(?:-[{_UP}][{_LOW}'’]+)?"
_INITIALS = rf"(?:[{_UP}][{_LOW}]?(?:\.?-[{_UP}][{_LOW}]?)?\.\s?)+"
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
# The same with its author and work: the tagger reads the author of
# "Estrabón, Geografía, XVI, 17" and takes the work for no title.
_CLASSICAL_WORK_RE = re.compile(
    rf"(?P<author>{_NAME}(?:\s(?:de|of|von|d['’])\s?{_NAME})?),\s(?P<work>[^,;]{{3,80}}),\s"
    r"(?:[IVXLCDM]{1,6}|\d{1,3})\b(?![.]\d)"
)
# A title that opens on a year, before the bracketed imprint whose year dates
# the work: "…, 1977 Consumer Credit Survey (Board of Governors …, 1978)".
_YEAR_LED_TITLE_RE = re.compile(
    rf",\s(?P<title>{_YEAR}\s[{_UP}][^,;:()«»“”‘’\"]{{2,120}}?)\s"
    rf"\((?:[^()]*?,\s)?(?P<year>{_YEAR})\)"
)
# Where commentary hands over to a citation: a lead-in word, a preposition
# naming the author ("by", "par", "por", "de", "von"), with the author's title
# if any ("par le professeur"), or an opening bracket before a name, or a
# comma or a plain word before initials and a surname ("... como bien ha
# demostrado J. Kany-Turpin, “Notre passé ...").
_HANDOVER_RE = re.compile(
    rf"(?:(?<![\w.])(?P<lead>(?i:{_LEAD_WORDS}))\.?(?:\s*[,:]\s*|\s+)"
    rf"|(?<![\w.])(?i:by|par|por|de|von|chez|bei|przez)\s+"
    rf"(?:(?:(?:le|la)\s+)?(?i:professeure?|professor|prof\.|dr\.?)\s+)?"
    rf"|\(\s*)"
    rf"(?=(?:{_ONSET}|{_CLASSICAL}))"
    rf"|(?:[,:]\s*|(?<=[{_LOW}])\s+)"
    rf"(?=(?:{_INITIALS_NAME}|{_UPPER_SURNAME},\s{_NAME}(?:\s{_NAME})?)[,:.])"
)
# How much of the text after a hand-over is read to tell whether a citation
# follows: a citation shows its year and cues well within it. Reading each
# hand-over's whole remainder, at every break of a split, made a long note's
# split cubic in its length.
_HANDOVER_READ = 1000
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
# The Nature/Science locator: journal, volume, pages, then the year in
# parentheses ("Devl Biol. 81, 286-300 (1981).", "Science 228, 1210 (1985).").
# A single-author entry of that style has no other cue: "Kimble, J." before a
# capitalised journal word is no name onset, and the year does not close it.
# The capitalised word before the volume and a page range or a page of three
# or more digits keep prose ("rates were 45, 52 (2019)", "Tables 3, 4 (2015)")
# out.
_VOLUME_PAGES_YEAR_RE = re.compile(
    rf"\b[A-Z][^\W\d_]*\.?\s+\d{{1,4}},\s*"
    rf"(?:[A-Z]?\d{{1,6}}\s?[-–]\s?[A-Z]?\d{{1,6}}|[A-Z]?\d{{3,6}})\s*\(\s*{_YEAR}[a-z]?\s*\)"
)
_WORD_RE = re.compile(r"\w+")

# Court decisions. A decision is cited by its court, a chamber, the date and
# at times a case number or the parties, then the reporters that print it:
# "Civ. 1re, 16 juill. 1998, D. 1999. 306, note Dreyer", "CA Paris, Pôle 5,
# ch. 2, 12 janvier 2018, RG n° 16/19375, SAS Les Éditions du net c/ Victima
# H.", "R. c. Deschamps, C.A. Québec, n° 500-10-000003-887, 11 mars 1988".
# The date of the decision, day, month and year.
_MONTH = (
    r"(?:janv(?:ier|\.)|f[ée]vr(?:ier|\.)|f[ée]v\.|mars|avr(?:il|\.)|mai|juin|"
    r"juil(?:let|l?\.)|ao[ûu]t|sept(?:embre|\.)|oct(?:obre|\.)|nov(?:embre|\.)|"
    r"d[ée]c(?:embre|\.)|January|February|March|April|May|June|July|August|"
    r"September|October|November|December|(?:Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec)\.)"
)
_DECISION_DATE_RE = re.compile(rf"(?<![\w.–-])(?:1er|[1-3]?\d)\s{_MONTH}\s({_YEAR})(?!\d)")
# A court, a chamber or two parties in what precedes the date: "CA Paris",
# "TGI Laval", "Cass. 1re civ.", "4e ch.", "Hoge Raad", "R. c. Deschamps".
# A newspaper ("Le Figaro, 18 déc. 2014") has none of them.
_COURT_RE = re.compile(
    rf"(?<![\w.])(?:CA|CAA|TGI|TJ|TI|TA|T\.G\.I\.|C\.A\.)\s[{_UP}]"
    r"|(?<![\w.])(?:CE|CJUE|CJCE|CEDH|C\.E\.|C\.S\.|C\.S\.C\.|Cass\.|Civ\.|Crim\.|Com\.|"
    r"Soc\.|Req\.|Cons\.\s?const\.|Trib\.|T\.\s?com\.|BGH|BVerfG|OLG|Hoge\s+Raad|Cour|Court|"
    r"Tribunal)(?!\w)"
    r"|\b\d{1,2}(?:re|er|e|ème)\s(?:ch\.|civ\.|chambre)|\bch\.\s?\d|\bP[ôo]le\s\d"
    rf"|\s(?:c\.|c/|v\.|vs\.?)\s[{_UP}]"
)
# A reporter printing the decision right after its date, "DH 1934. 385":
# the court is then the city alone ("Paris, 27 avr. 1934, DH 1934. 385.").
_DATED_REPORTER_RE = re.compile(rf",\s*[{_UP}][\w.]*(?:\s[\w.]+){{0,2}}\s{_YEAR}\.\s?\d")
# What may stand between the court and the date: capitalised words, a chamber,
# a case number ("n° 500-10-000003-887"), never a prose word ("Voir par
# exemple CA Paris" starts at "CA").
_DESIGNATION_RE = re.compile(rf"(?:[{_UP}\d]|n[°º]\s|no\s|ch\.)[^;:()«»“”\"]*")
_DESIGNATION_WORDS = frozenset(
    ("chambre", "civile", "criminelle", "commerciale", "sociale", "réunies", "mixte")
)
_LONG_LOWER_RE = re.compile(rf"(?<![\w{_UP}])[{_LOW}]{{5,}}\b")
_DESIGNATION_BREAK_RE = re.compile(r"[,;:()«»“”\"]|\s[–—]\s")
# After the date, the decision goes on through a case number, the parties or
# a one- or two-word case name ("CE, 19 nov. 2001, Titanic").
_CASE_NUMBER_RE = re.compile(r"(?:(?:RG|pourvoi|req\.|aff\.)\s)?n[°º]\s?\d[\w./-]*")
_PARTIES_RE = re.compile(rf"\s(?:c/|c\.|v\.|vs\.?|contre)\s[{_UP}]")
# A comma-separated segment naming the parties holds at most this many words.
_MAX_PARTIES_WORDS = 16
_CASE_NAME_RE = re.compile(rf"[{_UP}][{_LOW}]+(?:\s[{_UP}][{_LOW}]+)?")
# A second reporter of the decision a note cites, after a semicolon: "RLDI
# 2009, n° 50, p. 8, obs. Fontaine", "JCP E 2000, p. 77", "RTD com. 1999.
# 394", "JurisData no 1996-000044". It is no work of its own.
_REPORTER_WORD = rf"(?:[{_UP}]{{1,6}}|[{_UP}][{_LOW}]{{0,8}}\.|[{_LOW}]{{1,6}}\.|JurisData)"
_PARALLEL_REPORT_RE = re.compile(
    rf"\s*{_REPORTER_WORD}(?:\s{_REPORTER_WORD}){{0,3}}(?:\sn[°oº])?\s{_YEAR}(?![\d/])"
)

# A surname the text layer spaces out, as it does small capitals: "Christine
# Ze l le r ,", "J.F. Bo u l a is,", "L. M c Le o d ,". After a first name or
# initials: a capital and up to two lowercase letters, then lowercase pieces
# (a capital only after the "c" of Mc), up to a comma, a period or a second
# author. :func:`_join_spaced_surnames` joins it when two pieces or more are
# single letters, which no run of words has ("Y. Wu et al.").
_SPACED_PIECE = rf"(?:c\s[{_UP}][{_LOW}]{{0,2}}|[{_LOW}][{_LOW}{_UP}]*)"
_SPACED_SURNAME_RE = re.compile(
    rf"(?<![\w.])(?P<given>{_INITIALS}|{_NAME}\s)"
    rf"(?P<run>[{_UP}][{_LOW}]{{0,2}}(?:\s{_SPACED_PIECE}){{2,}})"
    rf"(?:\s(?=[,.;:])|(?=[,.;:]|\s(?:et|and|y|e|&)\s))"
)


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


def _citation_start(clause: str, verdicts: dict[str, bool] | None = None) -> int:
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

    *verdicts* keeps, across the calls of one split, whether the text after a
    hand-over leads to a citation.
    """
    if clause[:1] in '“«„‘"' or _CLASSICAL_RE.match(clause):
        return 0
    if _ONSET_RE.match(clause) and not _NAME_IN_PROSE_RE.match(clause):
        return 0
    if verdicts is None:
        verdicts = {}
    for match in _HANDOVER_RE.finditer(clause):
        # A lead-in word marks what precedes it as commentary; other
        # hand-overs need a prose word before them.
        if match.group("lead") is None and not _PROSE_WORD_RE.search(clause, 0, match.start()):
            continue
        rest = clause[match.end() : match.end() + _HANDOVER_READ].strip()
        if rest not in verdicts:
            verdicts[rest] = not _NAME_IN_PROSE_RE.match(rest) and looks_like_citation(rest)
        if verdicts[rest]:
            return match.end()
    return 0


def _cut_commentary(clause: str, verdicts: dict[str, bool] | None = None) -> str:
    return clause[_citation_start(clause, verdicts) :].strip()


def _clean(piece: str, verdicts: dict[str, bool] | None = None) -> str:
    return _cut_commentary(_strip_lead_in(piece), verdicts).strip(" ,;:")


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
    # The breaks read the same hand-overs again and again: what follows each
    # is judged once.
    verdicts: dict[str, bool] = {}
    for begin, end, sentence in breaks:
        if begin < start:
            continue
        left, right = clause[start:begin], clause[end:]
        if sentence and _ABBREVIATION_END_RE.search(left):
            continue
        led = sentence and _LEAD_IN_RE.match(right) is not None
        left_clean = _clean(left, verdicts)
        left_kind = _citation_kind(left_clean)
        # A repeat ("ABRAMS, ref. 6.") closes a citation as a full one does.
        left_cites = left_kind is not None or is_repeat_citation(left_clean)
        cleaned = _clean(right, verdicts)
        # Commentary that hands over to a citation within its first sentence.
        stripped = _strip_lead_in(right)
        cited_from = _citation_start(stripped, verdicts)
        handed_over = 0 < cited_from <= len(_next_piece(stripped))
        # After a sentence break the first sentence must read as a citation;
        # "or SURNAME, Name" opens a work whose byline is a sentence of its own.
        opened = right if not sentence else _next_piece(right)
        if (
            (led and _starts_work(stripped))
            or (led and left_cites and looks_like_citation(cleaned))
            or (left_cites and _starts_work(right) and _citation_kind(_clean(opened, verdicts)))
            or (left_cites and handed_over and looks_like_citation(cleaned))
        ):
            pieces.append(left)
            start = end
        elif sentence and left_kind == "full" and not looks_like_citation(cleaned):
            pieces.append(left)
            return pieces
    pieces.append(clause[start:])
    return pieces


def _join_spaced_surnames(text: str) -> str:
    """*text* with its spaced-out surnames joined: "Christine Ze l le r , Des
    enfants" → "Christine Zeller, Des enfants". A surname left spaced out
    reads as no name, so the tagger took byline and title for one title."""

    def join(match: re.Match[str]) -> str:
        pieces = match.group("run").split()
        joined = "".join(pieces)
        if sum(len(piece) == 1 for piece in pieces[1:]) < 2 or len(joined) < 4:
            return match.group()
        return f"{match.group('given')}{joined}"

    return _SPACED_SURNAME_RE.sub(join, text)


def _split_citations(note: str) -> list[str]:
    """The citation clauses of one note, lead-ins and commentary removed."""
    text = collapse_ws(_NOTE_MARK_RE.sub("", note, count=1))
    text = _join_spaced_surnames(text)
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
    after_decision = False
    for clause in clauses:
        # A court decision is a citation of its own, whatever surrounds it;
        # the reporters printing it follow it and are no works of their own.
        start = 0
        for decision in _decisions(clause):
            before = clause[start : decision.start]
            if not (after_decision and _PARALLEL_REPORT_RE.match(before)):
                out.extend(_clause_citations(before))
            out.append(_strip_lead_in(clause[decision.start : decision.end]).strip(" ,;"))
            start = decision.end
            after_decision = True
        rest = clause[start:]
        if start == 0 and after_decision and _PARALLEL_REPORT_RE.match(rest):
            continue
        after_decision = after_decision and start > 0
        out.extend(_clause_citations(rest))
    return out


def _clause_citations(clause: str) -> list[str]:
    """The citations of one clause, split where a new one starts and cleaned."""
    return [piece for piece in map(_clean, _soft_split(clause)) if piece]


@dataclass(frozen=True)
class _Decision:
    """Where a court decision stands in a clause."""

    start: int  # the court
    title_end: int  # past the date, case number and parties
    end: int  # past the reporters printing it
    year: int


def _is_designation(text: str, *, whole: bool = False) -> bool:
    """Whether *text* can stand between the court and the date of a decision.

    A *whole* comma-separated segment naming the parties may run longer
    ("Bashar Ibrahim and Others v. Bundesrepublik Deutschland and ... v. Taus
    Magamadov") than a tail cut out of running text.
    """
    parties = whole and _PARTIES_RE.search(f" {text}") is not None
    return (
        0 < len(text.split()) <= (_MAX_PARTIES_WORDS if parties else 8)
        and _DESIGNATION_RE.fullmatch(text) is not None
        and all(word in _DESIGNATION_WORDS for word in _LONG_LOWER_RE.findall(text))
    )


def _designation_start(clause: str, date_start: int) -> int | None:
    """Where the court of the decision dated at *date_start* is named.

    Walks back from the date over comma-separated designations ("CA Paris,
    4e ch.,"), and in the first text that is none keeps its longest
    capitalised tail ("Voir par exemple CA Paris" → "CA Paris").
    """
    end = len(clause[:date_start].rstrip(" ,"))
    start = None
    while end > 0:
        head = clause[:end]
        breaks = list(_DESIGNATION_BREAK_RE.finditer(head))
        cut = breaks[-1] if breaks else None
        segment = head[cut.end() if cut else 0 :]
        offset = end - len(segment.lstrip())
        segment = segment.strip()
        if not segment:
            break
        if _is_designation(segment, whole=True):
            start = offset
            if cut is None or cut.group() != ",":
                break
            end = cut.start()
            continue
        for word in re.finditer(rf"(?<!\S)[{_UP}]", segment):
            if _is_designation(segment[word.start() :]):
                start = offset + word.start()
                break
        break
    return start


def _decision_title_end(clause: str, date_end: int) -> int:
    """Past the case number, the parties or the case name after the date."""
    end = date_end
    while (segment := re.match(r",\s*([^,;]+)", clause[end:])) is not None:
        text = segment.group(1).strip()
        bare = text.rstrip(".")
        if not (
            _CASE_NUMBER_RE.fullmatch(bare)
            or _PARTIES_RE.search(f" {text}")
            or _CASE_NAME_RE.fullmatch(bare)
        ):
            break
        end += segment.end()
    return end


def _decisions(clause: str) -> list[_Decision]:
    """The court decisions *clause* cites, in order.

    A decision is a date ("16 juill. 1998") after a designation that names a
    court, a chamber or two parties, or one a dated reporter follows ("Paris,
    27 avr. 1934, DH 1934. 385."). What follows it after a comma, up to the
    end of the sentence, is its reporters ("D. 1999. 306, note Dreyer").
    """
    found: list[_Decision] = []
    for date in _DECISION_DATE_RE.finditer(clause):
        start = _designation_start(clause, date.start())
        if start is None or (found and start < found[-1].end):
            continue
        designation = clause[start : date.start()]
        if not (
            _COURT_RE.search(f" {designation}") or _DATED_REPORTER_RE.match(clause, date.end())
        ):
            continue
        title_end = _decision_title_end(clause, date.end())
        end = title_end
        rest = clause[title_end:]
        if rest.lstrip().startswith(","):
            breaks = [
                found
                for found in _sentence_breaks(rest)
                if not _ABBREVIATION_END_RE.search(rest[: found.start()])
            ]
            end += breaks[0].start() + 1 if breaks else len(rest)
        found.append(_Decision(start, title_end, end, int(date.group(1))))
    return found


def legal_decision(citation: str) -> tuple[str, int] | None:
    """Title and year of a citation that opens on a court decision, else None.

    The title is the decision as cited, court to case name ("Cass. 1re civ.,
    16 mai 2018, n° 15-14.023"), and the year that of its date. The
    reference tagger reads neither: a decision has no author, and its date
    no year the tagger knows.
    """
    text = collapse_ws(citation).strip()
    found = _decisions(text)
    if not found or found[0].start != 0:
        return None
    title = text[: found[0].title_end].rstrip(" ,;")
    if title.endswith(".") and title[-2:-1].isdigit():
        title = title[:-1]
    return title, found[0].year


def is_repeat_citation(clause: str) -> bool:
    """True for "Ibid.", "Id., p. 75", "L. McLeod, op. cit., supra, note 8", "BAY, ref. 27"."""
    return bool(_REPEAT_START_RE.match(clause) or _REPEAT_ANY_RE.search(clause))


def _citation_kind(clause: str) -> str | None:
    """ "full" for a dated citation, "short" for an undated one, None for commentary.

    A full citation needs a publication year (a lifespan such as
    "(1880-1943)" or a date stamp does not count) and one bibliographic cue:
    a name opening the clause, a quoted title, a "Place: Publisher" imprint, a
    container, edition or locator marker ("In", "(ed.)", "vol. 3", "pp. 12-19",
    "ISBN", "81, 286-300 (1981)"), a publisher word, or the year closing the
    clause. An undated citation needs three of those cues, the
    book-and-chapter shape of a classical work ("Estrabón, Geografía, XVI,
    17"), or an ISO 690 byline ("COLLALTO, Marie Therese.") and a container
    or locator. A court decision ("Civ. 1re, 16 juill. 1998, D. 1999. 306")
    is a full citation.
    """
    if len(_WORD_RE.findall(clause)) < 4 or _TABLE_NOTE_RE.match(clause):
        return None
    if legal_decision(clause) is not None:
        return "full"
    probe = _DATE_STAMP_RE.sub(" ", _LIFESPAN_RE.sub(" ", clause))
    cues = sum(
        bool(found)
        for found in (
            _STRICT_ONSET_RE.match(clause),
            _QUOTED_RE.search(clause),
            _PLACE_PUBLISHER_RE.search(clause),
            _CONTAINER_RE.search(clause),
            _VOLUME_PAGES_YEAR_RE.search(clause),
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


def classical_work(citation: str) -> tuple[str, str] | None:
    """Author and work of an undated citation of a classical work by book and
    chapter ("Diodoro de Sicilia, Biblioteca histórica, III, 32, 4."), else None."""
    text = collapse_ws(citation).strip()
    match = _CLASSICAL_WORK_RE.match(text)
    if match is None or _YEAR_RE.search(_DATE_STAMP_RE.sub(" ", _LIFESPAN_RE.sub(" ", text))):
        return None
    return match.group("author"), match.group("work").strip()


def year_led_title(citation: str) -> tuple[str, int] | None:
    """Title and year of a citation whose title opens on a year and whose
    bracketed imprint dates the work ("…, 1977 Consumer Credit Survey (Board
    of Governors of the Federal Reserve System, 1978), p. 72."), else None."""
    match = _YEAR_LED_TITLE_RE.search(collapse_ws(citation))
    if match is None:
        return None
    return match.group("title").strip(), int(match.group("year"))


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
