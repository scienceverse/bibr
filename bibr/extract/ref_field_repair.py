"""Deterministic repairs of NER reference fields (post-parse, before finalize).

The NER reference parser (ModernBERT emissions + CRF) segments a reference
correctly far more often than it places the field boundaries correctly. Its
recurring boundary errors are systematic enough to repair from the reference
text itself, without retraining:

* a web reference whose name (the page or product name before the URL) was
  tagged as the author, cut short, or split into an author and a title,
  leaving no usable title;
* a title that runs on past its own end: through the closing quote of a
  quoted title, into the URL, into a "Container, DD.MM.YYYY" dateline or a
  "Place, YYYY" imprint, into a bracketed English translation, a language note
  ("(Hindi)"), an ISBD statement of responsibility ("/ A. A. Yuldashev"), or a
  ", 2024. <notes>" year-and-notes tail;
* an author span that swallowed the title after a dash ("LIPSZYC, Delia —
  Domínio Público"), or a "SURNAME, Given — Title" byline the tagger left
  untagged or cut at the surname ("CHAVES, Antônio — Direito Autoral de
  Radiodifusão, S. Paulo, …");
* a year taken from an access date ("Acesso em: 14 abr. 2022") although the
  reference prints its own publication year, or no year at all although the
  reference prints exactly one;
* an editorial note printed as a list entry of its own ("(This is a series of
  short articles by …)"), which is no reference at all;
* a Vancouver byline that lost its last author's single initial ("Newnham M."
  tagged "Newnham": the tagger reads "M." as the field delimiter), kept the
  colon of a colon-style byline ("…, et al.:"), or a byline tagged as the title
  ("Deppe, U. et al.");
* a title that kept the closing quote of a quoted-title style ("…in Sumatra,”"),
  or a short quoted title right after the byline left untagged ("Ardupilot,
  “MissionPlanner,” https://…").

Each rule fires only on a clear textual signal and leaves fields that look well
formed alone: a cut keeps the part of the title before the signal, sibling
fields are filled only when empty (a year read off an access date is the one
value replaced), and a cut is refused when it would leave less than two words
of title. Rules run on the parser's flat output (PaperReference field names)
together with the parser-input text the spans were cut from.
:func:`repair_ner_reference_fields` is the hook the NER parse path calls.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

# A URL, tolerating the "https: //host" split that PDF text extraction leaves
# behind and a URL glued to the preceding word ("Retrieved fromhttp://...").
_URL_RE = re.compile(r"(?:<|〈)?(?:https?\s?:\s?//|www\.)\S+", re.IGNORECASE)

# Phrases that introduce a URL or an access date in a web reference.
_WEB_PHRASE_RE = re.compile(
    r"(?:\b(?:also\s+)?(?:it\s+is\s+)?available\s+(?:at|from|on|online)\b\s*:?"
    r"|\bretrieved\s+(?:from|on)\b"
    r"|\bretrieved\b"
    r"|\b(?:last\s+)?accessed(?:\s+on)?\b\s*:?"
    r"|\baccess\s+date\b\s*:?"
    r"|\[online\]"
    r"|\bonline\s*:"
    r"|\burl\s*:"
    r"|\bdisponível\s+em\b\s*:?"
    r"|\bacesso\s+em\b\s*:?)",
    re.IGNORECASE,
)

_EN_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

# Access-date phrases: the date after them is when the page was read, never
# the publication year. The match runs from the phrase to the first year
# within 40 characters ("Acesso em: 15 de jun. 2020", "accessed 17 Dec 2017").
_ACCESS_PHRASE_RE = re.compile(
    r"(?:\b(?:last\s+)?accessed(?:\s+on)?|\baccess\s+date|\bretrieved(?!\s+from)(?:\s+on)?"
    r"|\bacess?o\s+em|\bconsultado\s+(?:el|em)|\bconsulté\s+le|\babgerufen\s+am"
    r"|\bgeraadpleegd(?:\s*op)?|\bдата\s+обращения|\bviewed(?:\s+on)?|\bvisited(?:\s+on)?)"
    r"[\s\S]{0,40}?(?<!\d)(?:19|20)\d{2}(?!\d)",
    re.IGNORECASE,
)

# A reference marked in press prints the status as a field of its own: in
# brackets ("(in press)", "[Epub ahead of print]"), between field punctuation
# (". In press.", ", forthcoming,"), opening "Forthcoming in <venue>" or a dated
# "In press 2002", or at the very end ("J Synth Garden Res in press."). A bare
# substring search also read "caveats in press releases" and "The Darwin Press"
# as in press, which cost those references their year. ref_extractor imports
# this pattern; it cannot be imported the other way.
_IN_PRESS_STATUS = (
    r"(?:article\s+)?in\s*press(?:-?[a-z])?|forthcoming|advance\s*online(?:\s+publication)?"
    r"|manuscript\s*submitted(?:\s+for\s+publication)?|epub\s*ahead(?:\s+of\s+print)?"
)
_IN_PRESS_RE = re.compile(
    rf"(?:^|[(\[.,;:/])\s*(?:{_IN_PRESS_STATUS})"
    r"(?=\s*(?:[.,;:()\[\]]|$)|\s+in\b|\s+(?:19|20)\d\d\b)"
    rf"|\b(?:{_IN_PRESS_STATUS})[\s.)\]]*$",
    re.IGNORECASE,
)
# Mirrors bibr.extract.ref_extractor._VANCOUVER_YEAR_RE (not imported:
# ref_extractor imports this module).
_VANCOUVER_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[;:]\s*[eE]?\d")

_YEAR_TOKEN_RE = re.compile(r"(?<![\w.,/-])(1[5-9]\d\d|20\d\d)(?!\d)")
# A full numeric date ("21-01-1983", "18.11.2011", "3/5/2011") prints its year.
_NUMERIC_DATE_RE = re.compile(r"(?<![\w./-])\d{1,2}([-./])\d{1,2}\1(1[5-9]\d\d|20\d\d)(?!\d)")
# Article-history and standard-enactment labels whose dates leak into a
# reference ("Date of acceptance: 29-3-2016", "Recebido: 12/7/2025",
# "Vved. 01.04.2020"): the date is not the cited work's year.
_HISTORY_LABEL_RE = re.compile(
    r"(?:\bdate\s+of\s+\w+|\breceived|\baccepted|\brevised|\bsubmitted|\brecebido|\baceito"
    r"|\bpublicado|\bvved(?:en)?|\bвведен?)\W{0,3}$",
    re.IGNORECASE,
)
_MONTH_END_RE = re.compile(
    rf"\b(?:{_EN_MONTHS}|spring|summer|autumn|fall|winter)\.?$", re.IGNORECASE
)


def _dated_years(text: str, shielded: list[tuple[int, int]]) -> set[int]:
    """Year values printed in a date position outside the ``shielded`` spans.

    A date position is the start of the text, right after an opening bracket,
    a comma / period / semicolon / colon, or a month name ("(2013)", ", 2001.",
    "jan-jun. 2007", "December 2013"), and the year is followed by the end of
    the text, a space or closing punctuation. "Coronavirus Disease 2019
    (COVID-19)" and "Strategy 2014-2023" are names, not dates. Only when no
    year stands in such a position do full numeric dates ("21-01-1983") count.
    """
    years = set()
    for match in _YEAR_TOKEN_RE.finditer(text):
        pos = match.start()
        if any(a <= pos < b for a, b in shielded):
            continue
        after = text[match.end() :]
        after = after[1:] if after[:1].isalpha() and after[:1].islower() else after
        if after and after[0] not in " \t\n.,;:)]":
            continue
        if after[:1] == "." and after[1:2].isdigit():
            continue  # an identifier ("arXiv:1808.02594"), not a date
        before = text[:pos].rstrip()
        if before and before[-1] not in "([,.;:" and not _MONTH_END_RE.search(before):
            continue
        years.add(int(match.group(1)))
    if years:
        return years
    # No year in a date position: fall back to full numeric dates, which also
    # print article-history and enactment dates, so they only count alone.
    for match in _NUMERIC_DATE_RE.finditer(text):
        if any(a <= match.start() < b for a, b in shielded):
            continue
        after = text[match.end() :]
        if after and after[0] not in " \t\n.,;:)]":
            continue
        if _HISTORY_LABEL_RE.search(text[: match.start()].rstrip()):
            continue
        years.add(int(match.group(2)))
    return years


def _year_shields(fields: dict[str, Any], text: str) -> list[tuple[int, int]]:
    """Spans of ``text`` whose years are not the reference's own.

    Access dates, URLs and the tagged title: a year inside the title is the
    title's ("… in China, 2016", "Cassiopea xamachana (Bigelow, 1892)
    jellyfish").
    """
    shielded = [m.span() for m in _ACCESS_PHRASE_RE.finditer(text)]
    shielded += [m.span() for m in _URL_RE.finditer(text)]
    title = fields.get("title")
    if isinstance(title, str) and title.strip():
        at = text.find(title.strip())
        if at >= 0:
            shielded.append((at, at + len(title.strip())))
    return shielded


# Typographic double quotes that open a quoted title, with the characters that
# close each. Single quotes are left alone (they double as apostrophes and as
# the inner quotes of a title that starts with a quotation), and so are
# guillemets: Slavic and French styles use them for a name quoted inside the
# title («Biuletyn Informacyjny» żołnierzy …), not to delimit the title.
_DOUBLE_OPEN = "“„"
_DOUBLE_CLOSE = {"“": "”", "„": "“”"}

_LETTER_RE = re.compile(r"[^\W\d_]")

_LANGUAGES = (
    "afrikaans|albanian|arabic|armenian|azerbaijani|basque|belarusian|bengali|bosnian|bulgarian"
    "|catalan|chinese|croatian|czech|danish|dutch|english|estonian|farsi|finnish|french"
    "|georgian|german|greek|gujarati|hebrew|hindi|hungarian|icelandic|indonesian|italian"
    "|japanese|kannada|kazakh|korean|latin|latvian|lithuanian|macedonian|malay|malayalam"
    "|marathi|mongolian|nepali|norwegian|persian|polish|portuguese|punjabi|romanian|russian"
    "|sanskrit|serbian|slovak|slovenian|spanish|swahili|swedish|tamil|telugu|thai|turkish"
    "|ukrainian|urdu|uzbek|vietnamese"
)


def _has_word(text: str, min_letters: int = 2) -> bool:
    """True when ``text`` holds a run of at least ``min_letters`` letters."""
    return re.search(rf"[^\W\d_]{{{min_letters},}}", text or "") is not None


def _clean_cut(text: str) -> str:
    """Strip the separator punctuation a cut leaves on the right of a title."""
    return text.rstrip(" \t\n,;:–—-/").strip()


# ---------------------------------------------------------------------------
# Rules. Each takes (fields, text), mutates ``fields`` in place and returns
# True when it fired.
# ---------------------------------------------------------------------------


def _web_only(tail: str) -> bool:
    """True when ``tail`` carries nothing but URLs, web phrases and dates."""
    rest = _URL_RE.sub(" ", tail)
    # URL residue a line break split off ("…/courses/f all2007/sb5002/…").
    rest = re.sub(r"\S*(?:[/=&_#~%]|\.(?:html?|php|aspx?|pdf|jsp)\b)\S*", " ", rest)
    rest = _WEB_PHRASE_RE.sub(" ", rest)
    rest = re.sub(rf"\b(?:{_EN_MONTHS})\b\.?", " ", rest, flags=re.IGNORECASE)
    rest = re.sub(r"\bn\.\s?d\.?", " ", rest, flags=re.IGNORECASE)
    rest = re.sub(r"\d+(?:st|nd|rd|th)?\b", " ", rest, flags=re.IGNORECASE)
    return not _LETTER_RE.search(rest)


# Where the name of a web resource ends: at its URL, at a web phrase, or at a
# publication date printed right after it ("(n.d.)", "(2015, December 14)",
# ", 2022", ". 2022").
_LEAD_STOP_RE = re.compile(
    rf"{_URL_RE.pattern}|{_WEB_PHRASE_RE.pattern}"
    r"|\(\s*n\.\s?d\.?\s*\)|\(\s*(?:19|20)\d{2}|[.,]\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


_OTHER_SPAN_FIELDS = ("container", "publisher", "editors", "series", "note", "edition")


def _rule_web_lead_title(fields: dict[str, Any], text: str) -> bool:
    """A web reference's lone name is its title.

    "[4] AWS Wavelength. https://aws.amazon.com/wavelength/. Accessed
    2025-07-15." — the tagger reads the page name as an author and emits no
    title, cuts the name short ("Timeline - Overview for "Electrical imp..."
    without "in Publications - Dimensions"), or splits it into an author and a
    title ("Saldırganlığı Önlemeye Yönelik" + "The Effect of Psycho-Education
    ..."). The name is the text from the first tagged span up to the URL, web
    phrase or date that ends it. Fires only when the text starts with that
    span, a split author and title are separated by nothing but spaces,
    everything after the name is URLs, web phrases and dates (so nothing else
    could be the title), the name is one phrase (no sentence break, at most 20
    words), what it gains past the tagged spans is plain words (no comma,
    semicolon or quote), and no other tagged field (container, publisher, ...)
    lies inside it. The name becomes the title; the author span is left as
    tagged.
    """
    authors = fields.get("authors")
    authors = authors.strip() if isinstance(authors, str) else ""
    title = fields.get("title")
    title = title.strip() if isinstance(title, str) else ""
    first = authors or title
    if not first:
        return False
    start = text.find(first)
    if start < 0 or _LETTER_RE.search(text[:start]):
        return False
    pos = start + len(first)
    if authors and title:
        title_at = text.find(title, pos)
        if title_at < 0 or text[pos:title_at].strip():
            return False
        pos = title_at + len(title)
    stop = _LEAD_STOP_RE.search(text, pos)
    if stop is None:
        return False
    # What the name gains past the tagged spans must be plain words: a comma,
    # semicolon or quote there starts the next element ("“Title,” Publisher").
    if re.search(r"[,;“”«»„\"]", _clean_cut(text[pos : stop.start()])):
        return False
    lead = _clean_cut(text[start : stop.start()]).rstrip(".").strip()
    tail = text[stop.start() :]
    if not lead.startswith(first.rstrip(".")) or not _has_word(lead):
        return False
    if lead.rstrip(".") == title.rstrip("."):
        return False
    if not (_URL_RE.search(tail) or _WEB_PHRASE_RE.search(tail)):
        return False
    if not _web_only(tail):
        return False
    # One name, not "Author. Title": a sentence break inside the lead means
    # the title is missing for some other reason.
    if re.search(r"[.!?]\s+[A-ZÀ-ÖØ-ÞА-ЯЁ]", lead) or len(lead.split()) > 20:
        return False
    for name in _OTHER_SPAN_FIELDS:
        value = fields.get(name)
        if isinstance(value, str) and _has_word(value, 3) and value.strip(" ,.;:") in lead:
            return False
    fields["title"] = lead
    return True


# "LIPSZYC, Delia — Domínio Público": surname, given name(s), a spaced dash,
# then the title the tagger kept inside the author span.
_AUTHOR_DASH_TITLE_RE = re.compile(
    r"^(?P<author>[^\W\d_][\w'’.-]*(?:\s[\w'’.-]+)?,\s*[^\W\d_][\w'’.-]*(?:\s[\w'’.-]+){0,3})"
    r"\s+[—–]\s+(?P<title>\S.*)$"
)
# "SCHMIDT, Ondřej", "Clark, K. B", "FORNASIN, Alessio – MANFREDINI, Matteo":
# a co-author list joined by spaced dashes (Czech, Slovak, Hungarian and
# Romanian styles), not a title.
_DASH_NAME = r"[^\W\d_][\w'’‟-]*(?:\s[^\W\d_][\w'’‟-]*)?,\s*[^\W\d_][\w'’.-]*(?:\s[\w'’.-]+){0,2}"
_DASH_NAME_LIST_RE = re.compile(rf"{_DASH_NAME}(?:\s+[—–]\s+{_DASH_NAME})*\.?")


def _rule_author_dash_title(fields: dict[str, Any], _text: str) -> bool:
    """Untitled reference whose author span ran on through a dash into the title."""
    if fields.get("title"):
        return False
    authors = fields.get("authors")
    if not isinstance(authors, str):
        return False
    match = _AUTHOR_DASH_TITLE_RE.match(authors.strip())
    if match is None or _DASH_NAME_LIST_RE.fullmatch(match.group("title").strip()):
        return False
    title = _clean_cut(match.group("title")).rstrip(".")
    if not _has_word(title, 3):
        return False
    fields["authors"] = match.group("author").strip()
    fields["title"] = title
    return True


# A web phrase that still waits for its URL ("disponível em", "Retrieved
# from", "Available at:"): no title ends in one.
_DANGLING_LOCATOR_RE = re.compile(r"(?:\b(?:at|from|on|em)|:)\s*$", re.IGNORECASE)


def _url_cut(title: str) -> int | None:
    """Index where a URL or web phrase starts the tail of ``title``, if it does."""
    cut = None
    for rx in (_URL_RE, _WEB_PHRASE_RE):
        match = rx.search(title)
        if match is not None and (cut is None or match.start() < cut):
            cut = match.start()
    if cut is None:
        return None
    if title[:cut].rstrip().endswith("("):
        # "Program (www.seer.cancer.gov) SEER*Stat Database": a URL in
        # parentheses is part of the name when the name goes on after it.
        close = title.find(")", cut)
        if close >= 0 and _LETTER_RE.search(title[close + 1 :]):
            return None
    tail = title[cut:]
    if not _web_only(tail):
        return None
    if not (_URL_RE.search(tail) or re.search(r"\d", tail) or _DANGLING_LOCATOR_RE.search(tail)):
        # "Episodic memories are reconstructed, not retrieved": a web phrase
        # with neither a URL nor a date after it is the title's own last word,
        # unless it is left hanging ("Versão traduzida disponível em", the URL
        # tagged apart).
        return None
    return cut


def _rule_title_url_tail(fields: dict[str, Any], _text: str) -> bool:
    """A title that ends in a URL / access phrase ends before it.

    "Traditional, Complementary and Integrative Medicine https://www.who.int/…"
    and "Brexit: UK Leaves the European Union. Accessed March 20, 2020": the
    tagger ran the title into the web locator. Fires only when everything from
    the URL or web phrase on is web locator and date, and that tail holds a URL
    or a date (so "Terrorist Content Online: Safeguards …", "PAN retrieved from
    MIPAS …" and "… reconstructed, not retrieved" keep their titles).
    """
    title = fields.get("title")
    if not isinstance(title, str):
        return False
    cut = _url_cut(title)
    if cut is None:
        return False
    head = _clean_cut(title[:cut]).rstrip(".").rstrip(" (<〈[").strip()
    if len(head.split()) < 2 or not _has_word(head, 3):
        return False
    fields["title"] = head
    return True


# A thesis or dissertation note ("524 f. Tese (Doutoramento em …)", "PhD
# thesis", "Dissertação (Mestrado …)", "Tesis doctoral", "Thèse"). The French
# word needs its accent: unaccented it is the English "these" ("in these
# proceedings").
_THESIS_NOTE_RE = re.compile(
    r"\b(?:tese|disserta[çc][ãa]o|tesis|thèse|thesis|dissertation)\b", re.IGNORECASE
)


def _thesis_subtitle(runon: str, title: str, text: str) -> bool:
    """True when ``runon`` (the title past its closing quote) is a thesis subtitle.

    A thesis has no container for the title to run into, so what follows the
    quotation is the title's own subtitle ("“Somos as pessoas que temos de
    escolher …”. Infância e cenários de participação pública: …. 2014. 524 f.
    Tese …"). Holds when the reference prints a thesis note after the tagged
    title and the run-on part is a phrase of at least three words without one.
    """
    if len(runon.split()) < 3 or _THESIS_NOTE_RE.search(runon):
        return False
    at = text.find(title)
    return at >= 0 and _THESIS_NOTE_RE.search(text, at + len(title)) is not None


def _rule_title_quote_runon(fields: dict[str, Any], text: str) -> bool:
    """A quoted title ends at its closing quote: cut what the tagger ran on into.

    Quoted-title styles print “Title,” Container … / “Title”. Container …; the
    tagger often keeps the container (and more) in the title. Fires only when
    no container was tagged, a typographic double quote opens the title (in
    the title or right before it in the reference), its first closing quote
    sits next to a comma or period (",”", ".”", "”.", "”,"), and more text
    follows. A quotation that merely starts the title ("“I spy with my little
    eye!”: Breadth of attention …") is not cut, nor is the subtitle that
    follows the quotation in a thesis title (:func:`_thesis_subtitle`), and
    titles the tagger already ended at the quote are left as they are. The
    kept title is the text between the quotes.
    """
    title = fields.get("title")
    if not isinstance(title, str) or not title.strip():
        return False
    if fields.get("container"):
        # A container was tagged after the title, so the text past the quote
        # is title ("“Mini-Mental State”. A practical method for grading …").
        return False
    stripped = title.strip()
    if stripped[0] in _DOUBLE_OPEN:
        opener, body = stripped[0], stripped[1:]
    else:
        at = text.find(stripped[:40])
        before = text[:at].rstrip() if at > 0 else ""
        if not before or before[-1] not in _DOUBLE_OPEN:
            return False
        opener, body = before[-1], stripped
    positions = [body.find(c) for c in _DOUBLE_CLOSE[opener] if c in body]
    if not positions:
        return False
    close = min(positions)
    head = body[:close]
    if opener in head:
        return False  # nested or re-opened quote: boundary unclear
    after = body[close + 1 :]
    if not (head.rstrip("’'").endswith((",", ".")) or after.startswith((",", "."))):
        return False
    if not _has_word(after, 2):
        return False
    if _thesis_subtitle(after, stripped, text):
        return False
    head = _clean_cut(head).strip()
    if head.endswith(".") and not head.endswith(".."):
        head = head[:-1].rstrip()
    if len(head.split()) < 2 or not _has_word(head, 3):
        return False
    fields["title"] = head
    return True


# DD.MM.YYYY, or "Month[/Month] [DD,] YYYY", closing a news item's dateline.
_DATELINE_DATE = (
    rf"(?:\d{{1,2}}\.\d{{1,2}}\.(?:19|20)\d{{2}}"
    rf"|(?:{_EN_MONTHS})\.?(?:\s*/\s*(?:{_EN_MONTHS})\.?)?\s+(?:\d{{1,2}},\s+)?(?:19|20)\d{{2}})"
)
# The outlet opens a new sentence after the title: a comma there is the
# title's own place-and-date ending ("… - Skagit County, Washington, March
# 2020", "Progress report, Phase II, March 2019").
_DATELINE_TAIL_RE = re.compile(
    rf"\.\s+(?P<container>[^.,\[\]]{{2,60}}?)[.,]\s+(?P<date>{_DATELINE_DATE})\.?\s*$",
    re.IGNORECASE,
)
_BARE_DATE_TAIL_RE = re.compile(
    r",\s+(?P<date>\d{1,2}\.\d{1,2}\.(?:19|20)\d{2})\.?\s*$",
)


def _fill_year(fields: dict[str, Any], date: str) -> None:
    if fields.get("year"):
        return
    match = re.search(r"(?:19|20)\d{2}", date)
    if match is not None:
        fields["year"] = int(match.group(0))


def _rule_title_dateline_tail(fields: dict[str, Any], _text: str) -> bool:
    """Cut a trailing ". Container, 18.11.2011" / ". Foreign Affairs. January 2013".

    News and magazine items print Title. Outlet, DD.MM.YYYY; the tagger keeps
    the outlet and the date in the title. The container and year are filled
    only when the tagger left them empty.
    """
    title = fields.get("title")
    if not isinstance(title, str):
        return False
    match = _DATELINE_TAIL_RE.search(title)
    if match is not None:
        head = _clean_cut(title[: match.start()])
        container = match.group("container").strip()
        if len(head.split()) >= 2 and len(container.split()) <= 6 and _has_word(container):
            fields["title"] = head
            if not fields.get("container"):
                fields["container"] = container
            _fill_year(fields, match.group("date"))
            return True
    match = _BARE_DATE_TAIL_RE.search(title)
    if match is not None:
        head = _clean_cut(title[: match.start()])
        if len(head.split()) >= 2:
            fields["title"] = head
            _fill_year(fields, match.group("date"))
            return True
    return False


# GOST 7.1 general material designations: part of the record, not the title.
_MATERIAL_MARK_RE = re.compile(
    r"\s*\[(?:электронный\s+ресурс|текст|electronic\s+resource|text|elektronnyi\s+resurs)\]\s*\.?\s*$",
    re.IGNORECASE,
)
_ENGLISH_FUNCTION_WORDS = frozenset(
    {"the", "of", "and", "in", "to", "for", "on", "with", "is", "are", "has", "have", "will"}
    | {"its", "as", "by", "from", "at", "an", "a"}
)


def _rule_title_translation_bracket(fields: dict[str, Any], _text: str) -> bool:
    """Strip a trailing bracketed English translation or a GOST material mark.

    "V Iuzhnoi Koree poiavitsia ... baza SShA [South Korea will have the
    world’s largest U.S. military base]." — transliterated and Cyrillic
    references print the English translation in brackets after the original
    title. Fires only when the bracket closes the title, the bracket reads as
    English (English function words) and the original does not.
    """
    title = fields.get("title")
    if not isinstance(title, str):
        return False
    fired = False
    mark = _MATERIAL_MARK_RE.search(title)
    if mark is not None:
        head = _clean_cut(title[: mark.start()])
        if not _has_word(head, 3):
            return False
        title = head
        fields["title"] = title
        fired = True
    match = re.search(r"\s\[(?P<tr>[^\[\]]{8,})\]\s*\.?\s*$", title)
    if match is None:
        return fired
    head = _clean_cut(title[: match.start()])
    words_head = re.findall(r"[^\W\d_]+", head.lower())
    words_tr = re.findall(r"[^\W\d_]+", match.group("tr").lower())
    if len(words_head) < 2 or len(words_tr) < 2:
        return fired
    if not any(w in _ENGLISH_FUNCTION_WORDS for w in words_tr):
        return fired
    if any(w in _ENGLISH_FUNCTION_WORDS and len(w) > 1 for w in words_head):
        return fired
    fields["title"] = head
    return True


# The note ends the title when a period, comma or semicolon (or nothing)
# follows it; a colon after it opens the subtitle ("… fabrica (Latin): On the
# fabric …"), so the note is inside the title.
_LANGUAGE_NOTE_RE = re.compile(
    rf"\s*[(\[]\s*(?:in\s+)?(?:{_LANGUAGES})(?:\s+(?:language|text|version))?\s*[)\]]"
    r"(?=\s*(?:[.,;]|$))",
    re.IGNORECASE,
)


def _rule_title_language_note(fields: dict[str, Any], _text: str) -> bool:
    """A language note ("(Hindi)", "(in Russian)") ends the title.

    "Netra Yog Chikitsa (Hindi). Yog Prakshikshan" — the note closes the title
    and whatever follows it (edition, series, publisher) is not title text.
    """
    title = fields.get("title")
    if not isinstance(title, str):
        return False
    match = _LANGUAGE_NOTE_RE.search(title)
    if match is None:
        return False
    head = _clean_cut(title[: match.start()])
    if not _has_word(head, 3):
        return False
    fields["title"] = head
    return True


_INITIALS_NAME_RE = re.compile(
    r"^\s*(?:[^\W\d_]\.\s?){1,3}\s*[^\W\d_]{2,}|^\s*[^\W\d_]{2,}\s(?:[^\W\d_]\.\s?){1,3}"
)


def _rule_title_responsibility(fields: dict[str, Any], _text: str) -> bool:
    """Cut an ISBD / GOST statement of responsibility (" / A. A. Yuldashev").

    Library-catalogue styles print Title / Author names; the tagger keeps the
    names (and whatever follows) in the title. Fires on a spaced slash followed
    by initials-and-surname, or by one of the reference's own author surnames.
    """
    title = fields.get("title")
    if not isinstance(title, str) or " / " not in title:
        return False
    idx = title.find(" / ")
    head = _clean_cut(title[:idx])
    rest = title[idx + 3 :]
    if len(head.split()) < 2:
        return False
    signal = _INITIALS_NAME_RE.match(rest) is not None
    if not signal:
        authors = fields.get("authors")
        if isinstance(authors, str):
            surnames = {
                w.lower()
                for w in re.findall(r"[^\W\d_]{3,}", authors)
                if w.lower() not in {"and", "und", "et", "al"}
            }
            first = re.findall(r"[^\W\d_]{3,}", rest[:60])
            signal = any(w.lower() in surnames for w in first[:4])
    if not signal:
        return False
    fields["title"] = head
    return True


_YEAR_NOTES_TAIL_RE = re.compile(
    r"^(?P<head>.{8,}?)[,.]\s+(?P<year>(?:19|20)\d{2})[a-z]?\.\s+(?P<rest>\S.*)$", re.DOTALL
)


def _rule_title_year_tail(fields: dict[str, Any], text: str) -> bool:
    """ "Title, 2024. <notes>" when no year was tagged: the year ends the title.

    The tagger found no year, and the title carries ", YYYY." / ". YYYY."
    followed by more text (product notes, "488 f. Tese ..."): the year is the
    reference's own and everything from it on is not title text. When a
    capitalised phrase follows the year and the reference prints a different
    year as a date outside the title ("… Niigata Earthquake of June 16, 1964.
    Part 2. … Bull. Earthq. Res. Inst. 43, 237–239 (1965)."), the year is the
    title's own and the rule leaves it to ``year_from_text``.
    """
    if fields.get("year"):
        return False
    title = fields.get("title")
    if not isinstance(title, str):
        return False
    match = _YEAR_NOTES_TAIL_RE.match(title)
    if match is None:
        return False
    head = _clean_cut(match.group("head"))
    if len(head.split()) < 2 or not _has_word(match.group("rest"), 2):
        return False
    year = int(match.group("year"))
    if match.group("rest")[:1].isupper():
        # Volume, pages or thesis notes after the year ("17, 386", "264f.",
        # "no 57") keep it the reference's even when a later year prints: a
        # segment holding two works shows the second work's year there.
        outside = _dated_years(text, _year_shields(fields, text))
        if outside and year not in outside:
            return False
    fields["title"] = head
    fields["year"] = year
    return True


_IMPRINT_TAIL_RE = re.compile(
    r"^(?P<head>.{8,}?)\.\s+(?P<imprint>[A-ZÀ-ÖØ-ÞА-ЯЁ][^.\d]{0,60}?),\s+"
    r"(?P<year>(?:19|20)\d{2})[a-z]?\.?\s*$",
    re.DOTALL,
)


def _rule_title_imprint_tail(fields: dict[str, Any], _text: str) -> bool:
    """ "Title. Brasília, DF, 2011" when no year was tagged: cut the imprint.

    Place-and-year (or place, publisher and year) imprints close book and
    report references; the tagger kept one inside the title and found no year.
    Fires only when no year was tagged and the title ends in a sentence of at
    most six words that starts with a capital and ends in ", YYYY".
    """
    if fields.get("year"):
        return False
    title = fields.get("title")
    if not isinstance(title, str):
        return False
    match = _IMPRINT_TAIL_RE.match(title.strip())
    if match is None or len(match.group("imprint").split()) > 6:
        return False
    head = _clean_cut(match.group("head"))
    if len(head.split()) < 3:
        return False
    fields["title"] = head
    fields["year"] = int(match.group("year"))
    return True


def _year_positions(text: str, year: int) -> list[int]:
    return [m.start() for m in re.finditer(rf"(?<!\d){year}(?!\d)", text)]


def _rule_year_not_access_date(fields: dict[str, Any], text: str) -> bool:
    """A year read off an access date yields to the reference's own year.

    "... Belo Horizonte, 31 dez. 1969. Disponível em: <...>. Acesso em: 14
    abr. 2022." — the tagger took 2022 from the access date. Fires only when
    every occurrence of the tagged year sits inside an access-date phrase or a
    URL, and exactly one other year value is printed outside them.
    """
    year = fields.get("year")
    if not isinstance(year, int):
        return False
    shielded: list[tuple[int, int]] = [m.span() for m in _ACCESS_PHRASE_RE.finditer(text)]
    if not shielded:
        return False
    shielded += [m.span() for m in _URL_RE.finditer(text)]

    def inside(pos: int) -> bool:
        return any(a <= pos < b for a, b in shielded)

    hits = _year_positions(text, year)
    if not hits or not all(inside(p) for p in hits):
        return False
    others = _dated_years(text, shielded)
    others.discard(year)
    if len(others) != 1:
        return False
    fields["year"] = others.pop()
    return True


def _rule_year_from_text(fields: dict[str, Any], text: str) -> bool:
    """No year tagged: take the one year the reference prints as a date.

    "MUNFORD, D; LIMA, M. E. C. De C. Ensinar Ciências por Investigação: …
    Revista ensaio, V. 9, n. 1, 72-89, jan-jun. 2007. Disponível em: <…>.
    Acesso em: 15 de jun. 2020." — the tagger emitted no year. Fires only when
    no year was tagged and exactly one year value stands in a date position
    outside the title, URLs and access dates; the other fields are left as
    they are.
    """
    if fields.get("year"):
        return False
    if _IN_PRESS_RE.search(text) or _VANCOUVER_YEAR_RE.search(text):
        # In-press references carry no year; Vancouver "2024;171:54" tails are
        # filled more precisely by the finalize step's backfill.
        return False
    years = _dated_years(text, _year_shields(fields, text))
    if len(years) != 1:
        return False
    fields["year"] = years.pop()
    return True


# "CHAVES, Antônio — Direito Autoral de Radiodifusão, S. Paulo, Ed. …": a
# byline of a surname in capitals and the given names (words, or initials
# such as "V." and "J.-P."), a spaced dash, then the title up to the comma
# before the imprint or source. A period after a given name ends the byline
# ("BAY, József. Lengyel – Kastélypark" is a title with a dash).
_GIVEN_NAME = r"(?:[^\W\d_]\.(?:-?[^\W\d_]\.)*|[^\W\d_][\w'’-]*)"
_DASH_BYLINE_RE = re.compile(
    r"(?P<byline>(?P<surname>[^\W\d_][\w'’-]*(?:\s[^\W\d_][\w'’-]*)?),\s*"
    rf"{_GIVEN_NAME}(?:\s{_GIVEN_NAME}){{0,3}})\s+[—–]\s+(?=\S)"
)
# What follows the dash is another "Surname, Given" byline (a co-author list
# joined by dashes, "EBEL, Petr – SCHMIDT, Ondřej. Z Trevisa …"): not a title.
_BYLINE_AFTER_DASH_RE = re.compile(
    r"[^\W\d_][\w'’‟-]*(?:\s[^\W\d_][\w'’‟-]*)?,\s*[^\W\d_][\w'’.-]*(?:\s[^\W\d_][\w'’.-]*){0,3}?"
    r"\s*(?:[—–(:;\[]|\.(?:\s|$)|$)"
)
# Where the title after a dash byline ends: at its first comma, or at a period
# closing a word ("… Colombiana. RIDI").
_DASH_TITLE_END_RE = re.compile(r",\s|(?<=[^\W\d_]{3})\.(?:\s|$)")


def _rule_dash_byline(fields: dict[str, Any], text: str) -> bool:
    """ "SURNAME, Given — Title, Place, Publisher Year": the dash ends the byline.

    The tagger misreads this style's byline: it tags nothing but the source
    ("CHAVES, Antônio — Direito Autoral de Radiodifusão, S. Paulo, Ed. Rev. dos
    Tribunais 1952"), only the surname ("HAMMES"), or the surname with the
    given names starting the title ("FIGUEIREDO" + "Guilherme — Defesa de
    alguns pontos …"). Fires only when the text opens with a surname in
    capitals, a comma, one to four given names and a spaced dash; the tagged
    author, if any, lies inside that byline; no title was tagged, or the tagged
    one starts before the dash; and what follows the dash is neither another
    byline nor a word in capitals ("KRÁL Pavel."). The byline becomes the
    author. A tagged title keeps its end and loses only what precedes the
    dash; an untagged one runs from the dash to its first comma, a period
    closing a word, or the first span tagged as another field, whichever comes
    first. A title the tagger started after the dash is left as it is.
    """
    lead = len(text) - len(text.lstrip())
    match = _DASH_BYLINE_RE.match(text, lead)
    if match is None:
        return False
    surname = match.group("surname")
    if surname != surname.upper():
        return False
    body = match.end()
    rest = text[body:]
    first_word = rest.split(maxsplit=1)[0].strip(".,;:")
    if _BYLINE_AFTER_DASH_RE.match(rest) or (
        len(first_word) >= 3 and first_word.isalpha() and first_word.isupper()
    ):
        return False
    byline = match.group("byline").strip()
    authors = fields.get("authors")
    authors = authors.strip() if isinstance(authors, str) else ""
    if authors and not byline.startswith(authors.rstrip(".").rstrip()):
        return False
    title = fields.get("title")
    title = title.strip() if isinstance(title, str) else ""
    end = len(text)
    tagged_end = None
    if title:
        at = text.find(title, lead)
        if at < 0 or at >= body:
            return False
        if at + len(title) > body:
            tagged_end = at + len(title)
    if tagged_end is not None:
        end = tagged_end
    else:
        stop = _DASH_TITLE_END_RE.search(text, body)
        if stop is not None:
            end = stop.start()
    for name in (*_OTHER_SPAN_FIELDS, "year"):
        value = fields.get(name)
        if value is None or value == "":
            continue
        at = text.find(str(value).strip(" ,.;:"), body)
        if at >= 0:
            end = min(end, at)
    new_title = _clean_cut(text[body:end]).rstrip(".").strip()
    if len(new_title.split()) < 2 or not _has_word(new_title, 3):
        return False
    fields["authors"] = byline
    fields["title"] = new_title
    return True


# One parenthesised sentence and nothing else, with no number in it.
_EDITORIAL_NOTE_RE = re.compile(r"\((?P<body>[^()\d]+[.!?])\s*\)\.?")


def _rule_editorial_note(fields: dict[str, Any], text: str) -> bool:
    """A list entry that is one sentence in parentheses is a note, not a reference.

    "(This is a series of short articles by Adler and various pupils which
    show in more detail the general theory.)" annotates the entry above it;
    the tagger reads it as a title and the reference list gains an entry that
    matches nothing. Fires when the whole text is a parenthesised sentence of
    at least four words with no digit (so no year, volume or page) and a field
    was tagged; every field is cleared, so the entry is dropped.
    """
    match = _EDITORIAL_NOTE_RE.fullmatch(text.strip())
    if match is None or len(match.group("body").split()) < 4:
        return False
    tagged = [name for name, value in fields.items() if value not in (None, "", [], False)]
    if not tagged:
        return False
    for name in tagged:
        fields[name] = None
    return True


# A byline that ends in a family name ("Newnham", "Al-Shahi", "Ferrières"): a
# capitalised word holding a lowercase letter, so not an initial.
_SURNAME_END_RE = re.compile(
    r"(?:^|[\s,;])[A-ZÀ-ÖØ-Þ][^\W\d_]*[a-zß-öø-ÿ][^\W\d_]*(?:[-'’][^\W\d_]+)*$"
)
# One or two initials right after it, closed by "." or ":" and a space (or the
# end): "Newnham M. COVID-19 …", "Gauja A: Party reform …".
_FINAL_INITIAL_RE = re.compile(r"\s+([A-ZÀ-ÖØ-Þ]{1,2})(?=[.:](?:\s|$))")


def _rule_author_final_initial(fields: dict[str, Any], text: str) -> bool:
    """Give the last author back the single initial the tagger left out.

    Vancouver bylines close with the last author's initials and a period or a
    colon. With one initial ("Loo J, Spittle DA, Newnham M. COVID-19 …") the
    tagger reads "M." as the field delimiter and tags it O, so the byline ends
    at "Newnham"; with two ("Smith JK.") it does not. Fires only when the
    tagged byline is found verbatim, ends in a family name (not an initial),
    and is followed directly by one or two capitals closed by "." or ":", and
    the title does not start at those capitals ("M. tuberculosis …"). The
    byline is extended over them.
    """
    authors = fields.get("authors")
    if not isinstance(authors, str):
        return False
    byline = authors.rstrip()
    if not byline or not _SURNAME_END_RE.search(byline):
        return False
    at = text.find(byline)
    if at < 0:
        return False
    match = _FINAL_INITIAL_RE.match(text, at + len(byline))
    if match is None:
        return False
    title = fields.get("title")
    if isinstance(title, str) and title.strip() and text.find(title.strip(), at) == match.start(1):
        return False
    fields["authors"] = f"{byline} {match.group(1)}"
    return True


def _rule_author_trailing_colon(fields: dict[str, Any], _text: str) -> bool:
    """ "Kling CC, Kunegis J, Hartmann H, et al.: Voting behaviour …": the colon
    that closes a colon-style byline is punctuation, not part of the author list."""
    authors = fields.get("authors")
    if not isinstance(authors, str) or not authors.rstrip().endswith(":"):
        return False
    cut = authors.rstrip().rstrip(":").rstrip()
    if not _has_word(cut):
        return False
    fields["authors"] = cut
    return True


# A byline tagged as the title: "Surname, I. et al." and nothing else.
_ET_AL_BYLINE_RE = re.compile(
    r"[A-ZÀ-ÖØ-Þ][^\W\d_]*[a-zß-öø-ÿ][^\W\d_]*(?:[-'’][^\W\d_]+)*,"
    r"\s*(?:[A-ZÀ-ÖØ-Þ]\.\s?-?){1,3}\s*,?\s*et\s+al\.?"
)


def _rule_title_et_al_byline(fields: dict[str, Any], _text: str) -> bool:
    """ "Deppe, U. et al. Nature 326, 1-2 (1987).": the tagger reads the byline of
    a Nature-style note as its title and leaves the authors empty. Fires only
    when no authors were tagged and the whole title is "Surname, I. et al.";
    it moves to the authors and the title is cleared."""
    if fields.get("authors"):
        return False
    title = fields.get("title")
    if not isinstance(title, str) or not _ET_AL_BYLINE_RE.fullmatch(title.strip()):
        return False
    fields["authors"] = title.strip()
    fields["title"] = None
    return True


# The punctuation a quoted-title style puts on either side of a closing quote.
_QUOTE_PUNCT = ",.;"


def _cut_closing_quote(title: str) -> tuple[str, str] | None:
    """*title* without a closing quote at its end and one comma, period or
    semicolon on either side of it, and the quote; None when it does not end
    in one. String operations, not a regex: no backtracking on long spaces."""
    rest = title.rstrip()
    if rest[-1:] and rest[-1] in _QUOTE_PUNCT:
        rest = rest[:-1].rstrip()
    if rest[-1:] not in ("”", '"'):
        return None
    quote, rest = rest[-1], rest[:-1].rstrip()
    if rest[-1:] and rest[-1] in _QUOTE_PUNCT:
        rest = rest[:-1].rstrip()
    return rest, quote


def _rule_title_closing_quote(fields: dict[str, Any], _text: str) -> bool:
    """ "Assessment of effect of haze … in Sumatra,”": drop the closing quote
    (and the comma or period before or after it) of a quoted title when the
    tagged title holds no opening quote of its own."""
    title = fields.get("title")
    cut_quote = _cut_closing_quote(title) if isinstance(title, str) else None
    if cut_quote is None:
        return False
    cut, quote = cut_quote
    if quote == '"':
        if title.count('"') % 2 == 0:
            return False
    else:
        # „…” (Polish, Romanian, Hungarian, Croatian) and ”…” (Swedish,
        # Finnish) titles close their own quotes: only an odd surplus of ”
        # over its openers is the residue of a quoted-title style.
        surplus = title.count("”") - title.count("“") - title.count("„")
        if surplus <= 0 or surplus % 2 == 0:
            return False
    if not _has_word(cut, 3):
        return False
    fields["title"] = cut.strip()
    return True


# The quoted title right after a byline: "Ardupilot, “MissionPlanner,” https://…".
_QUOTED_AFTER_BYLINE_RE = re.compile(r"\s*[,.:]?\s*[“\"„](?P<title>[^”\"“„]{2,300}?)[”\"“ˮ]")


def _rule_title_quoted_after_byline(fields: dict[str, Any], text: str) -> bool:
    """An untitled reference whose title is the quotation right after its byline.

    IEEE-style "[30] P. Ekman, “Microexpression training tool (METT),” Stanford
    Univ., …" and web entries ("USA, “AirForceTimes,” https://…"): the tagger
    tags the byline and leaves the short quoted title out. Fires only when no
    title was tagged, the tagged byline is found verbatim and the next thing
    after it (past a comma) is a double-quoted phrase; the phrase, without its
    closing comma or period, becomes the title.
    """
    if fields.get("title"):
        return False
    authors = fields.get("authors")
    if not isinstance(authors, str) or not authors.strip():
        return False
    byline = authors.rstrip()
    at = text.find(byline)
    if at < 0:
        return False
    match = _QUOTED_AFTER_BYLINE_RE.match(text, at + len(byline))
    if match is None:
        return False
    title = _clean_cut(match.group("title")).rstrip(".").strip()
    if not _has_word(title, 2):
        return False
    fields["title"] = title
    return True


_RULES: tuple[tuple[str, Callable[[dict[str, Any], str], bool]], ...] = (
    ("author_final_initial", _rule_author_final_initial),
    ("author_trailing_colon", _rule_author_trailing_colon),
    ("title_et_al_byline", _rule_title_et_al_byline),
    ("title_closing_quote", _rule_title_closing_quote),
    ("title_quoted_after_byline", _rule_title_quoted_after_byline),
    ("web_lead_title", _rule_web_lead_title),
    ("author_dash_title", _rule_author_dash_title),
    ("dash_byline", _rule_dash_byline),
    ("title_url_tail", _rule_title_url_tail),
    ("title_quote_runon", _rule_title_quote_runon),
    ("title_dateline_tail", _rule_title_dateline_tail),
    ("title_translation_bracket", _rule_title_translation_bracket),
    ("title_language_note", _rule_title_language_note),
    ("title_responsibility", _rule_title_responsibility),
    ("title_year_tail", _rule_title_year_tail),
    ("title_imprint_tail", _rule_title_imprint_tail),
    ("year_not_access_date", _rule_year_not_access_date),
    ("year_from_text", _rule_year_from_text),
    ("editorial_note", _rule_editorial_note),
)


def repair_ner_reference_fields(fields: dict[str, Any], text: str) -> list[str]:
    """Repair NER field boundaries in place; return the names of the rules that fired.

    ``fields`` is one :meth:`OnnxRefParser.parse_batch` output (PaperReference
    field names); ``text`` is the parser input it was decoded from.
    """
    if not text or not fields:
        return []
    fired = []
    for name, rule in _RULES:
        try:
            if rule(fields, text):
                fired.append(name)
        except Exception:  # noqa: BLE001 - a repair must never break parsing
            logger.debug("reference field repair %s failed", name, exc_info=True)
    if fired:
        logger.debug("reference field repairs %s on %r", fired, text[:120])
    return fired
