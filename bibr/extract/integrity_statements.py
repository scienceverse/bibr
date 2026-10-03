"""Unified candidate resolution for research-integrity statements.

Section classifications are retrieval evidence, not permission to copy.  The
resolver builds section and anchored-paragraph candidates together, records
their source IDs, and delays rendering until callers have finished late text
cleaning.  ``shadow`` exports the compatibility values while keeping typed
comparison evidence for debugging; ``active`` materializes the bounded
selection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from bibr.extract.statement_scan import (
    _ANCHORS,
    _BARE_CATEGORY_LABELS,
    _BOILERPLATE_BOUNDARY,
    _COI_TOPICAL_PROSE,
    _FUNDING_AMBIGUOUS_ANCHOR,
    _FUNDING_STRONG_ANCHORS,
    _author_aliases,
    _boilerplate_boundary_for_field,
    _boundary_is_declaration_text,
    _bounded_sentence_for_field,
    _categories_in,
    _has_assertive_declaration,
    _has_field_anchor,
    _has_funder_hint,
    _has_funding_negative_declaration,
    _has_unresolved_author_funding_declaration,
    lexical_fallback_warning,
)
from bibr.paper_contents import CanonicalSection
from bibr.utils.text import normalize_text
from bibr.validation import IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from bibr.models import PaperMetadata
    from bibr.paper_contents import PaperContents, PaperSection, PaperSentence

IntegrityStatementMode = Literal["legacy", "shadow", "active"]

_FIELD_SECTION_TYPES: dict[str, CanonicalSection] = {
    "funding_statement": CanonicalSection.FUNDING,
    "coi_statement": CanonicalSection.COI,
    "ethics_statement": CanonicalSection.ETHICS,
    "data_availability": CanonicalSection.OPEN_DATA,
}
_FIELDS = tuple(_FIELD_SECTION_TYPES)
_SECTION_TYPE_FIELDS = {section_type: field for field, section_type in _FIELD_SECTION_TYPES.items()}

_STRONG_HEADINGS: dict[str, frozenset[str]] = {
    "funding_statement": frozenset(
        {
            "funding statement",
            "sources of funding",
            "source of funding",
            "financial support",
            "funding information",
        }
    ),
    "coi_statement": frozenset(
        {
            "competing interests",
            "conflict of interest statement",
            "conflicts of interest statement",
            "declaration of competing interest",
            "declaration of competing interests",
            "declaration of conflicting interest",
            "declaration of conflicting interests",
        }
    ),
    "ethics_statement": frozenset(
        {
            "ethical approval",
            "ethics approval",
            "ethics statement",
            "ethics approval and consent to participate",
            "institutional review board statement",
        }
    ),
    "data_availability": frozenset(
        {
            "data availability statement",
            "availability of data and materials",
            "data and code availability",
            "code availability statement",
        }
    ),
}
_GENERIC_HEADINGS: dict[str, frozenset[str]] = {
    "funding_statement": frozenset({"funding", "funding sources"}),
    "coi_statement": frozenset({"conflict of interest", "conflicts of interest"}),
    "ethics_statement": frozenset({"ethics", "ethical considerations"}),
    "data_availability": frozenset(
        {"data availability", "code availability", "materials availability"}
    ),
}

_UNLABELED_NEGATIVE_DECLARATION = re.compile(
    r"^(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
    re.IGNORECASE,
)
_FIELD_LABELED_NEGATIVE_DECLARATION: dict[str, re.Pattern[str]] = {
    "funding_statement": re.compile(
        r"^(?:funding|financial support)(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
    "coi_statement": re.compile(
        r"^(?:conflicts? of interest|competing interests?)(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
    "ethics_statement": re.compile(
        r"^(?:ethics|ethical approval|ethics approval)(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
    "data_availability": re.compile(
        r"^(?:data|code|materials?)\s+availability(?:\s*:\s*|\s+)"
        r"(?:not applicable|none(?: declared)?|no(?:ne)?)\.?$",
        re.IGNORECASE,
    ),
}
_FIELD_NEGATIVE_DECLARATION: dict[str, re.Pattern[str]] = {
    "funding_statement": re.compile(
        r"\b(?:no\s+(?:external\s+)?(?:funding|financial support)|"
        r"received no (?:funding|financial support))\b",
        re.IGNORECASE,
    ),
    "coi_statement": re.compile(
        r"^no\s+(?:potential\s+)?"
        r"(?:conflicts?(?:\s+of\s+interest)?|competing interests?)"
        r"(?:\s+(?:were\s+)?(?:declared|reported))?\.?$",
        re.IGNORECASE,
    ),
    "ethics_statement": re.compile(
        r"\b(?:ethical approval|ethics approval|consent)\s+(?:was\s+)?not\s+(?:required|applicable)\b",
        re.IGNORECASE,
    ),
    "data_availability": re.compile(
        r"\b(?:no data|data (?:are|were) not (?:generated|available))\b",
        re.IGNORECASE,
    ),
}
_TOPICAL_ETHICS_HEADING = re.compile(
    r"\b(?:ethical problem|ethics of|ethical implications?|ethics? .{0,30} debate|"
    r"ethical and scientific judgements?)\b",
    re.IGNORECASE,
)
_PUBLICATION_CONSENT = re.compile(r"^consent for publication$", re.IGNORECASE)
# End-matter types a strong statement heading is copied from as is; under a
# body type ("Financial support" typed as results) the copy must be compact
# and pass the lexical prose guards.
_STRONG_HEADING_COPY_TYPES = frozenset(
    {
        None,
        CanonicalSection.ENDNOTE,
        CanonicalSection.FOOTNOTE,
        CanonicalSection.UNKNOWN,
        CanonicalSection.ACKNOWLEDGMENT,
    }
)
_AMBIGUOUS_CLASSIFICATION_SOURCES = frozenset(
    {"model", "llm", "alias_prior", "substring_alias", "parent_context"}
)
_MAX_COMPACT_CHARS = 1200
_MAX_COMPACT_PARAGRAPHS = 3


def _legacy_compile(*phrases: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(r"\b" + phrase, re.IGNORECASE) for phrase in phrases)


# Compatibility ("legacy") ownership, which legacy and shadow mode export: a
# copy of each statement-typed section, else the first anchored row plus up to
# two following rows of its section. The anchors are the pre-resolver ones; the
# copies are matched on whitespace-collapsed rows, bounded at publisher
# furniture, licence blocks, the next label and thanks (the _legacy_* guards
# below), and rendered from the final, late-cleaned sentence text.
_LEGACY_ANCHORS: dict[str, tuple[re.Pattern[str], ...]] = {
    "funding_statement": _legacy_compile(
        r"this work was supported",
        r"supported by",
        r"funded by",
        r"funding for this",
        r"financial support",
        r"grant no\.",
    ),
    "coi_statement": _legacy_compile(
        r"conflicts? of interest",
        r"competing interests?",
        r"no potential conflict",
    ),
    "ethics_statement": _legacy_compile(
        r"ethical approval",
        r"ethics committee",
        r"ethics approval",
        r"institutional review board",
        r"irb approval",
        r"informed consent",
        r"complies with ethical",
    ),
    "data_availability": _legacy_compile(
        r"data availability",
        r"data are available",
        r"data are openly available",
        r"code availability",
        r"materials are available",
    ),
}
_LEGACY_FUNDING_STRONG_ANCHORS = _legacy_compile(
    r"this work was supported",
    r"funded by",
    r"funding for this",
    r"financial support",
    r"grant no\.",
)
_LEGACY_FUNDING_AMBIGUOUS_ANCHOR = re.compile(r"\bsupported by", re.IGNORECASE)
_LEGACY_FUNDER_KEYWORD = re.compile(
    r"\b(?:grant|grants|foundation|council|university|ministry|fellowship|"
    r"scholarship|endowment|nsf|erc|nih|nserc|dfg|funded|funding|award)\b",
    re.IGNORECASE,
)
_LEGACY_FUNDER_STRONG = re.compile(
    r"#\d|\b[A-Z]{2,}\b|\b[A-Z][A-Za-z]+\s+"
    r"(?:Foundation|Council|Trust|Institute|Fund|Agency)\b"
)
# "supported by" is the one ambiguous funding anchor: what follows it must name
# a funder ("supported by the base AR model" and "supported by expression of
# PODXL" do not), so a bare capitalised acronym is not enough.
_SUPPORTED_BY_FUNDER_WORD = re.compile(
    r"\b(?:grants?|foundation|council|university|ministry|fellowships?|scholarships?|"
    r"endowment|funded|funding|funds|awards?|awarded|contract|charity|"
    r"fundação|fundación|fondazione|fondation|stiftung|fonds|fondo|consejo|"
    r"conselho|ministerio|ministère|agencia|agência|universidad|universidade|"
    r"università|université|universität)\b",
    re.IGNORECASE,
)
_SUPPORTED_BY_FUNDER_NAME = re.compile(
    r"\b(?:Programme|Program|Association|Society|Academy|Commission|Department|Agency|"
    r"Institute|Trust|Fund|Centre|Center)\b"
)
_SUPPORTED_BY_GRANT_ID = re.compile(
    r"#\s?\d|\b[A-Z]{2,}[\s_-]?\d{4,}|\b\d{1,4}/\d{1,4}/\d{1,6}\b|"
    r"\bNo\.?\s+[A-Z0-9][\w./-]*\d|\b[Nn]o\.\s*\d{3,}|\b[Nn]o\s+\d{4,}|"
    r"\b[A-Z]{1,6}\d{2,}(?:[-/_][A-Z0-9]+)+|"
    # digit-led ("01GL1234") and slash ("UIDB/04501/2020") grant IDs
    r"\b\d{2,}[A-Z]{1,4}\d{2,}\b|\b[A-Z]{2,}/\d{3,}(?:/\d{2,4})?\b"
)
# Funder acronyms that are rarely anything else (the bounded scan's own list
# covers NSF, ERC, NIH, NSERC and DFG).
_SUPPORTED_BY_FUNDER_ACRONYM = re.compile(
    r"\b(?:NSFC|JSPS|KAKENHI|AMED|CONICET|FAPESP|FAPERJ|FAPEMIG|CNPq|CAPES|ANR|SNSF|NHMRC|"
    r"ESRC|EPSRC|BBSRC|AHRC|MRC|NERC|UKRI|NIHR|CIHR|SSHRC|NWO|FWF|DAAD|NCN|NCBiR|"
    r"BMBF|FCT|JST|CREST)\b"
)

# Row-start statement labels: "Funding:", "Conflict of interest statement:",
# run-in "Ethics approval The study was ...". A label-led row is tried before a
# row that only mentions an anchor in prose, and another field's label ends a
# copy.
_STATEMENT_LABEL_WORDS: dict[str, str] = {
    "funding_statement": (
        r"funding(?: sources?| information)?|sources? of funding|financial support|"
        r"grant information|grant support"
    ),
    "coi_statement": (
        r"conflicts? of interests?|competing (?:financial )?interests?|"
        r"conflicting interests?|declarations? of (?:conflicting |competing )?interests?|"
        r"(?:financial )?disclosures?|duality of interests?"
    ),
    "ethics_statement": (
        r"ethics(?: approval| statement)?|ethical approval|"
        r"ethics approval and consent to participate|"
        r"human and animal rights(?: and informed consent)?|"
        r"institutional review board(?: statement)?|informed consent(?: statement)?|"
        r"patient consent|human subjects|ethical considerations?"
    ),
    "data_availability": (
        r"data (?:availability|sharing)|availability of data(?: and materials?)?|"
        r"data and code availability|code availability|materials availability"
    ),
}
_STATEMENT_LABEL: dict[str, re.Pattern[str]] = {
    field: re.compile(
        rf"^\W*(?:{words})(?:\s+statements?)?"
        r"(?:\s*[:.—–/-]|\s+(?-i:[A-Z])|\s*$)",
        re.IGNORECASE,
    )
    for field, words in _STATEMENT_LABEL_WORDS.items()
}
# Inside the reference list only a row that opens with a punctuated label
# ("Conflict of interest statement: ...") is a statement: end matter that
# the layout model put after the last reference.
_REFERENCE_STATEMENT_LABEL: dict[str, re.Pattern[str]] = {
    field: re.compile(rf"^\W*(?:{words})(?:\s+statements?)?\s*[:.—–-]", re.IGNORECASE)
    for field, words in _STATEMENT_LABEL_WORDS.items()
}
# ... and only when the text after the label declares something: a reference
# title split into rows ("Conflicts of interest: a hidden threat to science.")
# is not a statement.
_REFERENCE_STATEMENT_CUE = {
    "funding_statement": re.compile(
        r"\b(?:fund\w*|grants?|support\w*|award\w*|sponsor\w*|none|no|not)\b", re.IGNORECASE
    ),
    "coi_statement": re.compile(
        r"\b(?:no|not|none|nothing|declare[sd]?|report\w*|disclose[sd]?|receive[sd]?|"
        r"consult\w*|honorari\w*|employee|shareholder|unaware|free of)\b",
        re.IGNORECASE,
    ),
    "ethics_statement": re.compile(
        r"\b(?:approv\w*|consent\w*|waive[sd]?|exempt\w*|committee|review board|irb|"
        r"helsinki|not (?:applicable|required))\b",
        re.IGNORECASE,
    ),
    "data_availability": re.compile(
        r"\b(?:availab\w*|deposit\w*|request|access\w*|repositor\w*|osf|zenodo|github|doi|"
        r"shared|not applicable)\b|https?://",
        re.IGNORECASE,
    ),
}
# A heading region that also holds the one-line statement ("Disclosure and
# competing interests statement The authors declare no competing interests.").
_HEADING_TAIL_START = re.compile(r"\s(?=[A-Z][a-z]*\s+[a-z])")
_MAX_HEADING_LABEL_CHARS = 80
_HEADING_NUMBER = re.compile(r"^\s*(?:\d+(?:[.-]\d+)*(?:[.-]\s*|\s+)|[IVX]+\.\s+)")
# Rows that open other end matter (or are page furniture) end a statement copy.
_END_MATTER_LABEL = re.compile(
    r"^\W*(?:(?:patient )?consent for publication|patient and public involvement|"
    r"provenance and peer review|transparency(?: declaration)?|acknowledg(?:e)?ments?|"
    r"authors?'? contributions?|author's contributions?|contributors|credit authorship|"
    r"keywords?|key words|e-?mail|correspondence|how to cite|cite this article|"
    r"open access(?!\s+(?:funding|publication|fees?|charges?|costs?|was|is|has)\b)|"
    r"copyright(?=\s*(?:©|\(c\)|:|\d{4}|notice|holder))|abbreviations|supplementary (?:material|information|data)|orcid|"
    r"article history|disclaimer|publisher'?s note|additional information|"
    r"ai tool disclosure|use of (?:ai|artificial intelligence)|declaration of generative ai|"
    r"all rights reserved|no reuse allowed|the copyright holder)\b|^\W*©",
    re.IGNORECASE,
)
_THANKS = re.compile(
    r"^\W*(?:additionally,?\s+|also,?\s+|finally,?\s+|further(?:more)?,?\s+|"
    r"moreover,?\s+)?(?:we|the authors?|i)\s+(?:would\s+(?:also\s+)?like\s+to\s+)?"
    r"(?:also\s+)?(?:thank|are\s+(?:very\s+|deeply\s+|also\s+)?grateful|is\s+grateful|"
    r"am\s+grateful|wish\s+to\s+thank|express\s+(?:our|my)\s+(?:sincere\s+|deep\s+)?"
    r"(?:gratitude|thanks))\b",
    re.IGNORECASE,
)
# Furniture boundaries the statement-scan list lacks: licence and preprint
# sidebars, BMJ end-matter labels, F1000 "Grant information:".
_EXTRA_BOUNDARY = re.compile(
    r"\bopen access\b(?=\s+this\b)|\ball rights reserved\b|\bno reuse allowed\b|"
    r"\bthe copyright holder\b|\bpatient consent for publication\b|"
    r"\bprovenance and peer review\b|\bpatient and public involvement\b|"
    r"\bgrant information\s*:|\bpublisher[’']s note\b|\ball claims expressed in this article\b",
    re.IGNORECASE,
)
# Furniture words ("copyright", "licence", "published by", "received ...") also
# occur inside statements ("Due to copyright restrictions ...", "MW is the
# copyright holder of ...", "published under a CC BY licence"). They end a
# statement only at a row or sentence start, inside a licence sentence ("This
# article is licensed under ..."), or in an unmistakable furniture shape: "©",
# "Copyright © 2024", "Open Access This ...", "Received: 3 May 2019".
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
)
_DATE = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\s*,?\s+\d{{4}}|"
    rf"{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+\d{{4}}|"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\w)"
)
# "received"/"accepted" followed by a day, a whole month name or a year (not
# "received separate funding" or "received 15,000 CHF").
_HISTORY_WORD = re.compile(
    rf"(?:received|accepted)\s*(?:on\b|:)?\s*(?:\d{{1,2}}(?![\d,])|{_MONTH}|(?:19|20)\d{{2}}\b)",
    re.IGNORECASE,
)
_HISTORY_SHAPE = re.compile(
    rf"(?:received|accepted|revised|published(?:\s+online)?)\s*(?:on\s+|:\s*)?{_DATE}",
    re.IGNORECASE,
)
_FURNITURE_SHAPE = re.compile(
    r"©|copyright\s*(?:©|\(c\)|:|\d{4})|(?-i:Open Access\s+This)\b", re.IGNORECASE
)
_LICENCE_SENTENCE = re.compile(
    r"\W*(?:open access\b|this is an open[- ]access|it is made available under|licensee\b|"
    r"this (?:article|work|paper|chapter|manuscript|preprint|version) is (?:an open[- ]access|"
    r"licensed|distributed|published under|made available under)|"
    r"(?:published|distributed|licensed) under\b|to view a copy of this licen[cs]e|"
    r"the images or other third[- ]party material|which permits (?:unrestricted )?use|"
    # rights-retention statements: "For the purpose of open access, the author
    # has applied a CC BY licence ...", "A CC-BY license is applied to the Author
    # Accepted Manuscript ..."
    r"for the purpose of open access|(?=[^.]{0,200}\bauthor accepted manuscript))",
    re.IGNORECASE,
)
# "correspond..." is furniture only as a contact line; a statement may point to
# the corresponding author ("Proposals should be directed to the corresponding
# author").
_CONTACT_FURNITURE = re.compile(
    r"correspondence\s+(?:may|should|concerning|regarding|to|address)\b|"
    r"correspond(?:ence|ing authors?)\s*(?::|\*)|"
    r"corresponding authors?'?s?\s+(?:e-?mail|address)",
    re.IGNORECASE,
)
_LICENCE_TEXT = re.compile(
    r"\b(?:open access(?!\s+(?:funding|publication|fees?|charges?|costs?|was|is|has)\b)|"
    r"creative commons|licen[cs]ed?|copyright|all rights reserved)\b|©",
    re.IGNORECASE,
)
# The rest of a licence paragraph: a continuation row or more licence terms.
_LICENCE_CONTINUATION = re.compile(
    r"^\W*(?-i:[a-z])|\b(?:permit\w*|reproduc\w*|distribut\w*|adapt\w*|credit|licen[cs]\w*|"
    r"copyright|creative ?commons|changes were made|statutory|third[- ]party material|"
    r"open access|waiver|public domain|re-?use|attribution)\b|https?://|©",
    re.IGNORECASE,
)
# A section copy from a classifier source that can be wrong (model, LLM, prior)
# that is longer than this is a chapter, not a statement: use the lexical path.
_MAX_AMBIGUOUS_SECTION_CHARS = 3000
# A lexical COI row must declare something, not just mention conflicts of
# interest ("To maintain credibility, avoid conflicts of interest, ..."): it is
# label-led, or a declaration cue sits near the anchor.
_LEGACY_COI_LABEL = re.compile(
    r"^\W*(?:statement re:? |declarations? of |disclosure of )?(?:potential )?"
    r"(?:conflicts? of interests?|competing (?:financial )?interests?|conflicting interests?|"
    r"declarations? of (?:conflicting|competing) interests?|declaration of interests?|"
    r"(?:financial )?disclosures?|duality of interest)\b",
    re.IGNORECASE,
)
_LEGACY_COI_CUE = re.compile(
    r"\b(?:no|not|none|nothing|without|declares?|declared|declaring|reports?|reported|"
    r"discloses?|disclosed|affirms?|absence of|received|receives|has served|serves? as|"
    r"consult\w*|honorari\w*|personal fees|employee|shareholder|stock|unaware|free of)\b",
    re.IGNORECASE,
)
# A cue that declares even in a sentence that reads like topical prose
# ("... no conflict of interest between the authors and the funding body").
_LEGACY_COI_STRONG_CUE = re.compile(
    r"\b(?:declare[sd]?|declaring|unaware|free (?:of|from)|"
    r"nothing to (?:disclose|declare|report)|"
    r"(?:has|have|had|there (?:is|are|was|were)) no (?:\w+ ){0,3}?"
    r"(?:conflicts?|competing|financial|relevant|known|potential|interests?)\b)",
    re.IGNORECASE,
)
_COI_CUE_WINDOW = 100
# "financial support" and "data/code availability" are label nouns: in running
# prose ("failed to manage the financial support provided by donors", "regardless
# of their school type or data availability") they are not a statement. Outside
# a label they need the paper's own funding (a recipient, a grant number or a
# named funder) or must open a sentence.
_LABEL_NOUN_ANCHORS = {
    "funding_statement": re.compile(r"\bfinancial support", re.IGNORECASE),
    "data_availability": re.compile(r"\b(?:data|code) availability", re.IGNORECASE),
}
_LABEL_NOUN_PATTERNS = frozenset(
    {r"\bfinancial support", r"\bdata availability", r"\bcode availability"}
)
_FUNDING_RECIPIENT = re.compile(
    r"\b(?:we|our|us|the authors?|this (?:work|study|research|project|paper|article|"
    r"publication|trial|review)|the (?:study|research|project|work) (?:was|is|has|received)|"
    r"acknowledg\w*|grateful(?:ly)?|thanks?)\b",
    re.IGNORECASE,
)
_NAMED_FUNDER_BODY = re.compile(
    r"\b[A-Z][A-Za-z]+\s+(?:Foundation|Council|Trust|Institute|Fund|Agency|Ministry|"
    r"Programme|Program|Commission|Society|Academy)\b"
)
# An ethics row whose only anchor is "informed consent" must say the consent
# was obtained, given or waived, not list it among topics.
_LEGACY_CONSENT_ACTION = re.compile(
    r"\b(?:obtained|obtain|provided|provide|gave|given|give|signed|sign|received|receive|"
    r"waived|required|sought|secured|documented|approved|consented|conforms?|accordance|"
    r"helsinki|voluntar\w*|collected|collect|taken|acquired)\b",
    re.IGNORECASE,
)
# Statement-typed sections that are not that statement: the Lancet-style
# "Role of the funding source" (what the funder did not do), AI-use
# disclosures, and topical ethics chapters.
_ROLE_OF_FUNDER_HEADING = re.compile(
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?(?:the )?role of (?:the )?"
    r"(?:funding sources?|funders?|sponsors?)\b",
    re.IGNORECASE,
)
_AI_USE_HEADING = re.compile(
    r"\b(?:ai|artificial intelligence|generative|large language models?|llms?|chatgpt)\b",
    re.IGNORECASE,
)
# An AI-use heading still names the statement it holds when it carries the
# field's own words ("Data and code availability for the LLM benchmark").
_FIELD_HEADING_WORD = {
    "funding_statement": re.compile(r"\b(?:fund\w*|grants?|financ\w*)\b", re.IGNORECASE),
    "coi_statement": re.compile(r"\b(?:interests?|conflicts?|duality)\b", re.IGNORECASE),
    "ethics_statement": re.compile(
        r"\b(?:ethic\w*|consent|approval|review board|irb)\b", re.IGNORECASE
    ),
    "data_availability": re.compile(r"\b(?:data|code|materials?|availability)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class IntegrityStatementCandidate:
    field: str
    method: str
    heading: str | None
    section_ids: tuple[int, ...]
    text_ids: tuple[int, ...]
    paragraph_ids: tuple[int, ...]
    pages: tuple[int, ...]
    classification_source: str | None
    classification_score: float | None
    reason_flags: tuple[str, ...]
    accepted: bool


@dataclass(frozen=True)
class IntegrityStatementResolution:
    """Candidate IDs per field; values are rendered later, from cleaned text.

    ``legacy_statement_snapshots`` holds the compatibility values rendered at
    resolve time, before late cleaning. Shadow mode compares them with the
    bounded selection; exports render the same IDs after late cleaning.
    """

    mode: IntegrityStatementMode
    candidates: tuple[IntegrityStatementCandidate, ...]
    legacy_candidate_indices: tuple[tuple[str, tuple[int, ...]], ...]
    selected_candidate_indices: tuple[tuple[str, tuple[int, ...]], ...]
    legacy_statement_snapshots: tuple[tuple[str, str | None], ...]
    issues: tuple[ValidationIssue, ...]

    def selected_indices(self, field: str) -> tuple[int, ...]:
        return dict(self.selected_candidate_indices).get(field, ())

    def legacy_indices(self, field: str) -> tuple[int, ...]:
        return dict(self.legacy_candidate_indices).get(field, ())

    def candidate_indices(self, field: str, *, effective: bool = True) -> tuple[int, ...]:
        if effective and self.mode == "active":
            return self.selected_indices(field)
        return self.legacy_indices(field)


def _ordered_unique(values) -> tuple:
    return tuple(dict.fromkeys(values))


def _section_rows(contents: PaperContents) -> dict[int, list[PaperSentence]]:
    rows: dict[int, list[PaperSentence]] = {}
    for sentence in contents.sentences:
        if not sentence.is_display_formula:
            rows.setdefault(sentence.section_id, []).append(sentence)
    return rows


def _legacy_categories_in(text: str) -> set[str]:
    return {
        field
        for field, patterns in _LEGACY_ANCHORS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _legacy_has_funder_hint(text: str) -> bool:
    return bool(_LEGACY_FUNDER_KEYWORD.search(text) or _LEGACY_FUNDER_STRONG.search(text))


def _supported_by_names_funder(text: str) -> bool:
    """Whether the text after "supported by" names a funder."""
    return bool(
        _SUPPORTED_BY_FUNDER_WORD.search(text)
        or _SUPPORTED_BY_FUNDER_NAME.search(text)
        or _SUPPORTED_BY_GRANT_ID.search(text)
        or _SUPPORTED_BY_FUNDER_ACRONYM.search(text)
        or _has_funder_hint(text)
    )


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _matching_rows(contents: PaperContents) -> list[PaperSentence]:
    """Whitespace-collapsed copies of the sentence rows (same IDs) to match on.

    Rows still carry raw PDF line breaks here, and a break inside a phrase
    ("supported \\r\\nby") would hide the anchor.
    """
    return [
        replace(sentence, text=_collapse_whitespace(sentence.text))
        for sentence in contents.sentences
    ]


def _sentence_start(text: str, position: int) -> int:
    """Where the sentence holding ``text[position]`` starts."""
    head = max(
        text.rfind(". ", 0, position), text.rfind("? ", 0, position), text.rfind("! ", 0, position)
    )
    return head + 2 if head >= 0 else 0


def _history_shape_at(text: str, position: int) -> bool:
    """A dated "Received ...", "Manuscript received ...; accepted ..." line."""
    if not _HISTORY_SHAPE.match(text, position):
        return False
    before = text[:position].rstrip()
    return bool(
        text[position].isupper()
        or not before
        or before.endswith((";", ","))
        or re.search(r"\b(?:manuscript|article|paper)$", before, re.IGNORECASE)
    )


def _furniture_boundaries(field: str, text: str) -> list[tuple[int, int]]:
    """Publisher-furniture boundaries ``(start, end)`` in one row, in order."""
    candidates: list[tuple[int, int]] = []
    for match in _BOILERPLATE_BOUNDARY.finditer(text):
        if _boundary_is_declaration_text(field, match, text):
            continue
        word = match.group().casefold()
        if word.startswith("correspond"):
            if _CONTACT_FURNITURE.match(text, match.start()):
                candidates.append((match.start(), match.end()))
            continue
        if word.startswith(("received", "accepted")) and not (
            _HISTORY_WORD.match(text, match.start()) or _history_shape_at(text, match.start())
        ):
            continue
        candidates.append((match.start(), match.end()))
    candidates.extend(
        (match.start(), match.end())
        for match in _EXTRA_BOUNDARY.finditer(text)
        # "Grant information:" is the funding label of F1000-family journals.
        if not (field == "funding_statement" and match.group().casefold().startswith("grant"))
    )
    candidates.extend((match.start(), match.end()) for match in _FURNITURE_SHAPE.finditer(text))
    found = set()
    for start, end in candidates:
        sentence = _sentence_start(text, start)
        if (
            not re.search(r"\w", text[sentence:start])
            or _FURNITURE_SHAPE.match(text, start)
            or _history_shape_at(text, start)
        ):
            found.add((start, end))
        elif _LICENCE_SENTENCE.match(text, sentence):
            found.add((sentence, end))
    return sorted(found)


def _opens_other_end_matter(field: str, text: str) -> bool:
    """Whether a row starts end matter other than ``field``'s own statement."""
    return bool(
        _END_MATTER_LABEL.match(text)
        or any(label.match(text) for other, label in _STATEMENT_LABEL.items() if other != field)
    )


def _legacy_follow_stops(field: str, text: str) -> bool:
    """Whether a row after an anchored row is no longer part of its statement."""
    return bool(
        _furniture_boundaries(field, text)
        or _opens_other_end_matter(field, text)
        or _thanks_ends(field, text)
    )


def _thanks_ends(field: str, text: str) -> bool:
    """A thanks sentence ends a statement unless it states the field itself
    ("We also thank the Jacobs Foundation for financial support (grant ...)")."""
    return bool(_THANKS.match(text)) and field not in _legacy_categories_in(text)


def _legacy_section_rows(field: str, rows: list[PaperSentence]) -> list[PaperSentence]:
    """The rows of a statement-typed section that belong to the statement.

    Leading furniture rows (a review watermark, a licence block) are skipped;
    once a row is kept, the copy ends at the next furniture boundary, end-matter
    label or thanks sentence. A row holding the declaration and then furniture
    is kept and clipped when rendered.
    """
    kept: list[PaperSentence] = []
    licence_seen = False
    in_licence = False
    for row in rows:
        text = row.text
        if not text:
            continue
        if in_licence:
            if _LICENCE_CONTINUATION.search(text):
                continue
            # The licence block ends at the first row that does not continue
            # it, even in the same paragraph ("© The Author(s) 2024. This
            # research received no external funding.").
            in_licence = False
        boundaries = _furniture_boundaries(field, text)
        label = _opens_other_end_matter(field, text)
        if not boundaries and not label:
            if kept and _thanks_ends(field, text):
                break
            kept.append(row)
            continue
        if kept:
            break
        if _LICENCE_TEXT.search(text) and (licence_seen or label or boundaries[0][0] == 0):
            # A licence block: drop it and the licence rows that follow it.
            licence_seen = in_licence = True
            continue
        if not label and boundaries[0][0] > 0:
            kept.append(row)
            break
        if label and not boundaries and field in _legacy_categories_in(text):
            # "Acknowledgements This work was supported by ...": the label
            # introduces this field's own anchored sentence.
            kept.append(row)
    return kept


def _text_before_boundary(text: str, start: int, boundary: int) -> str:
    """``text[start:boundary]``, ending at the last full sentence when there is one.

    Boundaries sit at a sentence start or at a furniture shape, so the text
    before one is whole sentences, or a declaration that runs into furniture
    without a full stop ("... no competing interests Copyright: © 2024").
    """
    segment = text[start:boundary]
    cut = max(segment.rfind(". "), segment.rfind("? "), segment.rfind("! "))
    return (segment[: cut + 1] if cut >= 0 else segment).strip()


def _clip_legacy_row(field: str, text: str) -> str:
    """Keep a row's declaration: from its anchor to the next furniture boundary.

    Furniture before the anchor ("Manuscript received ...; This work was
    supported by ...") is cut off from the sentence that holds the anchor.
    """
    boundaries = _furniture_boundaries(field, text)
    if not boundaries:
        return text
    anchors = [
        match.start()
        for pattern in (*_LEGACY_ANCHORS[field], *_ANCHORS[field])
        if (match := pattern.search(text))
    ]
    if not anchors:
        if boundaries[0][0] == 0:
            return text
        return _text_before_boundary(text, 0, boundaries[0][0])
    anchor = min(anchors)
    before = [boundary for boundary in boundaries if boundary[0] < anchor]
    after = [boundary for boundary in boundaries if boundary[0] >= anchor]
    start = 0
    if before:
        floor = before[-1][1]
        if text[anchor : anchor + 1].isupper() or _STATEMENT_LABEL[field].match(text[anchor:]):
            start = anchor
        else:
            segment = text[floor:anchor]
            cut = max(segment.rfind(". "), segment.rfind("; "))
            start = floor + (cut + 2 if cut >= 0 else 0)
    if not after:
        return text[start:].strip()
    return _text_before_boundary(text, start, after[0][0])


def _legacy_coi_declares(text: str) -> bool:
    """Whether a row with a COI anchor declares interests, not just names them."""
    if _LEGACY_COI_LABEL.match(text):
        return True
    windows = [
        text[max(0, match.start() - _COI_CUE_WINDOW) : match.end() + _COI_CUE_WINDOW]
        for pattern in _LEGACY_ANCHORS["coi_statement"]
        for match in pattern.finditer(text)
    ]
    if any(_LEGACY_COI_STRONG_CUE.search(window) for window in windows):
        return True
    if _COI_TOPICAL_PROSE.search(text):
        return False
    return any(_LEGACY_COI_CUE.search(window) for window in windows)


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence of ``text`` that holds ``text[start:end]``."""
    head = max(text.rfind(". ", 0, start), text.rfind("? ", 0, start), text.rfind("! ", 0, start))
    tail = min(
        (i for i in (text.find(". ", end), text.find("? ", end), text.find("! ", end)) if i >= 0),
        default=len(text),
    )
    return text[head + 2 if head >= 0 else 0 : tail + 1]


def _label_noun_only_in_prose(field: str, text: str) -> bool:
    """A row whose only anchor is a label noun used inside running prose."""
    noun = _LABEL_NOUN_ANCHORS.get(field)
    if noun is None or _STATEMENT_LABEL[field].match(text):
        return False
    if any(
        pattern.search(text)
        for pattern in _LEGACY_ANCHORS[field]
        if pattern.pattern not in _LABEL_NOUN_PATTERNS
    ):
        return False
    for match in noun.finditer(text):
        sentence = _sentence_around(text, match.start(), match.end())
        if sentence.lower().startswith(match.group().lower()):
            return False
        if field == "data_availability" and re.match(
            r"\s*(?:statement|section|:)", text[match.end() :], re.IGNORECASE
        ):
            return False
        if field == "funding_statement" and (
            _FUNDING_RECIPIENT.search(sentence)
            or _SUPPORTED_BY_GRANT_ID.search(sentence)
            or _NAMED_FUNDER_BODY.search(sentence)
            or _SUPPORTED_BY_FUNDER_ACRONYM.search(sentence)
        ):
            return False
    return True


def _legacy_ethics_declares(text: str) -> bool:
    """Whether an ethics-anchored row states an approval or consent."""
    if _STATEMENT_LABEL["ethics_statement"].match(text):
        return True
    matched = [pattern for pattern in _LEGACY_ANCHORS["ethics_statement"] if pattern.search(text)]
    if [pattern.pattern for pattern in matched] != [r"\binformed consent"]:
        return True
    return bool(_LEGACY_CONSENT_ACTION.search(text))


def _legacy_anchor_row_passes(field: str, text: str) -> bool:
    """Whether a row holding a legacy anchor of ``field`` passes the prose guards."""
    if field == "coi_statement" and not _legacy_coi_declares(text):
        return False
    if field == "ethics_statement" and not _legacy_ethics_declares(text):
        return False
    if _label_noun_only_in_prose(field, text):
        return False
    if field == "funding_statement" and not (
        any(pattern.search(text) for pattern in _LEGACY_FUNDING_STRONG_ANCHORS)
        # A "Funding:" label already says what the sentence is.
        or _STATEMENT_LABEL[field].match(text)
    ):
        ambiguous = _LEGACY_FUNDING_AMBIGUOUS_ANCHOR.search(text)
        if ambiguous is None or not _supported_by_names_funder(text[ambiguous.end() :]):
            return False
    return True


def _legacy_body_copy_passes(field: str, rows: list[PaperSentence]) -> bool:
    """Whether a body section under a strong heading reads as the statement.

    It is compact, and every row holding an anchor of the field passes the
    lexical prose guards ("Students who received financial support from
    parents ..." under "Financial support" does not).
    """
    if (
        len({row.paragraph_id for row in rows}) > _MAX_COMPACT_PARAGRAPHS
        or len(" ".join(row.text for row in rows)) > _MAX_COMPACT_CHARS
    ):
        return False
    return all(
        _legacy_anchor_row_passes(field, row.text)
        for row in rows
        if any(pattern.search(row.text) for pattern in _LEGACY_ANCHORS[field])
    )


def _legacy_capture_at(
    field: str,
    rows: list[PaperSentence],
    index: int,
    section_by_id: dict[int, PaperSection],
) -> list[PaperSentence]:
    """The lexical capture anchored at ``rows[index]``, or ``[]``."""
    sentence = rows[index]
    section = section_by_id.get(sentence.section_id)
    in_references = section is not None and section.section_type == CanonicalSection.REFERENCES
    if in_references:
        label = _REFERENCE_STATEMENT_LABEL[field].match(sentence.text)
        if not label or not _REFERENCE_STATEMENT_CUE[field].search(sentence.text[label.end() :]):
            return []
    if not any(pattern.search(sentence.text) for pattern in _LEGACY_ANCHORS[field]):
        return []
    if not _legacy_anchor_row_passes(field, sentence.text):
        return []
    strong_funding_anchor = field == "funding_statement" and any(
        pattern.search(sentence.text) for pattern in _LEGACY_FUNDING_STRONG_ANCHORS
    )

    captured = [sentence]
    for following in rows[index + 1 : index + 3]:
        if in_references or following.section_id != sentence.section_id:
            break
        if _legacy_categories_in(following.text) - {field}:
            break
        if _legacy_follow_stops(field, following.text):
            break
        captured.append(following)

    joined = " ".join(row.text.strip() for row in captured if row.text.strip())
    if (
        field == "funding_statement"
        and strong_funding_anchor
        and not _legacy_has_funder_hint(joined)
    ):
        return []
    return captured if joined else []


def _legacy_lexical_rows(
    field: str,
    rows: list[PaperSentence],
    section_by_id: dict[int, PaperSection],
) -> list[PaperSentence]:
    """The first label-led lexical capture, else the first in reading order.

    A capture may run on into the next paragraph of the same section.
    """
    label = _STATEMENT_LABEL[field]
    for index, sentence in enumerate(rows):
        if label.match(sentence.text) and (
            captured := _legacy_capture_at(field, rows, index, section_by_id)
        ):
            return captured
    for index in range(len(rows)):
        if captured := _legacy_capture_at(field, rows, index, section_by_id):
            return captured
    return []


def _heading_declaration_tail(field: str, header: str | None) -> str | None:
    """The declaration sentence a heading region carries after its label, if any."""
    header = _collapse_whitespace(header or "")
    for match in _HEADING_TAIL_START.finditer(header):
        label, tail = header[: match.start()], header[match.end() :]
        if len(label) > _MAX_HEADING_LABEL_CHARS:
            return None
        if re.search(r"[.!?]", label) or not _FIELD_HEADING_WORD[field].search(label):
            continue
        if not tail.endswith(".") or not any(
            pattern.search(tail) for pattern in _LEGACY_ANCHORS[field]
        ):
            return None
        if field == "coi_statement" and not _legacy_coi_declares(tail):
            return None
        if field == "ethics_statement" and not _legacy_ethics_declares(tail):
            return None
        return tail
    return None


def _heading_label(heading: str | None) -> str:
    """A heading as a statement-part label: no numbering, fused tail or colon."""
    label = _collapse_whitespace(heading or "")
    for field in _FIELDS:
        tail = _heading_declaration_tail(field, label)
        if tail:
            label = label[: -len(tail)]
            break
    label = _HEADING_NUMBER.sub("", label)
    return label.rstrip(" :.-–—")


def _legacy_section_qualifies(field: str, section: PaperSection) -> bool:
    """Whether a section is copied as ``field``'s statement.

    Its type is the field's statement type, or the heading is one of the
    field's strong headings and the section has no other statement type
    ("Ethics approval and consent to participate" typed as an endnote). A body
    section under such a heading must also pass ``_legacy_body_copy_passes``.
    """
    if section.section_type == _FIELD_SECTION_TYPES[field]:
        return not _legacy_section_is_other_matter(field, section)
    if section.section_type in _SECTION_TYPE_FIELDS or section.section_type in {
        CanonicalSection.REFERENCES,
        CanonicalSection.TITLE,
        CanonicalSection.ABSTRACT,
    }:
        return False
    heading_field, _normalized, strong, _consent = _heading_field(section)
    return heading_field == field and strong


def _legacy_section_is_other_matter(field: str, section: PaperSection) -> bool:
    """A statement-typed section whose heading says it is something else."""
    heading = section.header or ""
    if field == "funding_statement" and _ROLE_OF_FUNDER_HEADING.search(heading):
        return True
    if field == "ethics_statement" and _TOPICAL_ETHICS_HEADING.search(heading):
        return True
    return bool(_AI_USE_HEADING.search(heading) and not _FIELD_HEADING_WORD[field].search(heading))


def _legacy_section_is_long(section: PaperSection, rows: list[PaperSentence]) -> bool:
    """A long copy of a section whose type may be wrong (a thesis chapter)."""
    return (
        section.classification_source in _AMBIGUOUS_CLASSIFICATION_SOURCES
        and len(" ".join(row.text for row in rows)) > _MAX_AMBIGUOUS_SECTION_CHARS
    )


def _leading_paragraph_rows(rows: list[PaperSentence]) -> list[PaperSentence]:
    """The rows of the first ``_MAX_COMPACT_PARAGRAPHS`` paragraphs."""
    paragraph_ids: list[int] = []
    kept = []
    for row in rows:
        if row.paragraph_id not in paragraph_ids:
            if len(paragraph_ids) == _MAX_COMPACT_PARAGRAPHS:
                break
            paragraph_ids.append(row.paragraph_id)
        kept.append(row)
    return kept


def _build_legacy_snapshot_candidates(
    sentences: list[PaperSentence],
    sections: list[PaperSection],
    section_by_id: dict[int, PaperSection],
) -> list[IntegrityStatementCandidate]:
    """Compatibility ownership: bounded section copies, else a lexical capture.

    ``sentences`` are the whitespace-collapsed matching rows.
    """
    candidates: list[IntegrityStatementCandidate] = []
    linear_rows = [sentence for sentence in sentences if not sentence.is_display_formula]
    rows_by_section: dict[int, list[PaperSentence]] = {}
    for sentence in linear_rows:
        rows_by_section.setdefault(sentence.section_id, []).append(sentence)
    for field in _FIELDS:
        canonical_start = len(candidates)
        long_copies: list[tuple[PaperSection, list[PaperSentence], str | None]] = []
        for section in sections:
            if not _legacy_section_qualifies(field, section):
                continue
            rows = _legacy_section_rows(field, rows_by_section.get(section.section_id, []))
            heading_tail = _heading_declaration_tail(field, section.header)
            if not rows and not heading_tail:
                continue
            if (
                section.section_type != _FIELD_SECTION_TYPES[field]
                and section.section_type not in _STRONG_HEADING_COPY_TYPES
                and not _legacy_body_copy_passes(field, rows)
            ):
                continue
            if _legacy_section_is_long(section, rows):
                long_copies.append((section, rows, heading_tail))
                continue
            candidates.append(_legacy_section_candidate(field, section, rows, heading_tail))
        if len(candidates) > canonical_start:
            continue

        rows = _legacy_lexical_rows(field, linear_rows, section_by_id)
        if not rows:
            # No statement sentence in a long, possibly mistyped section: keep
            # its opening paragraphs rather than nothing.
            if long_copies:
                section, rows, heading_tail = long_copies[0]
                candidates.append(
                    _legacy_section_candidate(
                        field, section, _leading_paragraph_rows(rows), heading_tail
                    )
                )
            continue
        section = section_by_id.get(rows[0].section_id)
        text_ids, paragraph_ids, pages = _candidate_location(rows)
        candidates.append(
            IntegrityStatementCandidate(
                field=field,
                method="legacy_lexical_capture",
                heading=section.header if section is not None else None,
                section_ids=(rows[0].section_id,),
                text_ids=text_ids,
                paragraph_ids=paragraph_ids,
                pages=pages,
                classification_source=(
                    section.classification_source if section is not None else None
                ),
                classification_score=(
                    section.classification_score if section is not None else None
                ),
                reason_flags=("legacy_snapshot", "lexical_anchor"),
                accepted=True,
            )
        )
    return candidates


def _legacy_section_candidate(
    field: str,
    section: PaperSection,
    rows: list[PaperSentence],
    heading_tail: str | None,
) -> IntegrityStatementCandidate:
    text_ids, paragraph_ids, pages = _candidate_location(rows)
    return IntegrityStatementCandidate(
        field=field,
        method="legacy_section_copy",
        heading=section.header,
        section_ids=(section.section_id,),
        text_ids=text_ids,
        paragraph_ids=paragraph_ids,
        pages=pages,
        classification_source=section.classification_source,
        classification_score=section.classification_score,
        reason_flags=(
            "legacy_snapshot",
            "canonical_type"
            if section.section_type == _FIELD_SECTION_TYPES[field]
            else "strong_heading",
            *(("heading_tail",) if heading_tail else ()),
        ),
        accepted=True,
    )


def _candidate_location(
    rows: list[PaperSentence],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(sentence.text_id for sentence in rows),
        _ordered_unique(sentence.paragraph_id for sentence in rows),
        _ordered_unique(
            sentence.page_number for sentence in rows if sentence.page_number is not None
        ),
    )


def _strip_outer_punctuation(text: str) -> str:
    start = 0
    end = len(text)
    while start < end and (
        text[start].isspace() or unicodedata.category(text[start]).startswith("P")
    ):
        start += 1
    while end > start and (
        text[end - 1].isspace() or unicodedata.category(text[end - 1]).startswith("P")
    ):
        end -= 1
    return text[start:end]


def _heading_field(section: PaperSection) -> tuple[str | None, str, bool, bool]:
    normalized = _strip_outer_punctuation(normalize_text(section.header))
    publication_consent = bool(_PUBLICATION_CONSENT.fullmatch(normalized))
    for field in _FIELDS:
        if normalized in _STRONG_HEADINGS[field]:
            return field, normalized, True, publication_consent
        if normalized in _GENERIC_HEADINGS[field]:
            return field, normalized, False, publication_consent
    if "funding" in normalized or "financial support" in normalized:
        return "funding_statement", normalized, False, publication_consent
    if "conflict" in normalized or "competing interest" in normalized:
        return "coi_statement", normalized, False, publication_consent
    if "ethic" in normalized or publication_consent:
        return "ethics_statement", normalized, False, publication_consent
    if any(
        token in normalized
        for token in ("data availability", "code availability", "materials availability")
    ):
        return "data_availability", normalized, False, publication_consent
    return None, normalized, False, publication_consent


def _declaration_predicate(
    field: str,
    text: str,
    *,
    author_aliases: frozenset[str],
    source_text: str | None = None,
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if field != "funding_statement" and _is_negative_declaration(field, stripped):
        return True
    return _has_assertive_declaration(
        field,
        stripped,
        author_aliases=author_aliases,
        source_text=source_text,
    )


def _is_negative_declaration(field: str, text: str) -> bool:
    stripped = text.strip()
    return bool(
        _UNLABELED_NEGATIVE_DECLARATION.fullmatch(stripped)
        or _FIELD_LABELED_NEGATIVE_DECLARATION[field].fullmatch(stripped)
        or _FIELD_NEGATIVE_DECLARATION[field].search(stripped)
    )


def _bounded_section_rows(field: str, rows: list[PaperSentence]) -> list[PaperSentence]:
    bounded: list[PaperSentence] = []
    for sentence in rows:
        other_category = _categories_in(sentence.text) - {field}
        if bounded and (_boilerplate_boundary_for_field(field, sentence.text) or other_category):
            break
        bounded.append(sentence)
        if other_category or _boilerplate_boundary_for_field(field, sentence.text):
            break
    return bounded


def _build_section_candidate_for_field(
    section: PaperSection,
    rows: list[PaperSentence],
    *,
    field: str,
    heading_field: str | None,
    strong_heading: bool,
    publication_consent: bool,
    author_aliases: frozenset[str],
) -> IntegrityStatementCandidate:
    canonical_field = _SECTION_TYPE_FIELDS.get(section.section_type)

    bounded_rows = _bounded_section_rows(field, rows)
    text = " ".join(sentence.text.strip() for sentence in bounded_rows if sentence.text.strip())
    text_ids, paragraph_ids, pages = _candidate_location(bounded_rows)
    compact = len(text) <= _MAX_COMPACT_CHARS and len(paragraph_ids) <= _MAX_COMPACT_PARAGRAPHS
    predicate = _declaration_predicate(field, text, author_aliases=author_aliases)
    author_grounding_failed = bool(
        field == "funding_statement"
        and _has_unresolved_author_funding_declaration(text, author_aliases)
    )
    explicit_negative = _is_negative_declaration(field, text)
    category_boundary_clipped = any(_categories_in(row.text) - {field} for row in bounded_rows)
    boilerplate_boundary_clipped = any(
        _boilerplate_boundary_for_field(field, row.text) for row in bounded_rows
    )
    topical_heading = field == heading_field == "ethics_statement" and bool(
        _TOPICAL_ETHICS_HEADING.search(section.header)
    )
    ambiguous_source = section.classification_source in _AMBIGUOUS_CLASSIFICATION_SOURCES

    accepted = bool(
        text_ids
        and not (publication_consent and field == heading_field)
        and not topical_heading
        and not author_grounding_failed
        and not (explicit_negative and not predicate)
        and not (boilerplate_boundary_clipped and not predicate)
        and not (category_boundary_clipped and not predicate)
    )
    if accepted:
        if ambiguous_source:
            accepted = predicate
        elif strong_heading:
            accepted = compact or predicate
        else:
            accepted = predicate and compact

    flags = []
    if canonical_field == field:
        flags.append("canonical_type")
    if heading_field == field:
        flags.append("heading_match")
    flags.append("strong_heading" if strong_heading else "ambiguous_heading")
    if compact:
        flags.append("compact")
    if predicate:
        flags.append("declaration_predicate")
    if explicit_negative:
        flags.append("explicit_negative")
    if author_grounding_failed:
        flags.append("author_grounding_failed")
    if category_boundary_clipped:
        flags.append("category_boundary_clipped")
    if boilerplate_boundary_clipped:
        flags.append("boilerplate_boundary_clipped")
    if topical_heading:
        flags.append("topical_heading")
    if publication_consent and field == heading_field:
        flags.append("publication_consent")
    if ambiguous_source:
        flags.append("ambiguous_classification_source")
    if not accepted:
        flags.append("rejected")
    return IntegrityStatementCandidate(
        field=field,
        method="trusted_section" if accepted else "classified_section",
        heading=section.header,
        section_ids=(section.section_id,),
        text_ids=text_ids,
        paragraph_ids=paragraph_ids,
        pages=pages,
        classification_source=section.classification_source,
        classification_score=section.classification_score,
        reason_flags=tuple(flags),
        accepted=accepted,
    )


def _build_section_candidates(
    section: PaperSection,
    rows: list[PaperSentence],
    *,
    author_aliases: frozenset[str],
) -> list[IntegrityStatementCandidate]:
    heading_field, _normalized_heading, strong_heading, publication_consent = _heading_field(
        section
    )
    canonical_field = _SECTION_TYPE_FIELDS.get(section.section_type)
    fields = _ordered_unique(
        field for field in (heading_field, canonical_field) if field is not None
    )
    return [
        _build_section_candidate_for_field(
            section,
            rows,
            field=field,
            heading_field=heading_field,
            strong_heading=strong_heading and field == heading_field,
            publication_consent=publication_consent,
            author_aliases=author_aliases,
        )
        for field in fields
    ]


def _funding_anchor_qualifies(text: str) -> bool:
    if any(pattern.search(text) for pattern in _FUNDING_STRONG_ANCHORS):
        return _has_funder_hint(text) or _has_funding_negative_declaration(text)
    ambiguous = _FUNDING_AMBIGUOUS_ANCHOR.search(text)
    return bool(ambiguous and _has_funder_hint(text[ambiguous.end() :]))


def _build_lexical_candidates(
    rows: list[PaperSentence],
    section_by_id: dict[int, PaperSection],
    *,
    author_aliases: frozenset[str],
) -> list[IntegrityStatementCandidate]:
    candidates: list[IntegrityStatementCandidate] = []
    for index, sentence in enumerate(rows):
        section = section_by_id.get(sentence.section_id)
        if section is not None and section.section_type == CanonicalSection.REFERENCES:
            continue
        for field in _ANCHORS:
            if not _has_field_anchor(field, sentence.text):
                continue
            bounded_rows = [sentence]
            for following in rows[index + 1 : index + 3]:
                if (
                    following.section_id != sentence.section_id
                    or following.paragraph_id != sentence.paragraph_id
                    or _boilerplate_boundary_for_field(field, following.text)
                    or _categories_in(following.text) - {field}
                ):
                    break
                bounded_rows.append(following)
            fragments = [_bounded_sentence_for_field(field, row.text) for row in bounded_rows]
            text = " ".join(fragment for fragment in fragments if fragment).strip()
            bare_label = sentence.text.strip().casefold() in _BARE_CATEGORY_LABELS[field]
            source_rows = bounded_rows[1:] if bare_label else bounded_rows
            source_text = " ".join(row.text.strip() for row in source_rows if row.text.strip())
            predicate = _declaration_predicate(
                field,
                text,
                author_aliases=author_aliases,
                source_text=source_text,
            )
            if field == "funding_statement" and not bare_label:
                predicate = predicate and _funding_anchor_qualifies(sentence.text)
            text_ids, paragraph_ids, pages = _candidate_location(bounded_rows)
            flags = ["lexical_anchor", "same_paragraph"]
            if any(_categories_in(row.text) - {field} for row in bounded_rows):
                flags.append("category_boundary_clipped")
            if any(_boilerplate_boundary_for_field(field, row.text) for row in bounded_rows):
                flags.append("boilerplate_boundary_clipped")
            if predicate:
                flags.append("declaration_predicate")
            else:
                flags.append("rejected")
            candidates.append(
                IntegrityStatementCandidate(
                    field=field,
                    method="anchored_paragraph",
                    heading=section.header if section is not None else None,
                    section_ids=(sentence.section_id,),
                    text_ids=text_ids,
                    paragraph_ids=paragraph_ids,
                    pages=pages,
                    classification_source=(
                        section.classification_source if section is not None else None
                    ),
                    classification_score=(
                        section.classification_score if section is not None else None
                    ),
                    reason_flags=tuple(flags),
                    accepted=bool(text_ids and predicate),
                )
            )
    return candidates


def _render_candidate(contents: PaperContents, candidate: IntegrityStatementCandidate) -> str:
    by_id = {sentence.text_id: sentence for sentence in contents.sentences}
    parts = []
    for text_id in candidate.text_ids:
        sentence = by_id.get(text_id)
        if sentence is None or sentence.is_display_formula:
            continue
        text = _bounded_sentence_for_field(candidate.field, sentence.text.strip())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _render_legacy_candidate(
    contents: PaperContents, candidate: IntegrityStatementCandidate
) -> str:
    """Render a compatibility value from the current sentence text.

    Each row is clipped at publisher furniture around the field's anchor, and
    rendering stops at the first clipped row.
    """
    by_id = {sentence.text_id: sentence for sentence in contents.sentences}
    parts = []
    if "heading_tail" in candidate.reason_flags:
        section = next(
            (s for s in contents.sections if s.section_id in candidate.section_ids), None
        )
        tail = _heading_declaration_tail(
            candidate.field, section.header if section is not None else candidate.heading
        )
        if tail:
            parts.append(tail)
    for text_id in candidate.text_ids:
        sentence = by_id.get(text_id)
        if sentence is None or sentence.is_display_formula:
            continue
        text = _collapse_whitespace(sentence.text)
        clipped = _clip_legacy_row(candidate.field, text)
        if clipped:
            parts.append(clipped)
        if clipped != text:
            break
    return " ".join(parts)


def _with_heading(candidate: IntegrityStatementCandidate, text: str) -> str:
    """``text`` led by its section heading, for one part of a joined statement."""
    label = _heading_label(candidate.heading)
    if not label or text.casefold().startswith(label.casefold()):
        return text
    return f"{label}: {text}"


def _render_indices(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    indices: tuple[int, ...],
) -> str | None:
    parts = [
        (resolution.candidates[index], rendered)
        for index in indices
        if (
            rendered := (
                _render_legacy_candidate(contents, resolution.candidates[index])
                if resolution.candidates[index].method.startswith("legacy_")
                else _render_candidate(contents, resolution.candidates[index])
            )
        )
    ]
    if len(parts) > 1:
        # Joined sections keep their headings, or "Not applicable." parts
        # lose their meaning.
        return "\n\n".join(_with_heading(candidate, text) for candidate, text in parts)
    return parts[0][1] if parts else None


def render_integrity_statement(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    field: str,
    *,
    effective: bool = True,
) -> str | None:
    """Render a resolved field from current sentence text (after late cleaning)."""
    if resolution.mode != "active" or not effective:
        return _render_indices(contents, resolution, resolution.legacy_indices(field))
    return _render_indices(
        contents, resolution, resolution.candidate_indices(field, effective=effective)
    )


def render_selected_integrity_statement(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    field: str,
) -> str | None:
    """Render the bounded selected candidate independent of rollout mode."""
    return _render_indices(contents, resolution, resolution.selected_indices(field))


def _legacy_indices(
    candidates: list[IntegrityStatementCandidate],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            field,
            tuple(
                index
                for index, candidate in enumerate(candidates)
                if candidate.field == field and candidate.method.startswith("legacy_")
            ),
        )
        for field in _FIELDS
    )


def _active_indices(
    contents: PaperContents,
    candidates: list[IntegrityStatementCandidate],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    section_position = {
        section.section_id: index for index, section in enumerate(contents.sections)
    }
    selected: list[tuple[str, tuple[int, ...]]] = []
    for field in _FIELDS:
        section_candidates = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.field == field
            and candidate.accepted
            and candidate.method == "trusted_section"
        ]
        if section_candidates:
            first = min(
                section_candidates,
                key=lambda index: (
                    0 if "strong_heading" in candidates[index].reason_flags else 1,
                    section_position.get(candidates[index].section_ids[0], 10**9),
                ),
            )
            chosen = [first]
            position = section_position.get(candidates[first].section_ids[0], -1)
            by_position = {
                section_position.get(candidates[index].section_ids[0], -1): index
                for index in section_candidates
            }
            previous_position = position - 1
            while previous_position in by_position:
                chosen.insert(0, by_position[previous_position])
                previous_position -= 1
            next_position = position + 1
            while next_position in by_position:
                chosen.append(by_position[next_position])
                next_position += 1
            selected.append((field, tuple(chosen)))
            continue
        lexical = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if candidate.field == field
                and candidate.method == "anchored_paragraph"
                and candidate.accepted
            ),
            None,
        )
        selected.append((field, () if lexical is None else (lexical,)))
    return tuple(selected)


def _comparison_issues(
    contents: PaperContents,
    candidates: tuple[IntegrityStatementCandidate, ...],
    legacy: tuple[tuple[str, tuple[int, ...]], ...],
    selected: tuple[tuple[str, tuple[int, ...]], ...],
    legacy_snapshots: tuple[tuple[str, str | None], ...],
) -> tuple[ValidationIssue, ...]:
    temporary = IntegrityStatementResolution(
        mode="shadow",
        candidates=candidates,
        legacy_candidate_indices=legacy,
        selected_candidate_indices=selected,
        legacy_statement_snapshots=legacy_snapshots,
        issues=(),
    )
    issues = []
    for field in _FIELDS:
        legacy_indices = dict(legacy).get(field, ())
        selected_indices = dict(selected).get(field, ())
        legacy_text = dict(legacy_snapshots).get(field)
        selected_text = _render_indices(contents, temporary, dict(selected).get(field, ()))
        if _comparison_key(candidates, legacy_indices, legacy_text) == _comparison_key(
            candidates, selected_indices, selected_text
        ):
            continue
        implicated = _ordered_unique((*legacy_indices, *selected_indices))
        section_ids = _ordered_unique(
            section_id for index in implicated for section_id in candidates[index].section_ids
        )
        text_ids = _ordered_unique(
            text_id for index in implicated for text_id in candidates[index].text_ids
        )
        evidence_ids = (
            field,
            *(f"section:{section_id}" for section_id in section_ids),
            *(f"text:{text_id}" for text_id in text_ids),
        )[:20]
        issues.append(
            ValidationIssue(
                code="VAL_STATEMENT_SUSPECT",
                severity=IssueSeverity.WARNING,
                message=f"Legacy and bounded integrity-statement ownership differ for {field}",
                origin_stage="post_parse",
                evidence_ids=evidence_ids,
            )
        )
    return tuple(issues)


def _comparison_key(
    candidates: tuple[IntegrityStatementCandidate, ...],
    indices: tuple[int, ...],
    text: str | None,
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    from bibr.input.consolidate_text import clean_text_content_late

    normalized = " ".join(clean_text_content_late(text or "").casefold().split())
    return (
        normalized,
        _ordered_unique(
            section_id for index in indices for section_id in candidates[index].section_ids
        ),
        _ordered_unique(text_id for index in indices for text_id in candidates[index].text_ids),
        _ordered_unique(
            paragraph_id for index in indices for paragraph_id in candidates[index].paragraph_ids
        ),
        _ordered_unique(page for index in indices for page in candidates[index].pages),
    )


def resolve_integrity_statements(
    contents: PaperContents,
    *,
    mode: IntegrityStatementMode,
    author_names: tuple[tuple[str, str], ...] = (),
) -> IntegrityStatementResolution:
    """Build section and lexical candidates, then select compatibility/safe IDs."""
    if mode not in {"legacy", "shadow", "active"}:
        raise ValueError(f"unsupported integrity statement mode: {mode}")
    author_aliases = _author_aliases(author_names)
    rows_by_section = _section_rows(contents)
    section_by_id = {section.section_id: section for section in contents.sections}
    candidates = _build_legacy_snapshot_candidates(
        _matching_rows(contents), contents.sections, section_by_id
    )
    for section in contents.sections:
        candidates.extend(
            _build_section_candidates(
                section,
                rows_by_section.get(section.section_id, []),
                author_aliases=author_aliases,
            )
        )
    linear_rows = [sentence for sentence in contents.sentences if not sentence.is_display_formula]
    candidates.extend(
        _build_lexical_candidates(
            linear_rows,
            section_by_id,
            author_aliases=author_aliases,
        )
    )
    legacy = _legacy_indices(candidates)
    selected = _active_indices(contents, candidates)
    frozen_candidates = tuple(candidates)
    snapshot_resolution = IntegrityStatementResolution(
        mode="legacy",
        candidates=frozen_candidates,
        legacy_candidate_indices=legacy,
        selected_candidate_indices=selected,
        legacy_statement_snapshots=(),
        issues=(),
    )
    legacy_snapshots = tuple(
        (
            field,
            _render_indices(contents, snapshot_resolution, dict(legacy).get(field, ())),
        )
        for field in _FIELDS
    )
    issues = (
        _comparison_issues(contents, frozen_candidates, legacy, selected, legacy_snapshots)
        if mode == "shadow"
        else ()
    )
    return IntegrityStatementResolution(
        mode=mode,
        candidates=frozen_candidates,
        legacy_candidate_indices=legacy,
        selected_candidate_indices=selected,
        legacy_statement_snapshots=legacy_snapshots,
        issues=issues,
    )


def apply_integrity_resolution(
    contents: PaperContents,
    metadata: PaperMetadata,
    resolution: IntegrityStatementResolution,
) -> None:
    """Decide the statement scalars: the mode-effective statement, never replacing native values.

    A statement of an input that declares its metadata (JATS, HTML) keeps the
    ``native`` source, since its text is the input's own.
    """
    from bibr.extract.field_decisions import FieldCandidate, apply_decision, decide_statement

    native_metadata = contents.preparsed_metadata is not None
    for field in _FIELDS:
        current = getattr(metadata, field)
        selected = resolution.candidate_indices(field)
        incumbent = (
            FieldCandidate(field, "native" if native_metadata else None, current)
            if current is not None
            else None
        )
        if current is not None and (native_metadata or resolution.mode != "active" or not selected):
            apply_decision(metadata, decide_statement(field, incumbent, None, keep_incumbent=True))
            continue
        rendered = render_integrity_statement(contents, resolution, field)
        lexical = bool(
            {resolution.candidates[index].method for index in selected}
            & {"legacy_lexical_capture", "anchored_paragraph"}
        )
        source = (
            "native" if native_metadata else "lexical_anchor" if lexical else "integrity_statement"
        )
        apply_decision(
            metadata,
            decide_statement(
                field,
                incumbent,
                FieldCandidate(field, source, rendered),
                keep_incumbent=False,
            ),
        )
        if rendered is None:
            continue
        if lexical:
            warning = lexical_fallback_warning(field)
            if warning not in contents.processing_warnings:
                contents.processing_warnings.append(warning)
