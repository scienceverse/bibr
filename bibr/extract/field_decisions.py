"""One decision point, with a receipt, for each metadata field.

A field's value used to be written by whichever step ran last: the model's
answer, a grounding repair, a fallback that fills an empty field, a guard that
blanks one. Producers now propose :class:`FieldCandidate` values instead, one
rule per field picks among them, and :func:`apply_decision` — the only code
that assigns these ``PaperMetadata`` fields (:data:`FIELD_ATTRIBUTES`) once
the record exists — writes the chosen value and keeps the
:class:`FieldDecision` as the field's receipt: every candidate considered, the
one used, and the rule that decided. The DOI is the identity stage's, and the
fields a record is built with (volume, pages, identifiers, ...) have one
producer each.

A field is decided where its last candidate becomes known, at most once per
paper (a run without an LLM decides no structured funding or affiliations):

* the core extractor decides the fields only it produces — authors, the
  publication date, journal, publisher and paper type — and proposes the
  title, abstract and keywords;
* post-parse decides the title, abstract and keywords after its fallbacks,
  and the fields of an input that declares its own metadata (JATS, HTML) or a
  run without an LLM;
* the integrity steps decide the statements, structured funding and parsed
  affiliations.

The receipts ride the metadata record (:class:`FieldDecisions`) and end up on
``Paper.field_decisions``; the export reads each field's ``source`` and
``rule`` from them.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from bibr.validation import ValidationIssue

logger = logging.getLogger(__name__)


class Classification(NamedTuple):
    """The paper-type decision's value: the type and the OECD fields decided with it."""

    paper_type: str = ""
    paper_type_confidence: float | None = None
    oecd_l1: str = ""
    oecd_l2: str = ""
    oecd_confidence: float | None = None


# Decided field -> the ``PaperMetadata`` attributes its decision writes. The
# paper-type decision writes the whole classification, since a correction
# notice blanks the OECD fields together with the type.
FIELD_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "author": ("authors",),
    "abstract": ("abstract",),
    "keywords": ("keywords",),
    "published": ("published",),
    "journal": ("journal",),
    "publisher": ("publisher",),
    "paper_type": Classification._fields,
    "funding_statement": ("funding_statement",),
    "coi_statement": ("coi_statement",),
    "ethics_statement": ("ethics_statement",),
    "data_availability": ("data_availability",),
    "funding": ("funding",),
    "affiliations": ("affiliations",),
}
DECIDED_ATTRIBUTES = frozenset(
    attribute for attributes in FIELD_ATTRIBUTES.values() for attribute in attributes
)


@dataclass(frozen=True)
class FieldCandidate:
    """One proposed value for a field, and where it came from."""

    field: str
    # The producing step; None for a value whose provenance is unknown.
    source: str | None
    value: Any
    evidence_ids: tuple[str, ...] = ()
    # Repairs applied on the way here: "regrounded", "subtitle_folded", ...
    transforms: tuple[str, ...] = ()
    # A guard's reason this value must not be used, such as a correction notice.
    veto: str | None = None
    # Issues the pipeline records when this candidate is the one used.
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """What a decision made of one candidate."""

    candidate: FieldCandidate
    accepted: bool
    reason: str


@dataclass(frozen=True)
class FieldDecision:
    """A field's receipt: the value written, the candidate it came from and why."""

    field: str
    value: Any
    selected: FieldCandidate | None
    rule: str
    considered: tuple[Verdict, ...] = ()
    # The step responsible for the field when no candidate was used; a failed
    # call is reported against it.
    producer: str | None = None
    # The gating inputs the rule was given (scope, abstention, notice, ...).
    flags: tuple[tuple[str, Any], ...] = ()

    @property
    def source(self) -> str | None:
        return self.selected.source if self.selected is not None else self.producer

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return self.selected.issues if self.selected is not None else ()


@dataclass
class FieldDecisions:
    """One paper's proposals and decisions, keyed by field."""

    _proposals: dict[str, FieldCandidate] = field(default_factory=dict)
    _decisions: dict[str, FieldDecision] = field(default_factory=dict)
    # Decisions a later decision of the same field replaced. The pipeline
    # decides each field once, so this stays empty there.
    superseded: list[FieldDecision] = field(default_factory=list)

    def propose(self, candidate: FieldCandidate) -> None:
        """Hand a candidate to the decision made later in the pipeline."""
        self._proposals[candidate.field] = candidate

    def proposal(self, name: str) -> FieldCandidate | None:
        return self._proposals.get(name)

    def record(self, decision: FieldDecision) -> None:
        previous = self._decisions.get(decision.field)
        if previous is not None:
            # The pipeline decides each field once; a second decision replaces
            # the written value, so the newer receipt is kept.
            logger.warning(
                "Field %r decided twice (rule %r, then %r)",
                decision.field,
                previous.rule,
                decision.rule,
            )
            self.superseded.append(previous)
        self._decisions[decision.field] = decision

    def copy(self) -> FieldDecisions:
        """An independent ledger with the same proposals and receipts."""
        return FieldDecisions(dict(self._proposals), dict(self._decisions), list(self.superseded))

    def forget_attributes(self, attributes: Iterable[str]) -> None:
        """Drop the receipts of the fields whose attributes were rewritten outside a decision."""
        changed = set(attributes)
        for name, field_attributes in FIELD_ATTRIBUTES.items():
            if changed.intersection(field_attributes):
                self._decisions.pop(name, None)
                self._proposals.pop(name, None)

    def get(self, name: str) -> FieldDecision | None:
        return self._decisions.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._decisions

    def __iter__(self) -> Iterator[FieldDecision]:
        return iter(self._decisions.values())

    def sources(self) -> dict[str, str]:
        return {
            name: decision.source
            for name, decision in self._decisions.items()
            if decision.source is not None
        }

    def rules(self) -> dict[str, str]:
        return {name: decision.rule for name, decision in self._decisions.items()}


def field_decisions_of(metadata: Any) -> FieldDecisions | None:
    """The ledger a metadata record carries, or None for a test double."""
    ledger = getattr(metadata, "_field_decisions", None)
    return ledger if isinstance(ledger, FieldDecisions) else None


def apply_decision(metadata: Any, decision: FieldDecision) -> None:
    """Write *decision*'s value to *metadata* and keep the receipt.

    The only writer of the decided ``PaperMetadata`` fields after construction
    (``tests/extract/test_field_decisions.py`` scans the package for others).
    """
    attributes = FIELD_ATTRIBUTES[decision.field]
    values = decision.value if len(attributes) > 1 else (decision.value,)
    for attribute, value in zip(attributes, values, strict=True):
        setattr(metadata, attribute, value)
    ledger = field_decisions_of(metadata)
    if ledger is not None:
        ledger.record(decision)


def with_transforms(decision: FieldDecision, *transforms: str) -> FieldDecision:
    """*decision* with repairs applied to the chosen value after it was decided."""
    if not transforms or decision.selected is None:
        return decision
    selected = replace(decision.selected, transforms=(*decision.selected.transforms, *transforms))
    considered = tuple(
        replace(verdict, candidate=selected) if verdict.candidate is decision.selected else verdict
        for verdict in decision.considered
    )
    return replace(decision, value=selected.value, selected=selected, considered=considered)


def _empty(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    return not value


def incumbent_candidate(metadata: Any, name: str, *, source: str | None) -> FieldCandidate:
    """The field's proposal, or else its constructed value as a candidate.

    A producer that builds the record (the core extractor, the JATS and HTML
    readers) constructs it with its values; the core extractor also proposes
    them with their provenance. Without a proposal the constructed value
    stands in under *source*, None when its provenance is unknown.
    """
    ledger = field_decisions_of(metadata)
    proposal = ledger.proposal(name) if ledger is not None else None
    if proposal is not None:
        return proposal
    (attribute,) = FIELD_ATTRIBUTES[name]
    return FieldCandidate(name, source, getattr(metadata, attribute))


# ── Rules ────────────────────────────────────────────────────────────────


def _with_flags(decide: Callable[..., FieldDecision]) -> Callable[..., FieldDecision]:
    """Keep the rule's keyword flags (booleans, a notice type) on its receipt."""

    @functools.wraps(decide)
    def wrapper(*args: Any, **kwargs: Any) -> FieldDecision:
        decision = decide(*args, **kwargs)
        flags = tuple(
            (name, value)
            for name, value in kwargs.items()
            if isinstance(value, bool) or name == "notice"
        )
        return replace(decision, flags=flags) if flags else decision

    return wrapper


def decide_value(name: str, candidate: FieldCandidate | None) -> FieldDecision:
    """A field with one producer: its value, empty or not."""
    if candidate is None:
        return FieldDecision(name, None, None, "no_candidate")
    if _empty(candidate.value):
        return FieldDecision(
            name,
            candidate.value,
            None,
            "empty",
            (Verdict(candidate, False, "empty"),),
            producer=candidate.source,
        )
    return FieldDecision(
        name,
        candidate.value,
        candidate,
        _incumbent_rule(candidate),
        (Verdict(candidate, True, "extracted"),),
    )


_AUTHOR_RULES = {
    "llm": "extracted",
    "llm_recovery": "empty_author_recovery",
    "credit_statement": "credit_statement_fallback",
    "native": "native",
    "doc_info": "doc_info_fill",
}


@_with_flags
def decide_authors(
    candidates: Sequence[FieldCandidate], *, notice: str | None = None
) -> FieldDecision:
    """Authors: the first usable list in the order the candidates were produced.

    The core extractor produces the model's list, then — only while nothing
    usable is in hand — the empty-author recovery and the CRediT statement;
    post-parse offers the input's own list and the PDF doc-info. A list a guard
    vetoed (fabricated, romanised) is never used. A correction notice has no
    authors, whatever was extracted (*notice* names the notice type).
    """
    considered: list[Verdict] = []
    selected: FieldCandidate | None = None
    for candidate in candidates:
        if candidate.veto is not None:
            considered.append(Verdict(candidate, False, candidate.veto))
        elif _empty(candidate.value):
            considered.append(Verdict(candidate, False, "empty"))
        elif selected is not None:
            considered.append(Verdict(candidate, False, f"{selected.source} already used"))
        else:
            selected = candidate
            considered.append(Verdict(candidate, True, "first usable list"))
    producer = selected or (candidates[0] if candidates else None)
    producer_source = producer.source if producer is not None else None
    if notice is not None:
        return FieldDecision(
            "author",
            [],
            None,
            "correction_notice",
            tuple(Verdict(v.candidate, False, f"correction notice ({notice})") for v in considered),
            producer=producer_source,
        )
    if selected is None:
        return FieldDecision("author", [], None, "none", tuple(considered), producer_source)
    return FieldDecision(
        "author",
        selected.value,
        selected,
        _AUTHOR_RULES.get(selected.source or "", "extracted"),
        tuple(considered),
    )


@_with_flags
def decide_abstract(
    incumbent: FieldCandidate | None,
    *,
    fallback: FieldCandidate | None,
    explicitly_absent: bool,
    printed_abstract: bool,
    abstained: bool,
) -> FieldDecision:
    """Abstract: the extracted string, else the text the layout labelled Abstract.

    The model's string wins because layout regions (running headers,
    copyright lines, affiliation blocks) routinely flow into the abstract
    section. The layout fallback is skipped when the model explicitly found no
    abstract, unless the selected record prints an Abstract heading. Nothing is
    chosen when front-matter selection abstained.
    """
    producer = incumbent.source if incumbent is not None else None
    if abstained:
        return FieldDecision(
            "abstract",
            incumbent.value if incumbent is not None and incumbent.veto is None else "",
            None,
            "abstained",
            producer=producer,
        )
    considered: list[Verdict] = []
    if incumbent is not None:
        text = (incumbent.value or "").strip()
        if incumbent.veto is not None:
            considered.append(Verdict(incumbent, False, incumbent.veto))
        elif not text:
            considered.append(Verdict(incumbent, False, "empty"))
        else:
            considered.append(Verdict(incumbent, True, "extracted"))
            return FieldDecision(
                "abstract", text, incumbent, _incumbent_rule(incumbent), tuple(considered)
            )
    if explicitly_absent and not printed_abstract:
        if fallback is not None:
            considered.append(Verdict(fallback, False, "the model found no abstract"))
        return FieldDecision("abstract", "", None, "explicitly_absent", tuple(considered), producer)
    if fallback is not None:
        text = (fallback.value or "").strip()
        if text:
            considered.append(Verdict(fallback, True, "no extracted abstract"))
            return FieldDecision(
                "abstract", text, fallback, "abstract_section_fallback", tuple(considered)
            )
        considered.append(Verdict(fallback, False, "empty"))
    return FieldDecision("abstract", "", None, "none", tuple(considered), producer)


@_with_flags
def decide_keywords(
    incumbent: FieldCandidate | None,
    *,
    doc_info: FieldCandidate | None,
    section: FieldCandidate | None,
    abstained: bool,
) -> FieldDecision:
    """Keywords: the extracted list, else the PDF doc-info, else the Keywords section."""
    producer = incumbent.source if incumbent is not None else None
    empty_value = incumbent.value if incumbent is not None and incumbent.veto is None else []
    if abstained:
        return FieldDecision("keywords", empty_value, None, "abstained", producer=producer)
    considered: list[Verdict] = []
    if incumbent is not None:
        if incumbent.veto is not None:
            considered.append(Verdict(incumbent, False, incumbent.veto))
        elif not incumbent.value:
            considered.append(Verdict(incumbent, False, "empty"))
        else:
            considered.append(Verdict(incumbent, True, "extracted"))
            return FieldDecision(
                "keywords",
                incumbent.value,
                incumbent,
                _incumbent_rule(incumbent),
                tuple(considered),
            )
    for candidate, rule in ((doc_info, "doc_info_fill"), (section, "keywords_section_fallback")):
        if candidate is None:
            continue
        if not candidate.value:
            considered.append(Verdict(candidate, False, "empty"))
            continue
        considered.append(Verdict(candidate, True, "no earlier keywords"))
        return FieldDecision("keywords", candidate.value, candidate, rule, tuple(considered))
    return FieldDecision("keywords", empty_value, None, "none", tuple(considered), producer)


_CLASSIFICATION_RULES = {
    "classifier": "classifier",
    "llm_label": "low_confidence_escalation",
    "llm": "model_classification",
}


@_with_flags
def decide_classification(
    candidates: Sequence[FieldCandidate], *, notice: str | None = None
) -> FieldDecision:
    """Paper type: a correction notice's type, else the classification made.

    The core extractor offers at most one classification (the trained
    classifier's, its low-confidence paper type relabelled by the LLM, or the
    LLM's) plus the predictions it replaced; a correction-notice title
    overrides all of them and blanks the OECD fields.
    """
    considered: list[Verdict] = []
    selected: FieldCandidate | None = None
    for candidate in candidates:
        if candidate.veto is not None:
            considered.append(Verdict(candidate, False, candidate.veto))
        elif selected is not None:
            considered.append(Verdict(candidate, False, f"{selected.source} already used"))
        else:
            selected = candidate
            considered.append(Verdict(candidate, True, "classification made"))
    producer = selected.source if selected is not None else "llm"
    if notice is not None:
        notice_candidate = FieldCandidate(
            "paper_type", "correction_notice", Classification(paper_type=notice)
        )
        return FieldDecision(
            "paper_type",
            notice_candidate.value,
            notice_candidate,
            "correction_notice",
            (
                *(Verdict(v.candidate, False, "correction notice") for v in considered),
                Verdict(notice_candidate, True, "the title is a correction notice"),
            ),
        )
    if selected is None:
        return FieldDecision(
            "paper_type", Classification(), None, "not_classified", tuple(considered), producer
        )
    return FieldDecision(
        "paper_type",
        selected.value,
        selected,
        _CLASSIFICATION_RULES.get(selected.source or "", "extracted"),
        tuple(considered),
    )


@_with_flags
def decide_statement(
    name: str,
    incumbent: FieldCandidate | None,
    rendered: FieldCandidate | None,
    *,
    keep_incumbent: bool,
) -> FieldDecision:
    """A research-integrity statement: the input's own, else the resolved statement.

    *keep_incumbent* carries the rollout rule: a present statement is kept for
    an input that declares its metadata, outside the active mode, or when the
    resolution selected nothing.
    """
    considered: list[Verdict] = []
    if incumbent is not None and incumbent.value is not None:
        if keep_incumbent:
            verdicts = [Verdict(incumbent, True, "present")]
            if rendered is not None:
                verdicts.append(Verdict(rendered, False, "present statement kept"))
            return FieldDecision(
                name, incumbent.value, incumbent, _incumbent_rule(incumbent), tuple(verdicts)
            )
        considered.append(Verdict(incumbent, False, "replaced by the active resolution"))
    if rendered is None:
        return FieldDecision(name, None, None, "none", tuple(considered))
    if rendered.value is None:
        considered.append(Verdict(rendered, False, "empty"))
        return FieldDecision(name, None, None, "none", tuple(considered), producer=rendered.source)
    considered.append(Verdict(rendered, True, "resolved statement"))
    return FieldDecision(name, rendered.value, rendered, "integrity_resolution", tuple(considered))


@_with_flags
def decide_title(
    incumbent: FieldCandidate | None,
    *,
    resolution: Any,
    detected_title: str | None,
    sections: Sequence[Any],
    journal: str | None,
    publisher: str | None,
    scoped: bool,
    abstained: bool,
    prefer_byline_adjacent: bool,
    doc_info: FieldCandidate | None,
) -> FieldDecision:
    """Title: the extracted title, the printed-row fallbacks, then the doc-info.

    *scoped* says the selected front-matter record owns the metadata (every
    LLM run on a PDF, DOCX or ePub; a run without one only when selection
    raised a blocking issue); *abstained* that selection chose no record.
    *doc_info* is offered only when the doc-info may fill (an unscoped run
    that has one). In order:

    1. Scoped and not abstained, no extracted title: the one safe title row of
       the selected record, else the layout-detected title under the same
       filters.
    2. Scoped and not abstained, a title in hand, byline adjacency enabled: the
       printed row directly above the byline when it disagrees.
    3. Unscoped, or the layout title is exactly a generic article label: the
       layout title replaces the title unless it is a generic label and the
       title is printed in the selected record, or it is a masthead the title
       disagrees with; then, still without a title, the first unclassified
       heading.
    4. Still without a title: the doc-info title.
    """
    from bibr.extract import title_candidates as titles
    from bibr.utils.metadata import is_exact_generic_article_label

    considered: list[Verdict] = []
    selected: FieldCandidate | None = None
    if incumbent is not None and incumbent.value:
        selected = incumbent
        rule = _incumbent_rule(incumbent)
        considered.append(Verdict(incumbent, True, "extracted"))
    else:
        rule = "abstained" if abstained else "none"
        if incumbent is not None:
            considered.append(Verdict(incumbent, False, "empty"))

    def use(candidate: FieldCandidate, new_rule: str, reason: str) -> None:
        nonlocal selected, rule
        if selected is not None:
            considered.append(Verdict(selected, False, f"replaced by {candidate.source}"))
        selected, rule = candidate, new_rule
        considered.append(Verdict(candidate, True, reason))

    def reject(source: str, reason: str) -> None:
        considered.append(Verdict(FieldCandidate("title", source, None), False, reason))

    title = selected.value if selected is not None else ""
    if scoped and not abstained and not title:
        candidate, reason = titles.selected_title_candidate(
            resolution, journal=journal, publisher=publisher
        )
        if candidate is not None:
            logger.info(
                "Selected-title fallback recovered the null model title from candidate %s",
                candidate.evidence_ids[0],
            )
            use(candidate, "selected_record_title", reason)
        else:
            reject("front_matter_candidate", reason)
            candidate, reason = titles.detected_title_candidate(
                detected_title, journal=journal, publisher=publisher
            )
            if candidate is not None:
                logger.info(
                    "Detected-title fallback recovered the null model title: %r", candidate.value
                )
                use(candidate, "layout_title_fallback", reason)
            else:
                reject("layout_title", reason)
    elif scoped and not abstained and prefer_byline_adjacent:
        candidate, reason = titles.byline_adjacent_candidate(
            resolution, title, journal=journal, publisher=publisher
        )
        if candidate is not None:
            logger.info(
                "Byline-adjacency title preference replaced %r with candidate %s",
                title,
                candidate.evidence_ids[0],
            )
            use(candidate, "byline_adjacent_title", reason)
        else:
            reject("byline_adjacent", reason)

    if not scoped or is_exact_generic_article_label(detected_title):
        title = selected.value if selected is not None else ""
        if detected_title:
            layout = FieldCandidate("title", "layout_title", detected_title)
            yields = titles.layout_title_yields(
                detected_title, title, journal=journal, publisher=publisher, resolution=resolution
            )
            if yields is not None:
                considered.append(Verdict(layout, False, yields))
                rule = "masthead_guard" if yields == "masthead" else "grounded_over_generic_label"
            elif title == detected_title:
                considered.append(Verdict(layout, False, "same as the title"))
            else:
                use(layout, "layout_title", "layout-detected title")
        if selected is None or not selected.value:
            header = titles.section_header_candidate(sections)
            if header is not None:
                use(header, "section_header", "first unclassified heading")

    if doc_info is not None and (selected is None or not selected.value):
        if doc_info.value:
            use(doc_info, "doc_info_fill", "no title from the document")
        else:
            considered.append(Verdict(doc_info, False, "empty"))

    if selected is None:
        producer = incumbent.source if incumbent is not None else None
        value = incumbent.value if incumbent is not None and incumbent.value is not None else ""
        return FieldDecision("title", value, None, rule, tuple(considered), producer)
    return FieldDecision("title", selected.value, selected, rule, tuple(considered))


def _incumbent_rule(candidate: FieldCandidate) -> str:
    return "native" if candidate.source == "native" else "extracted"


__all__ = [
    "DECIDED_ATTRIBUTES",
    "FIELD_ATTRIBUTES",
    "Classification",
    "FieldCandidate",
    "FieldDecision",
    "FieldDecisions",
    "Verdict",
    "apply_decision",
    "decide_abstract",
    "decide_authors",
    "decide_classification",
    "decide_keywords",
    "decide_statement",
    "decide_title",
    "decide_value",
    "field_decisions_of",
    "incumbent_candidate",
    "with_transforms",
]
