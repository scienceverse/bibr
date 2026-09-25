"""Per-field extraction states: what happened to each tracked field.

An empty exported field reads the same whether the paper has none, the
extractor declined to choose, the call that produces it failed, or the run
never tried. ``extraction.fields`` tells them apart for the fields consumers
act on. The states are built at export time from facts the pipeline already
records: the final values, each field's decision
(:mod:`bibr.extract.field_decisions`: the source of the value used and the
rule that chose it), the run's scope (:class:`FieldScope`), and the codes of
the validation issues and warnings that explain a missing value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FieldState(StrEnum):
    """What happened to one exported field."""

    EXTRACTED = "extracted"  # a value was exported
    ABSENT = "absent"  # the extractor ran and found none
    ABSTAINED = "abstained"  # the extractor declined to choose between candidates
    FAILED = "failed"  # the step that produces it failed; the value is missing
    NOT_ATTEMPTED = "not_attempted"  # the run did not try (no LLM, references off)


# Export name of each tracked field: a ``metadata`` key, or a root table.
TRACKED_FIELDS = (
    "title",
    "author",
    "abstract",
    "keywords",
    "doi",
    "published",
    "journal",
    "funding_statement",
    "funding",
    "paper_type",
    "bib",
)

# Chosen from the selected front-matter record; a record abstention empties them.
_FRONT_MATTER_FIELDS = frozenset(
    {"title", "author", "abstract", "keywords", "published", "journal", "paper_type"}
)
# Produced by LLM calls unless the input declares them (JATS, HTML meta tags).
_LLM_FIELDS = _FRONT_MATTER_FIELDS | {"funding"}

# Codes whose presence means a field's producing step failed.
_FAILURE_CODES: Mapping[str, frozenset[str]] = {
    "VAL_CORE_METADATA_DEGRADED": _FRONT_MATTER_FIELDS,
    "AUTHORS_LLM_FAILED": frozenset({"author"}),
    "PAPER_CLASSIFICATION_FAILED": frozenset({"paper_type"}),
    "RESEARCH_INTEGRITY_LLM_FAILED": frozenset({"funding"}),
    "VAL_REFERENCES_INCOMPLETE": frozenset({"bib"}),
    "REF_SEG_FAILED": frozenset({"bib"}),
}
# Codes that qualify a field's state without deciding it.
_QUALIFYING_CODES: Mapping[str, frozenset[str]] = {
    "AUTHORS_EMPTY": frozenset({"author"}),
    "AUTHORS_TRUNCATED": frozenset({"author"}),
    "AUTHORS_PARTIAL": frozenset({"author"}),
    "VAL_AUTHOR_RECOVERY_DEGRADED": frozenset({"author"}),
    "REF_SECTION_NOT_FOUND": frozenset({"bib"}),
    "REF_SECTION_INFERRED": frozenset({"bib"}),
    "VAL_DOI_AMBIGUOUS": frozenset({"doi"}),
    "VAL_EXPECTED_ID_MISSING": frozenset({"doi"}),
    "VAL_EXPECTED_ID_MISMATCH": frozenset({"doi"}),
}
_FIELD_FAILED = "VAL_METADATA_FIELD_FAILED"
_ABSTENTION = "VAL_METADATA_MULTI_ITEM"


@dataclass(frozen=True)
class FieldScope:
    """What the run attempted, set by post-parse on the ``Paper``.

    A Paper built outside the pipeline has none, and its export carries no
    ``extraction.fields``.
    """

    no_llm: bool = False
    native_metadata: bool = False  # front matter declared by the input (JATS, HTML)
    references_off: bool = False
    # Where the reference list came from: "native" (the input's structured
    # citations) or the configured parser.
    references_source: str | None = None


@dataclass(frozen=True)
class FieldRecord:
    state: FieldState
    source: str | None
    issues: tuple[str, ...] = ()
    rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": str(self.state),
            "source": self.source,
            "issues": list(self.issues),
            "rule": self.rule,
        }


def _issue_fields(code: str, evidence_ids: Iterable[str]) -> frozenset[str]:
    if code == _FIELD_FAILED:
        return frozenset(
            evidence.removeprefix("field:")
            for evidence in evidence_ids
            if evidence.startswith("field:")
        )
    return _FAILURE_CODES.get(code) or _QUALIFYING_CODES.get(code) or frozenset()


def build_field_states(
    *,
    present: Mapping[str, bool],
    sources: Mapping[str, str],
    scope: FieldScope,
    issues: Iterable[Any],
    warnings: Iterable[Any],
    doi_selected: bool = False,
    rules: Mapping[str, str] | None = None,
) -> dict[str, FieldRecord]:
    """One :class:`FieldRecord` per tracked field.

    *present* says whether each field has an exported value; *sources* and
    *rules* come from the field decisions (the source of the value used, or of
    the step that failed, and the rule that decided); *issues* are the
    validation issues (``code``, ``blocking``, ``evidence_ids``) and
    *warnings* the processing warnings (``code``). A present value is always
    ``extracted``; otherwise a failure beats an abstention, which beats a step
    the run did not attempt.
    """
    failed: dict[str, list[str]] = {field: [] for field in TRACKED_FIELDS}
    qualified: dict[str, list[str]] = {field: [] for field in TRACKED_FIELDS}
    abstained_front_matter = False
    doi_abstained = False
    for issue in issues:
        code = str(getattr(issue, "code", ""))
        if code == _ABSTENTION and getattr(issue, "blocking", False):
            abstained_front_matter = True
        if code == "VAL_DOI_AMBIGUOUS":
            doi_abstained = True
        for field in _issue_fields(code, getattr(issue, "evidence_ids", ())):
            target = failed if code == _FIELD_FAILED or code in _FAILURE_CODES else qualified
            if field in target:
                target[field].append(code)
    for warning in warnings:
        code = str(getattr(warning, "code", ""))
        for field in _issue_fields(code, ()):
            target = failed if code in _FAILURE_CODES else qualified
            target[field].append(code)
    # Structured funding is parsed from the funding statement; without one
    # there was nothing for the failed call to parse.
    if not present.get("funding_statement"):
        failed["funding"].clear()

    sources = {**sources, **({"bib": scope.references_source} if scope.references_source else {})}
    rules = rules or {}
    defaults = {
        "doi": "identity" if doi_selected else None,
    }
    records: dict[str, FieldRecord] = {}
    for field in TRACKED_FIELDS:
        codes = tuple(dict.fromkeys([*failed[field], *qualified[field]]))
        rule = rules.get(field)
        if present.get(field):
            source = sources.get(field) or defaults.get(field)
            records[field] = FieldRecord(FieldState.EXTRACTED, source, codes, rule)
        elif failed[field]:
            records[field] = FieldRecord(FieldState.FAILED, sources.get(field), codes, rule)
        elif (abstained_front_matter and field in _FRONT_MATTER_FIELDS) or (
            field == "doi" and doi_abstained
        ):
            records[field] = FieldRecord(
                FieldState.ABSTAINED,
                None,
                tuple(dict.fromkeys([*codes, *([_ABSTENTION] if field != "doi" else [])])),
                rule,
            )
        elif (
            field in _LLM_FIELDS
            and scope.no_llm
            and not (scope.native_metadata and field != "funding")
        ) or (
            # Without an LLM only the input's own structured citations are read.
            field == "bib"
            and (scope.references_off or (scope.no_llm and sources.get("bib") != "native"))
        ):
            records[field] = FieldRecord(FieldState.NOT_ATTEMPTED, None, codes, rule)
        else:
            records[field] = FieldRecord(FieldState.ABSENT, None, codes, rule)
    return records
