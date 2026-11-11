"""The statement tagger as one more source of bibr's statement resolution (opt-in).

Off by default: ``ML_STATEMENT_TAGGER_MODEL_ID`` is unset and nothing here runs
(:func:`apply_statement_tagger` returns the resolution it was given). Set to a
bundle (a local directory or a Hub repo id, at ``ML_STATEMENT_TAGGER_REVISION``),
the tagger reads the paper's blocks after the rules resolved the statements and
adds what they missed:

* **Blocks** (:func:`content_blocks`): the post-parse rows grouped into their
  paragraphs, in row order, each section's printed heading a block of its own
  before its first paragraph; display formulas, empty rows and the rows of the
  reference list are left out.
* **Funding, competing interests, ethics, data availability** (the four
  statement fields): where the tagger found statements of a field's type, the
  field takes them, all of them in reading order, each trimmed to its own label
  when its paragraph holds another statement before it and cut at the next
  end-matter label (:func:`statement_rows`); the rules' candidates stay in the
  resolution, unselected. A field the tagger found nothing for keeps the rules'
  value. The candidate says where it came from (method ``statement_tagger``,
  its score, ``replaces_rules`` or ``rules_found_none``) and the field's source
  is ``statement_tagger``. (Chosen on development papers with publisher-tagged
  statements: where both had a statement, the tagger's was the right one about
  eight times as often.)
* **Acknowledgments, author contributions** (section types): when the paper
  has no section of the type, the tagger's paragraphs of that type become one:
  a section whose rows all carry the type is retyped; otherwise the rows move
  to a new section right after the one they sat in, with a synthetic heading.
  Either way ``classification_source`` is ``statement_tagger``.
* Each statement the tagger supplied adds a ``STATEMENT_TAGGER`` warning naming
  the field or section type.

A setting that names a bundle that does not exist or cannot be read stops the
run (:class:`~bibr.exceptions.ModelOverrideError`, see
:mod:`bibr.utils.model_overrides`); a Hub that cannot be reached leaves this
paper without the tagger, as for any model.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from bibr.paper_contents import CanonicalSection, PaperSection
from bibr.processing_warnings import ProcessingWarning, WarningCode

if TYPE_CHECKING:
    from bibr.extract.integrity_statements import IntegrityStatementResolution
    from bibr.paper_contents import PaperContents, PaperSentence
    from bibr.statement_tagger.runtime import StatementTagger, TaggedStatement

logger = logging.getLogger(__name__)

SETTING = "ML_STATEMENT_TAGGER_MODEL_ID"
METHOD = "statement_tagger"

# Tagger type -> statement field, and -> section type.
FIELD_TYPES: dict[str, str] = {
    "funding": "funding_statement",
    "coi": "coi_statement",
    "ethics": "ethics_statement",
    "data_availability": "data_availability",
}
SECTION_TYPES: dict[str, CanonicalSection] = {
    "acknowledgment": CanonicalSection.ACKNOWLEDGMENT,
    "author_contributions": CanonicalSection.AUTHOR_CONTRIBUTIONS,
}
_SYNTHETIC_HEADERS = {
    CanonicalSection.ACKNOWLEDGMENT: "Acknowledgments",
    CanonicalSection.AUTHOR_CONTRIBUTIONS: "Author contributions",
}
# Blocks never read: the reference list (its rows are references, and the
# tagger's time on them is a third of a paper's).
_SKIPPED_SECTION_TYPES = frozenset({CanonicalSection.REFERENCES})
# Section headings that are bibr's own labels, not printed headings.
_LABEL_HEADING_TYPES = frozenset(
    {CanonicalSection.TABLE, CanonicalSection.FIGURE, CanonicalSection.FOOTNOTE}
)
# Sections whose rows a tagged statement never moves out of or retypes.
_KEPT_SECTION_TYPES = frozenset(
    {
        CanonicalSection.TITLE,
        CanonicalSection.ABSTRACT,
        CanonicalSection.KEYWORDS,
        CanonicalSection.REFERENCES,
        CanonicalSection.TABLE,
        CanonicalSection.FIGURE,
        CanonicalSection.FUNDING,
        CanonicalSection.COI,
        CanonicalSection.ETHICS,
        CanonicalSection.OPEN_DATA,
        CanonicalSection.ACKNOWLEDGMENT,
        CanonicalSection.AUTHOR_CONTRIBUTIONS,
    }
)


@dataclass(frozen=True)
class ContentBlock:
    """One tagger block: a section heading, or the rows of one paragraph."""

    text: str
    section_id: int
    text_ids: tuple[int, ...] = ()

    @property
    def heading(self) -> bool:
        return not self.text_ids


def content_blocks(contents: PaperContents) -> list[ContentBlock]:
    """The paper's tagger blocks in row order (see the module docstring)."""
    sections = {section.section_id: section for section in contents.sections}
    blocks: list[ContentBlock] = []
    opened: set[int] = set()
    key: tuple[int, int] | None = None
    texts: list[str] = []
    ids: list[int] = []

    def flush() -> None:
        if ids and key is not None:
            blocks.append(ContentBlock(" ".join(texts), key[0], tuple(ids)))

    for sentence in contents.sentences:
        text = " ".join(sentence.text.split())
        section = sections.get(sentence.section_id)
        if sentence.is_display_formula or not text:
            continue
        if section is not None and section.section_type in _SKIPPED_SECTION_TYPES:
            continue
        if sentence.section_id not in opened:
            opened.add(sentence.section_id)
            header = " ".join((section.header or "").split()) if section is not None else ""
            if (
                section is not None
                and header
                and section.level > 0
                and not section.header_is_synthetic
                and section.section_type not in _LABEL_HEADING_TYPES
                and section.section_type != CanonicalSection.TITLE
            ):
                flush()
                key, texts, ids = None, [], []
                blocks.append(ContentBlock(header, sentence.section_id))
        if (sentence.section_id, sentence.paragraph_id) != key:
            flush()
            key, texts, ids = (sentence.section_id, sentence.paragraph_id), [], []
        texts.append(text)
        ids.append(sentence.text_id)
    flush()
    return blocks


def statement_rows(
    field: str, statement: TaggedStatement, blocks: Sequence[ContentBlock], rows_by_id: dict
) -> tuple[int, ...]:
    """The row ids of one tagged statement of a statement field.

    The rows of its paragraph blocks, from the first row that opens the field's
    own label when a row before it opens another statement's label (a
    paragraph "Funding: ... Conflicts of interest: ..."), up to the next row
    that opens other end matter.
    """
    from bibr.extract.integrity_statements import _STATEMENT_LABEL, _opens_other_end_matter

    ids = [i for block in blocks[statement.start : statement.end + 1] for i in block.text_ids]
    texts = [" ".join(rows_by_id[i].text.split()) for i in ids]
    own = _STATEMENT_LABEL[field]
    start = 0
    if texts and _opens_other_end_matter(field, texts[0]):
        start = next((k for k, text in enumerate(texts) if own.match(text)), 0)
    end = len(texts)
    for k in range(start + 1, len(texts)):
        if _opens_other_end_matter(field, texts[k]):
            end = k
            break
    return tuple(ids[start:end])


def _field_candidates(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    statements: Sequence[TaggedStatement],
    blocks: Sequence[ContentBlock],
):
    """Tagger candidates per statement field, all of the tagger's statements of its type."""
    from bibr.extract.integrity_statements import IntegrityStatementCandidate

    rows_by_id = {row.text_id: row for row in contents.sentences}
    sections = {section.section_id: section for section in contents.sections}
    out: list[tuple[str, IntegrityStatementCandidate]] = []
    for tagger_type, field in FIELD_TYPES.items():
        flag = "replaces_rules" if resolution.candidate_indices(field) else "rules_found_none"
        for statement in statements:
            if statement.type != tagger_type:
                continue
            ids = statement_rows(field, statement, blocks, rows_by_id)
            if not ids:
                continue
            rows = [rows_by_id[i] for i in ids]
            section_ids = tuple(dict.fromkeys(row.section_id for row in rows))
            # A heading is the statement's own only when its heading block is in the run.
            first = next(
                (
                    sections.get(block.section_id)
                    for block in blocks[statement.start : statement.end + 1]
                    if block.heading
                ),
                None,
            )
            out.append(
                (
                    field,
                    IntegrityStatementCandidate(
                        field=field,
                        method=METHOD,
                        heading=first.header if first is not None else None,
                        section_ids=section_ids,
                        text_ids=ids,
                        paragraph_ids=tuple(dict.fromkeys(row.paragraph_id for row in rows)),
                        pages=tuple(
                            dict.fromkeys(r.page_number for r in rows if r.page_number is not None)
                        ),
                        classification_source=METHOD,
                        classification_score=round(statement.score, 4),
                        reason_flags=(METHOD, flag),
                        accepted=True,
                    ),
                )
            )
    return out


def merge_field_candidates(
    resolution: IntegrityStatementResolution, additions: Sequence[tuple[str, Any]]
) -> IntegrityStatementResolution:
    """The resolution with tagger candidates appended and chosen for their fields."""
    if not additions:
        return resolution
    candidates = list(resolution.candidates)
    chosen: dict[str, list[int]] = {}
    for field, candidate in additions:
        chosen.setdefault(field, []).append(len(candidates))
        candidates.append(candidate)

    def update(pairs):
        return tuple((field, tuple(chosen.get(field, indices))) for field, indices in pairs)

    return replace(
        resolution,
        candidates=tuple(candidates),
        legacy_candidate_indices=update(resolution.legacy_candidate_indices),
        selected_candidate_indices=update(resolution.selected_candidate_indices),
    )


def apply_section_statements(
    contents: PaperContents,
    statements: Sequence[TaggedStatement],
    blocks: Sequence[ContentBlock],
) -> list[str]:
    """Give tagged acknowledgments and author contributions a section of their type.

    Only for a type the paper has no section of. Returns the section types set.
    """
    from bibr.structure.implicit_sections import _insert_after_source, _rebuild_sections_text

    present = {section.section_type for section in contents.sections}
    sections = {section.section_id: section for section in contents.sections}
    rows_of: dict[int, list[PaperSentence]] = {}
    for row in contents.sentences:
        rows_of.setdefault(row.section_id, []).append(row)
    rows_by_id = {row.text_id: row for row in contents.sentences}
    next_id = max((section.section_id for section in contents.sections), default=0) + 1
    placed: set[int] = set()
    affected: set[int] = set()
    done: list[str] = []
    for tagger_type, section_type in SECTION_TYPES.items():
        if section_type in present:
            continue
        runs = [s for s in statements if s.type == tagger_type]
        if not runs:
            continue
        ids = [i for s in runs for block in blocks[s.start : s.end + 1] for i in block.text_ids]
        score = round(max(s.score for s in runs), 4)
        by_section: dict[int, list[int]] = {}
        for i in ids:
            by_section.setdefault(rows_by_id[i].section_id, []).append(i)
        changed = False
        for section_id, members in by_section.items():
            source = sections.get(section_id)
            if source is None or source.section_type in _KEPT_SECTION_TYPES:
                continue
            own = [row.text_id for row in rows_of.get(section_id, []) if row.text.strip()]
            if set(own) <= set(members) and source.level > 0:
                source.section_type = section_type
                source.classification_source = METHOD
                source.classification_score = score
                changed = True
                continue
            new = PaperSection(
                section_id=next_id,
                header=_SYNTHETIC_HEADERS[section_type],
                level=max(1, source.level),
                parent_section_id=source.parent_section_id
                if source.level > 0
                else source.section_id,
                section_type=section_type,
                classification_score=score,
                classification_source=METHOD,
                header_is_synthetic=True,
            )
            _insert_after_source(contents.sections, new, section_id, placed)
            placed.add(next_id)
            for i in members:
                rows_by_id[i].section_id = next_id
            affected |= {section_id, next_id}
            next_id += 1
            changed = True
        if changed:
            present.add(section_type)
            done.append(str(section_type))
    if affected:
        _rebuild_sections_text(contents, affected)
    return done


# -- loading ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_TAGGER: StatementTagger | None = None
_KEY: tuple[str, str | None] | None = None


def get_statement_tagger(settings: Any) -> StatementTagger | None:
    """The configured tagger, loaded once per setting value; ``None`` when unset.

    ``None`` also when the Hub cannot be reached (not remembered: the next
    paper tries again). A bundle that does not exist or cannot be read raises
    :class:`~bibr.exceptions.ModelOverrideError`.
    """
    global _TAGGER, _KEY
    from bibr.utils.model_overrides import override_load_error, record_model_load

    ml = getattr(settings, "ml", None)
    value = getattr(ml, "statement_tagger_model_id", None)
    if not isinstance(value, str) or not value:
        return None
    revision = getattr(ml, "statement_tagger_revision", None)
    revision = revision if isinstance(revision, str) and revision else None
    key = (value, revision)
    with _LOCK:
        if _TAGGER is None or key != _KEY:
            try:
                _TAGGER = _load(value, revision)
                _KEY = key
            except Exception as exc:
                if (strict := override_load_error(settings, SETTING, exc)) is not None:
                    raise strict from exc
                logger.warning(
                    "statement tagger %s unavailable (%r); paper runs without it", value, exc
                )
                return None
        tagger = _TAGGER
    record_model_load(settings, SETTING, tagger.files)
    return tagger


def _load(value: str, revision: str | None) -> StatementTagger:
    from bibr.exceptions import ConfigurationError
    from bibr.statement_tagger.runtime import StatementTagger
    from bibr.utils.ml_runtime import find_onnx_bundle
    from bibr.utils.onnx_providers import apply_cpu_threading

    bundle = find_onnx_bundle(value, revision, label="statement tagger", raise_transient=True)
    if bundle is None:
        raise ConfigurationError(
            f"no statement tagger bundle at {value!r} (want bibr_onnx.json, model.onnx and "
            "tokenizer.json in the directory or its onnx/ subdirectory)"
        )
    import onnxruntime as ort

    options = ort.SessionOptions()
    apply_cpu_threading(options)
    options.inter_op_num_threads = 1
    logger.info("loading statement tagger: %s", value)
    return StatementTagger(bundle, session_options=options)


def reset_statement_tagger() -> None:
    """Test hook: forget the loaded tagger."""
    global _TAGGER, _KEY
    with _LOCK:
        _TAGGER, _KEY = None, None


# -- the stage ---------------------------------------------------------------------------


def apply_statement_tagger(
    contents: PaperContents,
    resolution: IntegrityStatementResolution,
    settings: Any,
    *,
    tagger: StatementTagger | None = None,
) -> IntegrityStatementResolution:
    """Add the tagger's statements to the resolution and the sections; unset, a no-op."""
    tagger = tagger if tagger is not None else get_statement_tagger(settings)
    if tagger is None:
        return resolution
    blocks = content_blocks(contents)
    if not blocks:
        return resolution
    statements = tagger.tag([block.text for block in blocks])
    additions = _field_candidates(contents, resolution, statements, blocks)
    resolution = merge_field_candidates(resolution, additions)
    sections = apply_section_statements(contents, statements, blocks)
    for name in dict.fromkeys([field for field, _ in additions] + sections):
        warning = ProcessingWarning(
            WarningCode.STATEMENT_TAGGER, f"{name} filled by the statement tagger"
        )
        if warning not in contents.processing_warnings:
            contents.processing_warnings.append(warning)
    return resolution


__all__ = [
    "FIELD_TYPES",
    "METHOD",
    "SECTION_TYPES",
    "SETTING",
    "ContentBlock",
    "apply_section_statements",
    "apply_statement_tagger",
    "content_blocks",
    "get_statement_tagger",
    "merge_field_candidates",
    "reset_statement_tagger",
    "statement_rows",
]
