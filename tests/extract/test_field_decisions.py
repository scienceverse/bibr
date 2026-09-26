"""One decision point, with a receipt, per metadata field (``bibr.extract.field_decisions``).

Each rule table is exercised on its own, then the one-writer property (no
module but the decision writes a decided field once the record exists) and
the receipts a pipeline run leaves on the ``Paper``.
"""

import ast
import copy
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_origin

import pytest

import bibr
import bibr.extract.field_decisions as field_decisions_module
from bibr.extract.field_decisions import (
    DECIDED_ATTRIBUTES,
    FIELD_ATTRIBUTES,
    Classification,
    FieldCandidate,
    FieldDecision,
    apply_decision,
    decide_abstract,
    decide_authors,
    decide_classification,
    decide_keywords,
    decide_statement,
    decide_title,
    decide_value,
    incumbent_candidate,
    with_transforms,
)
from bibr.models import PaperAuthor, PaperMetadata

# ---------------------------------------------------------------------------
# One writer
# ---------------------------------------------------------------------------

_PACKAGE = Path(bibr.__file__).parent
# The writer, and the JATS/HTML readers, which build the record they hand over.
_WRITERS = {"extract/field_decisions.py", "input/jats_native.py", "input/html_native.py"}
# The same attribute names on objects that are not the paper's metadata.
_OTHER_OBJECTS = {
    ("extract/ref_extractor.py", "ref.authors"),
    # The model's response, before any candidate is built from it.
    ("extract/core_metadata.py", "llm_metadata.model_copy"),
    # PaperMetadata's own copy, which drops the receipts of the fields an
    # update rewrites.
    ("models.py", "super().model_copy"),
    # References, their candidates and matches, the model's responses, the
    # doc-info parse: lists handed to helpers that are not the paper's.
    ("clients/llm.py", "authors.authors"),
    ("clients/llm.py", "title_kw.keywords"),
    ("enrich/references.py", "cand.authors"),
    ("enrich/references.py", "ref.authors"),
    ("export/json_export.py", "m.authors"),
    ("export/json_export.py", "r.authors"),
    ("extract/core_metadata.py", "llm_metadata.authors"),
    ("extract/core_metadata.py", "recovered.authors"),
    ("extract/research_integrity.py", "result.affiliations"),
    ("structure/citation_matcher.py", "r.authors"),
}
_MUTATORS = {"append", "extend", "insert", "remove", "clear", "pop", "sort", "reverse"}
_METADATA_NAME = re.compile(r"meta(?:data)?$|preparsed$|record$")
_CONSTRUCTORS = {"model_validate", "model_construct"}
# The decided attributes that hold a list, which a helper handed it can change.
_LIST_ATTRIBUTES = frozenset(
    attribute
    for attribute in DECIDED_ATTRIBUTES
    if get_origin(PaperMetadata.model_fields[attribute].annotation) is list
)
# Builtins that only read what they are handed, and the decision module's API.
_READERS = {
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "frozenset",
    "isinstance",
    "iter",
    "len",
    "list",
    "max",
    "min",
    "reversed",
    "set",
    "sorted",
    "str",
    "tuple",
    "zip",
    *field_decisions_module.__all__,
}
_ITERATORS = {"enumerate", "iter", "list", "reversed", "sorted", "tuple", "zip"}
# Other functions a decided list may be handed to, by module.
_HANDED_TO = {
    # Read the authors' affiliation strings.
    ("enrich/organizations.py", "collect_affiliations"),
    ("export/json_export.py", "collect_affiliations"),
    ("extract/research_integrity.py", "collect_affiliations"),
    # Adds contribution roles to the decided authors in place; the author
    # receipt lists the "contribution_roles" transform (record_transforms).
    ("extract/research_integrity.py", "_apply_contributions"),
}


def _decided_constant(node):
    return isinstance(node, ast.Constant) and node.value in DECIDED_ATTRIBUTES


def _decided_path(node):
    """Whether *node* is, or reaches into, a decided attribute: ``m.authors[0]``."""
    return any(
        isinstance(sub, ast.Attribute) and sub.attr in DECIDED_ATTRIBUTES for sub in ast.walk(node)
    )


def _root_name(node):
    """The local name an attribute or item chain starts from: ``a`` in ``a.role[0]``."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _target_writes(target, aliases):
    """The decided-field writes one assignment or ``del`` target makes."""
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_writes(element, aliases)
    elif isinstance(target, ast.Starred):
        yield from _target_writes(target.value, aliases)
    elif isinstance(target, ast.Attribute):
        # ``m.title = x``, and a write into a decided value: ``m.authors[0].family = x``.
        if (
            target.attr in DECIDED_ATTRIBUTES
            or _decided_path(target.value)
            or _root_name(target.value) in aliases
        ):
            yield ast.unparse(target)
    elif isinstance(target, ast.Subscript):
        container = target.value
        dunder = (isinstance(container, ast.Attribute) and container.attr == "__dict__") or (
            isinstance(container, ast.Call)
            and isinstance(container.func, ast.Name)
            and container.func.id == "vars"
        )
        if (
            _decided_path(container)
            or (dunder and _decided_constant(target.slice))
            or _root_name(container) in aliases
        ):
            yield ast.unparse(target)


def _decided_value(node, relative):
    """The decided attribute *node* reads, as in ``meta.authors``, or None."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr in DECIDED_ATTRIBUTES
        and (relative, ast.unparse(node)) not in _OTHER_OBJECTS
    ):
        return node.attr
    return None


def _aliases(function, relative):
    """Local names a function binds to a decided value, and the attribute each holds.

    ``authors = meta.authors``, and the targets of a loop over a decided value
    or an alias of one: ``for author in meta.authors``, ``for i, a in
    enumerate(authors)``.
    """
    aliases = {
        target.id: attribute
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and (attribute := _decided_value(node.value, relative))
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension))
    ]

    def iterated(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _ITERATORS:
                return next(filter(None, (iterated(argument) for argument in node.args)), None)
            return None
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        return _decided_value(node, relative)

    changed = True
    while changed:
        changed = False
        for loop in loops:
            attribute = iterated(loop.iter)
            if attribute is None:
                continue
            for name in ast.walk(loop.target):
                if isinstance(name, ast.Name) and name.id not in aliases:
                    aliases[name.id] = attribute
                    changed = True
    return aliases


def _handed_list(node, aliases, relative):
    """Whether a call argument is a decided list, or a local bound to one or its items."""
    if isinstance(node, ast.Starred):
        node = node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id) in _LIST_ATTRIBUTES
    return _decided_value(node, relative) in _LIST_ATTRIBUTES


def _writes_in(relative, source):
    """Every form that writes a decided field once the record exists.

    Assignments and ``del`` (attributes, items and slices of a decided value,
    ``__dict__``/``vars()`` items), in-place list mutations, including through a
    local alias of a decided value or a loop over one, a decided list (or an
    alias of it or its items) handed to a function that could change it,
    ``setattr``/``object.__setattr__``, and record rebuilds
    (``model_copy(update=...)``, ``PaperMetadata.model_validate`` of a dump,
    ``PaperMetadata.model_construct``).
    """
    tree = ast.parse(source)
    enclosing = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            for node in ast.walk(function):
                enclosing[node] = function
    aliases_of = {function: _aliases(function, relative) for function in set(enclosing.values())}
    for node in ast.walk(tree):
        function = enclosing.get(node)
        aliases = aliases_of[function] if function is not None else {}
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            for text in _target_writes(target, aliases):
                if (relative, text) not in _OTHER_OBJECTS:
                    yield relative, node.lineno, text
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _MUTATORS:
            receiver = func.value
            if (isinstance(receiver, ast.Attribute) and receiver.attr in DECIDED_ATTRIBUTES) or (
                _root_name(receiver) in aliases
            ):
                yield relative, node.lineno, ast.unparse(func)
        callee = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ast.unparse(func)
        )
        if callee not in _READERS and (relative, callee) not in _HANDED_TO:
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if _handed_list(argument, aliases, relative):
                    yield relative, node.lineno, f"{callee}({ast.unparse(argument)})"
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "model_copy"
            and (relative, ast.unparse(func)) not in _OTHER_OBJECTS
        ):
            for keyword in node.keywords:
                if keyword.arg != "update":
                    continue
                value = keyword.value
                literal = isinstance(value, ast.Dict) and None not in value.keys
                if not literal or any(_decided_constant(key) for key in value.keys):
                    yield relative, node.lineno, ast.unparse(func)
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _CONSTRUCTORS
            and ast.unparse(func.value).endswith("PaperMetadata")
        ):
            arguments = [*node.args, *(k.value for k in node.keywords if k.arg is None)]
            rebuilt = any(
                isinstance(argument, ast.Dict) and None in argument.keys for argument in arguments
            )
            decided_keywords = any(k.arg in DECIDED_ATTRIBUTES for k in node.keywords)
            if rebuilt or (func.attr == "model_construct" and decided_keywords):
                yield relative, node.lineno, ast.unparse(func)
        setter = (isinstance(func, ast.Name) and func.id == "setattr") or (
            isinstance(func, ast.Attribute)
            and func.attr == "__setattr__"
            and isinstance(func.value, ast.Name)
            and func.value.id == "object"
        )
        if setter and node.args:
            owner = ast.unparse(node.args[0])
            attribute = node.args[1] if len(node.args) > 1 else None
            if _METADATA_NAME.search(owner) or _decided_constant(attribute):
                yield relative, node.lineno, ast.unparse(node)


def test_only_the_decision_writes_the_decided_fields():
    writes = [
        write
        for path in sorted(_PACKAGE.rglob("*.py"))
        if (relative := path.relative_to(_PACKAGE).as_posix()) not in _WRITERS
        for write in _writes_in(relative, path.read_text(encoding="utf-8"))
    ]
    assert writes == []


@pytest.mark.parametrize(
    "source",
    [
        "def f(paper_metadata):\n    paper_metadata.title = 'x'\n",
        "def f(meta):\n    meta.keywords.append('y')\n",
        "def f(meta, field):\n    setattr(meta, field, None)\n",
        "def f(record, name):\n    setattr(record, name, None)\n",
        "def f(meta):\n    object.__setattr__(meta, 'title', 'X')\n",
        "def f(meta):\n    meta = meta.model_copy(update={'abstract': ''})\n",
        "def f(meta, update):\n    return meta.model_copy(update=update)\n",
        "def f(meta):\n    return meta.model_copy(update=dict(title='X'))\n",
        "def f(meta, a):\n    meta.authors[0] = a\n",
        "def f(meta):\n    meta.keywords[:] = []\n",
        "def f(meta):\n    del meta.keywords[0]\n",
        "def f(meta):\n    del meta.title\n",
        "def f(meta, a):\n    authors = meta.authors\n    authors.append(a)\n",
        "def f(meta):\n    meta.authors[0].family = 'X'\n",
        "def f(meta):\n    meta.__dict__['title'] = 'X'\n",
        "def f(meta):\n    vars(meta)['title'] = 'X'\n",
        "def f(meta, PaperMetadata):\n"
        "    return PaperMetadata.model_validate({**meta.model_dump(), 'title': 'X'})\n",
        "def f(PaperMetadata):\n    return PaperMetadata.model_construct(title='X', doi='')\n",
        # A loop over a decided value binds its items.
        "def f(meta):\n    for a in meta.authors:\n        a.family = 'X'\n",
        "def f(meta):\n    for a in meta.authors:\n        a.role.append('x')\n",
        "def f(meta):\n    for i, a in enumerate(meta.authors):\n        a.role[0] = 'x'\n",
        "def f(meta):\n    authors = meta.authors\n    for a in authors:\n        a.email = ''\n",
        # A decided list handed to a function that could change it.
        "def g(authors):\n    authors[0].role.append('x')\ndef f(meta):\n    g(meta.authors)\n",
        "def f(meta, g):\n    g(authors=meta.keywords)\n",
        "def f(meta, g):\n    authors = meta.authors\n    g(authors)\n",
        "def f(meta, g):\n    for a in meta.authors:\n        g(a)\n",
        "def f(metadata, result):\n"
        "    _apply_contributions(metadata.authors, result.contributions)\n",
    ],
)
def test_the_scan_sees_every_kind_of_write(source):
    assert list(_writes_in("pipeline/stages/post_parse.py", source))


def test_the_scan_allows_reading_and_the_listed_handovers():
    reading = (
        "def f(meta, g):\n"
        "    n = len(meta.authors)\n"
        "    g(meta.title)\n"
        "    for a in meta.authors:\n"
        "        g(a.family)\n"
        "    return FieldCandidate('author', 'native', meta.authors), n\n"
    )
    assert list(_writes_in("pipeline/stages/post_parse.py", reading)) == []
    # The role mapping changes the decided authors in place; it is allowed by
    # name in its own module, where the author receipt records it.
    handover = (
        "def f(metadata, result):\n"
        "    _apply_contributions(metadata.authors, result.contributions)\n"
    )
    assert list(_writes_in("extract/research_integrity.py", handover)) == []


def test_the_role_mapping_is_flagged_without_its_listed_handover(monkeypatch):
    listed = ("extract/research_integrity.py", "_apply_contributions")
    monkeypatch.setattr(sys.modules[__name__], "_HANDED_TO", _HANDED_TO - {listed})
    relative = listed[0]
    source = (_PACKAGE / relative).read_text(encoding="utf-8")
    writes = [text for _, _, text in _writes_in(relative, source)]
    assert writes == ["_apply_contributions(metadata.authors)"]


def test_the_scan_ignores_the_listed_other_objects():
    source = "def f(ref, resolved):\n    ref.authors = resolved\n"
    assert list(_writes_in("extract/ref_extractor.py", source)) == []


# ---------------------------------------------------------------------------
# Receipts and the writer
# ---------------------------------------------------------------------------


def test_apply_writes_the_value_and_keeps_the_receipt():
    metadata = PaperMetadata(doi="", title="")
    candidate = FieldCandidate("title", "llm", "A Printed Title")
    decision = FieldDecision("title", candidate.value, candidate, "extracted")

    apply_decision(metadata, decision)

    assert metadata.title == "A Printed Title"
    assert metadata._field_decisions.get("title") is decision
    assert metadata._field_decisions.sources() == {"title": "llm"}
    assert metadata._field_decisions.rules() == {"title": "extracted"}


def test_the_paper_type_decision_writes_the_whole_classification():
    metadata = PaperMetadata(doi="", title="", paper_type="article", oecd_l1="Social Sciences")
    apply_decision(metadata, decide_classification([], notice="erratum"))

    assert (metadata.paper_type, metadata.paper_type_confidence) == ("erratum", None)
    assert (metadata.oecd_l1, metadata.oecd_l2, metadata.oecd_confidence) == ("", "", None)


def test_a_second_decision_supersedes_the_first():
    metadata = PaperMetadata(doi="", title="")
    first = decide_value("journal", FieldCandidate("journal", "llm", "Journal A"))
    second = decide_value("journal", FieldCandidate("journal", "native", "Journal B"))

    apply_decision(metadata, first)
    apply_decision(metadata, second)

    assert metadata.journal == "Journal B"
    assert metadata._field_decisions.superseded == [first]


def _journal_decided():
    metadata = PaperMetadata(doi="", title="")
    apply_decision(metadata, decide_value("journal", FieldCandidate("journal", "llm", "Journal A")))
    return metadata


def _decide_journal_b(metadata):
    apply_decision(
        metadata, decide_value("journal", FieldCandidate("journal", "native", "Journal B"))
    )


@pytest.mark.parametrize(
    "clone",
    [
        lambda metadata: metadata.model_copy(),
        lambda metadata: metadata.model_copy(deep=True),
        copy.copy,
        copy.deepcopy,
        lambda metadata: pickle.loads(pickle.dumps(metadata)),
    ],
)
def test_a_copy_has_its_own_receipts(clone):
    original = _journal_decided()
    copied = clone(original)
    assert copied._field_decisions.get("journal").value == "Journal A"

    _decide_journal_b(copied)

    assert original._field_decisions.get("journal").value == "Journal A"
    assert original._field_decisions.superseded == []


def test_a_copy_that_rewrites_a_field_drops_its_stale_receipt():
    copied = _journal_decided().model_copy(update={"journal": "Journal C"})
    assert copied.journal == "Journal C"
    assert copied._field_decisions.get("journal") is None


def test_a_second_decision_is_logged(caplog):
    metadata = _journal_decided()
    with caplog.at_level("WARNING", logger="bibr.extract.field_decisions"):
        _decide_journal_b(metadata)
    assert "decided twice" in caplog.text


def test_a_test_double_is_written_without_a_ledger():
    metadata = SimpleNamespace(title="")
    candidate = FieldCandidate("title", "llm", "T")
    apply_decision(metadata, FieldDecision("title", "T", candidate, "extracted"))
    assert metadata.title == "T"


def test_incumbent_prefers_the_proposal():
    metadata = PaperMetadata(doi="", title="Constructed")
    assert incumbent_candidate(metadata, "title", source="native") == FieldCandidate(
        "title", "native", "Constructed"
    )
    proposal = FieldCandidate("title", "title_grounding", "Constructed", transforms=("regrounded",))
    metadata._field_decisions.propose(proposal)
    assert incumbent_candidate(metadata, "title", source="native") is proposal


def test_transforms_after_the_decision_are_on_the_receipt():
    authors = [PaperAuthor(author_id=1, given="Ada", family="Quill", affiliation="")]
    decision = decide_authors([FieldCandidate("author", "llm", authors, transforms=("x",))])

    updated = with_transforms(decision, "emails_harvested")

    assert updated.selected.transforms == ("x", "emails_harvested")
    assert updated.considered[0].candidate is updated.selected
    assert updated.value is authors
    assert with_transforms(decide_authors([]), "emails_harvested").selected is None


# ---------------------------------------------------------------------------
# Single-producer fields and statements
# ---------------------------------------------------------------------------


def test_single_value_rule():
    assert decide_value("journal", None).rule == "no_candidate"
    empty = decide_value("journal", FieldCandidate("journal", "llm", None))
    assert (empty.value, empty.selected, empty.source, empty.rule) == (None, None, "llm", "empty")
    used = decide_value("journal", FieldCandidate("journal", "llm", "Journal of Tests"))
    assert (used.value, used.source, used.rule) == ("Journal of Tests", "llm", "extracted")


@pytest.mark.parametrize(
    ("incumbent", "rendered", "keep", "value", "source", "rule"),
    [
        ("Native text.", "Rendered text.", True, "Native text.", "native", "native"),
        (
            "Native text.",
            "Rendered text.",
            False,
            "Rendered text.",
            "resolver",
            "integrity_resolution",
        ),
        (None, "Rendered text.", True, "Rendered text.", "resolver", "integrity_resolution"),
        ("Native text.", None, False, None, "resolver", "none"),
        (None, None, False, None, "resolver", "none"),
    ],
)
def test_statement_rule(incumbent, rendered, keep, value, source, rule):
    decision = decide_statement(
        "funding_statement",
        FieldCandidate("funding_statement", "native", incumbent) if incumbent else None,
        FieldCandidate("funding_statement", "resolver", rendered),
        keep_incumbent=keep,
    )
    assert (decision.value, decision.source, decision.rule) == (value, source, rule)


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------


def _authors(*families):
    return [
        PaperAuthor(author_id=i, given="A.", family=family, affiliation="")
        for i, family in enumerate(families, start=1)
    ]


def test_authors_take_the_first_usable_list():
    llm = FieldCandidate("author", "llm", _authors("Quill"))
    credit = FieldCandidate("author", "credit_statement", _authors("Other"))
    decision = decide_authors([llm, credit])
    assert (decision.value, decision.source, decision.rule) == (llm.value, "llm", "extracted")
    assert [v.accepted for v in decision.considered] == [True, False]


def test_a_vetoed_list_is_never_used():
    llm = FieldCandidate("author", "llm", _authors("Invented"), veto="fabricated")
    recovery = FieldCandidate("author", "llm_recovery", _authors("Quill"))
    decision = decide_authors([llm, recovery])
    assert (decision.source, decision.rule) == ("llm_recovery", "empty_author_recovery")
    assert decision.considered[0].reason == "fabricated"


def test_nothing_usable_reports_the_first_producer():
    decision = decide_authors(
        [FieldCandidate("author", "llm", []), FieldCandidate("author", "credit_statement", [])]
    )
    assert (decision.value, decision.selected, decision.source) == ([], None, "llm")
    assert decision.rule == "none"


def test_a_correction_notice_has_no_authors():
    recovery = FieldCandidate("author", "llm_recovery", _authors("Quill"))
    decision = decide_authors([FieldCandidate("author", "llm", []), recovery], notice="erratum")
    assert (decision.value, decision.selected, decision.rule) == ([], None, "correction_notice")
    # The failed-field source stays the step that produced the list.
    assert decision.source == "llm_recovery"


def test_doc_info_authors_fill_only_an_empty_list():
    native = FieldCandidate("author", "native", _authors("Quill"))
    doc_info = FieldCandidate("author", "doc_info", _authors("Docinfo"))
    assert decide_authors([native, doc_info]).source == "native"
    empty = FieldCandidate("author", None, [])
    assert decide_authors([empty, doc_info]).rule == "doc_info_fill"


# ---------------------------------------------------------------------------
# Abstract and keywords
# ---------------------------------------------------------------------------


def _abstract(value, *, veto=None):
    return FieldCandidate("abstract", "llm", value, veto=veto)


_SECTION = FieldCandidate("abstract", "abstract_section", "  Section abstract.  ")


@pytest.mark.parametrize(
    ("incumbent", "fallback", "absent", "printed", "value", "rule"),
    [
        (_abstract(" Model abstract. "), _SECTION, False, False, "Model abstract.", "extracted"),
        (_abstract(""), _SECTION, False, False, "Section abstract.", "abstract_section_fallback"),
        (_abstract(""), _SECTION, True, False, "", "explicitly_absent"),
        (_abstract(""), _SECTION, True, True, "Section abstract.", "abstract_section_fallback"),
        (
            _abstract("Notice body.", veto="correction notice"),
            _SECTION,
            False,
            False,
            "Section abstract.",
            "abstract_section_fallback",
        ),
        (_abstract(""), None, False, False, "", "none"),
    ],
)
def test_abstract_rule(incumbent, fallback, absent, printed, value, rule):
    decision = decide_abstract(
        incumbent,
        fallback=fallback,
        explicitly_absent=absent,
        printed_abstract=printed,
        abstained=False,
    )
    assert (decision.value, decision.rule) == (value, rule)


def test_abstention_decides_no_abstract():
    decision = decide_abstract(
        _abstract(""),
        fallback=_SECTION,
        explicitly_absent=False,
        printed_abstract=False,
        abstained=True,
    )
    assert (decision.value, decision.selected, decision.rule) == ("", None, "abstained")


@pytest.mark.parametrize(
    ("incumbent", "doc_info", "section", "value", "source"),
    [
        (["model"], ["doc"], ["section"], ["model"], "llm"),
        ([], ["doc"], ["section"], ["doc"], "doc_info"),
        ([], None, ["section"], ["section"], "keywords_section"),
        ([], None, None, [], "llm"),
    ],
)
def test_keyword_rule(incumbent, doc_info, section, value, source):
    decision = decide_keywords(
        FieldCandidate("keywords", "llm", incumbent),
        doc_info=FieldCandidate("keywords", "doc_info", doc_info) if doc_info else None,
        section=FieldCandidate("keywords", "keywords_section", section) if section else None,
        abstained=False,
    )
    assert (decision.value, decision.source) == (value, source)


def test_abstention_decides_no_keywords():
    decision = decide_keywords(
        FieldCandidate("keywords", None, []),
        doc_info=FieldCandidate("keywords", "doc_info", ["doc"]),
        section=FieldCandidate("keywords", "keywords_section", ["section"]),
        abstained=True,
    )
    assert (decision.value, decision.selected, decision.rule) == ([], None, "abstained")


def test_notice_keywords_are_never_used():
    decision = decide_keywords(
        FieldCandidate("keywords", "llm", ["notice"], veto="correction notice"),
        doc_info=None,
        section=None,
        abstained=False,
    )
    assert (decision.value, decision.selected, decision.source) == ([], None, "llm")


# ---------------------------------------------------------------------------
# Paper type
# ---------------------------------------------------------------------------


def test_classification_rule():
    classifier = Classification("article", 0.4, "Social Sciences", "", 0.9)
    label = Classification("review", 0.8, "Social Sciences", "", 0.9)
    decision = decide_classification(
        [
            FieldCandidate("paper_type", "llm_label", label),
            FieldCandidate("paper_type", "classifier", classifier, veto="low confidence"),
        ]
    )
    assert (decision.value, decision.source, decision.rule) == (
        label,
        "llm_label",
        "low_confidence_escalation",
    )
    skipped = decide_classification([])
    assert (skipped.value, skipped.source, skipped.rule) == (
        Classification(),
        "llm",
        "not_classified",
    )


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


def _candidate(candidate_id, text, roles, *, reading_order=0):
    from bibr.extract.front_matter import FrontMatterCandidate

    return FrontMatterCandidate(
        candidate_id=candidate_id,
        source_kind="paragraph",
        reading_order=reading_order,
        page=1,
        bbox=None,
        region_label="text",
        font_size=None,
        font_bold=None,
        section_id=1,
        text_ids=(),
        paragraph_id=1,
        raw_text=text,
        normalized_text=" ".join(text.casefold().split()),
        roles=frozenset(roles),
    )


def _resolution(*candidates, selected=True):
    from bibr.extract.front_matter import FrontMatterBlock, FrontMatterResolution

    block = FrontMatterBlock(
        block_id="selected",
        candidate_ids=tuple(c.candidate_id for c in candidates),
        title_candidate_ids=tuple(c.candidate_id for c in candidates if "title" in c.roles),
    )
    return FrontMatterResolution(
        candidates=tuple(candidates),
        blocks=(block,),
        selected_block_id="selected" if selected else None,
        selection_method="test",
        reason_flags=(),
        allowed_text_ids=frozenset(),
        allowed_section_ids=frozenset(),
    )


_PRINTED = "Effects of Quiet Rooms on Reading Speed"


def _title(incumbent="", **kwargs):
    options = {
        "resolution": _resolution(_candidate("t", _PRINTED, {"title"})),
        "detected_title": None,
        "sections": [],
        "journal": None,
        "publisher": None,
        "scoped": True,
        "abstained": False,
        "prefer_byline_adjacent": False,
        "doc_info": None,
    }
    options.update(kwargs)
    return decide_title(FieldCandidate("title", "llm", incumbent), **options)


def _heading(header):
    from bibr.paper_contents import CanonicalSection

    return SimpleNamespace(
        section_id=3, level=1, header=header, section_type=CanonicalSection.UNKNOWN
    )


_PARAPHRASE = "An Unprinted Paraphrase"
_GENERIC = "Research Article"
_LAYOUT = "A Layout Title Row"


@pytest.mark.parametrize(
    ("scoped", "abstained", "incumbent", "detected", "value", "rule"),
    [
        # Ownership scope: the null-title safety net, then the generic-label rule.
        (True, False, "", None, _PRINTED, "selected_record_title"),
        (True, False, "", _LAYOUT, _PRINTED, "selected_record_title"),
        (True, False, "", _GENERIC, _PRINTED, "grounded_over_generic_label"),
        (True, False, _PRINTED, _GENERIC, _PRINTED, "grounded_over_generic_label"),
        (True, False, _PARAPHRASE, _GENERIC, _GENERIC, "layout_title"),
        (True, False, _PARAPHRASE, _LAYOUT, _PARAPHRASE, "extracted"),
        (True, False, _PARAPHRASE, None, _PARAPHRASE, "extracted"),
        # An abstained record: only the generic-label rule runs.
        (True, True, "", None, "", "abstained"),
        (True, True, "", _LAYOUT, "", "abstained"),
        (True, True, "", _GENERIC, _GENERIC, "layout_title"),
        # Outside ownership scope the layout title wins unless it yields.
        (False, False, "", None, "", "none"),
        (False, False, "", _LAYOUT, _LAYOUT, "layout_title"),
        (False, False, _PARAPHRASE, _LAYOUT, _LAYOUT, "layout_title"),
        (False, False, _PARAPHRASE, _GENERIC, _GENERIC, "layout_title"),
        (False, False, _PRINTED, _GENERIC, _PRINTED, "grounded_over_generic_label"),
    ],
)
def test_title_rule_table(scoped, abstained, incumbent, detected, value, rule):
    decision = _title(incumbent, scoped=scoped, abstained=abstained, detected_title=detected)
    assert (decision.value, decision.rule) == (value, rule)


def test_title_keeps_the_extracted_title():
    decision = _title("A Model Title", detected_title="Some Layout Title")
    assert (decision.value, decision.source, decision.rule) == ("A Model Title", "llm", "extracted")
    # The receipt keeps the gating inputs the rule was given.
    assert dict(decision.flags) == {
        "scoped": True,
        "abstained": False,
        "prefer_byline_adjacent": False,
    }


def test_title_null_recovers_the_one_selected_title_row():
    decision = _title("", detected_title="Some Layout Title")
    assert (decision.value, decision.source, decision.rule) == (
        _PRINTED,
        "front_matter_candidate",
        "selected_record_title",
    )
    assert [issue.code for issue in decision.issues] == ["VAL_TITLE_RECOVERED"]


def test_title_null_falls_back_to_a_safe_layout_title():
    decision = _title("", resolution=_resolution(), detected_title="  A Layout Title Row  ")
    assert (decision.value, decision.source, decision.rule) == (
        "A Layout Title Row",
        "layout_title",
        "layout_title_fallback",
    )


def test_title_null_stays_null_on_an_unsafe_layout_title():
    decision = _title(
        "",
        resolution=_resolution(),
        detected_title="SCIENTIFIC REPORTS",
        journal="Scientific Reports",
    )
    assert (decision.value, decision.selected, decision.source, decision.rule) == (
        "",
        None,
        "llm",
        "none",
    )


def test_title_abstention_skips_the_scoped_fallbacks():
    decision = _title("", abstained=True, detected_title="A Layout Title Row")
    assert (decision.value, decision.rule) == ("", "abstained")


def test_title_generic_layout_label_is_used_under_abstention():
    # Behaviour kept as found: the unscoped layout-title rule runs whenever the
    # layout title is exactly a generic label, even for an abstained record.
    decision = _title("", abstained=True, detected_title="Research Article")
    assert (decision.value, decision.rule) == ("Research Article", "layout_title")


@pytest.mark.parametrize(
    ("incumbent", "value", "rule"),
    [
        (_PRINTED, _PRINTED, "grounded_over_generic_label"),
        # Not printed inside the selected title row: the generic label replaces it.
        ("An Unprinted Paraphrase", "Research Article", "layout_title"),
    ],
)
def test_title_generic_layout_label_yields_only_to_a_printed_title(incumbent, value, rule):
    decision = _title(incumbent, detected_title="Research Article")
    assert (decision.value, decision.rule) == (value, rule)


def test_title_byline_adjacency_is_off_by_default():
    resolution = _resolution(
        _candidate("fr", "Le contrôle des lois de finances", {"title"}, reading_order=0),
        _candidate("by", "Alice Auteur", {"byline"}, reading_order=1),
    )
    assert _title("Review of finance acts", resolution=resolution).rule == "extracted"
    enabled = _title("Review of finance acts", resolution=resolution, prefer_byline_adjacent=True)
    assert (enabled.value, enabled.source, enabled.rule) == (
        "Le contrôle des lois de finances",
        "byline_adjacent",
        "byline_adjacent_title",
    )


def test_unscoped_title_prefers_the_layout_title():
    decision = _title("A Model Title", scoped=False, detected_title="The Layout Title")
    assert (decision.value, decision.source, decision.rule) == (
        "The Layout Title",
        "layout_title",
        "layout_title",
    )
    # Each candidate appears once on the receipt.
    assert [(v.candidate.source, v.accepted) for v in decision.considered] == [
        ("llm", False),
        ("layout_title", True),
    ]
    same = _title("The Layout Title", scoped=False, detected_title="The Layout Title")
    assert (same.source, same.rule) == ("llm", "extracted")


def test_unscoped_title_masthead_guard_keeps_the_model_title():
    decision = _title(
        "A Model Title",
        scoped=False,
        detected_title="SCIENTIFIC REPORTS",
        journal="Scientific Reports",
    )
    assert (decision.value, decision.rule) == ("A Model Title", "masthead_guard")


def test_unscoped_title_scans_headings_then_the_doc_info():
    heading = _title("", scoped=False, sections=[_heading("Flurbulations Revisited")])
    assert (heading.value, heading.source, heading.rule) == (
        "Flurbulations Revisited",
        "section_header",
        "section_header",
    )
    doc_info = FieldCandidate("title", "doc_info", "Docinfo Title")
    filled = _title("", scoped=False, doc_info=doc_info)
    assert (filled.value, filled.source, filled.rule) == (
        "Docinfo Title",
        "doc_info",
        "doc_info_fill",
    )
    kept = _title("A Model Title", scoped=False, doc_info=doc_info)
    assert kept.value == "A Model Title"


def test_native_title_rule():
    decision = decide_title(
        FieldCandidate("title", "native", "Declared Title"),
        resolution=None,
        detected_title=None,
        sections=[],
        journal=None,
        publisher=None,
        scoped=False,
        abstained=False,
        prefer_byline_adjacent=False,
        doc_info=None,
    )
    assert (decision.source, decision.rule) == ("native", "native")


# ---------------------------------------------------------------------------
# A pipeline run leaves one receipt per field
# ---------------------------------------------------------------------------


async def test_pipeline_decides_each_field_once(tmp_path, monkeypatch):
    import bibr.pipeline.stages.post_parse as post_parse_module
    from tests.test_field_states import _smoke_export

    papers = []
    original = post_parse_module.post_parse

    async def capture(*args, **kwargs):
        paper = await original(*args, **kwargs)
        papers.append(paper)
        return paper

    monkeypatch.setattr(post_parse_module, "post_parse", capture)
    result = await _smoke_export(tmp_path, monkeypatch)

    (paper,) = papers
    decisions = paper.field_decisions
    assert decisions is paper.metadata._field_decisions
    assert decisions.superseded == []
    decided = {decision.field for decision in decisions}
    assert {"title", "author", "abstract", "keywords", "published", "journal", "publisher"} <= (
        decided
    )
    assert {"paper_type", "funding_statement", "funding", "affiliations"} <= decided
    # Every receipt describes the value the record ends up with.
    for decision in decisions:
        attributes = FIELD_ATTRIBUTES[decision.field]
        final = tuple(getattr(paper.metadata, attribute) for attribute in attributes)
        written = tuple(decision.value) if len(attributes) > 1 else (decision.value,)
        assert final == written, decision.field
    fields = result["extraction"]["fields"]
    assert fields["title"]["rule"] == decisions.get("title").rule
    assert "rule" not in fields["bib"]
