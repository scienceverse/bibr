"""The statement tagger in bibr: its blocks, how its statements join the resolution, its loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bibr.config import MlOptions, snapshot_settings
from bibr.exceptions import ModelOverrideError
from bibr.extract.field_decisions import field_decisions_of
from bibr.extract.integrity_statements import (
    apply_integrity_resolution,
    resolve_integrity_statements,
)
from bibr.models import PaperMetadata
from bibr.paper_contents import CanonicalSection, PaperContents, PaperSection, PaperSentence
from bibr.processing_warnings import WarningCode
from bibr.statement_tagger import pipeline
from bibr.statement_tagger.pipeline import (
    apply_section_statements,
    apply_statement_tagger,
    content_blocks,
    get_statement_tagger,
    statement_rows,
)
from bibr.statement_tagger.runtime import TaggedStatement
from bibr.utils.model_overrides import (
    begin_run,
    check_model_overrides,
    loaded_model_overrides,
    models_in_use,
    paper_scope,
    reset_loaded_model_overrides,
    run_records,
)

KEYWORDS = {
    "funded": "funding",
    "conflicts": "coi",
    "competing": "coi",
    "thank": "acknowledgment",
    "contributed": "author_contributions",
}


class KeywordTagger:
    """A stand-in for the model: a block is of a type when it holds the type's keyword."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def tag(self, texts):
        self.calls.append(list(texts))
        out: list[TaggedStatement] = []
        for kind in dict.fromkeys(KEYWORDS.values()):
            words = [w for w, k in KEYWORDS.items() if k == kind]
            hit = [any(w in t.lower() for w in words) for t in texts]
            k = 0
            while k < len(hit):
                if not hit[k]:
                    k += 1
                    continue
                end = k
                while end + 1 < len(hit) and hit[end + 1]:
                    end += 1
                out.append(TaggedStatement(kind, k, end, 0.97))
                k = end + 1
        return sorted(out, key=lambda s: (s.start, s.type))


class AllRowsTagger:
    """Tags every paragraph block acknowledgment, and the funding one funding too."""

    def tag(self, texts):
        out = [TaggedStatement("acknowledgment", k, k, 0.95) for k in range(len(texts))]
        out += [
            TaggedStatement("funding", k, k, 0.95)
            for k, text in enumerate(texts)
            if "funded" in text.lower()
        ]
        return sorted(out, key=lambda s: (s.start, s.type))


def _paper(spec) -> PaperContents:
    """``spec``: ``(section_id, header, level, type, [(text, paragraph_id), ...])`` per section."""
    sections = [PaperSection(0, "Root", 0, None, CanonicalSection.UNKNOWN)]
    rows: list[PaperSentence] = []
    for section_id, header, level, kind, texts in spec:
        sections.append(PaperSection(section_id, header, level, 0, kind, 1.0, "exact_alias"))
        for text, paragraph_id in texts:
            rows.append(PaperSentence(len(rows) + 1, text, section_id, paragraph_id, page_number=1))
    return PaperContents(sentences=rows, sections=sections, tables=[], links=[], sections_text={})


def _discussion_paper() -> PaperContents:
    return _paper(
        [
            (1, "Introduction", 1, CanonicalSection.INTRODUCTION, [("We study things.", 1)]),
            (
                2,
                "Discussion",
                1,
                CanonicalSection.DISCUSSION,
                [
                    ("The results hold.", 2),
                    ("They hold widely.", 2),
                    ("We thank the staff for their help.", 3),
                    ("This study was funded by the Science Council.", 4),
                    ("Competing interests: none declared.", 5),
                ],
            ),
            (3, "References", 1, CanonicalSection.REFERENCES, [("Doe J. Work funded.", 6)]),
        ]
    )


def _shared_paragraph_paper() -> PaperContents:
    """The rules bound the funding statement to its sentence; the tagger reads the paragraph."""
    return _paper(
        [
            (1, "Introduction", 1, CanonicalSection.INTRODUCTION, [("We study things.", 1)]),
            (
                2,
                "Discussion",
                1,
                CanonicalSection.DISCUSSION,
                [
                    ("The results hold.", 2),
                    ("We thank the staff for their help.", 3),
                    ("This study was funded by the Science Council.", 3),
                    ("Competing interests: none declared.", 4),
                ],
            ),
        ]
    )


def _settings(**ml):
    settings = snapshot_settings()
    for name, value in ml.items():
        setattr(settings.ml, name, value)
    return settings


@pytest.fixture(autouse=True)
def _fresh():
    pipeline.reset_statement_tagger()
    reset_loaded_model_overrides()
    yield
    pipeline.reset_statement_tagger()
    reset_loaded_model_overrides()


# -- blocks ------------------------------------------------------------------------------


def test_blocks_are_paragraphs_after_their_printed_headings_without_the_reference_list():
    contents = _discussion_paper()
    contents.sentences.append(PaperSentence(99, "E = mc^2", 2, 9, is_display_formula=True))
    blocks = content_blocks(contents)
    assert [(b.text, b.text_ids) for b in blocks] == [
        ("Introduction", ()),
        ("We study things.", (1,)),
        ("Discussion", ()),
        ("The results hold. They hold widely.", (2, 3)),
        ("We thank the staff for their help.", (4,)),
        ("This study was funded by the Science Council.", (5,)),
        ("Competing interests: none declared.", (6,)),
    ]
    assert blocks[0].heading and not blocks[1].heading


def test_synthetic_and_label_headings_are_not_blocks():
    contents = _paper(
        [
            (1, "Table 1", 1, CanonicalSection.TABLE, [("a b c", 1)]),
            (2, "Body", 1, CanonicalSection.DISCUSSION, [("Text.", 2)]),
        ]
    )
    contents.sections[2].header_is_synthetic = True
    assert [b.text for b in content_blocks(contents)] == ["a b c", "Text."]


def test_a_statement_is_trimmed_to_its_own_label_and_cut_at_the_next():
    contents = _paper(
        [
            (
                1,
                "Declarations",
                1,
                CanonicalSection.DISCUSSION,
                [
                    ("Funding: The work was funded by the Council.", 1),
                    ("Conflicts of interest: none.", 1),
                    ("The authors declare nothing else.", 1),
                ],
            )
        ]
    )
    blocks = content_blocks(contents)
    rows = {r.text_id: r for r in contents.sentences}
    run = TaggedStatement("funding", 1, 1, 0.9)
    assert statement_rows("funding_statement", run, blocks, rows) == (1,)
    assert statement_rows("coi_statement", run, blocks, rows) == (2, 3)


# -- the combination rule -----------------------------------------------------------------


def _resolve(contents, tagger):
    resolution = resolve_integrity_statements(contents, mode="shadow")
    before = resolution
    resolution = apply_statement_tagger(contents, resolution, None, tagger=tagger)
    metadata = PaperMetadata(doi="", title="T")
    apply_integrity_resolution(contents, metadata, resolution)
    return before, resolution, metadata


def test_the_tagger_statement_takes_the_field_and_says_so():
    contents = _shared_paragraph_paper()
    before, resolution, metadata = _resolve(contents, KeywordTagger())
    (rules,) = before.candidate_indices("funding_statement")
    assert before.candidates[rules].text_ids == (4,)
    # The tagger's statement is its paragraph, the acknowledgment sentence with it.
    assert metadata.funding_statement == (
        "We thank the staff for their help. This study was funded by the Science Council."
    )
    sources = field_decisions_of(metadata).sources()
    assert sources["funding_statement"] == "statement_tagger"
    (index,) = resolution.selected_indices("funding_statement")
    candidate = resolution.candidates[index]
    assert candidate.method == "statement_tagger"
    assert candidate.classification_source == "statement_tagger"
    assert candidate.classification_score == pytest.approx(0.97)
    assert candidate.text_ids == (3, 4)
    assert candidate.reason_flags == ("statement_tagger", "replaces_rules")
    # The rules' candidates stay in the resolution.
    assert resolution.candidates[: len(before.candidates)] == before.candidates
    codes = [(w.code, w.message) for w in contents.processing_warnings]
    assert (
        WarningCode.STATEMENT_TAGGER,
        "funding_statement filled by the statement tagger",
    ) in codes


def test_a_tagger_statement_that_reads_as_the_rules_changes_nothing():
    contents = _discussion_paper()
    before, resolution, metadata = _resolve(contents, KeywordTagger())
    expected = PaperMetadata(doi="", title="T")
    apply_integrity_resolution(_discussion_paper(), expected, before)
    # The tagger reads the funding and competing interests rows the rules chose.
    assert metadata.funding_statement == "This study was funded by the Science Council."
    assert metadata.coi_statement == "Competing interests: none declared."
    assert field_decisions_of(metadata).sources() == field_decisions_of(expected).sources()
    assert field_decisions_of(metadata).sources()["funding_statement"] != "statement_tagger"
    for field in ("funding_statement", "coi_statement"):
        assert resolution.selected_indices(field) == before.selected_indices(field)
        assert resolution.legacy_indices(field) == before.legacy_indices(field)
    assert not [c for c in resolution.candidates if c.method == "statement_tagger"]
    messages = [w.message for w in contents.processing_warnings]
    assert "funding_statement filled by the statement tagger" not in messages
    assert "coi_statement filled by the statement tagger" not in messages


def test_rows_that_differ_in_whitespace_only_read_as_the_same_statement():
    from dataclasses import replace

    contents = _discussion_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    (rules,) = resolution.candidate_indices("funding_statement")
    # The same sentence printed twice, the second time with other spacing.
    contents.sentences.append(
        PaperSentence(50, "This  study was funded\nby the Science  Council. ", 2, 9)
    )
    same = replace(resolution.candidates[rules], method="statement_tagger", text_ids=(50,))
    other = replace(same, text_ids=(4,))
    assert pipeline.unchanged_fields(contents, resolution, [("funding_statement", same)]) == {
        "funding_statement"
    }
    assert pipeline.unchanged_fields(contents, resolution, [("funding_statement", other)]) == set()


def test_flags_say_whether_the_tagger_replaced_the_rules_or_filled_a_gap():
    contents = _paper(
        [
            (
                1,
                "Discussion",
                1,
                CanonicalSection.DISCUSSION,
                [
                    ("We thank the staff for their help.", 1),
                    ("This study was funded by the Science Council.", 1),
                ],
            ),
            (2, "Notes", 1, CanonicalSection.DISCUSSION, [("Competing claims were set.", 2)]),
        ]
    )
    _before, resolution, _metadata = _resolve(contents, KeywordTagger())
    flags = {
        c.field: c.reason_flags for c in resolution.candidates if c.method == "statement_tagger"
    }
    assert flags["funding_statement"] == ("statement_tagger", "replaces_rules")
    assert flags["coi_statement"] == ("statement_tagger", "rules_found_none")


def test_an_acknowledgment_inside_a_body_section_gets_a_section_of_its_own():
    contents = _discussion_paper()
    _resolve(contents, KeywordTagger())
    by_id = {s.section_id: s for s in contents.sections}
    row = next(r for r in contents.sentences if r.text.startswith("We thank"))
    section = by_id[row.section_id]
    assert section.section_type == CanonicalSection.ACKNOWLEDGMENT
    assert section.header == "Acknowledgments" and section.header_is_synthetic
    assert section.classification_source == "statement_tagger"
    assert section.parent_section_id == 0 and section.level == 1
    # Right after the section the rows sat in; the other rows stay where they were.
    order = [s.section_id for s in contents.sections]
    assert order.index(section.section_id) == order.index(2) + 1
    assert {r.section_id for r in contents.sentences if r.text_id in (2, 3, 5, 6)} == {2}
    assert "We thank" in contents.sections_text[section.section_id]
    assert "We thank" not in contents.sections_text.get(2, "")
    codes = [(w.code, w.message) for w in contents.processing_warnings]
    assert (WarningCode.STATEMENT_TAGGER, "acknowledgment filled by the statement tagger") in codes
    # The tagger read the funding row the rules chose: no change, no warning.
    assert (
        WarningCode.STATEMENT_TAGGER,
        "funding_statement filled by the statement tagger",
    ) not in codes


def test_a_section_of_tagged_rows_only_is_retyped():
    contents = _paper(
        [
            (1, "Body", 1, CanonicalSection.DISCUSSION, [("Text.", 1)]),
            (2, "Notes", 1, CanonicalSection.UNKNOWN, [("Each author contributed equally.", 2)]),
        ]
    )
    statements = KeywordTagger().tag([b.text for b in content_blocks(contents)])
    done = apply_section_statements(contents, statements, content_blocks(contents))
    assert done == ["author_contributions"]
    notes = contents.sections[2]
    assert notes.header == "Notes" and notes.section_type == CanonicalSection.AUTHOR_CONTRIBUTIONS
    assert notes.classification_source == "statement_tagger"
    assert len(contents.sections) == 3


@pytest.mark.parametrize("heading", ["Discussion", "2. Methods", "Conclusions", "Appendix A"])
def test_a_section_under_a_printed_body_heading_is_never_retyped(heading):
    contents = _paper(
        [
            (1, "Introduction", 1, CanonicalSection.INTRODUCTION, [("Text.", 1)]),
            (
                2,
                heading,
                1,
                CanonicalSection.DISCUSSION,
                [("We thank the staff.", 2), ("This study was funded by the Council.", 3)],
            ),
        ]
    )
    before = [(s.section_id, s.header, s.section_type) for s in contents.sections]
    resolution = resolve_integrity_statements(contents, mode="shadow")
    resolution = apply_statement_tagger(contents, resolution, None, tagger=AllRowsTagger())
    # Every row is tagged acknowledgment, yet the heading says what the section is.
    assert [(s.section_id, s.header, s.section_type) for s in contents.sections] == before
    assert {r.section_id for r in contents.sentences if r.text_id in (2, 3)} == {2}
    messages = [w.message for w in contents.processing_warnings]
    assert "acknowledgment filled by the statement tagger" not in messages
    # Its rows still give the statement fields their values.
    metadata = PaperMetadata(doi="", title="T")
    apply_integrity_resolution(contents, metadata, resolution)
    assert "funded by the Council" in metadata.funding_statement


def test_a_section_under_an_unrecognised_printed_heading_is_still_retyped():
    contents = _paper(
        [
            (1, "Introduction", 1, CanonicalSection.INTRODUCTION, [("Text.", 1)]),
            (2, "Thanks", 1, CanonicalSection.UNKNOWN, [("We thank the staff.", 2)]),
        ]
    )
    statements = KeywordTagger().tag([b.text for b in content_blocks(contents)])
    assert apply_section_statements(contents, statements, content_blocks(contents)) == [
        "acknowledgment"
    ]
    assert contents.sections[2].section_type == CanonicalSection.ACKNOWLEDGMENT


def test_a_paper_that_has_the_section_keeps_it_untouched():
    contents = _paper(
        [
            (1, "Acknowledgements", 1, CanonicalSection.ACKNOWLEDGMENT, [("We thank A.", 1)]),
            (2, "Discussion", 1, CanonicalSection.DISCUSSION, [("We thank B too.", 2)]),
        ]
    )
    statements = KeywordTagger().tag([b.text for b in content_blocks(contents)])
    assert apply_section_statements(contents, statements, content_blocks(contents)) == []
    assert [r.section_id for r in contents.sentences] == [1, 2]
    assert len(contents.sections) == 3


def test_rows_of_a_footnote_never_become_a_section():
    contents = _paper(
        [
            (1, "Body", 1, CanonicalSection.DISCUSSION, [("Text.", 1)]),
            (2, "Footnote 3", 1, CanonicalSection.FOOTNOTE, [("Each author contributed.", 2)]),
        ]
    )
    statements = KeywordTagger().tag([b.text for b in content_blocks(contents)])
    assert statements
    assert apply_section_statements(contents, statements, content_blocks(contents)) == []
    assert [s.section_type for s in contents.sections][2] == CanonicalSection.FOOTNOTE
    assert [r.section_id for r in contents.sentences] == [1, 2]


def test_applying_twice_adds_each_warning_once():
    contents = _discussion_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    tagger = KeywordTagger()
    apply_statement_tagger(contents, resolution, None, tagger=tagger)
    apply_statement_tagger(contents, resolution, None, tagger=tagger)
    codes = [
        w.message for w in contents.processing_warnings if w.code == WarningCode.STATEMENT_TAGGER
    ]
    assert len(codes) == len(set(codes))


# -- off by default -------------------------------------------------------------------------


def test_the_tagger_is_off_by_default():
    assert MlOptions.model_fields["statement_tagger_model_id"].default is None
    settings = snapshot_settings()
    assert settings.ml.statement_tagger_model_id is None
    assert "ML_STATEMENT_TAGGER_MODEL_ID" not in models_in_use(settings)
    assert get_statement_tagger(settings) is None
    contents = _discussion_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    sections = [(s.section_id, s.section_type) for s in contents.sections]
    assert apply_statement_tagger(contents, resolution, settings) is resolution
    assert [(s.section_id, s.section_type) for s in contents.sections] == sections
    assert contents.processing_warnings == []
    assert loaded_model_overrides() == []


def test_the_dry_run_lists_the_tagger_only_when_set(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from bibr.config import Settings
    from bibr.local.cli import dry_run

    config = SimpleNamespace(no_llm=False, mode="fast", uses_llm=False)
    labels = [
        label for label, _ in dry_run._dry_run_model_specs(config, "geom", "ner", needs_ocr=False)
    ]
    assert "statement tagger" not in labels
    monkeypatch.setattr(Settings.ml, "statement_tagger_model_id", str(tmp_path))
    specs = dict(dry_run._dry_run_model_specs(config, "geom", "ner", needs_ocr=False))
    assert specs["statement tagger"] == f"local bundle: {tmp_path}"
    config.no_llm = True
    assert "statement tagger" not in dict(
        dry_run._dry_run_model_specs(config, "geom", "ner", needs_ocr=False)
    )


# -- loading and the override path -----------------------------------------------------------


def test_a_bundle_that_does_not_exist_stops_the_run(tmp_path):
    settings = _settings(statement_tagger_model_id=str(tmp_path / "missing"))
    with pytest.raises(ModelOverrideError, match="ML_STATEMENT_TAGGER_MODEL_ID"):
        get_statement_tagger(settings)


def test_the_preflight_names_a_missing_or_wrong_bundle(tmp_path):
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    settings = _settings(statement_tagger_model_id=str(tmp_path / "missing"))
    (problem,), _ = check_model_overrides(settings)
    assert problem.startswith("ML_STATEMENT_TAGGER_MODEL_ID=") and "no such file" in problem
    settings.ml.statement_tagger_model_id = str(
        write_bundle(tmp_path / "other", model="ner_parser")
    )
    (problem,), _ = check_model_overrides(settings)
    assert "'ner_parser' model, not a 'statement_tagger'" in problem
    # A graph that does not answer as the manifest says is caught before the run.
    settings.ml.statement_tagger_model_id = str(write_bundle(tmp_path / "short", n_out=3))
    (problem,), _ = check_model_overrides(settings)
    assert "shape" in problem
    settings.ml.statement_tagger_model_id = str(write_bundle(tmp_path / "good"))
    assert check_model_overrides(settings) == ([], [])


def test_a_bundle_is_loaded_once_and_recorded_by_checksum(tmp_path):
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    bundle = write_bundle(tmp_path / "statements-candidate")
    settings = _settings(statement_tagger_model_id=str(bundle))
    first = get_statement_tagger(settings)
    assert get_statement_tagger(settings) is first
    (record,) = loaded_model_overrides()
    assert record["setting"] == "ML_STATEMENT_TAGGER_MODEL_ID" and record["loaded"] is True
    by_name = {Path(f["path"]).name: f["sha256"] for f in record["files"]}
    assert set(by_name) == {"bibr_onnx.json", "model.onnx", "tokenizer.json"}
    for name, digest in by_name.items():
        assert digest == hashlib.sha256((bundle / "onnx" / name).read_bytes()).hexdigest()


def test_the_export_lists_the_tagger_by_name_and_checksums(tmp_path):
    pytest.importorskip("onnx")
    from bibr.utils.model_provenance import export_models, reset_model_provenance
    from tests.statement_tagger.bundle import write_bundle

    reset_model_provenance()
    bundle = write_bundle(tmp_path / "machine" / "statements-candidate")
    assert "ML_STATEMENT_TAGGER_MODEL_ID" not in {
        row["setting"] for row in export_models(snapshot_settings())
    }
    rows = export_models(_settings(statement_tagger_model_id=str(bundle)))
    (row,) = [r for r in rows if r["setting"] == "ML_STATEMENT_TAGGER_MODEL_ID"]
    # A local bundle is named by its directory and a checksum of its files.
    assert row["model"] == "statement tagger"
    assert row["id"].partition("@")[0] == "statements-candidate"
    assert row["override"] is True and row["revision"] is None
    assert {f["name"] for f in row["files"]} == {
        "onnx/bibr_onnx.json",
        "onnx/model.onnx",
        "onnx/tokenizer.json",
    }
    assert str(tmp_path) not in repr(row)
    rows = export_models(_settings(statement_tagger_model_id=str(bundle)), extraction=False)
    assert "ML_STATEMENT_TAGGER_MODEL_ID" not in {r["setting"] for r in rows}
    reset_model_provenance()


def test_end_to_end_with_a_bundle_on_disk(tmp_path):
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    settings = _settings(statement_tagger_model_id=str(write_bundle(tmp_path / "t")))
    contents = _shared_paragraph_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    resolution = apply_statement_tagger(contents, resolution, settings)
    metadata = PaperMetadata(doi="", title="T")
    apply_integrity_resolution(contents, metadata, resolution)
    assert field_decisions_of(metadata).sources()["funding_statement"] == "statement_tagger"
    assert metadata.funding_statement.startswith("We thank the staff")
    assert "funded by the Science Council" in metadata.funding_statement


# -- an override that does not load, or does not run -----------------------------------------


def _hub_from(bundle: Path, calls: list[str]):
    """``hf_download_or_cached`` answering from ``bundle``'s ``onnx/`` as a Hub repo would."""

    def download(repo_id, filename, revision=None):
        calls.append(f"{repo_id}@{revision}:{filename}")
        return str(bundle / filename)

    return download


def test_a_hub_bundle_is_fetched_at_its_revision_and_loaded(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    calls: list[str] = []
    bundle = write_bundle(tmp_path / "hub")
    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", _hub_from(bundle, calls))
    settings = _settings(
        statement_tagger_model_id="org/statements", statement_tagger_revision="abc123"
    )
    assert check_model_overrides(settings) == ([], [])
    assert calls[0] == "org/statements@abc123:onnx/bibr_onnx.json"
    assert set(calls[1:]) == {
        "org/statements@abc123:onnx/model.onnx",
        "org/statements@abc123:onnx/tokenizer.json",
    }
    tagger = get_statement_tagger(settings)
    assert tagger is not None and tagger.directory == bundle / "onnx"


def test_a_hub_bundle_offline_and_not_cached_stops_the_run(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(repo_id, filename, revision=None):
        raise LocalEntryNotFoundError(
            "Cannot find the requested files in the disk cache and outgoing traffic has been "
            "disabled."
        )

    monkeypatch.setattr("bibr.utils.hf_cache.hf_download_or_cached", offline)
    settings = _settings(statement_tagger_model_id="org/statements")
    (problem,), _ = check_model_overrides(settings)
    assert problem.startswith("ML_STATEMENT_TAGGER_MODEL_ID=")
    assert "the Hub could not be reached and the files are not in the local cache" in problem
    pipeline.reset_statement_tagger()
    begin_run(settings)
    contents = _discussion_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    with pytest.raises(ModelOverrideError, match="the Hub could not be reached"):
        apply_statement_tagger(contents, resolution, settings)
    (record,) = run_records()
    assert record["setting"] == "ML_STATEMENT_TAGGER_MODEL_ID" and record["loaded"] is False
    assert record["reason"].startswith("the Hub could not be reached")
    assert pipeline._TAGGER is None  # nothing in the tagger's place


def test_a_graph_that_does_not_open_stops_the_run(tmp_path):
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    bundle = write_bundle(tmp_path / "broken")
    (bundle / "onnx" / "model.onnx").write_bytes(b"not an onnx graph")
    settings = _settings(statement_tagger_model_id=str(bundle))
    (problem,), _ = check_model_overrides(settings)
    assert "does not open as an ONNX graph" in problem
    with pytest.raises(ModelOverrideError, match="does not open as an ONNX graph"):
        get_statement_tagger(settings)


def test_a_graph_that_fails_its_probe_stops_the_run_at_load(tmp_path):
    pytest.importorskip("onnx")
    from tests.statement_tagger.bundle import write_bundle

    settings = _settings(statement_tagger_model_id=str(write_bundle(tmp_path / "s", n_out=3)))
    with pytest.raises(ModelOverrideError, match=r"the graph returns probs of shape \(3, 3\)"):
        get_statement_tagger(settings)


def test_serve_refuses_to_start_on_a_broken_tagger(tmp_path):
    pytest.importorskip("onnx")
    from bibr.exceptions import ConfigurationError
    from bibr.serve.app import _check_model_overrides
    from tests.statement_tagger.bundle import write_bundle

    settings = _settings(statement_tagger_model_id=str(write_bundle(tmp_path / "s", n_out=3)))
    with pytest.raises(ConfigurationError, match="ML_STATEMENT_TAGGER_MODEL_ID"):
        _check_model_overrides(settings)
    settings.ml.statement_tagger_model_id = str(write_bundle(tmp_path / "good"))
    _check_model_overrides(settings)


class RaisingTagger:
    def tag(self, texts):
        raise RuntimeError("the graph fails on this input")


def test_an_inference_error_keeps_the_rules_and_three_in_a_row_stop_the_run(tmp_path):
    settings = _settings(statement_tagger_model_id=str(tmp_path / "candidate"))
    for paper in ("p1", "p2"):
        contents = _discussion_paper()
        resolution = resolve_integrity_statements(contents, mode="shadow")
        with paper_scope(paper):
            out = apply_statement_tagger(contents, resolution, settings, tagger=RaisingTagger())
        assert out is resolution  # the rules' statements, untouched
        assert [w.code for w in contents.processing_warnings] == [
            WarningCode.MODEL_OVERRIDE_INFERENCE_FAILED
        ]
        assert "ML_STATEMENT_TAGGER_MODEL_ID" in contents.processing_warnings[0].message
    contents = _discussion_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    with (
        paper_scope("p3"),
        pytest.raises(ModelOverrideError, match="raised at inference on 3 papers in a row"),
    ):
        apply_statement_tagger(contents, resolution, settings, tagger=RaisingTagger())


def test_a_paper_the_tagger_ran_on_ends_the_streak(tmp_path):
    settings = _settings(statement_tagger_model_id=str(tmp_path / "candidate"))
    for paper, tagger in (
        ("p1", RaisingTagger()),
        ("p2", RaisingTagger()),
        ("p3", KeywordTagger()),
        ("p4", RaisingTagger()),
        ("p5", RaisingTagger()),
    ):
        contents = _discussion_paper()
        resolution = resolve_integrity_statements(contents, mode="shadow")
        with paper_scope(paper):
            apply_statement_tagger(contents, resolution, settings, tagger=tagger)


def test_an_injected_tagger_without_the_setting_raises_as_before():
    contents = _discussion_paper()
    resolution = resolve_integrity_statements(contents, mode="shadow")
    with pytest.raises(RuntimeError, match="the graph fails"):
        apply_statement_tagger(contents, resolution, _settings(), tagger=RaisingTagger())
