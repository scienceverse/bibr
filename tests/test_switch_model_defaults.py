"""scripts/switch_model_defaults.py: moving a default model pin everywhere it is written."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import switch_model_defaults as switch

REPO = Path(__file__).resolve().parents[1]
FILES = (
    "bibr/config.py",
    "Dockerfile.serve",
    "docs/guides/configuration.md",
    "docs/guides/classifiers.md",
)
NEW = "a" * 40


@pytest.fixture
def tree(tmp_path):
    for name in FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, tmp_path / name)
    return tmp_path


def _default(root, setting, which):
    pin = switch.PINS[setting]
    text = (root / "bibr/config.py").read_text(encoding="utf-8")
    field = pin.id_field if which == "id" else pin.revision_field
    return switch._field_default(text, pin.config_class, field)[2]


def _changed_lines(root, name):
    old = (REPO / name).read_text(encoding="utf-8").splitlines()
    new = (root / name).read_text(encoding="utf-8").splitlines()
    assert len(old) == len(new)
    return [(a, b) for a, b in zip(old, new, strict=True) if a != b]


def test_a_new_revision_is_written_to_config_dockerfile_and_docs_table(tree, capsys):
    setting = "ML_SECTION_CLASSIFIER_MODEL_ID"
    repo = _default(tree, setting, "id")
    old = _default(tree, setting, "revision")
    assert switch.main([f"{setting}={repo}@{NEW}", "--root", str(tree)]) == 0
    assert _default(tree, setting, "revision") == NEW
    assert _default(tree, setting, "id") == repo
    config = _changed_lines(tree, "bibr/config.py")
    assert config == [(f'        "{old}",', f'        "{NEW}",')]
    docker = _changed_lines(tree, "Dockerfile.serve")
    assert docker == [
        (f"ARG SECTION_CLASSIFIER_REVISION={old}", f"ARG SECTION_CLASSIFIER_REVISION={NEW}")
    ]
    table = _changed_lines(tree, "docs/guides/configuration.md")
    assert len(table) == 1 and f"`{NEW[:8]}`" in table[0][1] and f"`{old[:8]}`" in table[0][0]
    assert _changed_lines(tree, "docs/guides/classifiers.md") == []
    assert "changed 3 file(s)" in capsys.readouterr().out


def test_a_model_card_link_follows_the_revision(tree):
    setting = "ML_PAPER_CLASSIFIER_MODEL_ID"
    repo = _default(tree, setting, "id")
    old = _default(tree, setting, "revision")
    assert switch.main([f"{setting}={repo}@{NEW}", "--root", str(tree)]) == 0
    link = _changed_lines(tree, "docs/guides/classifiers.md")
    assert len(link) == 1 and f"{repo}/blob/{NEW}/" in link[0][1] and old in link[0][0]


def test_several_pins_from_json_and_an_unset_revision_gets_one(tree):
    pins = tree / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "NER_PARSER_CKPT": ["org/parser-new", "b" * 40],
                "REF_TEXT_SEG_MODEL_ID": ["org/text-seg", "c" * 40],
                "ML_FRONT_TAGGER_MODEL_ID": ["org/front-tagger", "d" * 40],
            }
        )
    )
    assert switch.main(["--from-json", str(pins), "--root", str(tree)]) == 0
    assert _default(tree, "NER_PARSER_CKPT", "id") == "org/parser-new"
    assert _default(tree, "NER_PARSER_CKPT", "revision") == "b" * 40
    assert _default(tree, "REF_TEXT_SEG_MODEL_ID", "id") == "org/text-seg"
    assert _default(tree, "REF_TEXT_SEG_MODEL_ID", "revision") == "c" * 40
    assert _default(tree, "ML_FRONT_TAGGER_MODEL_ID", "revision") == "d" * 40
    assert "| Reference parser | `org/parser-new` | `bbbbbbbb` |" in (
        tree / "docs/guides/configuration.md"
    ).read_text(encoding="utf-8")


def test_an_old_pin_in_an_unknown_place_stops_it_and_nothing_is_written(tree, capsys):
    setting = "ML_SECTION_CLASSIFIER_MODEL_ID"
    repo = _default(tree, setting, "id")
    old = _default(tree, setting, "revision")
    extra = tree / "docs/guides/other.md"
    extra.write_text(f"Pinned at `{old[:8]}` for now.\n")
    before = {name: (tree / name).read_bytes() for name in FILES}
    assert switch.main([f"{setting}={repo}@{NEW}", "--root", str(tree)]) == 1
    err = capsys.readouterr().err
    assert "does not know how to update" in err and "docs/guides/other.md:1" in err
    assert {name: (tree / name).read_bytes() for name in FILES} == before


def test_a_new_repo_id_fails_on_every_other_mention_of_the_old_one(tree, capsys):
    setting = "ML_PAPER_CLASSIFIER_MODEL_ID"
    old_repo = _default(tree, setting, "id")
    assert switch.main([f"{setting}=org/new-paper@{NEW}", "--root", str(tree)]) == 1
    err = capsys.readouterr().err
    assert "docs/guides/classifiers.md" in err
    assert _default(tree, setting, "id") == old_repo


def test_a_dockerfile_arg_that_disagrees_with_the_config_is_refused(tree, capsys):
    docker = tree / "Dockerfile.serve"
    text = docker.read_text(encoding="utf-8")
    old = _default(tree, "LAYOUT_MODEL_ID", "revision")
    docker.write_text(
        text.replace(f"ARG LAYOUT_MODEL_REVISION={old}", "ARG LAYOUT_MODEL_REVISION=main"),
        encoding="utf-8",
    )
    repo = _default(tree, "LAYOUT_MODEL_ID", "id")
    assert switch.main([f"LAYOUT_MODEL_ID={repo}@{NEW}", "--root", str(tree)]) == 2
    assert "reconcile them first" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        (f"WTPSPLIT_MODEL=org/sat@{NEW}", "not a setting this script pins"),
        ("NER_PARSER_CKPT=org/parser@main", "not a 40-hex commit"),
        (f"NER_PARSER_CKPT=/models/parser@{NEW}", "not a Hub repo id"),
        ("NER_PARSER_CKPT", "expected SETTING=org/name@<40-hex>"),
    ],
)
def test_bad_input_is_refused(tree, capsys, arg, message):
    assert switch.main([arg, "--root", str(tree)]) == 2
    assert message in capsys.readouterr().err


def test_check_writes_nothing(tree, capsys):
    setting = "ML_SECTION_CLASSIFIER_MODEL_ID"
    repo = _default(tree, setting, "id")
    before = {name: (tree / name).read_bytes() for name in FILES}
    assert switch.main([f"{setting}={repo}@{NEW}", "--root", str(tree), "--check"]) == 0
    assert {name: (tree / name).read_bytes() for name in FILES} == before
    assert "would change 3 file(s)" in capsys.readouterr().out


def test_the_table_covers_every_model_setting_and_reads_the_shipped_defaults():
    from bibr.config import GlobalSettings, LayoutOptions, MlOptions
    from bibr.utils.model_overrides import MODEL_SETTINGS

    assert set(switch.PINS) == {s.env for s in MODEL_SETTINGS} | {
        "LAYOUT_MODEL_ID",
        "LAYOUT_ONNX_MODEL_ID",
    }
    classes = {
        "GlobalSettings": GlobalSettings,
        "MlOptions": MlOptions,
        "LayoutOptions": LayoutOptions,
    }
    for spec in MODEL_SETTINGS:
        pin = switch.PINS[spec.env]
        assert (pin.id_field, pin.revision_field) == (spec.attr[-1], spec.revision_attr[-1])
    for setting, pin in switch.PINS.items():
        fields = classes[pin.config_class].model_fields
        assert _default(REPO, setting, "id") == fields[pin.id_field].default, setting
        assert _default(REPO, setting, "revision") == fields[pin.revision_field].default, setting


@pytest.mark.parametrize("setting", sorted(switch.PINS))
def test_the_real_tree_has_no_pin_the_script_cannot_move(setting):
    repo = _default(REPO, setting, "id")
    if repo is None:
        pytest.skip("no default model")
    _, stale, _ = switch.plan(REPO, {setting: (repo, NEW)})
    assert stale == []


def _section_pin(tree):
    setting = "ML_SECTION_CLASSIFIER_MODEL_ID"
    return setting, _default(tree, setting, "id"), _default(tree, setting, "revision")


@pytest.mark.parametrize(
    ("name", "line"),
    [
        ("bibr/fetch.py", 'URL = "https://huggingface.co/{repo}/resolve/{rev}/onnx/model.onnx"'),
        ("README.md", "See https://huggingface.co/{repo}/commit/{rev} for the export."),
        (".github/workflows/models.yml", "      - run: hf download {repo} --revision {rev}"),
        ("pyproject.toml", 'section-revision = "{rev}"'),
        ("entrypoint.sh", "REV={short}"),
        ("evaluation/pins.json", '{{"section": "{rev}"}}'),
        ("ci/models.yaml", "section: /models/{short}/onnx"),
    ],
)
def test_an_old_revision_anywhere_in_the_tree_stops_it(tree, capsys, name, line):
    setting, repo, old = _section_pin(tree)
    target = tree / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(line.format(repo=repo, rev=old, short=old[:8]) + "\n")
    before = {f: (tree / f).read_bytes() for f in FILES}
    assert switch.main([f"{setting}={repo}@{NEW}", "--root", str(tree)]) == 1
    err = capsys.readouterr().err
    assert f"{name}:1" in err
    assert {f: (tree / f).read_bytes() for f in FILES} == before


def test_a_model_card_link_to_the_old_repo_stops_a_repo_change(tree, capsys):
    setting, repo, old = _section_pin(tree)
    (tree / "docs/guides/other.md").write_text(f"The [card](https://huggingface.co/{repo}).\n")
    assert switch.main([f"{setting}=org/new-section@{NEW}", "--root", str(tree)]) == 1
    assert "docs/guides/other.md:1" in capsys.readouterr().err


def test_resolve_and_commit_links_in_the_docs_follow_the_revision(tree):
    setting, repo, old = _section_pin(tree)
    doc = tree / "docs/guides/other.md"
    doc.write_text(
        f"[graph](https://huggingface.co/{repo}/resolve/{old}/onnx/model.onnx) and "
        f"[the commit](https://huggingface.co/{repo}/commit/{old}).\n"
    )
    assert switch.main([f"{setting}={repo}@{NEW}", "--root", str(tree)]) == 0
    text = doc.read_text(encoding="utf-8")
    assert old not in text
    assert f"{repo}/resolve/{NEW}/onnx" in text and f"{repo}/commit/{NEW})" in text


def test_a_longer_hex_string_or_name_is_not_the_old_pin(tree):
    setting, repo, old = _section_pin(tree)
    (tree / "bibr/other.py").write_text(f'A = "{old[:8]}ff"\nB = "{repo}-v2"\nC = "x{old}"\n')
    _, stale, _ = switch.plan(tree, {setting: ("org/new-section", NEW)})
    assert not any(entry.startswith("bibr/other.py") for entry in stale)


def test_in_a_git_checkout_only_tracked_files_are_searched(tree):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    import subprocess

    setting, repo, old = _section_pin(tree)
    (tree / "notes.md").write_text(f"pinned at {old}\n")
    (tree / "tracked.md").write_text(f"pinned at {old}\n")
    subprocess.run([git, "init", "-q", str(tree)], check=True)
    subprocess.run([git, "-C", str(tree), "add", "tracked.md", *FILES], check=True)
    _, stale, _ = switch.plan(tree, {setting: (repo, NEW)})
    assert [entry.split(":")[0] for entry in stale] == ["tracked.md"]


@pytest.mark.parametrize("length", [7, 10, 12, 39])
@pytest.mark.parametrize(
    ("name", "line"),
    [
        ("docs/guides/zz_short.md", "Pinned at `{short}` until the retrain."),
        ("evaluation/zz_short.json", '{{"rev": "{short}"}}'),
        ("bibr/zz_short.py", "URL = 'https://huggingface.co/{repo}/resolve/{short}/x'"),
    ],
)
def test_any_abbreviation_of_seven_or_more_digits_stops_it(tree, capsys, length, name, line):
    setting, repo, old = _section_pin(tree)
    target = tree / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(line.format(repo=repo, short=old[:length]) + "\n")
    assert switch.main([f"{setting}={repo}@{NEW}", "--check", "--root", str(tree)]) == 1
    assert f"{name}:1" in capsys.readouterr().err


def test_six_digits_or_another_commit_of_the_same_prefix_is_not_the_old_pin(tree):
    setting, repo, old = _section_pin(tree)
    other = old[:7] + ("0" if old[7] != "0" else "1") + old[8:]
    (tree / "bibr/zz_other.py").write_text(f'A = "{old[:6]}"\nB = "{other}"\nC = "{other[:9]}"\n')
    _, stale, _ = switch.plan(tree, {setting: (repo, NEW)})
    assert not any(entry.startswith("bibr/zz_other.py") for entry in stale)
