"""Input discovery: manifests (comments, relative paths, missing entries),
recursive directories, mixed inputs, and collision-free paper ids."""

from __future__ import annotations

from pathlib import Path

from bibr.batch.manifest import (
    BatchItem,
    assign_paper_ids,
    discover_inputs,
    parse_manifest_lines,
    sha256_file,
)


def _pdf(path: Path, body: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + body)
    return path


def test_parse_manifest_lines_skips_blanks_and_comments():
    text = "# header\n\n  a.pdf  \n#b.pdf\n/abs/c.pdf\n"
    assert parse_manifest_lines(text) == ["a.pdf", "/abs/c.pdf"]


def test_manifest_relative_entries_resolve_against_manifest_dir(tmp_path):
    a = _pdf(tmp_path / "corpus" / "a.pdf")
    b = _pdf(tmp_path / "corpus" / "sub" / "b.pdf")
    manifest = tmp_path / "corpus" / "list.txt"
    manifest.write_text(f"# pilot\ncorpus-does-not-matter\n{a.name}\nsub/b.pdf\n{b}\n")
    # ``corpus-does-not-matter`` is a missing entry; ``b`` appears twice (dedupe).
    found = discover_inputs([str(manifest)])
    assert found.files == [a, b]
    assert found.manifests == [manifest]
    assert len(found.missing) == 1
    assert "corpus-does-not-matter" in found.missing[0]
    assert found.unsupported == []


def test_manifest_directory_entry_is_walked_recursively(tmp_path):
    _pdf(tmp_path / "d" / "z.pdf")
    _pdf(tmp_path / "d" / "deep" / "a.pdf")
    (tmp_path / "d" / "notes.txt").write_text("ignored")
    manifest = tmp_path / "m.lst"
    manifest.write_text("d\n")
    found = discover_inputs([str(manifest)])
    assert [p.name for p in found.files] == ["a.pdf", "z.pdf"]
    assert found.directories == [tmp_path / "d"]


def test_manifest_unsupported_entry_is_reported_not_processed(tmp_path):
    bad = tmp_path / "paper.tex"
    bad.write_text("\\documentclass{article}")
    manifest = tmp_path / "m.txt"
    manifest.write_text(f"{bad}\n")
    found = discover_inputs([str(manifest)])
    assert found.files == []
    assert found.unsupported == [bad]


def test_directory_discovery_is_recursive_sorted_and_filtered(tmp_path):
    _pdf(tmp_path / "b.pdf")
    _pdf(tmp_path / "a" / "x.pdf")
    (tmp_path / "a" / "x.docx").write_bytes(b"PK\x03\x04")
    (tmp_path / "a" / "README.md").write_text("no")
    found = discover_inputs([tmp_path])
    assert found.files == [tmp_path / "a" / "x.docx", tmp_path / "a" / "x.pdf", tmp_path / "b.pdf"]


def test_empty_directory_and_missing_path_are_reported(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    found = discover_inputs([str(empty), str(tmp_path / "nope.pdf")])
    assert found.files == []
    assert found.empty_dirs == [empty]
    assert found.missing == [str(tmp_path / "nope.pdf")]
    assert found.problems == 2


def test_mixed_inputs_keep_order_and_dedupe(tmp_path):
    a = _pdf(tmp_path / "a.pdf")
    b = _pdf(tmp_path / "dir" / "b.pdf")
    manifest = tmp_path / "m.txt"
    manifest.write_text(f"{a}\n{b}\n")
    found = discover_inputs([str(b), str(manifest), str(tmp_path / "dir")])
    assert found.files == [b, a]


def test_assign_paper_ids_uses_stems_when_unique(tmp_path):
    a = _pdf(tmp_path / "a.pdf")
    b = _pdf(tmp_path / "b.pdf")
    items = assign_paper_ids([a, b])
    assert [i.paper_id for i in items] == ["a", "b"]
    assert all(not i.disambiguated for i in items)
    assert items[0] == BatchItem(path=a, paper_id="a", stem="a")


def test_assign_paper_ids_disambiguates_stem_collisions_with_sha_prefix(tmp_path):
    a1 = _pdf(tmp_path / "x" / "paper.pdf", b"one")
    a2 = _pdf(tmp_path / "y" / "Paper.pdf", b"two")  # case-insensitive collision
    other = _pdf(tmp_path / "other.pdf")
    items = assign_paper_ids([a1, a2, other])
    assert items[0].paper_id == f"paper-{sha256_file(a1)[:8]}"
    assert items[1].paper_id == f"Paper-{sha256_file(a2)[:8]}"
    assert items[2].paper_id == "other"
    assert items[0].disambiguated and items[1].disambiguated
    assert items[0].stem == "paper"
    # Deterministic: same inputs, same ids.
    assert [i.paper_id for i in assign_paper_ids([a1, a2, other])] == [i.paper_id for i in items]


def test_assign_paper_ids_identical_bytes_get_an_ordinal(tmp_path):
    a1 = _pdf(tmp_path / "x" / "same.pdf", b"same")
    a2 = _pdf(tmp_path / "y" / "same.pdf", b"same")
    items = assign_paper_ids([a1, a2])
    sha8 = sha256_file(a1)[:8]
    assert [i.paper_id for i in items] == [f"same-{sha8}", f"same-{sha8}-2"]
