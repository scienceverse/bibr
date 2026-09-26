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


def test_recorded_ids_stay_when_a_same_named_file_joins_the_inputs(tmp_path):
    """Growing a corpus must not rename a paper the ledger already has."""
    a = _pdf(tmp_path / "a" / "paper.pdf", b"a")
    b = _pdf(tmp_path / "b" / "paper.pdf", b"b")
    recorded = [{"paper_id": "paper", "path": str(a), "status": "ok"}]

    items = assign_paper_ids([a, b], recorded=recorded)

    assert [i.paper_id for i in items] == ["paper", f"paper-{sha256_file(b)[:8]}"]
    # ...and on the next run both keep their ids, whatever the input order.
    recorded.append({"paper_id": items[1].paper_id, "path": str(b), "status": "ok"})
    again = assign_paper_ids([b, a], recorded=recorded)
    assert [i.paper_id for i in again] == [items[1].paper_id, "paper"]


def test_recorded_suffixed_id_stays_after_its_twin_leaves(tmp_path):
    a = _pdf(tmp_path / "a" / "paper.pdf", b"a")
    recorded = [{"paper_id": "paper-0badc0de", "path": str(a), "status": "ok"}]
    assert [i.paper_id for i in assign_paper_ids([a], recorded=recorded)] == ["paper-0badc0de"]


def test_unrecorded_inputs_keep_the_stem_rules(tmp_path):
    """A moved corpus (paths not in the ledger) resolves exactly as before."""
    a = _pdf(tmp_path / "moved" / "a.pdf")
    recorded = [{"paper_id": "a", "path": "/old/place/a.pdf", "status": "ok"}]
    assert [i.paper_id for i in assign_paper_ids([a], recorded=recorded)] == ["a"]


def test_the_run_info_stem_is_reserved_for_bibr_batch(tmp_path):
    """<out>/run_info.json is the runner's own file; the library writes none."""
    from bibr.batch.manifest import RESERVED_IDS
    from bibr.batch.runner import RUN_INFO_FILENAME

    info = _pdf(tmp_path / "Run_Info.pdf", b"paper")
    recorded = [{"paper_id": "Run_Info", "path": str(info)}]
    items = assign_paper_ids([info], recorded=recorded, reserved=RESERVED_IDS)
    assert [i.paper_id for i in items] == [f"Run_Info-{sha256_file(info)[:8]}"]
    assert Path(RUN_INFO_FILENAME).stem in RESERVED_IDS
    assert [i.paper_id for i in assign_paper_ids([info])] == ["Run_Info"]


def test_a_recorded_id_is_kept_by_the_first_input_that_claims_it(tmp_path):
    """Two paths recorded under one id (each ran alone once) would overwrite
    each other's export: the first keeps it, the other gets a suffix."""
    a = _pdf(tmp_path / "a" / "paper.pdf", b"a")
    b = _pdf(tmp_path / "b" / "Paper.pdf", b"b")
    recorded = [
        {"paper_id": "paper", "path": str(a), "status": "ok"},
        {"paper_id": "PAPER", "path": str(b), "status": "ok"},
    ]
    items = assign_paper_ids([a, b], recorded=recorded)
    assert [i.paper_id for i in items] == ["paper", f"Paper-{sha256_file(b)[:8]}"]


def test_the_latest_recorded_id_of_a_path_wins(tmp_path):
    a = _pdf(tmp_path / "a" / "paper.pdf", b"a")
    recorded = [
        {"paper_id": "paper", "path": str(a), "status": "failed"},
        {"paper_id": "paper-0badc0de", "path": str(a), "status": "ok"},
    ]
    assert [i.paper_id for i in assign_paper_ids([a], recorded=recorded)] == ["paper-0badc0de"]


def test_an_unreadable_colliding_file_still_gets_a_unique_id(tmp_path):
    a = _pdf(tmp_path / "a" / "paper.pdf", b"a")
    missing = tmp_path / "b" / "paper.pdf"  # vanished between discovery and planning
    items = assign_paper_ids([a, missing])
    assert [i.paper_id for i in items] == [f"paper-{sha256_file(a)[:8]}", "paper"]


# --- audit S8: only manifest-like files are read as manifests ---


def test_top_level_stray_files_are_unsupported_not_manifests(tmp_path):
    """A shell glob sweeping up a binary or prose file must not abort the batch."""
    binary = tmp_path / "old.doc"
    binary.write_bytes(bytes([0xD0, 0xCF, 0x11, 0xE0, 0x80, 0x41]) * 30)
    prose = tmp_path / "notes.md"
    prose.write_text("See paper a.pdf for details\n", encoding="utf-8")

    found = discover_inputs([binary, prose])

    assert found.files == []
    assert found.missing == []
    assert found.unsupported == [binary, prose]
    assert found.problems == 2


def test_unreadable_manifest_like_file_names_itself(tmp_path):
    """A binary .txt names itself in unreadable instead of raising a bare codec error."""
    binary = tmp_path / "list.txt"
    binary.write_bytes(bytes([0xD0, 0xCF, 0x11, 0xE0, 0x80, 0x41]) * 30)

    found = discover_inputs([binary])

    assert found.files == []
    assert found.manifests == []
    assert len(found.unreadable) == 1
    assert str(binary) in found.unreadable[0]
    assert found.problems == 1


def test_txt_and_extensionless_files_are_still_manifests(tmp_path):
    """The manifest-like suffixes keep working, including no suffix at all."""
    a = _pdf(tmp_path / "a.pdf")
    manifest = tmp_path / "m.list"
    manifest.write_text(f"{a}\n")
    assert discover_inputs([str(manifest)]).files == [a]

    bare = tmp_path / "manifest"
    bare.write_text(f"{a}\n")
    found = discover_inputs([str(bare)])
    assert found.files == [a]
    assert found.manifests == [bare]


def test_manifest_over_long_line_is_missing_not_a_crash(tmp_path):
    """A >255-byte line in a .txt manifest cannot be stated: record, don't raise."""
    manifest = tmp_path / "notes.txt"
    long_line = "x" * 300
    manifest.write_text(f"{long_line}\n", encoding="utf-8")

    found = discover_inputs([str(manifest)])

    assert found.files == []
    assert len(found.missing) == 1
    assert f"(from {manifest.name})" in found.missing[0]
    assert found.problems == 1


def test_uppercase_manifest_suffix_is_still_a_manifest(tmp_path):
    """Suffix matching is case-insensitive: LIST.TXT is a manifest, not prose."""
    a = _pdf(tmp_path / "a.pdf")
    manifest = tmp_path / "LIST.TXT"
    manifest.write_text(f"{a}\n")

    found = discover_inputs([str(manifest)])

    assert found.files == [a]
    assert found.manifests == [manifest]
