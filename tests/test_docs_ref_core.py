"""Generated-reference renderers produce the content the docs nav links to."""

import importlib.util
import re
from pathlib import Path

import pytest

_CORE_PATH = Path(__file__).parents[1] / "scripts" / "docs_ref_core.py"


@pytest.fixture(scope="module")
def core():
    spec = importlib.util.spec_from_file_location("docs_ref_core", _CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_settings_md_has_sections_and_fields(core):
    md = core.render_settings_md()
    assert "## LLM" in md
    assert "`LLM_PROVIDER`" in md
    assert "`OCR_BACKEND`" in md
    assert "`ENVIRONMENT`" in md  # top-level section
    # descriptions actually flow through
    assert "provider" in md.lower()


def test_settings_md_intro_cites_config_commands(core):
    """``bibr config show``/``config path`` now exist — the intro should

    point readers at them, in addition to the underlying env vars / .env
    mechanism (this pointer was intentionally dropped in a prior docs pass
    before the ``bibr config`` subcommand suite shipped; restored here).
    """
    md = core.render_settings_md()
    assert "bibr config show" in md
    assert "bibr config path" in md
    assert "environment variables" in md.lower()
    assert ".env" in md


def test_cli_md_covers_subcommands_and_flags(core):
    md = core.render_cli_md()
    assert "## bibr chew" in md
    assert "## bibr setup" in md
    assert "## bibr preset save" in md  # nested subparser flattened
    assert "`-o" in md or "`--output" in md


def test_cli_md_includes_delegated_help_and_demo_flags(core):
    md = core.render_cli_md()
    assert "bibr setup --advanced" in md
    assert "bibr serve --help" in md
    demo = md.split("## bibr demo\n", 1)[1].split("\n## ", 1)[0]
    assert "`--share`" in demo
    assert "`--presets`" in demo
    assert "`--port` | `7860`" in demo


def test_cli_md_no_bare_sections(core):
    """Every '## bibr <cmd>' heading must be followed by real content

    (prose and/or an options table) before the next heading — not a bare
    heading with nothing under it. Regression for subcommands built with
    only ``help=`` and no ``description=`` (bibr setup/serve/demo/doctor,
    preset list/deactivate), which used to render as heading + empty table.
    """
    md = core.render_cli_md()
    lines = md.splitlines()
    heading_idxs = [i for i, line in enumerate(lines) if line.startswith("## bibr")]
    assert heading_idxs, "no '## bibr ...' headings found"
    for pos, i in enumerate(heading_idxs):
        end = heading_idxs[pos + 1] if pos + 1 < len(heading_idxs) else len(lines)
        body = lines[i + 1 : end]
        assert any(line.strip() for line in body), f"bare section: {lines[i]!r}"


def test_cli_serve_has_description(core):
    """'bibr serve' is built with only help= (no description=); the renderer

    must still surface that help text as prose under the heading.
    """
    md = core.render_cli_md()
    lines = md.splitlines()
    i = lines.index("## bibr serve")
    following = next(line for line in lines[i + 1 :] if line.strip())
    assert not following.startswith("|"), "expected prose, not a table row"
    assert "litserve" in following.lower() or "http api" in following.lower()


def test_cli_md_no_empty_options_tables(core):
    """A '| Flag | Default | Description |' header must always be followed

    by at least one data row — never left dangling with zero rows (that's
    the "empty three-column table" bug, now handled by omitting the table
    entirely when a parser has nothing to document).
    """
    md = core.render_cli_md()
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line == "| Flag | Default | Description |":
            assert i + 2 < len(lines), "table header with no separator/rows at all"
            assert lines[i + 1] == "|---|---|---|"
            next_line = lines[i + 2] if i + 2 < len(lines) else ""
            assert next_line.startswith("|"), f"empty options table after: {line!r}"


def test_schema_md_pins_live_version(core):
    from bibr.export.json_export import _SCHEMA_VERSION

    md = core.render_schema_md()
    assert f"v{_SCHEMA_VERSION}" in md
    assert "`info`" in md
    assert "`validation`" in md


# Block/type names are backtick-delimited and never contain a backtick
# themselves, so this splits correctly on backticks even though a union type
# cell (e.g. ``OcrConfigExport | None``) contains a raw, unescaped pipe that
# a naive pipe-split would mistake for a column separator. Pipes inside a
# code span don't need escaping in Python-Markdown's `tables` extension
# (mkdocs-material's toolchain), which is code-span-aware when splitting
# row cells.
_SCHEMA_ROW_RE = re.compile(r"^\| `([^`]*)` \| `([^`]*)` \| (.*) \|$")


def _schema_table_rows(md: str) -> list[tuple[str, str, str]]:
    """Parse the `| Block | Type | Description |` rows out of rendered markdown.

    Cells are returned verbatim — no pipe-unescaping, since a correctly
    rendered code-span cell never contains a backslash-pipe escape.
    """
    rows = []
    top_level = md.split("## Top-level blocks\n", 1)[1].split("\n## ", 1)[0]
    for line in top_level.splitlines():
        if not line.startswith("| `"):
            continue
        m = _SCHEMA_ROW_RE.match(line)
        assert m, f"unexpected schema row shape: {line!r}"
        rows.append(m.groups())
    return rows


def test_schema_md_every_row_has_a_description(core):
    from bibr.export.json_export import PaperExport

    rows = _schema_table_rows(core.render_schema_md())
    assert len(rows) == len(PaperExport.model_fields)
    for block, _type_cell, desc in rows:
        assert desc, f"empty Description cell for {block}"


def test_schema_md_list_field_keeps_item_type(core):
    schema_rows = _schema_table_rows(core.render_schema_md())
    rows = {block: type_cell for block, type_cell, _desc in schema_rows}
    assert rows["author"] == "list[AuthorExport]"
    assert rows["text"] == "list[TextExport]"
    # unions render short names, not fully-qualified module paths
    assert rows["ocr_config"] == "OcrConfigExport | None"


def test_schema_md_type_cells_have_no_module_paths(core):
    rows = _schema_table_rows(core.render_schema_md())
    for block, type_cell, _desc in rows:
        assert "bibr." not in type_cell, (
            f"module path leaked into type cell for {block}: {type_cell}"
        )


def test_schema_md_union_type_cell_is_not_escaped(core):
    # Pipes inside a backtick code span don't need escaping for
    # Python-Markdown's `tables` extension (code-span-aware row splitter);
    # escaping them renders a literal backslash in the built HTML. No
    # schema field description contains a "|", so a bare substring search
    # for the two-char sequence "\|" anywhere in the render is equivalent
    # to checking every backticked Type cell is unescaped.
    md = core.render_schema_md()
    assert "\\|" not in md
    assert "`OcrConfigExport | None`" in md


def test_schema_md_uses_serialized_json_names(core):
    rows = {name for name, _, _ in _schema_table_rows(core.render_schema_md())}
    assert {"_regions", "_native_source"} <= rows
    assert not {"regions", "native_source"} & rows


def test_schema_md_explains_nested_records_and_links_schema(core):
    md = core.render_schema_md()
    assert "[Download the generated JSON Schema](paper.schema.json)" in md
    assert "### AuthorExport" in md
    assert "| `author_id` | `int` | Yes |" in md
    assert "### TextExport" in md
    assert "| `_bbox_2d` | `list[float] | None` | No | `None` |" in md
    assert "### FigurePartExport" in md
    assert "### ValidationIssueExport" in md


def test_settings_md_union_type_cell_is_not_escaped(core):
    # Same defect, same fix, in the settings table: AUTH_API_KEY (str | None)
    # must render its Type cell without an escaped pipe.
    md = core.render_settings_md()
    for line in md.splitlines():
        if line.startswith("| `AUTH_API_KEY`"):
            assert "\\|" not in line
            assert "`str | None`" in line
            break
    else:
        raise AssertionError("AUTH_API_KEY row not found in settings table")
