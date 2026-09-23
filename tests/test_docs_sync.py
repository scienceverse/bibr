"""Anti-drift guards keeping hand-maintained docs in lock-step with live code.

Doc audits repeatedly found enumerations (supported extensions, paper-type
labels, canonical sections, export keys, dependency pins) drifting from the
code they describe. Where a value list is rendered from code via a
mkdocs-macros variable, no guard is needed; where prose descriptions make full
generation awkward, these tests fail the moment the doc and the code diverge.

Fast, import-light (no torch/transformers), no network, not slow.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
DOCS = REPO_ROOT / "docs"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _load_docs_ref_core():
    """Import scripts/docs_ref_core.py by path (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / "docs_ref_core.py"
    spec = importlib.util.spec_from_file_location("docs_ref_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _table_col_codes(md: str, header: str, col: int) -> set[str]:
    """Codes (backtick-wrapped) in column ``col`` of the markdown table whose
    header row equals ``header`` (first match wins)."""
    lines = md.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == header)
    codes: set[str] = set()
    for line in lines[start + 2 :]:  # skip header + separator row
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = re.search(r"`([^`]+)`", cells[col])
        if m:
            codes.add(m.group(1))
    return codes


# --------------------------------------------------------------------------
# 1. Supported input extensions
# --------------------------------------------------------------------------


def test_supported_extensions_macro_matches_enum():
    """The mkdocs macro rendering the extension list must match the enum, in
    declaration order, and library.md must actually use the macro (not a
    hand-typed list that could drift)."""
    from types import SimpleNamespace

    from bibr.input.supported_files import SupportedFileType

    expected = "/".join(f"`{ft.value}`" for ft in SupportedFileType)

    docs_macros_path = REPO_ROOT / "scripts" / "docs_macros.py"
    spec = importlib.util.spec_from_file_location("docs_macros", docs_macros_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = SimpleNamespace(variables={})
    module.define_env(env)
    assert env.variables["supported_extensions"] == expected

    # The consuming page must reference the macro, not a frozen literal.
    assert "{{ supported_extensions }}" in _read("docs/guides/library.md")


def test_supported_formats_named_in_prose():
    """Prose pages that name accepted formats in friendly words (PDF, JATS,
    ePub, ...) rather than extensions must mention every supported format
    family. The keyword map below must cover every SupportedFileType member —
    adding a new file type without updating this map (and the prose) fails
    here."""
    from bibr.input.supported_files import SUPPORTED_EXTENSIONS

    # Each extension -> friendly keyword(s), any one of which satisfies it.
    format_keywords = {
        ".pdf": ["PDF"],
        ".docx": ["DOCX"],
        ".xml": ["XML", "JATS"],
        ".html": ["HTML"],
        ".htm": ["HTML"],  # subsumed by "HTML" in prose
        ".epub": ["ePub", "EPUB", "epub"],
    }
    assert set(format_keywords) == SUPPORTED_EXTENSIONS, (
        "format_keywords is out of sync with SupportedFileType; "
        "add the new extension here and mention it in the prose docs"
    )

    index = _read("docs/index.md")
    arch = _read("docs/guides/architecture.md")
    # Scope architecture to the Input stage, which is the canonical enumeration.
    input_block = arch.split("### 2. Input", 1)[1].split("### 3. Structure", 1)[0]

    for ext, keywords in format_keywords.items():
        assert any(k in index for k in keywords), f"{ext}: none of {keywords} in docs/index.md"
        assert any(k in input_block for k in keywords), (
            f"{ext}: none of {keywords} in architecture.md Input stage"
        )


# --------------------------------------------------------------------------
# 2. Paper-type labels
# --------------------------------------------------------------------------


def test_paper_type_labels_match_doc_table():
    from bibr.export.json_export import _snake
    from bibr.structure.paper_classifier import PAPER_TYPE_LABELS, PaperTypeLiteral

    md = _read("docs/guides/classifiers.md")
    doc_codes = _table_col_codes(md, "| Type | Description |", col=0)
    # The docs show the labels as the export spells them (snake_case).
    assert doc_codes == {_snake(label) for label in PAPER_TYPE_LABELS}
    # PaperTypeLiteral must stay in lock-step with the list too.
    literal_values = set(PaperTypeLiteral.__args__)
    assert literal_values == set(PAPER_TYPE_LABELS)


# --------------------------------------------------------------------------
# 3. CanonicalSection tables
# --------------------------------------------------------------------------


def test_canonical_section_tables_match_enum():
    from bibr.export.json_export import _EXPORT_SECTION_TYPES
    from bibr.paper_contents import CanonicalSection

    # As the export spells them (``open_data`` is ``data_availability``).
    values = {_EXPORT_SECTION_TYPES.get(m.value, m.value) for m in CanonicalSection}

    classifiers = _read("docs/guides/classifiers.md")
    arch = _read("docs/guides/architecture.md")

    # classifiers.md: "| Type | Value | Examples |" — codes in the Value column.
    assert _table_col_codes(classifiers, "| Type | Value | Examples |", col=1) == values
    # architecture.md: "| Value | Description |" — codes in the first column.
    assert _table_col_codes(arch, "| Value | Description |", col=0) == values


# --------------------------------------------------------------------------
# 4. Export top-level keys
# --------------------------------------------------------------------------


def test_export_top_level_keys_documented():
    """Every always-present PaperExport field must appear (as a code span) in
    both docs' "Top-level keys, grouped by role" enumerations. ``regions`` (the
    opt-in ``_regions`` debug payload) is documented separately, so it is
    excluded."""
    from bibr.export.json_export import PaperExport

    fields = set(PaperExport.model_fields) - {"regions"}

    for rel in ("docs/guides/architecture.md", "docs/reference/rest-api.md"):
        md = _read(rel)
        line = next(line for line in md.splitlines() if "Top-level keys, grouped by role" in line)
        missing = [name for name in fields if f"`{name}`" not in line]
        assert not missing, f"{rel}: keys not documented: {missing}"


# --------------------------------------------------------------------------
# 5. Extras version pins
# --------------------------------------------------------------------------


def _optional_deps() -> dict[str, list[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _base_req(req: str) -> str:
    """Strip an environment marker (``; sys_platform == ...``) from a
    requirement, leaving ``name<spec>``."""
    return req.split(";", 1)[0].strip()


def test_install_doc_version_pins_match_pyproject():
    """Version-bearing tokens cited in install.md must equal the current
    pyproject requirement for their single-package extra."""
    opt = _optional_deps()
    install = _read("docs/getting-started/install.md")

    # extra -> exact token the doc must cite (and that pyproject must declare).
    cited = {
        "vllm": "vllm==0.27.0",
        "cache": "redis>=5.0.0",
        "demo": "gradio>=6.15.0",
        "batch": "anthropic>=0.40.0",
    }
    for extra, token in cited.items():
        bases = {_base_req(r) for r in opt[extra]}
        assert token in bases, f"pyproject extra '{extra}' no longer declares '{token}': {bases}"
        assert token in install, f"install.md no longer cites '{token}' for extra '{extra}'"

    # The `uv tool run --from vllm==0.27.0` fallback line must match the pin too.
    vllm_pin = _base_req(opt["vllm"][0])
    sdk_floor = _base_req(opt["vllm"][1])
    assert f"uv tool run --from {vllm_pin} --with '{sdk_floor}' vllm serve" in install


def _all_extra_members() -> set[str]:
    """Members of the ``all`` extra, parsed from ``bibr[a,b,c]``."""
    reqs = _optional_deps()["all"]
    joined = " ".join(reqs)
    m = re.search(r"bibr\[([^\]]+)\]", joined)
    assert m, f"could not parse 'all' extra members from {reqs!r}"
    return {p.strip() for p in m.group(1).split(",")}


def test_all_extra_member_list_matches_docs():
    members = _all_extra_members()

    install = _read("docs/getting-started/install.md")
    # The `all` table row: "| `all` | `batch` + `cache` + `demo` + `ml` | ... |"
    row = next(line for line in install.splitlines() if line.strip().startswith("| `all` |"))
    adds_cell = row.strip().strip("|").split("|")[1]
    assert set(re.findall(r"`([^`]+)`", adds_cell)) == members

    # The prose restatement: "It bundles `batch` + `cache` + `demo` + `ml`".
    bundles = next(line for line in install.splitlines() if "It bundles" in line)
    bundles_tail = bundles.split("It bundles", 1)[1].split("—", 1)[0]
    assert set(re.findall(r"`([^`]+)`", bundles_tail)) == members


# --------------------------------------------------------------------------
# 6. _SECTION_TITLES completeness
# --------------------------------------------------------------------------


def test_section_titles_cover_every_config_prefix():
    """Every settings-section env prefix in bibr.config must have a
    _SECTION_TITLES entry, or its generated settings-reference heading falls
    back to the bare prefix (this drifted once when RAPID_MLX_ was missing)."""
    from bibr.config_introspect import iter_setting_docs

    docs_ref_core = _load_docs_ref_core()
    prefixes = {d.section for d in iter_setting_docs()}
    missing = sorted(p for p in prefixes if p not in docs_ref_core._SECTION_TITLES)
    assert not missing, f"_SECTION_TITLES missing entries for: {missing}"
