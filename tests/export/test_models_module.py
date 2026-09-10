"""bibr.export.models is the canonical home for the v11 export schema.

The package root (``bibr.export``) re-exports the public models straight from
``models``; ``json_export`` owns serialization and is no longer a general
re-export surface for types.
"""

from pathlib import Path

import bibr.export as export_pkg
import bibr.export.json_export as json_export
import bibr.export.models as models

# The only models ``json_export`` still re-exports purely for outside
# importers, i.e. names it does not itself use. Kept deliberately small — if
# this set grows, the models/json_export split has started to blur again.
SHIM_ONLY_REEXPORTS = {"LlmEngineExport", "OcrEngineExport"}


def test_models_module_holds_the_schema():
    assert models.PaperExport is json_export.PaperExport
    assert models._SCHEMA_VERSION == json_export._SCHEMA_VERSION


def test_package_root_reexports_models_not_json_export():
    """``from bibr.export import XExport`` must resolve to the models object."""
    names = [n for n in export_pkg.__all__ if n.endswith("Export")]
    assert len(names) > 20
    for name in names:
        assert getattr(export_pkg, name) is getattr(models, name), name


def test_person_name_export_is_reachable_from_the_package_root():
    """The one genuinely new row-level model of v11 must not be unreachable."""
    assert "PersonNameExport" in export_pkg.__all__
    assert export_pkg.PersonNameExport is models.PersonNameExport


def test_json_export_shim_stays_trimmed():
    """json_export may still expose a model only if it uses it, or if it is one
    of the few grandfathered re-exports."""
    exposed = {
        n
        for n in dir(json_export)
        if n.endswith("Export") and getattr(json_export, n, None) is getattr(models, n, None)
    }
    body = Path(json_export.__file__).read_text()
    for name in sorted(exposed - SHIM_ONLY_REEXPORTS):
        # Used by json_export itself: appears somewhere other than the import.
        assert body.count(name) > 1, f"{name} is re-exported but unused; drop it from the shim"
