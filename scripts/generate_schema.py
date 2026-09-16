"""Regenerate the machine-readable export schema artifact.

The artifact — not a hand-written copy — is what scienceverse/schema publishes
and metacheck vendors. Run after any change to bibr/export/models.py.
"""

import json
from pathlib import Path

from bibr.export.document_models import build_document_schema
from bibr.export.schema_artifact import build_export_schema

OUT = Path(__file__).resolve().parents[1] / "docs" / "schema" / "bibr-export-v11.schema.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    schema = build_export_schema()
    OUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    document_out = OUT.with_name("bibr-document-v1.schema.json")
    document_out.write_text(json.dumps(build_document_schema(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {document_out}")


if __name__ == "__main__":
    main()
