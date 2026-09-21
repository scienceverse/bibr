"""Regenerate the machine-readable export schema artifacts.

The artifacts — not hand-written copies — are what scienceverse/schema
publishes and metacheck vendors: the strict schema of what bibr writes, and the
lenient reader schema that accepts any 11.x export. Run after any change to
bibr/export/models.py.
"""

import json
from pathlib import Path

from bibr.export.schema_artifact import build_export_schema

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schema"
OUT = SCHEMA_DIR / "bibr-export-v11.schema.json"
READER_OUT = SCHEMA_DIR / "bibr-export-v11-reader.schema.json"


def main() -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for path, reader in ((OUT, False), (READER_OUT, True)):
        schema = build_export_schema(reader=reader)
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
