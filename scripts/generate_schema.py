"""Regenerate the machine-readable export schema artifact.

The artifact — not a hand-written copy — is what scienceverse/schema publishes
and metacheck vendors. Run after any change to bibr/export/models.py.
"""

import json
from pathlib import Path

from bibr.export.schema_artifact import build_export_schema

OUT = Path(__file__).resolve().parents[1] / "docs" / "schema" / "bibr-export-v11.schema.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    schema = build_export_schema()
    OUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
