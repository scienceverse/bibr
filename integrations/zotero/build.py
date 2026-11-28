"""Package the Zotero plugin: ``python integrations/zotero/build.py`` writes
``integrations/zotero/dist/bibr-zotero-<version>.xpi``."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def build() -> Path:
    version = json.loads((SRC / "manifest.json").read_text())["version"]
    out = HERE / "dist" / f"bibr-zotero-{version}.xpi"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as xpi:
        for path in sorted(SRC.rglob("*")):
            if path.is_file():
                xpi.write(path, path.relative_to(SRC).as_posix())
    return out


if __name__ == "__main__":
    print(build())
