"""Gate: built dist artifacts must contain only public package material.

Only runtime files and package metadata belong in an sdist or wheel.
CI runs main() against dist/ after `uv build`; find_violations() is unit-tested.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

# sdist members, after stripping the "bibr-<version>/" prefix
SDIST_ALLOWED_PREFIXES = ("bibr/",)
SDIST_ALLOWED_FILES = {"PKG-INFO", "README.md", "LICENSE.md", "pyproject.toml", ".gitignore"}
# wheel members
WHEEL_ALLOWED_PREFIXES = ("bibr/",)  # plus *.dist-info/
WHEEL_REQUIRED_FILES = {"bibr/py.typed"}
# binary-ish payloads may only live in the shipped package's data dir
PDF_ALLOWED_PREFIX = "bibr/data/"


def _strip_root(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def find_violations(sdist_names: list[str], wheel_names: list[str]) -> list[str]:
    violations: list[str] = []
    for raw in sdist_names:
        name = _strip_root(raw)
        if not name:  # the root dir entry itself
            continue
        ok = name in SDIST_ALLOWED_FILES or name.startswith(SDIST_ALLOWED_PREFIXES)
        if not ok:
            violations.append(f"sdist: unexpected member {raw}")
        elif name.endswith(".pdf") and not name.startswith(PDF_ALLOWED_PREFIX):
            violations.append(f"sdist: PDF outside {PDF_ALLOWED_PREFIX}: {raw}")
    for name in wheel_names:
        parts = name.split("/", 1)
        if parts[0].endswith(".dist-info") or parts[0].endswith(".data"):
            continue
        if not name.startswith(WHEEL_ALLOWED_PREFIXES):
            violations.append(f"wheel: unexpected member {name}")
        elif name.endswith(".pdf") and not name.startswith(PDF_ALLOWED_PREFIX):
            violations.append(f"wheel: PDF outside {PDF_ALLOWED_PREFIX}: {name}")
    for name in sorted(WHEEL_REQUIRED_FILES - set(wheel_names)):
        violations.append(f"wheel: missing required member {name}")
    return violations


def main() -> int:
    dist = Path("dist")
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        # A stale artifact from a prior version could mask a violation in
        # the fresh one — refuse to guess which pair to check.
        print(
            f"error: expected exactly one sdist and one wheel in {dist}/, "
            f"found {len(sdists)} sdist(s) and {len(wheels)} wheel(s) — clean dist/ and rebuild",
            file=sys.stderr,
        )
        return 2
    with tarfile.open(sdists[0]) as tf:
        sdist_names = [m.name for m in tf.getmembers() if m.isfile()]
    with zipfile.ZipFile(wheels[0]) as zf:
        wheel_names = [n for n in zf.namelist() if not n.endswith("/")]
    violations = find_violations(sdist_names, wheel_names)
    for v in violations:
        print(v, file=sys.stderr)
    print(
        f"checked {sdists[-1].name} ({len(sdist_names)} files) and "
        f"{wheels[-1].name} ({len(wheel_names)} files): "
        f"{len(violations)} violation(s)"
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
