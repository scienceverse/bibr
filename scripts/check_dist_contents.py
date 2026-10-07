"""Gate: built dist artifacts must contain only public package material.

Only runtime files and package metadata belong in an sdist or wheel.
CI runs main() against dist/ after `uv build`; find_violations() is unit-tested.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from check_public_tree import DATASET_SUFFIXES

# sdist members, after stripping the "bibr-<version>/" prefix
SDIST_ROOT = re.compile(r"bibr-[0-9][^/]*")
SDIST_ALLOWED_PREFIXES = ("bibr/",)
SDIST_ALLOWED_FILES = {"PKG-INFO", "README.md", "LICENSE.md", "pyproject.toml", ".gitignore"}
# wheel members, plus bibr's own metadata directory. A *.data/ directory is never
# allowed: its purelib/ .pth files run at interpreter startup and scripts/ land on PATH.
WHEEL_ALLOWED_PREFIXES = ("bibr/",)
WHEEL_DIST_INFO = re.compile(r"bibr-[0-9][^/-]*\.dist-info")
WHEEL_REQUIRED_FILES = {"bibr/py.typed"}
# binary-ish payloads may only live in the shipped package's data dir
PDF_ALLOWED_PREFIX = "bibr/data/"


def _strip_root(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def _payload_violation(kind: str, name: str, shown: str) -> str | None:
    suffixes = {suffix.casefold() for suffix in PurePosixPath(name).suffixes}
    if DATASET_SUFFIXES & suffixes:
        return f"{kind}: dataset payload {shown}"
    if ".pdf" in suffixes and not name.startswith(PDF_ALLOWED_PREFIX):
        return f"{kind}: PDF outside {PDF_ALLOWED_PREFIX}: {shown}"
    return None


def find_violations(sdist_names: list[str], wheel_names: list[str]) -> list[str]:
    violations: list[str] = []
    for raw in sdist_names:
        name = _strip_root(raw)
        if not name:  # the root dir entry itself
            continue
        ok = SDIST_ROOT.fullmatch(raw.split("/", 1)[0]) is not None and (
            name in SDIST_ALLOWED_FILES or name.startswith(SDIST_ALLOWED_PREFIXES)
        )
        if not ok:
            violations.append(f"sdist: unexpected member {raw}")
        elif violation := _payload_violation("sdist", name, raw):
            violations.append(violation)
    dist_info_dirs = set()
    for name in wheel_names:
        top = name.split("/", 1)[0]
        if "/" in name and WHEEL_DIST_INFO.fullmatch(top):
            dist_info_dirs.add(top)
        elif not name.startswith(WHEEL_ALLOWED_PREFIXES):
            violations.append(f"wheel: unexpected member {name}")
        elif violation := _payload_violation("wheel", name, name):
            violations.append(violation)
    if len(dist_info_dirs) != 1:
        violations.append(
            f"wheel: expected one bibr-<version>.dist-info/, found {sorted(dist_info_dirs)}"
        )
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
