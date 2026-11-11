"""Move bibr's default models to new Hub pins, in every place that pins them.

    python scripts/switch_model_defaults.py ML_SECTION_CLASSIFIER_MODEL_ID=org/name@<40-hex>
    python scripts/switch_model_defaults.py --from-json pins.json   # {"SETTING": ["org/name", "<40-hex>"]}

For each setting it rewrites:

* the pydantic defaults of the model's id and revision in ``bibr/config.py``;
* the matching ``ARG`` pair in ``Dockerfile.serve``, for the models the image
  pins itself (the others it reads from bibr's config at build time);
* the rows of Markdown tables under ``docs/`` that name the old repo and the
  old short revision, and ``huggingface.co/<repo>/blob|tree/<revision>`` links
  in the docs.

Then it looks for the old pin anywhere else (the full revision, its first eight
characters, and the old repo id when the repo changes) in the code, the
Dockerfiles, the README and the docs. A pin it finds there is one it does not
know how to update, and it stops with a list of them, writing nothing. The
CHANGELOG is history and is not searched; tests that assert the old pin are
listed for a hand update but do not stop it. ``--check`` reports without
writing; ``--root`` runs on a copy of the tree.

It changes text only. Whether a new default is better is decided before it runs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REVISION = re.compile(r"[0-9a-f]{40}")
_REPO = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+(:[\w./-]+)?")


@dataclass(frozen=True)
class Pin:
    """Where one model setting is pinned: its config fields and its Dockerfile.serve ARGs."""

    config_class: str
    id_field: str
    revision_field: str
    docker_args: tuple[str, str] | None = None


# The settings bibr resolves through ``bibr.utils.model_overrides.MODEL_SETTINGS``, plus the
# two layout pairs (a drift-guard test keeps this in step with that table).
PINS: dict[str, Pin] = {
    "NER_PARSER_CKPT": Pin("GlobalSettings", "NER_PARSER_CKPT", "NER_PARSER_REVISION"),
    "NER_SEG_CKPT": Pin("GlobalSettings", "NER_SEG_CKPT", "NER_SEG_REVISION"),
    "REF_GEOM_SEG_MODEL_ID": Pin(
        "GlobalSettings", "REF_GEOM_SEG_MODEL_ID", "REF_GEOM_SEG_REVISION"
    ),
    "REF_GEOM_TAGGER_MODEL_ID": Pin(
        "GlobalSettings", "REF_GEOM_TAGGER_MODEL_ID", "REF_GEOM_TAGGER_REVISION"
    ),
    "REF_TEXT_SEG_MODEL_ID": Pin(
        "GlobalSettings", "REF_TEXT_SEG_MODEL_ID", "REF_TEXT_SEG_REVISION"
    ),
    "ML_FRONT_ROLE_MODEL_ID": Pin("MlOptions", "front_role_model_id", "front_role_revision"),
    "ML_FRONT_TAGGER_MODEL_ID": Pin("MlOptions", "front_tagger_model_id", "front_tagger_revision"),
    "ML_SECTION_CLASSIFIER_MODEL_ID": Pin(
        "MlOptions",
        "section_classifier_model_id",
        "section_classifier_revision",
        ("SECTION_CLASSIFIER_MODEL", "SECTION_CLASSIFIER_REVISION"),
    ),
    "ML_PAPER_CLASSIFIER_MODEL_ID": Pin(
        "MlOptions",
        "paper_classifier_model_id",
        "paper_classifier_revision",
        ("PAPER_CLASSIFIER_MODEL", "PAPER_CLASSIFIER_REVISION"),
    ),
    "LAYOUT_ONNX_MODEL_ID": Pin("LayoutOptions", "onnx_model_id", "onnx_revision"),
    "LAYOUT_MODEL_ID": Pin(
        "LayoutOptions",
        "model_id",
        "model_revision",
        ("LAYOUT_MODEL_ID", "LAYOUT_MODEL_REVISION"),
    ),
}

CONFIG = Path("bibr/config.py")
DOCKERFILE = Path("Dockerfile.serve")


class PinError(Exception):
    """A pin the script cannot update, or an input it refuses."""


def _class_span(text: str, name: str) -> tuple[int, int]:
    match = re.search(rf"^class {re.escape(name)}\b.*$", text, re.M)
    if match is None:
        raise PinError(f"{CONFIG}: class {name} not found")
    following = re.search(r"^class \w", text[match.end() :], re.M)
    end = match.end() + following.start() if following else len(text)
    return match.end(), end


def _field_default(text: str, cls: str, field: str) -> tuple[int, int, str | None]:
    """``(start, end, value)`` of the literal default of ``field`` in class ``cls``."""
    start, end = _class_span(text, cls)
    pattern = re.compile(rf"^    {re.escape(field)}: [^\n=]+= Field\(\s*(None|\"[^\"\n]*\")", re.M)
    found = list(pattern.finditer(text, start, end))
    if len(found) != 1:
        raise PinError(
            f"{CONFIG}: expected one `{field}: ... = Field(<literal>, ...)` in {cls}, "
            f"found {len(found)}"
        )
    literal = found[0].group(1)
    value = None if literal == "None" else literal[1:-1]
    return found[0].start(1), found[0].end(1), value


def _replace(text: str, start: int, end: int, value: str) -> str:
    return text[:start] + json.dumps(value) + text[end:]


def _parse_pin(setting: str, repo: object, revision: object) -> tuple[str, str]:
    if setting not in PINS:
        known = ", ".join(sorted(PINS))
        raise PinError(f"{setting}: not a setting this script pins (known: {known})")
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise PinError(f"{setting}: {repo!r} is not a Hub repo id (org/name)")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise PinError(f"{setting}: revision {revision!r} is not a 40-hex commit")
    return repo, revision


def parse_args_pins(items: list[str], from_json: Path | None) -> dict[str, tuple[str, str]]:
    pins: dict[str, tuple[str, str]] = {}
    if from_json is not None:
        data = json.loads(from_json.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PinError(f"{from_json}: expected an object of SETTING: [repo, revision]")
        for setting, value in data.items():
            if not isinstance(value, list | tuple) or len(value) != 2:
                raise PinError(f"{setting}: expected [repo, revision], got {value!r}")
            pins[setting] = _parse_pin(setting, *value)
    for item in items:
        setting, sep, rest = item.partition("=")
        repo, at, revision = rest.rpartition("@")
        if not sep or not at:
            raise PinError(f"{item!r}: expected SETTING=org/name@<40-hex>")
        if setting in pins:
            raise PinError(f"{setting}: given twice")
        pins[setting] = _parse_pin(setting, repo, revision)
    if not pins:
        raise PinError("no pins given")
    return pins


def _docker_arg(text: str, name: str) -> re.Match[str]:
    found = list(re.finditer(rf"^ARG {re.escape(name)}=(\S*)$", text, re.M))
    if len(found) != 1:
        raise PinError(f"{DOCKERFILE}: expected one `ARG {name}=...`, found {len(found)}")
    return found[0]


def _search_files(root: Path) -> list[Path]:
    paths = [root / "README.md", *sorted(root.glob("Dockerfile*"))]
    for pattern in ("bibr/**/*.py", "scripts/**/*.py", "docs/**/*.md", "docs/**/*.yml"):
        paths.extend(sorted(root.glob(pattern)))
    return [p for p in paths if p.is_file()]


def plan(
    root: Path, pins: dict[str, tuple[str, str]]
) -> tuple[dict[Path, str], list[str], list[str]]:
    """``(new text per changed file, stale pins outside the known places, tests to update)``."""
    files: dict[Path, str] = {}

    def text_of(path: Path) -> str:
        if path not in files:
            files[path] = path.read_text(encoding="utf-8")
        return files[path]

    config = root / CONFIG
    olds: dict[str, tuple[str | None, str | None]] = {}
    for setting, (repo, revision) in pins.items():
        pin = PINS[setting]
        text = text_of(config)
        _, _, old_repo = _field_default(text, pin.config_class, pin.id_field)
        _, _, old_rev = _field_default(text, pin.config_class, pin.revision_field)
        olds[setting] = (old_repo, old_rev)
        # Revision first: the id literal sits before it, so its offsets stay valid.
        start, end, _ = _field_default(text, pin.config_class, pin.revision_field)
        text = _replace(text, start, end, revision)
        start, end, _ = _field_default(text, pin.config_class, pin.id_field)
        files[config] = _replace(text, start, end, repo)

        if pin.docker_args is not None:
            docker = root / DOCKERFILE
            text = text_of(docker)
            for name, value, old in (
                (pin.docker_args[0], repo, old_repo),
                (pin.docker_args[1], revision, old_rev),
            ):
                match = _docker_arg(text, name)
                if match.group(1) != old:
                    raise PinError(
                        f"{DOCKERFILE}: ARG {name}={match.group(1)} does not match the config "
                        f"default {old!r}; reconcile them first"
                    )
                text = text[: match.start(1)] + value + text[match.end(1) :]
            files[docker] = text

        for doc in sorted((root / "docs").glob("**/*.md")):
            text = text_of(doc)
            new = text
            if old_repo and old_rev:
                rows = []
                for line in new.splitlines(keepends=True):
                    if (
                        line.lstrip().startswith("|")
                        and f"`{old_repo}`" in line
                        and f"`{old_rev[:8]}`" in line
                    ):
                        line = line.replace(f"`{old_repo}`", f"`{repo}`")
                        line = line.replace(f"`{old_rev[:8]}`", f"`{revision[:8]}`")
                    rows.append(line)
                new = "".join(rows)
                new = re.sub(
                    rf"huggingface\.co/{re.escape(old_repo)}/(blob|tree)/{old_rev}",
                    lambda m, repo=repo, revision=revision: (
                        f"huggingface.co/{repo}/{m.group(1)}/{revision}"
                    ),
                    new,
                )
            files[doc] = new

    # Anything still naming an old pin is a place this script does not know.
    stale: list[str] = []
    needles: list[str] = []
    for setting, (repo, revision) in pins.items():
        old_repo, old_rev = olds[setting]
        if old_rev and old_rev != revision:
            needles += [old_rev, old_rev[:8]]
        if old_repo and old_repo != repo:
            needles.append(old_repo)
    needles = sorted(set(needles), key=len, reverse=True)
    for path in _search_files(root):
        text = files.get(path) if path in files else path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            for needle in needles:
                if re.search(rf"(?<![\w/-]){re.escape(needle)}(?![\w-])", line):
                    stale.append(f"{path.relative_to(root)}:{number}: {line.strip()[:120]}")
                    break
    tests = []
    for path in sorted((root / "tests").glob("**/*.py")):
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            tests.append(str(path.relative_to(root)))
    changed = {p: t for p, t in files.items() if t != p.read_text(encoding="utf-8")}
    return changed, stale, tests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pins", nargs="*", metavar="SETTING=org/name@REVISION")
    parser.add_argument("--from-json", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="report the changes, write nothing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        pins = parse_args_pins(args.pins, args.from_json)
        changed, stale, tests = plan(root, pins)
    except (PinError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if stale:
        print("error: pins this script does not know how to update:", file=sys.stderr)
        for line in stale:
            print(f"  {line}", file=sys.stderr)
        print("Update them by hand (or teach the script), then run it again.", file=sys.stderr)
        return 1
    for path, text in changed.items():
        rel = str(path.relative_to(root))
        old = path.read_text(encoding="utf-8").splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(old, text.splitlines(keepends=True), rel, rel))
        if not args.check:
            path.write_text(text, encoding="utf-8")
    if tests:
        print("tests that name the old pin (update by hand):")
        for name in tests:
            print(f"  {name}")
    print(("would change" if args.check else "changed") + f" {len(changed)} file(s)")
    if changed:
        print("Read the prose next to each changed line: it may describe the old model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
