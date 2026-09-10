#!/usr/bin/env python3
"""Validate expiring vulnerability exceptions and run the locked dependency audit."""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parents[2]
DEFAULT_POLICY = ROOT / "ci" / "vulnerability-exceptions.toml"
REQUIRED_FIELDS = ("id", "reason", "exposure", "review_by")


@dataclass(frozen=True)
class VulnerabilityException:
    id: str
    reason: str
    exposure: str
    review_by: date


def load_exceptions(path: Path, today: date | None = None) -> list[VulnerabilityException]:
    """Load active exceptions, rejecting incomplete, duplicate, or expired policy."""

    current_date = today or date.today()
    with path.open("rb") as policy_file:
        document = tomllib.load(policy_file)

    raw_entries = document.get("exception")
    if not isinstance(raw_entries, list):
        raise ValueError("policy must contain at least one [[exception]] entry")

    entries: list[VulnerabilityException] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"exception {index} must be a TOML table")
        for field in REQUIRED_FIELDS:
            if field not in raw:
                raise ValueError(f"exception {index} is missing {field}")

        identifier = raw["id"]
        reason = raw["reason"]
        exposure = raw["exposure"]
        review_by = raw["review_by"]
        for field, value in (("id", identifier), ("reason", reason), ("exposure", exposure)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"exception {index} has an invalid {field}")
        if not isinstance(review_by, date) or isinstance(review_by, datetime):
            raise ValueError(f"exception {identifier} has an invalid review_by date")
        if identifier in seen:
            raise ValueError(f"duplicate vulnerability exception: {identifier}")
        if review_by < current_date:
            raise ValueError(
                f"vulnerability exception {identifier} expired on {review_by.isoformat()}"
            )

        seen.add(identifier)
        entries.append(
            VulnerabilityException(
                id=identifier,
                reason=reason.strip(),
                exposure=exposure.strip(),
                review_by=review_by,
            )
        )
    return entries


def build_command(entries: Sequence[VulnerabilityException]) -> list[str]:
    """Build the locked pip-audit command for validated exceptions."""

    command = ["uv", "run", "--locked", "pip-audit", "--desc", "on"]
    for entry in entries:
        command.extend(("--ignore-vuln", entry.id))
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    entries = load_exceptions(args.policy)
    print(f"validated {len(entries)} active vulnerability exceptions")
    if args.validate_only:
        return 0
    subprocess.run(  # noqa: S603 - fixed executable and validated vulnerability IDs
        build_command(entries),  # noqa: S607 - uv is an explicit CI prerequisite
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
