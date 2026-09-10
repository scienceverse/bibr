"""JSONL manifest input for identity-aware local runs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from bibr.extract.doi_identity import doi_sha256
from bibr.pipeline.identity import ExpectedIdentity
from bibr.utils.text import normalize_doi


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestRecord:
    input_path: Path
    output_path: Path
    expected_identity: ExpectedIdentity
    content_sha256: str


def _path_collision_key(path: Path) -> str:
    """Return a conservative cross-platform alias key for a resolved path."""

    return unicodedata.normalize("NFC", str(path)).casefold()


def _optional_string(row: dict, key: str, line_number: int) -> str | None:
    value = row.get(key)
    if value is not None and not isinstance(value, str):
        raise ManifestError(f"line {line_number}: {key} must be a string")
    return value


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    records: list[ManifestRecord] = []
    record_lines: list[int] = []
    record_ids: set[str] = set()
    output_keys: set[str] = set()
    content_hashes: dict[Path, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ManifestError(f"line {line_number}: manifest row must be an object")
        raw_queue_record_id = row.get("queue_record_id")
        if raw_queue_record_id is None:
            raise ManifestError(f"line {line_number}: queue_record_id is required")
        if not isinstance(raw_queue_record_id, str):
            raise ManifestError(f"line {line_number}: queue_record_id must be a string")
        queue_record_id = raw_queue_record_id.strip()
        if not queue_record_id:
            raise ManifestError(f"line {line_number}: queue_record_id is required")
        if queue_record_id in record_ids:
            raise ManifestError(f"line {line_number}: duplicate record {queue_record_id!r}")
        record_ids.add(queue_record_id)

        raw_input = row.get("input_path", row.get("input"))
        raw_output = row.get("output_path", row.get("output"))
        if not isinstance(raw_input, str) or not raw_input.strip():
            raise ManifestError(f"line {line_number}: input_path is required")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ManifestError(f"line {line_number}: output_path is required")
        input_path = (manifest_path.parent / raw_input).resolve()
        output_path = (manifest_path.parent / raw_output).resolve()
        if not input_path.is_file():
            raise ManifestError(f"line {line_number}: input not found: {input_path}")
        output_key = _path_collision_key(output_path)
        if output_key in output_keys:
            raise ManifestError(f"line {line_number}: duplicate output {output_path}")
        output_keys.add(output_key)

        raw_expected_doi = _optional_string(row, "expected_doi", line_number)
        expected_doi = normalize_doi(raw_expected_doi) if raw_expected_doi else None
        if raw_expected_doi and expected_doi is None:
            raise ManifestError(f"line {line_number}: invalid expected DOI")
        expected_doi = expected_doi.casefold() if expected_doi else None
        expected_doi_sha256 = _optional_string(row, "expected_doi_sha256", line_number)
        if expected_doi_sha256 is not None:
            expected_doi_sha256 = expected_doi_sha256.casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_doi_sha256):
                raise ManifestError(f"line {line_number}: expected DOI hash must be SHA-256")
            if expected_doi and doi_sha256(expected_doi) != expected_doi_sha256:
                raise ManifestError(f"line {line_number}: expected DOI hash does not match DOI")
        source_sha256 = _optional_string(row, "source_sha256", line_number)
        if source_sha256 is not None:
            source_sha256 = source_sha256.casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
                raise ManifestError(f"line {line_number}: source_sha256 must be SHA-256")
        expected_title = _optional_string(row, "expected_title", line_number)
        target_block_hint = row.get("target_block_hint")
        if target_block_hint is not None and not isinstance(target_block_hint, dict):
            raise ManifestError(f"line {line_number}: target_block_hint must be an object")
        if "doi_required" in row and not isinstance(row["doi_required"], bool):
            raise ManifestError(f"line {line_number}: doi_required must be a boolean")

        content_sha256 = content_hashes.get(input_path)
        if content_sha256 is None:
            content_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
            content_hashes[input_path] = content_sha256

        records.append(
            ManifestRecord(
                input_path=input_path,
                output_path=output_path,
                expected_identity=ExpectedIdentity(
                    queue_record_id=queue_record_id,
                    expected_doi=expected_doi,
                    expected_doi_sha256=expected_doi_sha256,
                    expected_title=(expected_title.strip() or None) if expected_title else None,
                    target_block_hint=target_block_hint,
                    source_sha256=source_sha256,
                    doi_required=row.get("doi_required", bool(expected_doi or expected_doi_sha256)),
                ),
                content_sha256=content_sha256,
            )
        )
        record_lines.append(line_number)
    if not records:
        raise ManifestError("manifest contains no records")

    input_keys = {_path_collision_key(record.input_path) for record in records}
    manifest_key = _path_collision_key(manifest_path)
    for line_number, record in zip(record_lines, records, strict=True):
        output_key = _path_collision_key(record.output_path)
        if output_key == manifest_key:
            raise ManifestError(
                f"line {line_number}: output path aliases manifest: {record.output_path}"
            )
        if output_key in input_keys:
            raise ManifestError(
                f"line {line_number}: output path aliases manifest input: {record.output_path}"
            )
    return records


def resolve_cli_source_mode(inputs: list[str], manifest: str | None) -> str:
    if bool(inputs) == bool(manifest):
        raise ManifestError("provide exactly one input mode: positional inputs or --manifest")
    return "manifest" if manifest else "inputs"
