"""Install the immutable, fine-tuned reference adapter without ML dependencies."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from bibr.exceptions import BibrError

MODEL_TAG = "model-ref-qwen35-08b-balanced-v1"
MODEL_URL = (
    f"https://github.com/scienceverse/bibr/releases/download/{MODEL_TAG}/"
    "bibr-reference-qwen35-08b-balanced-v1.zip"
)
MODEL_SHA256 = "d827c8c7dc96df8f2ab98d722d37339ecd356b1f06c069f99bc672a35c11c650"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ADAPTER_SHA256 = "dc7380ae4125d36134c6071c99896a616bb37e6a1ac8fc93fef5de2824ef934f"
MANIFEST_SHA256 = "e32404bec2ad66e04f8a662dccf66489467469b45914c75e7ebefaed4057198f"
_MAX_BYTES = 128 * 1024 * 1024
_FILES = frozenset(
    {
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE.md",
        "BASE_MODEL_LICENSE",
        "README.md",
        "manifest.json",
    }
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(directory: Path) -> None:
    if {p.name for p in directory.iterdir()} != _FILES:
        raise ValueError("unexpected or missing adapter files")
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError("adapter files must be regular files")
    if _digest(directory / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("adapter manifest checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("model") != "Qwen/Qwen3.5-0.8B"
        or manifest.get("base_revision") != BASE_REVISION
        or manifest.get("adapter") != "balanced-r32-v1"
        or manifest.get("schema") != "ref-llm-sft@1"
        or set(manifest.get("files", {})) != _FILES - {"manifest.json"}
        or manifest["files"].get("adapter_model.safetensors") != ADAPTER_SHA256
    ):
        raise ValueError("adapter manifest identity mismatch")
    for name, expected in manifest["files"].items():
        if _digest(directory / name) != expected:
            raise ValueError(f"adapter file checksum mismatch: {name}")


def _unpack(archive: Path, directory: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if (
            len(members) != len(_FILES)
            or {m.filename for m in members} != _FILES
            or sum(m.file_size for m in members) > _MAX_BYTES
        ):
            raise ValueError("unexpected adapter archive contents")
        for member in members:
            # No extractall: the fixed flat allowlist excludes paths and links.
            (directory / member.filename).write_bytes(bundle.read(member))
    _verify(directory)


def download(output: Path) -> Path:
    """Verify an existing install, or download and atomically install the bundle."""
    output = output.expanduser().absolute()
    try:
        if output.exists():
            _verify(output)
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".bibr-ref-", dir=output.parent) as temporary:
            root = Path(temporary)
            archive = root / "model.zip"
            # The immutable download URL is HTTPS; no user-supplied URL is accepted.
            request = urllib.request.Request(  # noqa: S310
                MODEL_URL, headers={"User-Agent": "bibr-ref-model"}
            )
            with (
                urllib.request.urlopen(request, timeout=30) as response,  # noqa: S310
                archive.open("wb") as stream,
            ):
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        raise ValueError("adapter download exceeds the size limit")
                    stream.write(chunk)
            if _digest(archive) != MODEL_SHA256:
                raise ValueError("adapter download checksum mismatch")
            staging = root / "adapter"
            staging.mkdir()
            _unpack(archive, staging)
            # Refuse replacement, including an empty directory created concurrently.
            if output.exists():
                raise ValueError("destination was created during download")
            staging.rename(output)
        return output
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        raise BibrError(
            "Could not install the pinned reference adapter. Check network access and "
            "the destination; existing files are preserved. " + str(exc)
        ) from exc
