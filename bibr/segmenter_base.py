"""Shared wtpsplit-lite SaT sentence-segmenter core.

``BaseSentenceSegmenter`` owns model loading (ONNX provider selection, GPU
memory cap, warmup) and the sync split core with OOM-halving retry. Variants
layer concurrency + lifecycle on top — ``bibr.local.segmenter`` serializes via
a per-loop asyncio lock and supports ``unload()``, while
``bibr.serve.deployments.segmenter`` coalesces concurrent requests through a
GpuBatcher.
"""

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bibr.config import GlobalSettings, snapshot_settings

logger = logging.getLogger(__name__)
_FALLBACK_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SEGMENTER_MANIFEST = "segmenter_manifest.json"
_SEGMENTER_STAGING_MANIFEST = "segmenter_staging.json"
_TOKENIZER_MODEL = "FacebookAI/xlm-roberta-base"
_TOKENIZER_FILES = {
    "tokenizer/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/sentencepiece.bpe.model",
}
_SEALED_RUNTIME_FILES = {
    "model.onnx",
    "model_optimized.onnx",
    "config.json",
} | _TOKENIZER_FILES
_SEALED_PATH_FIELDS = {
    "metrics_path": "evaluation/metrics.json",
    "calibration_path": "evaluation/calibration.json",
    "evaluation_report_path": "evaluation/report.md",
    "gate_path": "evaluation/gate.json",
    "performance_path": "evaluation/performance.json",
    "release_evidence_path": "provenance/release_evidence.json",
    "source_manifest_path": "provenance/source_manifest.yaml",
    "build_manifest_path": "provenance/build_manifest.json",
    "training_run_path": "provenance/training_run.json",
    "environment_lock_path": "provenance/uv.lock",
    "environment_project_path": "provenance/pyproject.toml",
    "compat_probe_path": "provenance/compat_probe.py",
    "parity_report_path": "parity/report.json",
    "model_card_path": "MODEL_CARD.md",
    "staging_manifest_path": "segmenter_staging.json",
}


def _validate_threshold(value: object, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Sentence-segmenter threshold from {source} must be a number in [0, 1]")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Sentence-segmenter threshold from {source} must be in [0, 1]")
    return threshold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_bundle_file(model_dir: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Sentence-segmenter manifest {field} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Sentence-segmenter manifest {field} escapes the bundle")
    path = model_dir / relative
    current = model_dir
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Sentence-segmenter manifest {field} uses a symlink")
    if not path.is_file():
        raise ValueError(f"Sentence-segmenter manifest {field} is missing: {value}")
    return path


def _validate_sealed_manifest(
    model_dir: Path, manifest: dict[str, Any]
) -> tuple[float, str, int, int]:
    required = {
        "schema_version",
        "threshold",
        "model_revision",
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_checksums",
        "language_scope",
        "windowing",
        "checksums",
    }
    required.update(_SEALED_PATH_FIELDS)
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Sentence-segmenter manifest is missing: {', '.join(missing)}")
    if manifest["schema_version"] != 1 or manifest["language_scope"] != "en":
        raise ValueError("Sentence-segmenter manifest must be schema 1 with language_scope en")
    if manifest["tokenizer_model"] != _TOKENIZER_MODEL:
        raise ValueError(f"Sentence-segmenter manifest tokenizer_model must be {_TOKENIZER_MODEL}")
    tokenizer_revision = manifest["tokenizer_revision"]
    if (
        not isinstance(tokenizer_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision) is None
    ):
        raise ValueError(
            "Sentence-segmenter manifest tokenizer_revision must be a 40-character "
            "lowercase hexadecimal revision"
        )
    threshold = _validate_threshold(
        manifest["threshold"], source=str(model_dir / _SEGMENTER_MANIFEST)
    )
    windowing = manifest["windowing"]
    if not isinstance(windowing, dict) or set(windowing) != {"block_size", "eval_stride"}:
        raise ValueError(
            "Sentence-segmenter manifest windowing must define only block_size and eval_stride"
        )
    block_size = windowing["block_size"]
    eval_stride = windowing["eval_stride"]
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or not isinstance(eval_stride, int)
        or isinstance(eval_stride, bool)
        or block_size <= 0
        or eval_stride <= 0
        or eval_stride > block_size
    ):
        raise ValueError("Sentence-segmenter manifest windowing values are invalid")

    declared = {field: manifest[field] for field in _SEALED_PATH_FIELDS}
    for field, expected in _SEALED_PATH_FIELDS.items():
        if declared[field] != expected:
            raise ValueError(
                f"Sentence-segmenter manifest {field} must use canonical path {expected}"
            )
    if len(set(declared.values())) != len(declared):
        raise ValueError("Sentence-segmenter manifest artifact paths must be distinct")
    if set(declared.values()) & (_SEALED_RUNTIME_FILES | {_SEGMENTER_MANIFEST}):
        raise ValueError("Sentence-segmenter manifest artifact paths overlap runtime files")
    for field, relative in declared.items():
        _local_bundle_file(model_dir, relative, field=field)

    bundle_paths = list(model_dir.rglob("*"))
    if symlinks := sorted(
        str(path.relative_to(model_dir)) for path in bundle_paths if path.is_symlink()
    ):
        raise ValueError("Sentence-segmenter bundle contains symlink files: " + ", ".join(symlinks))
    actual_files = {str(path.relative_to(model_dir)) for path in bundle_paths if path.is_file()}
    expected_files = _SEALED_RUNTIME_FILES | set(declared.values()) | {_SEGMENTER_MANIFEST}
    if unexpected_files := sorted(actual_files - expected_files):
        raise ValueError(
            "Sentence-segmenter bundle contains unexpected files: " + ", ".join(unexpected_files)
        )

    checksums = manifest["checksums"]
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Sentence-segmenter manifest checksums must be a non-empty object")
    required_checksums = expected_files - {_SEGMENTER_MANIFEST}
    if not all(isinstance(relative, str) for relative in checksums):
        raise ValueError("Sentence-segmenter manifest checksum paths must be strings")
    if set(checksums) != required_checksums:
        raise ValueError("Sentence-segmenter manifest must checksum every bundle file exactly")
    for relative, expected in checksums.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"Sentence-segmenter manifest checksum is invalid: {relative}")
        path = _local_bundle_file(model_dir, relative, field=f"checksums[{relative}]")
        if _sha256(path) != expected:
            raise ValueError(f"Sentence-segmenter bundle checksum mismatch: {relative}")

    tokenizer_checksums = manifest["tokenizer_checksums"]
    if not isinstance(tokenizer_checksums, dict) or set(tokenizer_checksums) != _TOKENIZER_FILES:
        raise ValueError(
            "Sentence-segmenter manifest tokenizer_checksums must contain exactly: "
            + ", ".join(sorted(_TOKENIZER_FILES))
        )
    for relative, expected in tokenizer_checksums.items():
        if expected != checksums[relative] or _sha256(model_dir / relative) != expected:
            raise ValueError(f"Sentence-segmenter tokenizer checksum mismatch: {relative}")

    revision = manifest["model_revision"]
    optimized_hash = _sha256(model_dir / "model_optimized.onnx")
    if revision != optimized_hash:
        raise ValueError(
            "Sentence-segmenter manifest model_revision does not match model_optimized.onnx"
        )
    return threshold, revision, block_size, eval_stride


#: Hub revisions for the default segmenter (HF API, 2026-09-02). Other Hub
#: models load ``main`` unless ``WTPSPLIT_MODEL_REVISION`` pins them.
_WTPSPLIT_PINNED_REVISIONS: dict[str, str] = {
    "sat-6l-sm": "d85d2b6ddfb19036c4c8e8b3b7ca45da684b0905",
}


@dataclass(frozen=True)
class ResolvedSegmenterModel:
    """Resolved wtpsplit model source and optional local bundle metadata."""

    model_name: str
    hub_prefix: str | None
    is_local: bool
    #: Hub revision to load (``None`` = the repo head); local bundles use the manifest.
    revision: str | None = None
    manifest_threshold: float | None = None
    manifest_revision: str | None = None
    manifest_block_size: int | None = None
    manifest_eval_stride: int | None = None
    tokenizer_name_or_path: str | None = None

    @property
    def source(self) -> str:
        if self.is_local:
            return f"local:{self.model_name}"
        return f"huggingface:{self.repo_id}"

    @property
    def repo_id(self) -> str:
        if self.hub_prefix is None:
            return self.model_name
        return f"{self.hub_prefix}/{self.model_name}"


def _read_local_manifest(
    model_dir: Path,
) -> tuple[float | None, str | None, int | None, int | None]:
    manifest_path = model_dir / _SEGMENTER_MANIFEST
    if not manifest_path.is_file():
        return None, None, None, None
    if manifest_path.is_symlink():
        raise ValueError(f"Sentence-segmenter manifest {manifest_path} uses a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read sentence-segmenter manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Sentence-segmenter manifest {manifest_path} must contain a JSON object")

    return _validate_sealed_manifest(model_dir, manifest)


def materialize_hub_snapshot(repo_id: str, revision: str) -> tuple[str, str | None]:
    """Fetch the files wtpsplit-lite opens at ``revision``; return (model dir, tokenizer dir).

    wtpsplit-lite forwards ``from_pretrained_kwargs`` to its own config loader,
    which takes no ``revision``, so a pinned Hub model cannot be requested
    through ``SaT`` itself. Download (or, for a commit hash already in the
    cache, merely locate — no network round-trip) ``model_optimized.onnx``
    and ``config.json`` at the pinned commit and hand ``SaT`` the snapshot
    directory instead. The tokenizer directory is ``None`` when the repo
    ships no ``tokenizer.json`` (the ``sat-*`` repos), in which case
    wtpsplit-lite loads its XLM-R base tokenizer as it does for any Hub name.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    onnx_path = Path(hf_hub_download(repo_id, "model_optimized.onnx", revision=revision))
    config_path = Path(hf_hub_download(repo_id, "config.json", revision=revision))
    if config_path.parent != onnx_path.parent:
        raise RuntimeError(
            f"Pinned wtpsplit files for {repo_id}@{revision} resolved to different snapshots"
        )
    try:
        tokenizer_path = Path(hf_hub_download(repo_id, "tokenizer.json", revision=revision))
    except EntryNotFoundError:
        # Covers both the remote 404 and the offline cache miss for a repo
        # that has no tokenizer.json at this commit.
        return str(onnx_path.parent), None
    return str(onnx_path.parent), str(tokenizer_path.parent)


def resolve_wtpsplit_model(model_name: str, revision: str | None = None) -> ResolvedSegmenterModel:
    """Resolve a short wtpsplit name, full Hub ID, or existing local bundle.

    ``revision`` pins a Hub model to a commit; unset, the default short name
    resolves to its audited commit and anything else to the repo head.
    """
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        if local_path.is_symlink():
            raise ValueError(f"Sentence-segmenter bundle {local_path} uses a symlink")
        if (local_path / _SEGMENTER_STAGING_MANIFEST).is_file() and not (
            local_path / _SEGMENTER_MANIFEST
        ).is_file():
            raise ValueError(
                f"Sentence-segmenter bundle {local_path} is an unsealed staging export"
            )
        normalized_path = str(local_path)
        threshold, revision, block_size, eval_stride = _read_local_manifest(local_path)
        tokenizer_dir = local_path / "tokenizer"
        if (tokenizer_dir / "tokenizer.json").is_file():
            tokenizer_name_or_path = str(tokenizer_dir)
        elif (local_path / "tokenizer.json").is_file():
            # Explicit manifestless legacy directories used a root tokenizer.
            tokenizer_name_or_path = normalized_path
        else:
            tokenizer_name_or_path = None
        return ResolvedSegmenterModel(
            model_name=normalized_path,
            hub_prefix=None,
            is_local=True,
            manifest_threshold=threshold,
            manifest_revision=revision,
            manifest_block_size=block_size,
            manifest_eval_stride=eval_stride,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
    is_full_repo_id = "/" in model_name
    return ResolvedSegmenterModel(
        model_name=model_name,
        hub_prefix=None if is_full_repo_id else "segment-any-text",
        is_local=False,
        revision=revision or _WTPSPLIT_PINNED_REVISIONS.get(model_name),
        tokenizer_name_or_path=model_name if is_full_repo_id else None,
    )


def _coerce_split_text(text: Any) -> str:
    """Normalize OCR/parser scalars before handing them to tokenizer code."""
    if isinstance(text, str):
        return text
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return str(text)


def _is_tokenizer_textencode_error(exc: Exception) -> bool:
    return "TextEncodeInput" in str(exc)


def _fallback_split_text(text: str) -> list[str]:
    """Conservative splitter for a paragraph wtpsplit/tokenizers cannot encode."""
    text = text.strip()
    if not text:
        return []
    return [piece for piece in (p.strip() for p in _FALLBACK_SENTENCE_RE.split(text)) if piece]


class BaseSentenceSegmenter:
    """wtpsplit-lite SaT wrapper: loading, warmup, and the sync split core.

    Subclasses set ``_variant`` (used in device reporting), may extend
    ``__init__`` to add their concurrency state, and implement their own
    async ``segment_batch``.
    """

    _variant = "base"

    def __init__(
        self,
        model_name: str | None = None,
        use_gpu: bool | None = None,
        threshold: float | None = None,
        settings: GlobalSettings | None = None,
    ):
        from wtpsplit_lite import SaT

        from bibr.utils.device import report_device
        from bibr.utils.onnx_providers import cuda_provider_available, get_ort_providers

        self._settings = settings if settings is not None else snapshot_settings()
        self._resolved_model = resolve_wtpsplit_model(
            model_name or self._settings.WTPSPLIT_MODEL,
            revision=self._settings.WTPSPLIT_MODEL_REVISION,
        )
        self._model_name = self._resolved_model.model_name
        if threshold is not None:
            self._threshold = _validate_threshold(threshold, source="constructor")
            self._threshold_source = "constructor"
        elif self._settings.WTPSPLIT_THRESHOLD is not None:
            self._threshold = _validate_threshold(
                self._settings.WTPSPLIT_THRESHOLD,
                source="WTPSPLIT_THRESHOLD",
            )
            self._threshold_source = "WTPSPLIT_THRESHOLD"
        elif self._resolved_model.manifest_threshold is not None:
            self._threshold = self._resolved_model.manifest_threshold
            self._threshold_source = _SEGMENTER_MANIFEST
        else:
            self._threshold = None
            self._threshold_source = "wtpsplit default"
        setting_block_size = self._settings.WTPSPLIT_BLOCK_SIZE
        setting_stride = self._settings.WTPSPLIT_STRIDE
        if (setting_block_size is None) != (setting_stride is None):
            raise ValueError("Wtpsplit block size and stride must be configured together")
        if setting_block_size is not None and setting_stride is not None:
            if setting_stride > setting_block_size:
                raise ValueError("Wtpsplit stride cannot exceed block size")
            self._block_size = setting_block_size
            self._eval_stride = setting_stride
            self._windowing_source = "WTPSPLIT_BLOCK_SIZE/WTPSPLIT_STRIDE"
        else:
            self._block_size = self._resolved_model.manifest_block_size
            self._eval_stride = self._resolved_model.manifest_eval_stride
            self._windowing_source = (
                _SEGMENTER_MANIFEST if self._block_size is not None else "wtpsplit default"
            )

        # Auto-detect: use the GPU when onnxruntime exposes the CUDA EP, unless
        # explicitly disabled (use_gpu=False) for VRAM-constrained deployments.
        if use_gpu is None:
            use_gpu = cuda_provider_available()

        mem_limit = (
            self._settings.SEGMENTER_GPU_MEM_LIMIT_MB * 1024 * 1024
            if self._settings.SEGMENTER_GPU_MEM_LIMIT_MB > 0 and use_gpu
            else None
        )
        providers = get_ort_providers(
            enable_cuda=use_gpu,
            model_name="wtpsplit-sat",
            gpu_mem_limit=mem_limit,
        )
        model_kwargs: dict[str, Any] = {
            "ort_providers": providers,
            "hub_prefix": self._resolved_model.hub_prefix,
        }
        if self._resolved_model.tokenizer_name_or_path is not None:
            model_kwargs["tokenizer_name_or_path"] = self._resolved_model.tokenizer_name_or_path
        sat_target = self._model_name
        if not self._resolved_model.is_local and self._resolved_model.revision is not None:
            # Pinned Hub model: materialise the snapshot at that commit and load
            # it as a directory (see materialize_hub_snapshot for why).
            sat_target, pinned_tokenizer = materialize_hub_snapshot(
                self._resolved_model.repo_id, self._resolved_model.revision
            )
            model_kwargs["hub_prefix"] = None
            if pinned_tokenizer is not None:
                model_kwargs["tokenizer_name_or_path"] = pinned_tokenizer
        self.model = SaT(sat_target, **model_kwargs)
        on_cuda = any(
            (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in providers
        )
        logger.info(
            "SentenceSegmenter ready (source=%s, threshold=%s, threshold_source=%s, "
            "windowing=%s/%s, windowing_source=%s, manifest_revision=%s, providers=%s)",
            self._resolved_model.source,
            self._threshold if self._threshold is not None else "default",
            self._threshold_source,
            self._block_size if self._block_size is not None else "default",
            self._eval_stride if self._eval_stride is not None else "default",
            self._windowing_source,
            self._resolved_model.manifest_revision or "none",
            [p if isinstance(p, str) else p[0] for p in providers],
        )
        report_device(
            f"SentenceSegmenter ({self._variant})", "cuda" if on_cuda else "cpu", gpu_capable=True
        )
        self._run_warmup()

    def _run_warmup(self):
        """Run warmup inference to initialize ONNX session and JIT caches."""
        if self.model is None:
            return
        try:
            self.model.split(
                ["This is a warmup sentence. It has two parts."], **self._split_kwargs()
            )
            logger.info("SentenceSegmenter warmup complete")
        except Exception as e:
            logger.warning("SentenceSegmenter warmup failed: %s", e)

    def _split_many(self, texts: list[str]) -> list[list[str]]:
        """Split each text into sentences using batch inference.

        Processes texts in sub-batches of SEGMENTER_SUB_BATCH_SIZE to prevent
        GPU OOM when many texts are batched together (large papers or
        concurrent requests sharing the same GPU).
        """
        if not texts:
            return []

        sub_batch = self._settings.SEGMENTER_SUB_BATCH_SIZE
        if len(texts) > sub_batch:
            logger.debug("Sub-batching %d texts into groups of %d", len(texts), sub_batch)

        results: list[list[str]] = []
        for start in range(0, len(texts), sub_batch):
            batch = texts[start : start + sub_batch]
            results.extend(self._split_batch_with_retry(batch))

        return results

    def _split_batch_with_retry(self, texts: list[str]) -> list[list[str]]:
        """Run model.split() with automatic batch halving on OOM."""
        try:
            return self._collect_split_result(texts)
        except Exception as exc:
            if "BFCArena" not in str(exc) and "out of memory" not in str(exc).lower():
                raise

            if len(texts) == 1:
                logger.error(
                    "OOM on single text (%d chars), cannot reduce batch further", len(texts[0])
                )
                raise

            half = max(1, len(texts) // 2)
            logger.warning(
                "OOM with batch size %d, retrying as %d + %d",
                len(texts),
                half,
                len(texts) - half,
            )
            left = self._split_batch_with_retry(texts[:half])
            right = self._split_batch_with_retry(texts[half:])
            return left + right

    def _collect_split_result(self, batch: list[str]) -> list[list[str]]:
        """Call model.split() and normalize the result to list[list[str]]."""
        assert self.model is not None  # noqa: S101 — segment_batch ensures load
        batch = [_coerce_split_text(text) for text in batch]
        try:
            return self._collect_model_split_result(batch)
        except TypeError as exc:
            if not _is_tokenizer_textencode_error(exc):
                raise
            return self._collect_split_result_with_tokenizer_fallback(batch, exc)

    def _collect_model_split_result(self, batch: list[str]) -> list[list[str]]:
        """Call wtpsplit and normalize its generator/list return shape."""
        assert self.model is not None  # noqa: S101 — segment_batch ensures load
        split_result = self.model.split(batch, **self._split_kwargs())
        if hasattr(split_result, "__next__"):
            return list(split_result)
        if len(batch) == 1 and split_result and not isinstance(split_result[0], list):
            return [split_result]
        return list(split_result)

    def _split_kwargs(self) -> dict[str, float | int]:
        kwargs: dict[str, float | int] = {}
        if self._threshold is not None:
            kwargs["threshold"] = self._threshold
        block_size = getattr(self, "_block_size", None)
        eval_stride = getattr(self, "_eval_stride", None)
        if block_size is not None and eval_stride is not None:
            kwargs["block_size"] = block_size
            kwargs["stride"] = eval_stride
        return kwargs

    def _collect_split_result_with_tokenizer_fallback(
        self, batch: list[str], exc: TypeError
    ) -> list[list[str]]:
        """Isolate wtpsplit tokenizer failures to one paragraph when possible."""
        if len(batch) == 1:
            text = batch[0]
            logger.warning(
                "SentenceSegmenter tokenizer rejected one text (%d chars); "
                "using punctuation fallback: %s",
                len(text),
                exc,
            )
            return [_fallback_split_text(text)]

        half = max(1, len(batch) // 2)
        left = self._collect_split_result(batch[:half])
        right = self._collect_split_result(batch[half:])
        return left + right
