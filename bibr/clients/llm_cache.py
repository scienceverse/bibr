"""Content-addressed disk cache for structured LLM responses.

The key covers everything that determines the answer — model, response schema,
system prompt, and the flattened user text — so an entry can only ever serve a
request that would have produced it. Nothing about the paper, the run, or the
call site enters the key, which is what lets one entry serve an identical
prompt across papers, runs, and machines.

That property is what makes this a safe *prefill* target: an offline Anthropic
Message Batch (``bibr/clients/batch.py``) answers a pile of requests at half
price and writes them here, and a later ordinary run finds them already
answered. A key that misses simply falls through to a live call, so a prefill
that is stale, partial, or absent costs correctness nothing.

Storage is one JSON file per key, fanned out by the first two hex characters to
keep directory sizes sane. Writes are atomic (tmp + ``os.replace``) and every
failure degrades to "uncached" rather than raising — same contract as the OCR
disk cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.config import GlobalSettings

logger = logging.getLogger(__name__)

# Bump when the stored payload shape changes. Entries carrying a different
# version are ignored (and rewritten on the next put) rather than migrated.
_CACHE_FORMAT_VERSION = 1

# 32 hex chars of SHA-256. Collision risk is negligible at any corpus size bibr
# will see, and short keys keep the batch custom_id well inside Anthropic's
# 64-character limit.
_KEY_CHARS = 32


# Prompts fence document data behind a per-call `uuid4().hex` boundary as an
# injection guard (bibr.clients.prompts.fence), so the same logical request has
# different bytes on every run. Marker lines look like `--- <hex32> START ---`
# or `--- <hex32> BEGIN REFERENCES ---`; matching the marker *line* rather than
# any bare 32-hex token keeps an MD5 that happens to appear in a paper's text
# from being canonicalised away and merging two genuinely different documents.
_FENCE_MARKER_RE = re.compile(r"^--- ([0-9a-f]{32}) [A-Z][A-Z ]*---$", re.MULTILINE)
_BOUNDARY_PLACEHOLDER = "<bibr:boundary>"


def canonical_user_text(text: str) -> str:
    """Replace per-call fence boundaries so the same request hashes the same.

    Without this the cache could never hit: 10 of bibr's 13 call sites mint a
    fresh boundary per call, so identical work produces a fresh key every run.
    """
    for boundary in set(_FENCE_MARKER_RE.findall(text)):
        text = text.replace(boundary, _BOUNDARY_PLACEHOLDER)
    return text


def request_key(
    *,
    model: str,
    schema_name: str,
    system: str,
    user_text: str,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    mode: str | None = None,
    schema_json: str | None = None,
    chat_template_json: str | None = None,
) -> str:
    """Stable identity for one structured request.

    Everything that can change the answer is in the key. ``max_tokens`` and
    ``reasoning_effort`` vary per task, and ``mode`` separates transports that
    are handed byte-identical text but are expected to answer differently — the
    JSON-mode re-roll resends the classification prompt verbatim precisely
    because the default transport returned something unusable.

    Fields are length-prefixed rather than concatenated so no rearrangement of
    adjacent fields can collide.
    """
    digest = hashlib.sha256()
    fields = (
        model,
        schema_name,
        system,
        canonical_user_text(user_text),
        "" if max_tokens is None else str(max_tokens),
        reasoning_effort or "",
        mode or "",
        schema_json or "",
        chat_template_json or "",
    )
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(encoded)
    return digest.hexdigest()[:_KEY_CHARS]


def _effective_settings(settings: GlobalSettings | None) -> GlobalSettings:
    if settings is not None:
        return settings
    from bibr.config import snapshot_settings

    return snapshot_settings()


def cache_dir(settings: GlobalSettings | None = None) -> Path:
    """Resolve the cache directory: ``CACHE_LLM_DIR`` else ``$XDG_CACHE_HOME``/~."""
    configured = _effective_settings(settings).cache.llm_dir
    if configured:
        return Path(configured).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "bibr" / "llm"


def is_enabled(settings: GlobalSettings | None = None) -> bool:
    """Whether the LLM response cache is turned on (``CACHE_LLM``)."""
    return _effective_settings(settings).cache.llm


class LlmResponseCache:
    """Read/write access to one cache root. Never raises on I/O failure."""

    def __init__(self, root: Path | str | None = None, *, settings: GlobalSettings | None = None):
        self.root = Path(root).expanduser() if root is not None else cache_dir(settings)

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return the cached response body, or ``None`` on any miss."""
        path = self.path_for(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.debug("LLM cache read failed for %s (%s); treating as a miss", key, e)
            return None
        if not isinstance(entry, dict) or entry.get("version") != _CACHE_FORMAT_VERSION:
            return None
        body = entry.get("response")
        return body if isinstance(body, dict) else None

    def put(
        self,
        key: str,
        response: dict,
        *,
        model: str,
        schema_name: str,
        label: str | None = None,
        source: str = "live",
    ) -> bool:
        """Store one response. Returns whether it was written.

        ``source`` records where the answer came from (``"live"`` or
        ``"batch"``) purely as provenance for anyone inspecting the cache.
        """
        path = self.path_for(key)
        payload = {
            "version": _CACHE_FORMAT_VERSION,
            "model": model,
            "schema": schema_name,
            "label": label,
            "source": source,
            "response": response,
        }
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.info("LLM cache write failed (%s); continuing uncached", e)
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
            return False
