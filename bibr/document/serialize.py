"""Lossless, canonical serialisation of a :class:`DocumentLayer`.

``to_dict`` gives plain JSON data: dataclasses carry their class name under
``"@"``, numpy arrays become ``{"@a": [dtype, shape, base64 bytes]}``, tuples
``{"@t": [...]}``, dicts ``{"@d": [[key, value], ...]}`` sorted by key, and
non-finite floats ``{"@f": "nan"}``. ``from_dict`` inverts it exactly, and
:func:`digest` hashes the canonical JSON bytes, so two layers are identical
exactly when their digests are.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from bibr.document import model

_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        model.Decided,
        model.PageColumns,
        model.RenderRecipe,
        model.Font,
        model.RoleTag,
        model.Furniture,
        model.Suppressed,
        model.OutlineEntry,
        model.OutlineGuard,
        model.Link,
        model.StructElem,
        model.DecisionRecord,
        model.Presence,
        model.Block,
        model.Page,
        model.DocumentLayer,
    )
}


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        data = base64.b64encode(array.tobytes()).decode("ascii")
        return {"@a": [array.dtype.str, list(array.shape), data]}
    if is_dataclass(value) and not isinstance(value, type):
        encoded = {"@": type(value).__name__}
        for item in fields(value):
            encoded[item.name] = _encode(getattr(value, item.name))
        return encoded
    if isinstance(value, dict):
        return {"@d": [[_encode(key), _encode(item)] for key, item in sorted(value.items())]}
    if isinstance(value, tuple):
        return {"@t": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"@f": repr(value)}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "@a" in value:
        dtype, shape, data = value["@a"]
        return np.frombuffer(base64.b64decode(data), dtype=np.dtype(dtype)).reshape(shape).copy()
    if "@t" in value:
        return tuple(_decode(item) for item in value["@t"])
    if "@d" in value:
        return {_decode(key): _decode(item) for key, item in value["@d"]}
    if "@f" in value:
        return float(value["@f"])
    cls = _TYPES[value["@"]]
    return cls(**{key: _decode(item) for key, item in value.items() if key != "@"})


def to_dict(layer: model.DocumentLayer) -> dict[str, Any]:
    return _encode(layer)


def from_dict(data: dict[str, Any]) -> model.DocumentLayer:
    """The layer *data* encodes.

    Raises ValueError for anything but a layer of this :data:`LAYER_VERSION`
    and :data:`INDEX_FRAME`: under other strip rules the same PDF has other
    glyph indexes, so such a layer must be rebuilt, not loaded.
    """
    if not isinstance(data, dict) or data.get("@") != "DocumentLayer":
        raise ValueError("not a document layer")
    version, frame = data.get("version"), data.get("index_frame")
    if version != model.LAYER_VERSION:
        raise ValueError(f"document layer version {version!r}, not {model.LAYER_VERSION!r}")
    if frame != model.INDEX_FRAME:
        raise ValueError(f"document layer index frame {frame!r}, not {model.INDEX_FRAME!r}")
    layer = _decode(data)
    if not isinstance(layer, model.DocumentLayer):
        raise ValueError("not a document layer")
    return layer


def canonical_bytes(layer: model.DocumentLayer) -> bytes:
    """Canonical JSON of the layer (the encoding of ``canonical_json_bytes``)."""
    return json.dumps(
        to_dict(layer),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(layer: model.DocumentLayer) -> str:
    return hashlib.sha256(canonical_bytes(layer)).hexdigest()
