"""Recover complete leading objects from a truncated JSON array.

A structured LLM call that hits its output-token cap mid-array leaves the array
unclosed, so a whole-string ``json.loads`` fails and the entire completion is
lost — even though the leading objects are usually complete and valid. This
scans a ``"<key>": [ {...}, {...}, ... `` array and returns every ``{...}``
object that closed before truncation, stopping at the first incomplete one.

Stdlib-only leaf module so any layer (the LLM client, the reference extractor)
can import it without pulling in heavy deps.
"""

import json
from typing import Any


def salvage_array_objects(text: str, key: str) -> list[dict[str, Any]]:
    """Return the complete ``{...}`` objects of a ``"<key>": [...]`` array.

    Brace-matches each object (string- and escape-aware) so a truncated
    trailing object, a mid-string cutoff, or a trailing comma stops the walk
    cleanly rather than corrupting the recovered prefix, and the walk ends at
    the array's closing bracket so trailing sibling values are never picked up.
    Returns ``[]`` when the key or its opening bracket is absent (garbage
    input), and — on a complete, well-formed array — every object, so it is a
    safe no-op on untruncated text.
    """
    marker = f'"{key}"'
    idx = text.find(marker)
    if idx == -1:
        return []
    array_start = text.find("[", idx)
    if array_start == -1:
        return []

    objects: list[dict[str, Any]] = []
    i = array_start + 1
    n = len(text)
    while i < n:
        while i < n and text[i] != "{":
            if text[i] == "]":
                return objects
            i += 1
        if i >= n:
            break
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0 or j >= n:
            break
        try:
            objects.append(json.loads(text[i : j + 1]))
        except json.JSONDecodeError:
            break
        i = j + 1
    return objects
