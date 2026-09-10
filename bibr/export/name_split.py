"""Split a verbatim printed author string into structured person names.

The verbatim string stays canonical (``bib[].authors``); this produces the
derived ``bib[].author`` list. Best-effort by contract: anything that does not
split cleanly is emitted whole as ``{"literal": ...}`` (the JATS
``string-name`` analog) rather than guessed at. The splitter never raises and
never invents characters — every emitted value is a substring of its input.
"""

from __future__ import annotations

import re

# Generational/honorific suffixes that trail a given name.
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

# Corporate authors have no comma-separated family/given structure and often
# carry an organisational keyword; treat them as unsplittable.
_CORPORATE_RE = re.compile(
    r"\b(organization|organisation|institute|university|association|society|"
    r"committee|department|ministry|agency|council|foundation|group|"
    r"consortium|collaboration|centre|center|laboratory|academy|academies|inc|ltd|llc)\b",
    re.IGNORECASE,
)

# "; " is unambiguous. Otherwise split on "&"/" and " first, then on the
# comma pairs that APA style leaves behind.
_AND_RE = re.compile(r"\s*(?:&|\band\b)\s*")

# Editor-list role markers ("(Ed.)"/"(Eds.)"/"(Editor)"/"(Editors)", any case)
# are structural noise, not part of a name — APA style tags them onto the
# last editor (or the whole editor list) the same way it tags a generational
# suffix onto an author. Strip them, with a leading comma if present, before
# any other splitting so they never leak into a given name.
#
# Anchored to a trailing lookahead (end of string, ";", or ",") so this is
# substring-safe *by construction*, not by assumption: a real trailing tag
# sits after a complete name and its removal only ever trims a prefix/suffix
# run, never glues two non-adjacent fragments together. A tag that is NOT in
# trailing position (e.g. "Van (Eds) Houten, K.") fails the lookahead and is
# left untouched — it flows through as (odd but honest) part of the name
# rather than risk stitching "Van" to "Houten" into text that never appeared
# in the source. Bare "Ed"/"Eds" without parentheses is out of scope — too
# ambiguous against real initials.
_ROLE_TAG_RE = re.compile(
    r",?\s*\(\s*(?:editors?|eds?)\.?\s*\)(?=\s*(?:[,;]|$))",
    re.IGNORECASE,
)


def _split_chunks(verbatim: str) -> list[str]:
    # ";" is the one unambiguous person separator, so it splits first; a
    # semicolon-joined list can legitimately mix persons and organizations
    # ("Smith, J.; World Health Organization"), so corporate detection must
    # not run on the whole, un-split verbatim string.
    chunks = verbatim.split(";") if ";" in verbatim else [verbatim]

    parts = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Check the whole (semicolon-bounded) chunk for an institutional
        # keyword *before* any "&"/"and" or comma splitting fragments it.
        # An institution's defining keyword can straddle an "and"
        # ("Department of Health and Human Services") or sit past more
        # commas than a person name would ("National Institute of Mental
        # Health, Bethesda, MD, USA") — splitting first would tear it into
        # fake persons.
        if _CORPORATE_RE.search(chunk):
            parts.append(chunk)
            continue
        for group in _AND_RE.split(chunk):
            parts.extend(_split_comma_pairs(group))
    return [p.strip().strip(",").strip() for p in parts if p.strip().strip(",").strip()]


def _split_comma_pairs(group: str) -> list[str]:
    """Regroup "Smith, J., Jones, K." into ["Smith, J.", "Jones, K."].

    Commas alternate between the family/given separator and the person
    separator, so pair the comma-delimited fields two at a time. A trailing
    generational suffix belongs to the preceding person, not to a new one
    ("King, M. L., Jr." is one author), so it gets re-joined before pairing.
    """
    fields = [f.strip() for f in group.split(",") if f.strip()]
    if len(fields) >= 3 and fields[-1].casefold() in _SUFFIXES:
        # Pop the suffix first: assigning to fields[-2] in the same expression
        # as the pop() call is an evaluation-order hazard (the RHS pop mutates
        # the list before the LHS index is resolved against it).
        suffix_field = fields.pop()
        fields[-1] = f"{fields[-1]}, {suffix_field}"
    if len(fields) <= 2:
        return [", ".join(fields)] if fields else []
    return [", ".join(fields[i : i + 2]) for i in range(0, len(fields), 2)]


def _parse_one(chunk: str) -> dict:
    if _CORPORATE_RE.search(chunk) or "," not in chunk:
        return {"literal": chunk}

    family, _, remainder = chunk.partition(",")
    family = family.strip()
    given = remainder.strip().strip(",").strip()
    if not family or not given:
        return {"literal": chunk}

    suffix = None
    tail = given.replace(",", " ").split()
    if tail and tail[-1].casefold() in _SUFFIXES:
        suffix = tail[-1]
        given = given[: given.rfind(suffix)].strip().strip(",").strip()

    if not given:
        return {"literal": chunk}

    person = {"family": family, "given": given}
    if suffix:
        person["suffix"] = suffix
    return person


def split_person_names(verbatim: str | None) -> list[dict]:
    """Return structured names for *verbatim*; ``[]`` when there is nothing to split."""
    if not verbatim or not verbatim.strip():
        return []
    try:
        cleaned = _ROLE_TAG_RE.sub("", verbatim)
        return [_parse_one(chunk) for chunk in _split_chunks(cleaned)]
    except Exception:  # noqa: BLE001 - a splitter failure must never break export
        return [{"literal": verbatim.strip()}]
