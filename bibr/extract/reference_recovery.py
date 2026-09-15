"""Ground optional reference recovery in one retained source entry."""

import re
import unicodedata


def _normalized(value: object) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", str(value)).casefold() if char.isalnum()
    )


def ungrounded_fields(fields: dict, source: str) -> tuple[str, ...]:
    """Accept typography normalization, never translation or invented fields.

    Categorical reference types and derived flags are not printed strings. Every
    asserted bibliographic identity field must occur in this entry's own source.
    """
    printed = _normalized(source)
    checked = (
        "title",
        "authors",
        "container",
        "editors",
        "publisher",
        "year",
        "volume",
        "issue",
        "first_page",
        "last_page",
        "doi",
        "url",
        "edition",
        "version",
        "date",
    )
    unsupported = []
    for name in checked:
        value = fields.get(name)
        if value is None or not str(value).strip():
            continue
        normalized = _normalized(value)
        if not normalized or normalized not in printed:
            unsupported.append(name)
        elif normalized.isdecimal() and not re.search(
            rf"(?<!\d){re.escape(normalized)}(?!\d)", unicodedata.normalize("NFKC", source)
        ):
            # A volume of 2 is not supported merely by a printed year of 2021.
            unsupported.append(name)
    return tuple(unsupported)
