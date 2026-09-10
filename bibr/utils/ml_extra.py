"""Clear ImportError for dependencies provided by the optional 'ml' extra."""


def ml_import_error(feature: str) -> ImportError:
    """Build the ImportError to raise when an 'ml'-extra dependency is missing."""
    return ImportError(
        f"{feature} requires the 'ml' extra: pip install 'bibr[ml]' (or uv sync --extra ml)"
    )
