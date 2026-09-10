"""Enrich subpackage - external API enrichment (CrossRef, etc.)."""

__all__ = ["enrich_references", "consolidate_bibs"]


def __getattr__(name):
    if name == "enrich_references":
        from bibr.enrich.references import enrich_references

        globals()[name] = enrich_references
        return enrich_references
    if name == "consolidate_bibs":
        from bibr.enrich.consolidate import consolidate_bibs

        globals()[name] = consolidate_bibs
        return consolidate_bibs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
