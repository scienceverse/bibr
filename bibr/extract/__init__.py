"""Extract subpackage - metadata extraction from paper content."""

__all__ = [
    "BylineGroup",
    "EquationExtractor",
    "FrontMatterBlock",
    "FrontMatterCandidate",
    "FrontMatterResolution",
    "MetadataExtractor",
    "assess_author_grounding",
    "build_byline_group",
    "render_block_context",
    "resolve_front_matter",
]


def __getattr__(name):
    if name == "MetadataExtractor":
        from bibr.extract.extractor import MetadataExtractor

        globals()["MetadataExtractor"] = MetadataExtractor
        return MetadataExtractor
    if name == "EquationExtractor":
        from bibr.extract.equation_extractor import EquationExtractor

        globals()["EquationExtractor"] = EquationExtractor
        return EquationExtractor
    if name in {
        "BylineGroup",
        "assess_author_grounding",
        "build_byline_group",
        "render_block_context",
    }:
        from bibr.extract import core_metadata

        value = getattr(core_metadata, name)
        globals()[name] = value
        return value
    if name in {
        "FrontMatterBlock",
        "FrontMatterCandidate",
        "FrontMatterResolution",
        "resolve_front_matter",
    }:
        from bibr.extract import front_matter

        value = getattr(front_matter, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
