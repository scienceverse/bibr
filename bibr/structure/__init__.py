"""Structure subpackage - document parsing and section classification."""

__all__ = [
    "classify_header",
    "classify_headers_batch",
    "classify_headers_batch_async",
    "detect_bib_xrefs",
    "detect_implicit_sections",
]


def __getattr__(name):
    if name in ("classify_header", "classify_headers_batch", "classify_headers_batch_async"):
        from bibr.structure.section_classifier import (
            classify_header,
            classify_headers_batch,
            classify_headers_batch_async,
        )

        globals()["classify_header"] = classify_header
        globals()["classify_headers_batch"] = classify_headers_batch
        globals()["classify_headers_batch_async"] = classify_headers_batch_async
        return globals()[name]
    if name == "detect_bib_xrefs":
        from bibr.structure.citation_linker import detect_bib_xrefs

        globals()["detect_bib_xrefs"] = detect_bib_xrefs
        return detect_bib_xrefs
    if name == "detect_implicit_sections":
        from bibr.structure.implicit_sections import detect_implicit_sections

        globals()["detect_implicit_sections"] = detect_implicit_sections
        return detect_implicit_sections
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
