"""Local single-machine pipeline for researchers.

Provides ``bibr chew`` CLI — runs the full paper processing pipeline
on a single machine without the LitServe API.
"""

__all__ = ["LocalPipeline"]


def __getattr__(name):
    if name == "LocalPipeline":
        from bibr.local.pipeline import LocalPipeline

        globals()["LocalPipeline"] = LocalPipeline
        return LocalPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
