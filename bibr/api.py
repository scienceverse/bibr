"""One-call library API: ``bibr.chew()`` / ``bibr.achew()`` → :class:`Result`.

.. code-block:: python

    import bibr

    result = bibr.chew("paper.pdf", ocr="glm")
    result.title             # from the metadata block
    result.references        # list of dicts (alias for the schema's "bib")
    result.references.df     # the same rows as a pandas DataFrame
    result.data              # the raw v11.0 export dict
    result.save("out.json")

Inside an already-running event loop (Jupyter, async apps) use the async
twin: ``result = await bibr.achew("paper.pdf")``.

Single paths return a :class:`Result`. Directories and lists of paths run as
a batch on one pipeline (models load once) and return an order-aligned
``list[Result | ChewFailure]`` — filter with ``[r for r in results if r.ok]``.

For repeated calls over time (notebooks, queue workers), :class:`Chewer`
keeps the pipeline warm across calls instead of reloading models each time::

    with bibr.Chewer(ocr="glm-rapid-mlx") as chewer:
        r1 = chewer.chew("a.pdf")
        r2 = chewer.chew("b.pdf")
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

if TYPE_CHECKING:
    import pandas as pd

    from bibr.config import GlobalSettings
    from bibr.export import PaperExport
    from bibr.pipeline.progress import ProgressTracker

__all__ = [
    "ChewFailure",
    "ChewOptions",
    "Chewer",
    "Records",
    "Result",
    "achew",
    "achew_file",
    "achew_many",
    "chew",
    "chew_file",
    "chew_many",
]

# Table-shaped top-level keys of the v11.0 export schema.
_TABLE_KEYS = (
    "author",
    "text",
    "section",
    "url",
    "bib",
    "bib_match",
    "xref",
    "figure",
    "table",
    "eq",
)

# Friendly attribute → schema key.
_ALIASES = {"references": "bib", "authors": "author", "sections": "section"}

# chew()/achew() option → LocalPipeline constructor argument. Identity
# mappings are accepted too, so both ``ocr=`` and ``ocr_backend=`` work.
_OPTION_TO_PIPELINE_ARG = {
    "ocr": "ocr_backend",
    "llm": "llm_backend",
    "memory": "memory_mode",
    "ocr_backend": "ocr_backend",
    "llm_backend": "llm_backend",
    "memory_mode": "memory_mode",
    "ocr_url": "ocr_url",
    "ocr_model": "ocr_model",
    "ocr_profile": "ocr_profile",
    "device": "device",
    "crossref": "crossref",
    "equations": "equations",
    "no_llm": "no_llm",
    "figure_images": "figure_images",
    "include_regions": "include_regions",
    "include_region_meta": "include_region_meta",
    "start_page": "start_page",
    "end_page": "end_page",
}


class ChewOptions(TypedDict, total=False):
    """Typed keyword options shared by the one-call and warm-pipeline APIs."""

    ocr: str
    llm: str
    memory: str
    ocr_backend: str
    llm_backend: str
    memory_mode: str
    ocr_url: str | None
    ocr_model: str | None
    ocr_profile: Literal["paddle", "glm"] | None
    device: str | None
    crossref: bool | None
    equations: bool
    no_llm: bool
    figure_images: bool | None
    include_regions: bool
    include_region_meta: bool
    start_page: int | None
    end_page: int | None
    pages: str
    ref_seg: Literal["geom", "region", "llm", "crf"] | None
    ref_seg_strategy: Literal["geom", "region", "llm", "crf"] | None
    consolidate: bool | Literal["off", "fill", "replace"] | None


class Records(list):
    """A list of row dicts with a ``.df`` pandas convenience view."""

    @property
    def df(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(list(self))


class ChewFailure:
    """Order-aligned placeholder for a file that failed in a batch chew().

    ``[r for r in results if r.ok]`` filters a batch down to successes.
    """

    ok = False

    def __init__(
        self,
        path: str | Path,
        error: str,
        error_code: str | None = None,
        failed_stage: str | None = None,
    ):
        self.path = Path(path)
        self.error = error
        self.error_code = error_code
        self.failed_stage = failed_stage

    def __repr__(self) -> str:
        stage = f" at {self.failed_stage}" if self.failed_stage else ""
        return f"<bibr.ChewFailure {self.path.name!r}{stage}: {self.error}>"


class Result:
    """Read-only view over a bibr v11.0 export dict.

    Table keys (``bib``, ``author``, ``text``, ...) and their friendly
    aliases (``references``, ``authors``, ``sections``) come back as
    :class:`Records`; ``metadata`` fields (``title``, ``doi``, ...) and
    ``source`` fields (``file_name``, ``file_hash``, ``input_format``) resolve
    as attributes, as do the remaining top-level keys (``paper_id``,
    ``extraction``, ...). The raw dict stays available as :attr:`data`.
    """

    def __init__(self, data: dict[str, Any] | PaperExport):
        from bibr.export import PaperExport

        if isinstance(data, PaperExport):
            self._model = data
            self._data = cast(dict[str, Any], data.model_dump(by_alias=True, exclude_unset=True))
        else:
            self._model = PaperExport.model_validate(data)
            self._data = data

    @property
    def model(self) -> PaperExport:
        """Validated v11.0 export model for statically typed consumers."""
        return self._model

    @property
    def data(self) -> dict[str, Any]:
        """The raw v11.0 export dict."""
        return self._data

    @property
    def ok(self) -> bool:
        """True — a successful extraction (its batch twin is ``ChewFailure``)."""
        return True

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        data = self.__dict__["_data"]
        key = _ALIASES.get(name, name)
        if key in _TABLE_KEYS:
            return Records(data.get(key) or [])
        if key in data:
            return data[key]
        # ``source`` is searched alongside ``metadata`` so ``result.file_hash``
        # keeps resolving after v11 split file identity out of the old ``info``.
        for container in ("metadata", "source"):
            block = data.get(container) or {}
            if key in block:
                return block[key]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        metadata = self._data.get("metadata") or {}
        source = self._data.get("source") or {}
        return sorted(
            set(super().__dir__())
            | set(_TABLE_KEYS)
            | set(_ALIASES)
            | set(self._data)
            | set(metadata)
            | set(source)
        )

    def __repr__(self) -> str:
        metadata = self._data.get("metadata") or {}
        title = metadata.get("title") or "untitled"
        if len(title) > 60:
            title = title[:57] + "..."
        n_refs = len(self._data.get("bib") or [])
        n_sections = len(self._data.get("section") or [])
        return (
            f"<bibr.Result {self._data.get('paper_id')!r}: {title!r} — "
            f"{n_refs} refs, {n_sections} sections>"
        )

    def save(self, path: str | Path, *, compact: bool = False) -> Path:
        """Write the raw export dict as JSON; returns the path written."""
        path = Path(path)
        kwargs: dict[str, Any] = {"separators": (",", ":")} if compact else {"indent": 2}
        path.write_text(json.dumps(self._data, ensure_ascii=False, **kwargs), encoding="utf-8")
        return path

    def consolidate(self, mode: Literal["fill", "replace"] = "fill") -> Result:
        """Return a new Result with ``bib_match`` data merged into ``bib``.

        ``mode="fill"`` only fills empty fields; ``mode="replace"`` also
        overwrites disagreeing ones, but only from a match carrying the
        reference's printed DOI. The original Result keeps the PDF-verbatim
        data untouched.
        """
        import copy

        from bibr.enrich.consolidate import consolidate_bibs

        data = copy.deepcopy(self._data)
        consolidate_bibs(data, mode=mode)
        return Result(data)


def _pipeline_kwargs(options: Mapping[str, Any]) -> dict[str, Any]:
    """Translate friendly chew() options into LocalPipeline kwargs."""
    kwargs: dict[str, Any] = {}
    for name, value in options.items():
        if name == "pages":
            from bibr.utils.pages import parse_pages

            kwargs["start_page"], kwargs["end_page"] = parse_pages(str(value))
            continue
        if name in ("ref_seg", "ref_seg_strategy"):
            # Mirrors CLI --ref-seg; rides RunConfig like refs= (never mutates
            # the process-global Settings). None defers to REF_SEG_STRATEGY.
            if value is None:
                continue
            if value not in ("geom", "region", "llm", "crf"):
                raise ValueError(
                    f"ref_seg must be 'geom', 'region', 'llm', or 'crf', got {value!r}"
                )
            kwargs["ref_seg_strategy"] = value
            continue
        if name == "consolidate":
            # True → "fill"; False → "off" (forces off even when the
            # CROSSREF_CONSOLIDATE env setting enables it); None → defer.
            if value is True:
                value = "fill"
            elif value is False:
                value = "off"
            elif value is None:
                continue
            if value not in ("off", "fill", "replace"):
                raise ValueError(f"consolidate must be a bool, 'fill' or 'replace', got {value!r}")
            kwargs["consolidate"] = value
            continue
        arg = _OPTION_TO_PIPELINE_ARG.get(name)
        if arg is None:
            valid = ", ".join(
                sorted(
                    {*_OPTION_TO_PIPELINE_ARG, "pages", "refs", "ref_seg", "batch_size"}
                    | {"consolidate"}
                )
            )
            raise TypeError(f"unknown chew() option {name!r}; valid options: {valid}")
        kwargs[arg] = value
    return kwargs


def _collect_batch(path: Any) -> list[Path] | None:
    """Return the file list for batch inputs, or None for single-file input.

    Lists/tuples pass through in order. A directory yields its sorted
    supported files (non-recursive), matching the CLI; an unsupported-only
    directory is a loud error rather than a silent [].
    """
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    p = Path(path)
    if not p.is_dir():
        return None
    from bibr.input.supported_files import SUPPORTED_EXTENSIONS

    files = [
        child
        for child in sorted(p.iterdir())
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"no supported files (.pdf/.docx) in directory: {p}")
    return files


async def _process_batch(
    pipeline: Any,
    files: list[Path],
    batch_size: int | None,
    progress: ProgressTracker | None = None,
) -> list[Result | ChewFailure]:
    """Run *files* through one pipeline in chunks; order-aligned results."""
    from bibr.local.pipeline import _auto_batch_size
    from bibr.pipeline.state import FileState

    if not batch_size or batch_size < 1:
        batch_size = _auto_batch_size(getattr(pipeline, "memory_mode", "balanced"))
    states = [FileState(path=f) for f in files]
    for i in range(0, len(states), batch_size):
        await pipeline.process_chunk(states[i : i + batch_size], progress=progress)
    results: list[Result | ChewFailure] = []
    for fs in states:
        if fs.error:
            results.append(
                ChewFailure(
                    path=fs.path,
                    error=fs.error,
                    error_code=fs.error_code,
                    failed_stage=fs.failed_stage,
                )
            )
        else:
            result_json = fs.result_json
            if result_json is None:
                raise RuntimeError(f"pipeline completed without an export result for {fs.path}")
            results.append(Result(result_json))
    return results


async def _chew_on(
    pipeline: Any,
    path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    paper_id: str | None = None,
    batch_size: int | None = None,
    progress: ProgressTracker | None = None,
) -> Result | list[Result | ChewFailure]:
    """chew() path semantics (single vs batch) on an already-built pipeline."""
    batch = _collect_batch(path)
    if batch is not None and paper_id is not None:
        raise TypeError("paper_id only applies to single-file calls")
    if batch is not None and not batch:
        return []
    if batch is None:
        data = await pipeline.process_file(path, paper_id=paper_id, progress=progress)
        return Result(data)
    return await _process_batch(pipeline, batch, batch_size, progress)


def _preflight_llm(settings: GlobalSettings | None, pipeline_kwargs: Mapping[str, Any]) -> None:
    """Fail before any model loads when the configured LLM cannot run.

    Mirrors ``bibr chew``: a missing cloud credential, or a managed local
    backend with no launcher or unsupported hardware, used to surface only on
    the first LLM call — after the caller had already paid for layout and
    OCR. Raises the provider's ``ValueError`` for credentials and
    :class:`~bibr.exceptions.ConfigurationError` for a local backend.
    """
    if pipeline_kwargs.get("no_llm"):
        return
    from bibr.config import snapshot_settings
    from bibr.local.pipeline import LOCAL_LLM_BACKENDS, resolve_llm_backend

    effective = settings if settings is not None else snapshot_settings()
    backend = resolve_llm_backend(pipeline_kwargs.get("llm_backend") or effective.llm.backend)
    if backend == "cloud":
        from bibr.clients.llm import preflight_credentials

        preflight_credentials(settings)
    elif backend in LOCAL_LLM_BACKENDS:
        from bibr.local.cli.run_config import _preflight_local_backend

        problem = _preflight_local_backend(backend)
        if problem:
            from bibr.exceptions import ConfigurationError

            raise ConfigurationError(problem)


def _refs_kwargs(refs: str | bool | None) -> dict[str, Any]:
    """Translate the ``refs`` option into per-run pipeline kwargs.

    ``refs`` selects the parse strategy ("ner" = local NER parser, "llm" =
    full-precision LLM parsing, "llm-chunked" = chunk-tolerant LLM parse over
    region-aligned chunks, "off" / ``False`` = skip reference extraction
    entirely); segmentation keeps its configured default. Carried on the
    pipeline's ``RunConfig`` — never mutates the process-global ``Settings``,
    so concurrent Chewers with different ``refs`` can't clobber each other.
    """
    if refs is None:
        return {}
    if refs is False:
        refs = "off"
    if refs not in ("ner", "llm", "llm-chunked", "off"):
        raise ValueError(f"refs must be 'ner', 'llm', 'llm-chunked', or 'off', got {refs!r}")
    return {"ref_parse_strategy": refs}


async def achew(
    path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    paper_id: str | None = None,
    refs: str | bool | None = None,
    batch_size: int | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> Result | list[Result | ChewFailure]:
    """Async :func:`chew` — for Jupyter and other running-loop contexts."""
    batch = _collect_batch(path)
    if batch is not None and paper_id is not None:
        raise TypeError("paper_id only applies to single-file calls")
    if batch is not None and not batch:
        return []
    kwargs = {**_pipeline_kwargs(options), **_refs_kwargs(refs)}

    from bibr.local.pipeline import LocalPipeline

    _preflight_llm(settings, kwargs)
    pipeline = LocalPipeline(settings=settings, **kwargs)
    try:
        return await _chew_on(pipeline, path, paper_id=paper_id, batch_size=batch_size)
    finally:
        await pipeline.aclose()


class Chewer:
    """Warm-pipeline session: models load once, then :meth:`chew` many times.

    .. code-block:: python

        with bibr.Chewer(ocr="glm-rapid-mlx") as chewer:
            r1 = chewer.chew("a.pdf")
            r2 = chewer.chew("b.pdf")

    Takes the same options as :func:`chew`, except per-call ``paper_id`` and
    ``batch_size``, which move to :meth:`chew`. The pipeline is built lazily
    on the first call and released on :meth:`close` / context exit. Inside a
    running event loop (Jupyter, async apps) use the async form:
    ``async with bibr.Chewer() as chewer: await chewer.achew(...)``.
    Use one mode per instance — sync calls drive a private event loop that
    async calls don't share.
    """

    def __init__(
        self,
        *,
        refs: str | bool | None = None,
        settings: GlobalSettings | None = None,
        **options: Unpack[ChewOptions],
    ) -> None:
        # Validate eagerly, fail fast.
        from bibr.config import snapshot_settings

        self._settings = snapshot_settings(settings)
        self._kwargs = {
            **_pipeline_kwargs(options),
            **_refs_kwargs(refs),
            "settings": self._settings,
        }
        _preflight_llm(self._settings, self._kwargs)
        self._pipeline: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def _ensure_pipeline(self) -> Any:
        if self._closed:
            raise RuntimeError("this Chewer is closed — create a new one")
        if self._pipeline is None:
            from bibr.local.pipeline import LocalPipeline

            self._pipeline = LocalPipeline(**self._kwargs)
        return self._pipeline

    async def achew(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        paper_id: str | None = None,
        batch_size: int | None = None,
        progress: ProgressTracker | None = None,
    ) -> Result | list[Result | ChewFailure]:
        """Async :meth:`chew` — same warm pipeline, running-loop friendly.

        ``progress`` takes any :class:`bibr.pipeline.progress.ProgressTracker`
        (e.g. ``RichProgress``) to observe stage transitions and per-region
        OCR progress; default is silent.
        """
        pipeline = self._ensure_pipeline()
        return await _chew_on(
            pipeline, path, paper_id=paper_id, batch_size=batch_size, progress=progress
        )

    async def achew_file(
        self,
        path: str | Path,
        *,
        paper_id: str | None = None,
        progress: ProgressTracker | None = None,
    ) -> Result:
        """Process exactly one file and return one typed result."""
        _require_file_path(path, api_name="Chewer.achew_file")
        result = await self.achew(path, paper_id=paper_id, progress=progress)
        if not isinstance(result, Result):
            raise RuntimeError("single-file processing returned a batch result")
        return result

    async def achew_many(
        self,
        paths: Sequence[str | Path],
        *,
        batch_size: int | None = None,
    ) -> list[Result | ChewFailure]:
        """Process an explicit sequence and return an order-aligned batch."""
        result = await self.achew(list(paths), batch_size=batch_size)
        if isinstance(result, Result):
            raise RuntimeError("batch processing returned a single-file result")
        return result

    def chew(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        paper_id: str | None = None,
        batch_size: int | None = None,
        progress: ProgressTracker | None = None,
    ) -> Result | list[Result | ChewFailure]:
        """Process paper(s) on the warm pipeline; semantics match :func:`chew`."""
        if self._closed:
            raise RuntimeError("this Chewer is closed — create a new one")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Chewer.chew() cannot run inside an active event loop "
                "(e.g. Jupyter) — use 'await chewer.achew(...)' instead."
            )
        if self._loop is None:
            # One loop for the session's lifetime, so loop-bound resources
            # (HTTP clients etc.) stay valid across calls.
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(
            self.achew(path, paper_id=paper_id, batch_size=batch_size, progress=progress)
        )

    def chew_file(
        self,
        path: str | Path,
        *,
        paper_id: str | None = None,
    ) -> Result:
        """Synchronous :meth:`achew_file`."""
        _require_file_path(path, api_name="Chewer.chew_file")
        result = self.chew(path, paper_id=paper_id)
        if not isinstance(result, Result):
            raise RuntimeError("single-file processing returned a batch result")
        return result

    def chew_many(
        self,
        paths: Sequence[str | Path],
        *,
        batch_size: int | None = None,
    ) -> list[Result | ChewFailure]:
        """Synchronous :meth:`achew_many`."""
        result = self.chew(list(paths), batch_size=batch_size)
        if isinstance(result, Result):
            raise RuntimeError("batch processing returned a single-file result")
        return result

    async def aclose(self) -> None:
        """Release the pipeline (models, clients). Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._pipeline is not None:
            pipeline, self._pipeline = self._pipeline, None
            await pipeline.aclose()

    def close(self) -> None:
        """Sync :meth:`aclose`; also closes the private event loop. Idempotent."""
        if self._closed:
            return
        if self._pipeline is not None:
            loop = self._loop or asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.aclose())
            finally:
                if loop is not self._loop:
                    loop.close()
        self._closed = True
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def __enter__(self) -> Chewer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Chewer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("warm" if self._pipeline else "cold")
        return f"<bibr.Chewer {state}>"


def chew(
    path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    paper_id: str | None = None,
    refs: str | bool | None = None,
    batch_size: int | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> Result | list[Result | ChewFailure]:
    """Process paper(s); a single path returns a :class:`Result`.

    Directories and lists of paths run as a batch on one pipeline (models
    load once) and return an order-aligned ``list[Result | ChewFailure]``.
    Options mirror the ``bibr chew`` CLI flags / ``LocalPipeline`` arguments:
    ``ocr`` (backend), ``llm`` (backend), ``memory`` (mode), ``refs``
    (``"llm"``/``"ner"``/``"llm-chunked"``, or ``"off"``/``False`` to skip
    reference extraction entirely), ``ref_seg`` (segmentation strategy:
    ``"geom"``/``"region"``/``"llm"``/``"crf"``), ``no_llm``, ``device``,
    ``crossref`` (tri-state: ``True`` runs Crossref/resolver reference
    enrichment for this call, ``False`` skips it, ``None``/omitted follows
    the ``CROSSREF_ENRICH`` setting, which is off by default),
    ``equations``, ``pages`` (1-based, e.g. ``"1-5"``), ``figure_images``,
    ``include_regions``, ``ocr_url``, ``ocr_model``, ``paper_id``
    (single-file only), ``batch_size`` (files per chunk, batch only),
    ``consolidate`` (``True``/``"fill"``/``"replace"`` — merge Crossref
    enrichment into bib rows before export; ``False`` forces it off even
    when the ``CROSSREF_CONSOLIDATE`` setting enables it).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "bibr.chew() cannot run inside an active event loop "
            "(e.g. Jupyter) — use 'await bibr.achew(...)' instead."
        )
    return asyncio.run(
        achew(
            path,
            paper_id=paper_id,
            refs=refs,
            batch_size=batch_size,
            settings=settings,
            **options,
        )
    )


def _require_file_path(path: str | Path, *, api_name: str) -> None:
    candidate = Path(path)
    if candidate.is_dir():
        raise IsADirectoryError(f"{api_name} requires a file, got directory: {candidate}")


async def achew_file(
    path: str | Path,
    *,
    paper_id: str | None = None,
    refs: str | bool | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> Result:
    """Process exactly one file asynchronously."""
    _require_file_path(path, api_name="achew_file")
    result = await achew(path, paper_id=paper_id, refs=refs, settings=settings, **options)
    if not isinstance(result, Result):
        raise RuntimeError("single-file processing returned a batch result")
    return result


def chew_file(
    path: str | Path,
    *,
    paper_id: str | None = None,
    refs: str | bool | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> Result:
    """Process exactly one file synchronously."""
    _require_file_path(path, api_name="chew_file")
    result = chew(path, paper_id=paper_id, refs=refs, settings=settings, **options)
    if not isinstance(result, Result):
        raise RuntimeError("single-file processing returned a batch result")
    return result


async def achew_many(
    paths: Sequence[str | Path],
    *,
    refs: str | bool | None = None,
    batch_size: int | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> list[Result | ChewFailure]:
    """Process an explicit sequence asynchronously."""
    result = await achew(
        list(paths), refs=refs, batch_size=batch_size, settings=settings, **options
    )
    if isinstance(result, Result):
        raise RuntimeError("batch processing returned a single-file result")
    return result


def chew_many(
    paths: Sequence[str | Path],
    *,
    refs: str | bool | None = None,
    batch_size: int | None = None,
    settings: GlobalSettings | None = None,
    **options: Unpack[ChewOptions],
) -> list[Result | ChewFailure]:
    """Process an explicit sequence synchronously."""
    result = chew(list(paths), refs=refs, batch_size=batch_size, settings=settings, **options)
    if isinstance(result, Result):
        raise RuntimeError("batch processing returned a single-file result")
    return result
