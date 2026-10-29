"""Launcher shim for ``vllm_mlx.server`` with conservative hybrid-cache handling.

Older vllm-mlx versions monkeypatched mlx-lm's cache factory for Mamba-style
models. With current mlx-lm, ArraysCache already supports the batch operations
needed by hybrid ArraysCache + KVCache models (Qwen3-Next / NuExtract3), and
the old monkeypatch can crash those models with
``ArraysCache.__init__() missing 1 required positional argument: 'size'``.

vllm-mlx 0.4.x no-ops this path upstream, but bibr still enters through this
shim so managed OCR and LLM launchers have one stable module target and remain
safe if a stale environment resolves an older vllm-mlx.

``python -m vllm_mlx.server`` has no ``--served-model-name`` (only the
``vllm-mlx serve`` CLI does), so a local model directory would answer to its
path. The shim takes ``--served-model-name <name>`` itself and passes it to
``vllm_mlx.server.load_model``, as ``vllm-mlx serve`` does, so a pinned build
bibr downloaded is served under its Hub id.

vllm-mlx chooses mlx-vlm (multimodal) or mlx-lm (text) with ``is_mllm_model``,
which reads a local directory's ``config.json`` but judges a Hub id by its name.
bibr's Qwen3.5 student builds keep ``image_token_id`` in their config with no
vision weights, so from a local directory they would load through mlx-vlm and
fail, while their Hub id loads as text. vllm-mlx has no switch that forces text
(``--mllm`` only forces multimodal), so without ``--mllm`` the shim makes every
copy of ``is_mllm_model`` answer for the ``--model`` path:

- with ``--served-model-name <name>``, what it answers for *name*;
- with the shim's own ``--no-mllm`` (a local model directory bibr serves as a
  text LLM, as Rapid-MLX does with its ``--no-mllm``), text.

Usage (drop-in for ``python -m vllm_mlx.server``):
    python -m bibr.local._vllm_mlx_server --model <model> --port <port> ...
"""

import functools
import importlib
import importlib.util
import os
import runpy
import sys
import types

#: The vllm-mlx 0.4.0 / 0.4.1 modules that hold ``is_mllm_model``: the engines,
#: the server and the model registry each import it by name, so each has its own
#: reference. They are imported before every copy in every loaded ``vllm_mlx``
#: module is replaced; a module imported later takes the replaced one from
#: ``vllm_mlx.api.utils``.
_MLLM_DETECTOR_MODULES = (
    "vllm_mlx.api.utils",
    "vllm_mlx.engine.simple",
    "vllm_mlx.engine.batched",
    "vllm_mlx.model_registry",
    "vllm_mlx.server",
)
_MLLM_DETECTOR = "is_mllm_model"


def _disable_legacy_mamba_patch() -> None:
    try:
        mamba_cache_spec = importlib.util.find_spec("vllm_mlx.utils.mamba_cache")
    except (ImportError, ModuleNotFoundError, ValueError):
        return
    if mamba_cache_spec is None:
        return

    from vllm_mlx.utils import mamba_cache

    def _ensure_mamba_support_noop():
        # mlx-lm >= 0.30.6: ArraysCache batches natively; the old patch breaks
        # hybrid ArraysCache + KVCache models. Mark patched without patching.
        mamba_cache._patched = True

    mamba_cache.ensure_mamba_support = _ensure_mamba_support_noop


def _pop_served_model_name(argv: list[str]) -> str | None:
    """Remove ``--served-model-name <name>`` (or ``=<name>``) from *argv*; return the name."""
    flag = "--served-model-name"
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            name = argv[index + 1]
            del argv[index : index + 2]
            return name
        if arg.startswith(flag + "="):
            del argv[index]
            return arg.split("=", 1)[1]
    return None


def _pop_flag(argv: list[str], flag: str) -> bool:
    """Remove every *flag* from *argv*; whether there was one."""
    found = flag in argv
    argv[:] = [arg for arg in argv if arg != flag]
    return found


def _option_value(argv: list[str], option: str) -> str | None:
    """The value given for *option* (``--model <value>`` or ``--model=<value>``)."""
    for index, arg in enumerate(argv[1:], start=1):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def _is_launch_model(name: str, model: str) -> bool:
    if name == model:
        return True
    try:
        return os.path.isdir(name) and os.path.samefile(name, model)
    except (OSError, TypeError, ValueError):
        return False


def _judged(original, model: str, served_model_name: str | None):
    """*original* ``is_mllm_model``, answering for *model* as for *served_model_name* (None: text)."""

    @functools.wraps(original)
    def is_mllm_model(model_name, *args, **kwargs):
        if _is_launch_model(model_name, model):
            if served_model_name is None:
                return False
            return original(served_model_name, *args, **kwargs)
        return original(model_name, *args, **kwargs)

    # Marks a replacement, so a second pass leaves it alone.
    is_mllm_model.__dict__["_bibr_launch_model"] = model
    return is_mllm_model


def _judge_launch_model(model: str, served_model_name: str | None) -> list[str]:
    """Make every vllm-mlx copy of ``is_mllm_model`` judge *model* by *served_model_name*.

    *served_model_name* None judges *model* as text. Any other model (a draft
    or embedding model) keeps vllm-mlx's own answer. Every function named
    ``is_mllm_model`` in a loaded ``vllm_mlx`` module is replaced, whatever
    attribute holds it (``is_vlm_model`` is an alias), including a module's own
    definition. Returns the replaced attributes as ``module.attribute``.
    """
    for name in _MLLM_DETECTOR_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Absent from this vllm-mlx; a missing dependency still fails.
            if exc.name is None or not (name == exc.name or name.startswith(exc.name + ".")):
                raise
    wrappers: dict[int, object] = {}
    replaced: list[str] = []
    for module_name, module in sorted(sys.modules.items(), key=lambda item: item[0]):
        if not isinstance(module, types.ModuleType):
            continue
        if module_name != "vllm_mlx" and not module_name.startswith("vllm_mlx."):
            continue
        for attr, value in list(vars(module).items()):
            if (
                not isinstance(value, types.FunctionType)
                or value.__name__ != _MLLM_DETECTOR
                or hasattr(value, "_bibr_launch_model")
            ):
                continue
            if id(value) not in wrappers:
                wrappers[id(value)] = _judged(value, model, served_model_name)
            setattr(module, attr, wrappers[id(value)])
            replaced.append(f"{module_name}.{attr}")
    judged_as = "text" if served_model_name is None else f"as {served_model_name}"
    if replaced:
        print(
            f"bibr: vllm-mlx judges {model} {judged_as} in {', '.join(replaced)}",
            file=sys.stderr,
        )
    else:
        print(
            f"bibr: found no vllm-mlx {_MLLM_DETECTOR} to judge {model} {judged_as}",
            file=sys.stderr,
        )
    return replaced


def _run_server(served_model_name: str | None) -> None:
    """Run vllm-mlx's server ``main()``, its model served as *served_model_name* when given."""
    from vllm_mlx import server

    if served_model_name is not None:
        load_model = server.load_model

        def _load_model_served_as(*args, **kwargs):
            kwargs.setdefault("served_model_name", served_model_name)
            return load_model(*args, **kwargs)

        # main() looks load_model up in its module's globals when it runs.
        server.load_model = _load_model_served_as
    server.main()


def _main(argv: list[str]) -> None:
    _disable_legacy_mamba_patch()
    served_model_name = _pop_served_model_name(argv)
    text_only = _pop_flag(argv, "--no-mllm")
    if served_model_name is None and not text_only:
        runpy.run_module("vllm_mlx.server", run_name="__main__")
        return
    model = _option_value(argv, "--model")
    # --mllm forces mlx-vlm whatever is_mllm_model answers, and wins over --no-mllm.
    if model is not None and "--mllm" not in argv:
        _judge_launch_model(model, None if text_only else served_model_name)
    _run_server(served_model_name)


if __name__ == "__main__":
    _main(sys.argv)
