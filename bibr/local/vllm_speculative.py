"""Choose the managed vLLM LLM server's speculative decoding at each start.

``LLM_VLLM_SPEC_DECODE`` (``llm.vllm_spec_decode``):

- ``auto`` (default): MTP, from the checkpoint's own multi-token-prediction
  layers, when the registry lists an MTP config for the model, the server
  stays up for the whole run, and the budget it starts with meets MTP's floor.
  Otherwise none. On the 252 dev papers (vLLM 0.32, one server for the run)
  MTP scored within 2 papers of no speculation and cut the LLM stage to 0.80x
  (RTX 3090) and 0.74x (RTX 5090). Where the LLM server restarts every chunk
  to take turns with a managed PaddleOCR-VL server (bibr's default on cards
  up to 64 GB), each MTP start took 51 s instead of 37 s, and batches on 12
  and 16 GB cards ran 5% and 9% slower, so it is off there. An auto choice
  never refuses a start: below MTP's floor the server starts without it.
- ``none``: never.
- ``mtp`` or ``dflash``: that config at every start, the swap path included,
  when the registry lists it for the model (DFlash only for the 4B student's
  bf16 build, with z-lab's drafter pinned by revision). The start then needs
  that config's floor, and the check before a chunk's OCR asks for it too
  (``bibr.local.vllm_share.managed_llm_floor_gb``). A model without that
  config starts without speculation, with a warning.

A ``--speculative-config`` in ``LLM_VLLM_EXTRA_ARGS`` is the user's own
choice: bibr then adds none.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bibr.config import GlobalSettings
    from bibr.local.llm_models import SpeculativeConfig

logger = logging.getLogger(__name__)

SETTING = "LLM_VLLM_SPEC_DECODE"

#: Methods ``LLM_VLLM_SPEC_DECODE`` can name explicitly.
EXPLICIT_METHODS = ("mtp", "dflash")

# Tolerance for budgets that meet a floor exactly (share x total in floats).
_FLOOR_EPSILON_GB = 1e-6

# (model, method) pairs already warned about: once per process each, since
# every start and every chunk's check reads the setting.
_unlisted_warned: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class SpecDecodeChoice:
    """The speculative config for one start (None: none) and why."""

    config: SpeculativeConfig | None
    reason: str

    def server_args(self) -> tuple[str, ...]:
        return () if self.config is None else self.config.server_args()

    def describe(self) -> str:
        if self.config is None:
            return f"none ({self.reason})"
        return f"{self.config.method} k={self.config.num_speculative_tokens} ({self.reason})"


def user_sets_speculative(extra_args: str | None) -> bool:
    """Whether ``LLM_VLLM_EXTRA_ARGS`` carries its own ``--speculative-config``."""
    for arg in shlex.split(extra_args or ""):
        flag = arg.split("=", 1)[0].replace("_", "-")
        if flag == "--speculative-config":
            return True
    return False


def _listed_config(model: str, method: str) -> SpeculativeConfig | None:
    from bibr.local.llm_models import vllm_registry_variant

    variant = vllm_registry_variant(model)
    return None if variant is None else variant.speculative_config(method)


def explicit_config(settings: GlobalSettings, model: str) -> SpeculativeConfig | None:
    """The config ``LLM_VLLM_SPEC_DECODE=mtp|dflash`` asks for, if *model* has it.

    None for ``auto`` and ``none``, for a model the registry lists without
    that config, and when ``LLM_VLLM_EXTRA_ARGS`` sets its own.
    """
    method = settings.llm.vllm_spec_decode
    if method not in EXPLICIT_METHODS or user_sets_speculative(settings.llm.vllm_extra_args):
        return None
    return _listed_config(model, method)


def choose_spec_decode(
    settings: GlobalSettings,
    model: str,
    *,
    swaps_with_ocr: bool,
    budget_gb: float | None,
) -> SpecDecodeChoice:
    """The speculative config a managed vLLM start for *model* launches with.

    *swaps_with_ocr*: a managed PaddleOCR-VL server takes turns with the LLM
    server, which then restarts every chunk. *budget_gb*: the start's
    ``--gpu-memory-utilization`` x CUDA's total (None: not known).
    """
    if user_sets_speculative(settings.llm.vllm_extra_args):
        return SpecDecodeChoice(None, "LLM_VLLM_EXTRA_ARGS sets --speculative-config")
    method = settings.llm.vllm_spec_decode
    if method == "none":
        return SpecDecodeChoice(None, f"{SETTING}=none")
    if method in EXPLICIT_METHODS:
        config = _listed_config(model, method)
        if config is not None:
            return SpecDecodeChoice(config, f"{SETTING}={method}")
        if (model, method) not in _unlisted_warned:
            _unlisted_warned.add((model, method))
            logger.warning(
                "%s=%s is ignored for %s: bibr's model registry lists no %s config for that "
                "model, so its vLLM server starts without speculative decoding. Pass your own "
                "--speculative-config in LLM_VLLM_EXTRA_ARGS to try one.",
                SETTING,
                method,
                model,
                method,
            )
        return SpecDecodeChoice(None, f"{SETTING}={method}, but no {method} config is listed")
    mtp = _listed_config(model, "mtp")
    if mtp is None:
        return SpecDecodeChoice(None, "auto: no MTP config is listed for the model")
    if swaps_with_ocr:
        return SpecDecodeChoice(
            None,
            "auto: the server restarts every chunk to take turns with PaddleOCR-VL, where "
            "MTP's slower starts cost more than it saved",
        )
    if budget_gb is None:
        return SpecDecodeChoice(None, "auto: the server's budget is not known")
    if budget_gb + _FLOOR_EPSILON_GB < mtp.min_budget_gb:
        return SpecDecodeChoice(
            None,
            f"auto: the {budget_gb:.2f} GiB budget is below MTP's {mtp.min_budget_gb:.2f} GiB "
            "floor",
        )
    return SpecDecodeChoice(
        mtp,
        f"auto: the server stays up for the run and its {budget_gb:.2f} GiB budget meets "
        f"MTP's {mtp.min_budget_gb:.2f} GiB floor",
    )
