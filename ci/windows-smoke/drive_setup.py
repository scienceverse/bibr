"""Drive `bibr setup` non-interactively on a CI runner and log every prompt.

SCENARIO=cloud: the runner's real hardware (no GPU) -> weak-hardware fork ->
no private server -> consent to Anthropic -> fake key.
SCENARIO=local-cuda: pretend an 8 GB NVIDIA GPU so the Windows fully-local
plan (llama.cpp) is what the wizard recommends.
"""

from __future__ import annotations

import os
import sys

import rich.prompt

SCENARIO = os.environ.get("SCENARIO", "cloud")
FAKE_KEY = os.environ.get("FAKE_ANTHROPIC_KEY", "sk-ant-api03-invalid-ci-key")

CONFIRM_RULES = [
    ("private OCR/LLM server", False),
    ("Send document-derived content to Anthropic", True),
    ("Install recommended extras", True),
    ("Use this configuration", True),
    ("Switch to the advanced wizard", False),
    ("Test the LLM connection", True),
    ("Retry with a different", False),
    ("Launch the local server once", True),
    ("Run a test extraction", True),
    ("named preset", False),
]


def _confirm(prompt, *args, default=False, **kwargs):
    text = str(prompt)
    answer = default
    for needle, value in CONFIRM_RULES:
        if needle.lower() in text.lower():
            answer = value
            break
    print(f"[drive] CONFIRM {text!r} -> {answer}", flush=True)
    return answer


def _prompt(prompt, *args, default=None, password=False, **kwargs):
    text = str(prompt)
    answer = FAKE_KEY if "API key" in text and password else default
    if answer is None:
        answer = ""
    shown = "<fake key>" if answer == FAKE_KEY else answer
    print(f"[drive] PROMPT {text!r} -> {shown!r}", flush=True)
    return answer


rich.prompt.Confirm.ask = classmethod(lambda cls, *a, **k: _confirm(*a, **k))
rich.prompt.Prompt.ask = classmethod(lambda cls, *a, **k: _prompt(*a, **k))

import bibr.setup_wizard as wizard  # noqa: E402

wizard.Confirm.ask = rich.prompt.Confirm.ask
wizard.Prompt.ask = rich.prompt.Prompt.ask

if SCENARIO == "local-cuda":
    wizard.detect_hardware = lambda: ("cuda", 8.0)

from bibr.local.cli import main as bibr_main  # noqa: E402

sys.argv = ["bibr", "setup"]
try:
    bibr_main()
except SystemExit as exc:
    print(f"[drive] wizard exited with {exc.code!r}", flush=True)
    raise
except BaseException as exc:
    print(f"[drive] wizard raised {type(exc).__name__}: {exc}", flush=True)
    raise
print("[drive] wizard finished", flush=True)
sys.exit(0)
