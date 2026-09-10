"""Shared cross-package constants.

Anchor lengths live here rather than in ``bibr.extract.anchor_snap`` so that
``bibr.clients`` can interpolate them into prompts without importing
``bibr.extract`` (which itself imports ``bibr.clients``).
"""

# Effective anchor length used for matching. The prompt requests extra characters so truncation
# still yields a complete anchor when a model under-delivers.
ANCHOR_LEN = 30
# Overshoot the prompt requests from the LLM (interpolated into the prompt).
ANCHOR_PROMPT_CHARS = ANCHOR_LEN + 10
