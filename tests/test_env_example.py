"""Every key in .env.example must be read by the Settings model (or allowlisted)."""

import re
from pathlib import Path

from bibr.config_introspect import iter_setting_docs

_ENV_EXAMPLE = Path(__file__).parents[1] / ".env.example"
# Matches both active and commented-out example lines: `KEY=` / `# KEY=value`
_KEY_RE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)

# Read by docker-compose / external tools, not by bibr.config — keep justified.
_EXTERNAL_KEYS = {
    "WITH_GPU",  # docker-compose build arg (onnxruntime-gpu image variant)
    "GRAFANA_ADMIN_USER",  # grafana container
    "GRAFANA_ADMIN_PASSWORD",  # grafana container
    "DISCORD_WEBHOOK_URL",  # alerting sidecar
    "HF_TOKEN",  # read by huggingface_hub directly
}


def test_env_example_keys_are_real_settings():
    known = set()
    for d in iter_setting_docs():
        known.add(d.env_name)
        known.update(d.aliases)
    keys = set(_KEY_RE.findall(_ENV_EXAMPLE.read_text()))
    unknown = keys - known - _EXTERNAL_KEYS
    assert not unknown, (
        f"Keys in .env.example not read by bibr.config: {sorted(unknown)}\n"
        "Fix the stale key, or add it to _EXTERNAL_KEYS with a justification "
        "comment if an external tool reads it."
    )
