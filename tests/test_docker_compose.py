"""Deployment defaults in docker-compose.yml."""

from pathlib import Path

import yaml


def _serve_service() -> dict:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    return compose["services"]["bibr-serve"]


def test_api_port_is_published_on_loopback_by_default():
    """Docker publishes 0.0.0.0 ports past host firewalls; opt in explicitly."""
    ports = [str(p) for p in _serve_service()["ports"]]
    assert ports == ["${BIBR_PUBLISH_HOST:-127.0.0.1}:8000:8000"]


def test_redis_password_is_not_interpolated_into_the_url():
    """bibr URL-encodes REDIS_PASSWORD into REDIS_URL itself; a raw password with
    URL metacharacters in the Compose URL silently disabled the cache."""
    entries = dict(item.split("=", 1) for item in _serve_service()["environment"])
    assert entries["REDIS_URL"] == "redis://redis:6379/0"
    assert entries["REDIS_PASSWORD"] == "${REDIS_PASSWORD:-}"
