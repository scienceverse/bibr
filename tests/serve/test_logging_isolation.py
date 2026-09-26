"""CLI-invoked logging must not leak between tests (x-tests-9).

``bibr chew``/``doctor`` run ``logging.basicConfig`` and pin
``bibr.local``/``bibr.pipeline``/``bibr.structure``/``bibr.extract`` to
INFO. Tests calling ``main()`` left those levels (and root handlers) set, so
``test_serve_log_level_applies_to_bibr_loggers`` failed whenever a CLI test
ran first. The autouse ``_restore_logging_state`` fixture snapshots the root
handlers/level and every logger's level per test.

Runs in definition order: the first test reproduces the CLI logging setup,
the second proves it is gone. On the pre-fix tree the second test fails
(``bibr.pipeline`` still at INFO, 'not shown' leaks through); with the
fixture both pass in any order. A second pair repeats the exercise with a
bare root-handler add plus level moves (no ``monkeypatch`` to hide behind),
covering handler removal and the restore of a pre-existing logger's level.
"""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace

import pytest

# A dedicated pre-existing logger: created at import, so the fixture must
# restore its level (not reset it as a mid-test creation).
_PROBE_LOGGER = logging.getLogger("bibr.iso_probe")
_PROBE_LOGGER.setLevel(logging.WARNING)

_ADDED_HANDLER: logging.Handler | None = None
_ROOT_LEVEL: int | None = None
_PROBE_LEVEL: int | None = None


def test_cli_logging_setup_pins_bibr_loggers_to_info():
    """Reproduce what ``main()`` does to logging (without running a CLI)."""
    logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler()])
    for name in ("bibr.local", "bibr.pipeline", "bibr.structure", "bibr.extract"):
        logging.getLogger(name).setLevel(logging.INFO)
    assert logging.getLogger("bibr.pipeline").level == logging.INFO


def test_serve_warning_level_hides_bibr_info_after_cli_test():
    from bibr.serve.logsetup import configure_serve_logging

    stream = io.StringIO()
    settings = SimpleNamespace(
        SERVE_LOG_LEVEL="warning",
        metering=SimpleNamespace(log_path=None, log_max_bytes=10_000, log_backup_count=1),
    )
    configure_serve_logging(settings, stream=stream)
    logging.getLogger("bibr.pipeline").info("not shown")
    logging.getLogger("bibr.pipeline").warning("shown")
    assert logging.getLogger("bibr").level == logging.WARNING
    assert "not shown" not in stream.getvalue()
    assert "WARNING bibr.pipeline: shown" in stream.getvalue()


def test_no_extra_root_handlers_leak():
    """Pollute logging without ``monkeypatch``: only the fixture cleans up.

    Adds a root handler the way ``basicConfig`` does and moves a
    pre-existing logger plus the root level. Runs in definition order
    before the verify test below.
    """
    global _ADDED_HANDLER, _ROOT_LEVEL, _PROBE_LEVEL
    _ROOT_LEVEL = logging.getLogger().level
    _PROBE_LEVEL = _PROBE_LOGGER.level
    _PROBE_LOGGER.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    _ADDED_HANDLER = logging.StreamHandler()
    logging.getLogger().addHandler(_ADDED_HANDLER)
    assert _ADDED_HANDLER in logging.getLogger().handlers


def test_root_handler_and_levels_are_restored():
    """The previous test's handler is removed and levels are put back."""
    if _ADDED_HANDLER is None:
        pytest.skip("polluting test did not run first")
    assert _ADDED_HANDLER not in logging.getLogger().handlers
    assert logging.getLogger().level == _ROOT_LEVEL
    assert _PROBE_LOGGER.level == _PROBE_LEVEL
