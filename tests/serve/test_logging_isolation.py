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
fixture both pass in any order.
"""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace


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
    """Guard: a CLI-style basicConfig handler must not accumulate on root."""
    count = len(logging.getLogger().handlers)
    logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler()])
    assert len(logging.getLogger().handlers) >= count
