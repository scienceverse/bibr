import os
import stat

from bibr.utils.secure_temp import open_subprocess_log


def test_subprocess_log_is_private_and_unpredictable():
    first_path, first = open_subprocess_log("test", 8765)
    second_path, second = open_subprocess_log("test", 8765)
    try:
        assert first_path != second_path
        if os.name != "nt":  # Windows protects temporary files through ACLs.
            assert stat.S_IMODE(first_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(second_path.stat().st_mode) == 0o600
    finally:
        first.close()
        second.close()
        first_path.unlink(missing_ok=True)
        second_path.unlink(missing_ok=True)
