"""Tests for bibr.utils.redact — secret scrubbing for logs and CLI output."""

import logging

from bibr.utils.redact import (
    SecretScrubbingFilter,
    redact_key,
    scrub_secrets,
)

# --- redact_key: mask a known literal key ------------------------------------


def test_redact_key_masks_literal_key():
    key = "AIzaSyA1234567890abcdefGHIJKLMNOPqrstuv"
    text = f"403 error for https://api.example/v1?key={key}&alt=json"
    out = redact_key(text, key)
    assert key not in out
    assert "***" in out


def test_redact_key_masks_prefix_fragment():
    """SDKs sometimes echo only a URL-encoded prefix of the key; the >=12-char
    prefix is scrubbed too."""
    key = "AIzaSyA1234567890abcdefGHIJKLMNOP"
    text = f"partial key seen: {key[:16]}…"
    out = redact_key(text, key)
    assert key[:16] not in out


def test_redact_key_noop_for_short_key():
    assert redact_key("nothing secret here", "abc") == "nothing secret here"


def test_redact_key_noop_for_empty_inputs():
    assert redact_key("", "AIzaSyAverylongkey12345") == ""
    assert redact_key("some text", "") == "some text"


# --- scrub_secrets: pattern-based, without knowing the key -------------------


def test_scrub_secrets_redacts_query_key_param():
    text = "GET https://generativelanguage.googleapis.com/v1?key=AIzaSyABCDEF1234567890xyz HTTP/1.1"
    out = scrub_secrets(text)
    assert "AIzaSyABCDEF1234567890xyz" not in out
    assert "key=" in out  # structure preserved, value masked


def test_scrub_secrets_redacts_bearer_token():
    out = scrub_secrets("Authorization: Bearer sk-ant-api03-abcDEF1234567890ghiJKLmno")
    assert "sk-ant-api03-abcDEF1234567890ghiJKLmno" not in out
    assert "Bearer" in out


def test_scrub_secrets_redacts_bare_google_key():
    out = scrub_secrets("boom: AIzaSyA1234567890abcdefGHIJKLMNOPqrstuvwx in trace")
    assert "AIzaSyA1234567890abcdefGHIJKLMNOPqrstuvwx" not in out


def test_scrub_secrets_redacts_url_userinfo():
    """Credentials embedded in a base URL are masked, scheme and host kept."""
    out = scrub_secrets("OCR base: https://svc:s3cr3t@ocr.internal:8002/v1 unreachable")
    assert "s3cr3t" not in out
    assert "svc:" not in out
    assert out.startswith("OCR base: https://***@ocr.internal:8002/v1")


def test_scrub_secrets_leaves_mailto_param_untouched():
    """The Crossref polite-pool ``mailto=`` address is not user-info — the ``@``
    sits after a path separator, so the user-info pattern must not fire."""
    text = "GET https://api.crossref.org/works?mailto=you@example.com"
    assert scrub_secrets(text) == text


def test_scrub_secrets_leaves_host_port_untouched():
    text = "connecting to http://gpu-box:2010/search/batch"
    assert scrub_secrets(text) == text


def test_scrub_secrets_leaves_clean_text_untouched():
    text = "LLM title extraction failed (hash=abc123): connection reset by peer"
    assert scrub_secrets(text) == text


# --- SecretScrubbingFilter: scrubs message + traceback on log records --------


def test_filter_scrubs_message():
    rec = logging.LogRecord(
        "bibr.test",
        logging.ERROR,
        __file__,
        1,
        "call failed key=AIzaSyABCDEF1234567890xyz done",
        None,
        None,
    )
    assert SecretScrubbingFilter().filter(rec) is True
    assert "AIzaSyABCDEF1234567890xyz" not in rec.getMessage()


def test_filter_scrubs_traceback_text():
    key = "AIzaSyA1234567890abcdefGHIJKLMNOPqrstuv"
    try:
        raise RuntimeError(f"blew up hitting ?key={key}")
    except RuntimeError:
        import sys

        exc_info = sys.exc_info()
    rec = logging.LogRecord(
        "bibr.test",
        logging.ERROR,
        __file__,
        1,
        "op failed",
        None,
        exc_info,
    )
    SecretScrubbingFilter().filter(rec)
    formatted = logging.Formatter("%(message)s").format(rec)
    assert key not in formatted


# --- install_secret_scrubbing: idempotent handler wiring (audit L12) ----------


def test_install_secret_scrubbing_attaches_once():
    from bibr.utils.redact import install_secret_scrubbing

    handler = logging.StreamHandler()
    install_secret_scrubbing(handler)
    install_secret_scrubbing(handler)  # idempotent
    scrubbers = [f for f in handler.filters if isinstance(f, SecretScrubbingFilter)]
    assert len(scrubbers) == 1


def test_installed_handler_scrubs_exc_traceback():
    import io

    key = "AIzaSyA1234567890abcdefGHIJKLMNOPqrstuv"
    from bibr.utils.redact import install_secret_scrubbing

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    install_secret_scrubbing(handler)
    log = logging.getLogger("bibr.test.scrub_wiring")
    log.propagate = False
    log.setLevel(logging.ERROR)
    log.addHandler(handler)
    try:
        raise RuntimeError(f"GET ?key={key}")
    except RuntimeError:
        log.error("upstream failed", exc_info=True)
    assert key not in stream.getvalue()
