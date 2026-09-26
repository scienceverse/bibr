"""Tests for bibr.utils.redact — secret scrubbing for logs and CLI output."""

import logging

import pytest

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


# --- exception arguments and password-only user-info (x-security-5) ----------


def _scrubbed_output(message: str, *args) -> str:
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(SecretScrubbingFilter())
    log = logging.getLogger("bibr.test.scrub_args")
    log.propagate = False
    log.handlers = [handler]
    log.warning(message, *args)
    return stream.getvalue().rstrip("\n")


def test_filter_scrubs_an_exception_passed_as_argument():
    """``logger.warning("...: %s", e)`` renders ``str(e)`` after the filter ran."""
    key = "AIzaSy" + "A" * 30
    err = RuntimeError(f"call failed at https://x/generate?key={key}")
    assert _scrubbed_output("upstream failed: %s", err) == (
        "upstream failed: call failed at https://x/generate?key=***"
    )


def test_filter_keeps_the_argument_tuple_a_formatter_unpacks():
    """uvicorn's AccessFormatter unpacks five ``record.args``; emptying them to
    freeze the masked text lost the access line with a logging error."""
    uvicorn_logging = pytest.importorskip("uvicorn.logging")
    import io

    key = "AIzaSy" + "B" * 30
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        uvicorn_logging.AccessFormatter(
            '%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False
        )
    )
    handler.addFilter(SecretScrubbingFilter())
    log = logging.getLogger("bibr.test.access")
    log.propagate = False
    log.handlers = [handler]
    log.setLevel(logging.INFO)
    # The call shape of uvicorn's HTTP protocols.
    log.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", f"/jobs?key={key}", "1.1", 200)
    assert stream.getvalue() == '127.0.0.1:5000 - "GET /jobs?key=*** HTTP/1.1" 200 OK\n'


def test_filter_freezes_the_message_when_arguments_cannot_be_masked_one_by_one():
    key = "AIzaSy" + "C" * 30
    # ``%r`` of the masked str would not render as the masked repr: freeze it.
    rec = logging.LogRecord(
        "bibr.test", logging.INFO, __file__, 1, "failed: %r", (ValueError(key),), None
    )
    SecretScrubbingFilter().filter(rec)
    assert (rec.msg, rec.args) == ("failed: ValueError('***')", ())
    # A mapping argument is frozen too.
    rec = logging.LogRecord(
        "bibr.test", logging.INFO, __file__, 1, "%(err)s", ({"err": RuntimeError(key)},), None
    )
    SecretScrubbingFilter().filter(rec)
    assert (rec.getMessage(), rec.args) == ("***", ())


def test_filter_leaves_a_clean_record_and_its_arguments_alone():
    rec = logging.LogRecord("bibr.test", logging.INFO, __file__, 1, "%s %d", ("GET /", 200), None)
    SecretScrubbingFilter().filter(rec)
    assert (rec.msg, rec.args) == ("%s %d", ("GET /", 200))


def test_filter_survives_a_malformed_record():
    rec = logging.LogRecord("bibr.test", logging.INFO, __file__, 1, "%d", ("token=abc",), None)
    assert SecretScrubbingFilter().filter(rec) is True
    assert rec.args == ("token=abc",)


def test_scrub_secrets_masks_password_only_user_info():
    assert scrub_secrets("redis://:hunter2hunter2@redis:6379/0") == "redis://***@redis:6379/0"


# --- text shown outside the process (x-security-1, config-12) -----------------


def test_describe_error_names_only_the_status_of_an_http_error():
    import httpx

    from bibr.utils.redact import describe_error

    url = "https://ocruser:ocr-s3cret@sglang.internal.corp:30000/v1/chat/completions"
    response = httpx.Response(400, request=httpx.Request("POST", url))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        response.raise_for_status()
    assert describe_error(exc.value) == "HTTPStatusError: HTTP 400 Bad Request"


def test_describe_error_replaces_urls_in_other_messages():
    from bibr.utils.redact import describe_error

    exc = ConnectionError("connect to https://svc:pw@ocr.internal:8002/v1 refused")
    assert describe_error(exc) == "ConnectionError: connect to <url> refused"
    assert describe_error(TimeoutError()) == "TimeoutError"


def test_redact_urls_masks_secrets_outside_urls_too():
    from bibr.utils.redact import redact_urls

    assert redact_urls("upstream said: Authorization: Bearer abcdefgh12345678 rejected") == (
        "upstream said: Authorization: Bearer *** rejected"
    )
    assert redact_urls("see git+ssh://git@host/repo or x_https://h/p or ...https://h/q") == (
        "see <url> or x_<url> or <url>"
    )


def test_redact_urls_is_linear_in_a_long_scheme_like_run():
    """A long run of scheme characters with no ``://`` used to be rescanned from
    every word boundary: seconds for 100k characters of exception text."""
    import time

    from bibr.utils.redact import describe_error

    text = "bad value " + "a." * 50_000
    started = time.perf_counter()
    assert describe_error(ValueError(text)) == f"ValueError: {text}"
    assert time.perf_counter() - started < 1.0


def test_redact_url_secrets_masks_the_password_and_keeps_the_user():
    from bibr.utils.redact import redact_url_secrets

    assert redact_url_secrets("redis://default:hunter2@redis:6379/0") == (
        "redis://default:***@redis:6379/0"
    )
    assert redact_url_secrets("https://llm.example/v1?api_key=abc") == (
        "https://llm.example/v1?api_key=***"
    )
    assert redact_url_secrets("redis://redis:6379/0") == "redis://redis:6379/0"
