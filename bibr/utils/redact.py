"""Secret scrubbing for user-facing output and server logs.

Two complementary tools:

* :func:`redact_key` — mask a *known* literal API key (and a long prefix of it)
  in a string. Used where the key is in scope (``bibr doctor``, setup wizard):
  precise, no false positives.
* :func:`scrub_secrets` / :class:`SecretScrubbingFilter` — pattern-based masking
  for cases where the key is *not* in scope, e.g. an SDK traceback logged with
  ``exc_info=True`` that embeds ``?key=AIza…`` in a request URL (audit L12).

:func:`describe_error` goes further for exception text that leaves the process
(export warnings, HTTP error bodies): it drops URLs altogether, since the
internal endpoint is itself what those readers must not learn.
"""

from __future__ import annotations

import logging
import re

# Query-string credential params: ``?key=...``, ``&api_key=...``, ``token=...``,
# and signed-URL signatures (Azure SAS ``sig=``, S3/GCS ``X-Amz-Signature=``).
_QUERY_SECRET_RE = re.compile(
    r"((?:[?&])(?:key|api[_-]?key|access[_-]?token|refresh[_-]?token|token|client[_-]?secret"
    r"|password|passwd|pwd|sig|signature|x-amz-signature|x-amz-security-token"
    r"|x-goog-signature)=)[^&\s\"'<>]+",
    re.IGNORECASE,
)
# A quoted JSON/dict entry: ``"api_key": "…"``, ``'token': '…'``. Same-line
# only, and the value must close with the quote that opened the key.
_MAPPING_SECRET_RE = re.compile(
    r"""((["'])(?:(?:x-(?:goog-)?)?api[_-]?key|access[_-]?token|refresh[_-]?token|token"""
    r"""|client[_-]?secret|secret|password|passwd|pwd|authorization)\2[ \t]*:[ \t]*\2)"""
    r"""(?:(?!\2)[^\\\n]|\\.)*\2""",
    re.IGNORECASE,
)
# ``Authorization: Bearer <token>`` / ``Basic <b64>``.
_BEARER_RE = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/\-]{8,}=*", re.IGNORECASE)
# API-key headers quoted in error text: ``x-api-key: …`` (Anthropic),
# ``x-goog-api-key: …`` (Google), ``api-key: …`` (Azure OpenAI).
_HEADER_SECRET_RE = re.compile(
    r"\b((?:x-(?:goog-)?)?api-key[ \t]*:[ \t]*)[^\s\"'<>,;]+", re.IGNORECASE
)
# URL user-info: ``https://user:pass@host`` — used by OCR/LLM SDKs that embed
# credentials in the base URL — and a token used as the user name
# (``https://<token>@github.com``). The user may be empty: the compose-style
# ``redis://:password@redis:6379/0`` carries only a password. Split as urllib
# does: the authority ends at ``/``, ``?`` or ``#`` (so ``?mailto=you@example.com``
# is left alone) and the user-info at its last ``@``, as clients accept an
# unencoded ``@`` or ``:`` in the password.
_URL_USERINFO_RE = re.compile(r"(://)[^/?#\s\"'<>]*@")
# Just the password of URL user-info (after the first ``:``, to the last ``@``),
# keeping the user name visible.
_URL_PASSWORD_RE = re.compile(r"(://[^/?#\s\"'<>:]*:)[^/?#\s\"'<>]*@")
# Any ``scheme://…`` URL, for text that must not name endpoints at all. A match
# takes the whole run of scheme characters before ``://`` (so ``-https://`` loses
# the dash too) and starts only where such a run starts: a long run without
# ``://`` is scanned once, not once per word boundary inside it.
_URL_RE = re.compile(r"(?<![a-z0-9+.\-])[a-z0-9+.\-]+://[^\s'\"<>]+", re.IGNORECASE)
# Vendor key shapes: Google (AIza…), OpenAI/Anthropic (sk-…), Groq (gsk_…),
# Hugging Face (hf_…), GitHub (ghp_…, gho_…, ghs_…, github_pat_…) and AWS
# access key ids (AKIA…, ASIA…).
_VENDOR_KEY_RE = re.compile(
    r"\b(?:AIza[0-9A-Za-z_\-]{20,}|sk-(?:ant-)?[0-9A-Za-z_\-]{16,}|gsk_[0-9A-Za-z_\-]{16,}"
    r"|hf_[0-9A-Za-z]{20,}|gh[pousr]_[0-9A-Za-z]{20,}|github_pat_[0-9A-Za-z_]{20,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}\b)"
)

_REDACTED = "***"


def redact_key(text: str, api_key: str) -> str:
    """Replace occurrences of *api_key* (and any >=12-char prefix) in *text*.

    SDK errors sometimes echo the request URL or auth header back unredacted.
    A no-op when the key is empty/short (<8 chars) or the text is empty, so it is
    always safe to wrap around ``str(exc)``.
    """
    if not text or not api_key or len(api_key) < 8:
        return text
    out = text.replace(api_key, _REDACTED)
    # Catch URL-encoded ``?key=<prefix>…`` fragments by scrubbing a long prefix.
    if len(api_key) >= 12:
        out = out.replace(api_key[:12], _REDACTED)
    return out


def scrub_secrets(text: str) -> str:
    """Mask credential-shaped substrings without knowing the specific key.

    Redacts query-string secret params, quoted ``"api_key": "…"`` entries,
    ``Bearer``/``Basic`` auth values, API-key headers, URL user-info and known
    vendor key shapes. Leaves clean text untouched.
    """
    if not text:
        return text
    out = _QUERY_SECRET_RE.sub(rf"\1{_REDACTED}", text)
    out = _MAPPING_SECRET_RE.sub(rf"\g<1>{_REDACTED}\2", out)
    out = _BEARER_RE.sub(rf"\1 {_REDACTED}", out)
    out = _HEADER_SECRET_RE.sub(rf"\1{_REDACTED}", out)
    out = _URL_USERINFO_RE.sub(rf"\1{_REDACTED}@", out)
    out = _VENDOR_KEY_RE.sub(_REDACTED, out)
    return out


def redact_url_secrets(text: str) -> str:
    """Mask a URL's user-info password and query-string secrets, keeping the rest.

    For showing configured URLs (``REDIS_URL``, ``LLM_BASE_URL``, …) to their
    operator: host, port, path and user name stay readable, unless the user
    name is a known token shape (``https://ghp_…@github.com``).
    """
    if not text:
        return text
    out = _URL_PASSWORD_RE.sub(rf"\1{_REDACTED}@", text)
    out = _QUERY_SECRET_RE.sub(rf"\1{_REDACTED}", out)
    return _VENDOR_KEY_RE.sub(_REDACTED, out)


def redact_urls(text: str) -> str:
    """Replace every ``scheme://…`` URL in *text* with ``<url>``; mask secrets."""
    if not text:
        return text
    return _URL_RE.sub("<url>", scrub_secrets(text))


def describe_error(exc: BaseException) -> str:
    """``TypeName: detail`` for exception text shown outside the process.

    Export warnings and HTTP error bodies reach callers who must not learn the
    internal endpoints behind them, and httpx/SDK status errors quote the full
    request URL, user-info and query included. A status error is reduced to
    ``HTTP <code> <reason>``; any other message keeps its text with URLs
    replaced by ``<url>`` and credential shapes masked. Log the raw exception
    separately for the operator.
    """
    name = type(exc).__name__
    try:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        reason = getattr(response, "reason_phrase", None)
    except Exception:  # noqa: BLE001 - some SDK properties raise when unset
        status = reason = None
    if isinstance(status, int) and not isinstance(status, bool):
        detail = (
            f"HTTP {status} {reason}" if isinstance(reason, str) and reason else f"HTTP {status}"
        )
    else:
        detail = redact_urls(str(exc))
    return f"{name}: {detail}" if detail else name


def _scrubbed_arg(arg: object) -> object:
    """*arg* with secrets masked: a str is scrubbed, any other object is
    replaced by its scrubbed text only when its text carries a secret."""
    if isinstance(arg, str):
        return scrub_secrets(arg)
    try:
        text = str(arg)
    except Exception:  # noqa: BLE001 - the whole-message check below decides
        return arg
    scrubbed = scrub_secrets(text)
    return scrubbed if scrubbed != text else arg


class SecretScrubbingFilter(logging.Filter):
    """Logging filter that scrubs credential-shaped substrings from records.

    Scrubs the interpolated message and, when a record carries ``exc_info``,
    pre-renders and scrubs the traceback into ``exc_text`` so the formatter emits
    the masked version instead of re-rendering the raw exception. Attach to the
    *handlers* that write logs so propagated ``bibr.*`` records are covered too.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            rendered: str | None = record.getMessage()
        except Exception:  # noqa: BLE001 - a malformed record is the handler's to report
            rendered = None
        if rendered is not None:
            scrubbed = scrub_secrets(rendered)
            if scrubbed != rendered:
                self._mask(record, scrubbed)
        else:
            if isinstance(record.msg, str):
                record.msg = scrub_secrets(record.msg)
            if isinstance(record.args, tuple):
                record.args = tuple(
                    scrub_secrets(a) if isinstance(a, str) else a for a in record.args
                )
        if record.exc_text:
            record.exc_text = scrub_secrets(record.exc_text)
        elif record.exc_info:
            record.exc_text = scrub_secrets(logging.Formatter().formatException(record.exc_info))
        return True

    @staticmethod
    def _mask(record: logging.LogRecord, scrubbed: str) -> None:
        """Make *record* render as *scrubbed*, the masked form of its message.

        The formatter renders ``msg % args`` only after this filter has run, so
        an argument that is not a str (an exception, the most common case) must
        be replaced, not skipped. Masking argument by argument keeps the tuple's
        shape, which formatters such as uvicorn's ``AccessFormatter`` unpack;
        when that does not render the same masked text, the message is frozen.
        """
        if isinstance(record.msg, str) and isinstance(record.args, tuple):
            record.msg = scrub_secrets(record.msg)
            record.args = tuple(_scrubbed_arg(a) for a in record.args)
            try:
                masked = record.getMessage() == scrubbed
            except Exception:  # noqa: BLE001 - masking split a format directive
                masked = False
            if masked:
                return
        record.msg = scrubbed
        record.args = ()


def install_secret_scrubbing(*targets: logging.Logger | logging.Handler) -> None:
    """Attach a :class:`SecretScrubbingFilter` to each logger/handler (idempotent)."""
    scrubber = SecretScrubbingFilter()
    for target in targets:
        if any(isinstance(f, SecretScrubbingFilter) for f in target.filters):
            continue
        target.addFilter(scrubber)
