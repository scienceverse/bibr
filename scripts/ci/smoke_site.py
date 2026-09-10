#!/usr/bin/env python3
"""Verify public or Access-protected bibr deployments and their source revision."""

from __future__ import annotations

import argparse
import os
import re
import sys
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
USER_AGENT = "bibr-ci-smoke/1.0"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def anonymous_is_challenged(status: int, location: str | None) -> bool:
    """Return whether an anonymous response proves an Access challenge."""

    if status in {401, 403}:
        return True
    if 300 <= status < 400 and location:
        lowered = location.lower()
        return ".cloudflareaccess.com/" in lowered or "/cdn-cgi/access/" in lowered
    return False


def verify_build_marker(body: str, expected_sha: str) -> None:
    """Require the marker body to equal the expected full Git SHA."""

    if FULL_SHA.fullmatch(expected_sha) is None:
        raise ValueError("expected revision must be a full lowercase Git SHA")
    if body.strip() != expected_sha:
        raise ValueError("build marker does not match the expected revision")


def _fetch(
    url: str, *, headers: dict[str, str] | None = None, follow_redirects: bool
) -> tuple[int, str | None, str]:
    opener = build_opener() if follow_redirects else build_opener(NoRedirect)
    # Identify both probes so Cloudflare does not reject Python's default client.
    request = Request(  # noqa: S310 - URL is validated
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}, method="GET"
    )
    try:
        with opener.open(request, timeout=20) as response:  # noqa: S310 - URL is validated
            body = response.read().decode("utf-8", errors="replace")
            return response.status, response.headers.get("Location"), body
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if error.code == 403 and body.strip() == "error code: 1010":
            raise ValueError(
                "Cloudflare rejected the client's browser signature (error 1010) "
                "before Access authentication; check the smoke client's User-Agent"
            ) from error
        return error.code, error.headers.get("Location"), body


def smoke(
    url: str,
    expected_sha: str,
    *,
    access: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("site URL must be an absolute HTTPS URL")
    if access not in {"public", "protected"}:
        raise ValueError("access must be public or protected")
    if access == "protected" and (not client_id or not client_secret):
        raise ValueError("CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET are required")
    marker_url = urljoin(url.rstrip("/") + "/", "bibr-build.txt")

    status, location, body = _fetch(marker_url, follow_redirects=False)
    if access == "public":
        if status != 200:
            raise ValueError(f"anonymous request returned HTTP {status}")
        verify_build_marker(body, expected_sha)
        return

    if not anonymous_is_challenged(status, location):
        raise ValueError(f"anonymous request was not challenged (HTTP {status})")

    status, _location, body = _fetch(
        marker_url,
        headers={
            "CF-Access-Client-Id": client_id,
            "CF-Access-Client-Secret": client_secret,
        },
        follow_redirects=True,
    )
    if status != 200:
        raise ValueError(f"authorized request returned HTTP {status}")
    verify_build_marker(body, expected_sha)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access", choices=("public", "protected"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-sha", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    client_id = os.environ.get("CF_ACCESS_CLIENT_ID") if args.access == "protected" else None
    client_secret = (
        os.environ.get("CF_ACCESS_CLIENT_SECRET") if args.access == "protected" else None
    )
    if args.access == "protected" and (not client_id or not client_secret):
        print("CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET are required", file=sys.stderr)
        return 2
    try:
        smoke(
            args.url,
            args.expected_sha,
            access=args.access,
            client_id=client_id,
            client_secret=client_secret,
        )
    except (OSError, ValueError) as error:
        print(f"{args.access}-site smoke failed: {error}", file=sys.stderr)
        return 1
    print(f"{args.access}-site smoke passed for {args.expected_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
