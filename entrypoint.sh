#!/usr/bin/env bash
# entrypoint.sh — bibr server container startup
# Runs the bibr serve HTTP API in the foreground.
# OCR runs in a separate container (bibr-ocr).
set -euo pipefail

echo "[entrypoint] Starting bibr serve..."
exec bibr serve --host 0.0.0.0 --port 8000
