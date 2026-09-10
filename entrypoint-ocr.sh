#!/usr/bin/env bash
# entrypoint-ocr.sh — SGLang OCR server startup
# Supports multi-GPU tensor parallelism via SGLANG_GPUS env var.
set -euo pipefail

SGLANG_GPUS="${SGLANG_GPUS:-1}"

TP_ARGS=()
if [ "$SGLANG_GPUS" -gt 1 ]; then
    TP_ARGS=(--tp "$SGLANG_GPUS")
    echo "[ocr] Starting SGLang with tensor parallelism (tp=${SGLANG_GPUS})..."
else
    echo "[ocr] Starting SGLang on single GPU..."
fi

# NOTE: NEXTN/MTP speculative decoding (--speculative-algorithm NEXTN ...) is
# DISABLED. It worked on the old (~April 2026) sglang but device-side-asserts
# during GLM-OCR warmup on recent sglang — both the floating :dev tag and the
# pinned v0.5.13.post1 (gather-index-out-of-bounds in the EAGLE V2 draft path).
# Plain decoding is stable; re-enable once upstream fixes GLM-OCR + NEXTN.
exec python -m sglang.launch_server \
    --model zai-org/GLM-OCR \
    --port 8080 \
    --host 0.0.0.0 \
    --revision "$GLM_OCR_REVISION" \
    --mem-fraction-static "${OCR_MEM_FRACTION_STATIC:-0.60}" \
    --served-model-name glm-ocr \
    "${TP_ARGS[@]}"
