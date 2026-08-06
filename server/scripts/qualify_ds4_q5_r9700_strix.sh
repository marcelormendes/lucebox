#!/usr/bin/env bash
set -euo pipefail

# Reproduce the qualified q=5 profile on the LuceBox R9700 + Strix Halo pair.
# The hipBLASLt table is tied to these GPU architectures and ROCm 7.2.4.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEFAULT_TUNING_FILE="$CHECKOUT/server/config/hipblaslt/r9700-strix-rocm-7.2.4.txt"
EXPECTED_TUNING_SHA256="7b909b84adaf8a24fdfb760a8d66cd5adf95bf6e8c0270e1a10e929f21499e98"
TUNING_FILE="${HIPBLASLT_TUNING_OVERRIDE_FILE:-$DEFAULT_TUNING_FILE}"

if [[ ! -f "$TUNING_FILE" ]]; then
    echo "missing hipBLASLt tuning file: $TUNING_FILE" >&2
    exit 2
fi

if [[ "$TUNING_FILE" == "$DEFAULT_TUNING_FILE" ]]; then
    actual_sha256=$(sha256sum "$TUNING_FILE" | awk '{print $1}')
    if [[ "$actual_sha256" != "$EXPECTED_TUNING_SHA256" ]]; then
        echo "unexpected hipBLASLt tuning table checksum: $actual_sha256" >&2
        exit 2
    fi
fi

exec env -u HIPBLASLT_TUNING_FILE \
    ROCBLAS_USE_HIPBLASLT=1 \
    HIPBLASLT_TUNING_OVERRIDE_FILE="$TUNING_FILE" \
    CRITICAL_PATH_PLACEMENT="${CRITICAL_PATH_PLACEMENT:-1}" \
    MAIN_TO_PEER_RATE="${MAIN_TO_PEER_RATE:-4.4}" \
    EXPERT_BUDGET_MB="${EXPERT_BUDGET_MB:-14350}" \
    DYNAMIC_ROUTE_BALANCE="${DYNAMIC_ROUTE_BALANCE:-1}" \
    DYNAMIC_MAIN_SLOTS="${DYNAMIC_MAIN_SLOTS:-3}" \
    DYNAMIC_MAIN_SLOTS_X4="${DYNAMIC_MAIN_SLOTS_X4:-13}" \
    FUSED_OWNER_RESIDUAL="${FUSED_OWNER_RESIDUAL:-1}" \
    TARGETS="${TARGETS:-2048}" \
    WARMUP="${WARMUP:-2}" \
    RUNS="${RUNS:-7}" \
    RUN_ID="${RUN_ID:-ds4-q5-r9700-strix-hipblaslt}" \
    "$SCRIPT_DIR/qualify_ds4_q5_amd.sh" "$@"
