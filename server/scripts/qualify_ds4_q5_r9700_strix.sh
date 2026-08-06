#!/usr/bin/env bash
set -euo pipefail

# Reproduce the qualified q=5 profile on the LuceBox R9700 + Strix Halo pair.
# The hipBLASLt table is tied to these GPU architectures and ROCm 7.2.4.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEFAULT_TUNING_FILE="$CHECKOUT/server/config/hipblaslt/r9700-strix-rocm-7.2.4.txt"
EXPECTED_TUNING_SHA256="7b909b84adaf8a24fdfb760a8d66cd5adf95bf6e8c0270e1a10e929f21499e98"
TUNING_FILE="${HIPBLASLT_TUNING_OVERRIDE_FILE:-$DEFAULT_TUNING_FILE}"
PROFILE_DIR="$CHECKOUT/server/config/ds4/r9700-strix"
DEFAULT_HOTNESS_CSV="$PROFILE_DIR/prefill-routing.csv"
DEFAULT_DECODE_HOTNESS_CSV="$PROFILE_DIR/decode-routing.csv"
EXPECTED_HOTNESS_SHA256="927d1449906d7881097e818c6e0210a7834b4231594e4daefb18f40e37955dc2"
EXPECTED_DECODE_HOTNESS_SHA256="f5989e0d5a5ef91d5ffd3068f81ab568c5b0bc6b755461acbfc32536dda47e15"
HOTNESS_CSV="${HOTNESS_CSV:-$DEFAULT_HOTNESS_CSV}"
DECODE_HOTNESS_CSV="${DECODE_HOTNESS_CSV:-$DEFAULT_DECODE_HOTNESS_CSV}"

verify_default_file() {
    local path=$1
    local expected_sha256=$2
    local label=$3
    if [[ ! -f "$path" ]]; then
        echo "missing $label: $path" >&2
        exit 2
    fi
    local actual_sha256
    actual_sha256=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "unexpected $label checksum: $actual_sha256" >&2
        exit 2
    fi
}

if [[ "$TUNING_FILE" == "$DEFAULT_TUNING_FILE" ]]; then
    verify_default_file \
        "$TUNING_FILE" "$EXPECTED_TUNING_SHA256" "hipBLASLt tuning table"
elif [[ ! -f "$TUNING_FILE" ]]; then
    echo "missing hipBLASLt tuning file: $TUNING_FILE" >&2
    exit 2
fi
if [[ "$HOTNESS_CSV" == "$DEFAULT_HOTNESS_CSV" ]]; then
    verify_default_file \
        "$HOTNESS_CSV" "$EXPECTED_HOTNESS_SHA256" "prefill routing profile"
fi
if [[ "$DECODE_HOTNESS_CSV" == "$DEFAULT_DECODE_HOTNESS_CSV" ]]; then
    verify_default_file \
        "$DECODE_HOTNESS_CSV" "$EXPECTED_DECODE_HOTNESS_SHA256" \
        "decode routing profile"
fi

exec env -u HIPBLASLT_TUNING_FILE \
    ROCBLAS_USE_HIPBLASLT=1 \
    HIPBLASLT_TUNING_OVERRIDE_FILE="$TUNING_FILE" \
    HOTNESS_CSV="$HOTNESS_CSV" \
    DECODE_HOTNESS_CSV="$DECODE_HOTNESS_CSV" \
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
