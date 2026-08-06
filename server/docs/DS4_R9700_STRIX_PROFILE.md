# DS4 q=5 profile for R9700 + Strix Halo

This profile reproduces the qualified single-process heterogeneous decode on
an R9700 (`gfx1201`) plus Strix Halo (`gfx1151`) running ROCm 7.2.4. It is a
hardware profile, not a portable default for unrelated GPU pairs.

## Checked-in inputs

The profile keeps every small input needed for the qualified placement in the
repository:

- `config/ds4/r9700-strix/prefill-routing.csv`: resident-expert profile;
- `config/ds4/r9700-strix/decode-routing.csv`: decode placement profile;
- `config/hipblaslt/r9700-strix-rocm-7.2.4.txt`: selected hipBLASLt solutions;
- `scripts/qualify_ds4_q5_r9700_strix.sh`: exact launch and measurement policy.

The wrapper verifies all three file checksums before loading the model. Custom
profile paths remain possible through `HOTNESS_CSV`, `DECODE_HOTNESS_CSV`, and
`HIPBLASLT_TUNING_OVERRIDE_FILE`, but they are a new qualification rather than
the checked-in result.

## Build

Compile one binary for both GPU architectures:

```bash
cmake -S server -B server/build-hip-dual \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1151;gfx1201' \
  -DDFLASH27B_TESTS=ON

cmake --build server/build-hip-dual -j \
  --target dflash_server test_tokenizer_harness test_deepseek4_unit
```

Before the model run, execute the focused unit suite:

```bash
server/build-hip-dual/test_deepseek4_unit
```

## Run the qualified profile

Only the two model artifacts and build directory are external inputs:

```bash
TARGET_MODEL=/path/to/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
DRAFT_MODEL=/path/to/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf \
BUILD_DIR="$PWD/server/build-hip-dual" \
server/scripts/qualify_ds4_q5_r9700_strix.sh
```

The default protocol is one exact 2,048-token prompt, 128 generated tokens,
two warmups, and seven measured requests. Every request must produce 128
tokens with response SHA-256
`0f785a7ffa406498aafb14553966eaed0f52220fed0f7cc016b66921d104d194`.
The runner fails on a different length or hash.

Each run directory records the source commit, server checksum, tuning-table
path and checksum, complete environment, model metadata, individual request
records, client summary, authoritative model-side decode summary, server log,
and ROCm/VRAM state. Set `OUT_ROOT` and `RUN_ID` to choose its location.

## Measured result

The final two-warmup/seven-run qualification used source
`ee160092ce106a017d50a11cf8110e3b21e3cc46` and server SHA-256
`2bd2282c5409f5d6573fbf0763c00bfcbdbe50c547d64231c8213eeee8ec5119`.
All seven measured requests returned 128 tokens with the required response
hash.

- authoritative server decode: **73.7 tok/s median**, 73.5-73.8 range;
- historical client diagnostic: 83.403 tok/s median, 83.122-83.509 range;
- speculative acceptance: 1.00 median.

An earlier experiment reported 89.876 tok/s in the client diagnostic, but its
model-side median was 73.2 tok/s. That client formula starts timing at the
first non-empty streamed text event while counting every completion token. A
slow first rejected speculative step therefore fell outside its time window
and inflated the result. The repaired capture path accepts the first block and
is slightly faster model-side, even though that legacy client number is lower.
Use `server-decode-summary.json` for engine performance claims.

## Qualified placement

The wrapper fixes the settings that produced the measured profile:

- critical-path expert placement with a main/peer rate ratio of `4.4`;
- a 14,350 MiB main-GPU expert budget;
- dynamic route balance with 3 main slots and 13 main slots at q=5;
- owner-local residual fusion;
- q=5 verification with the platform-selected ROCmFP4 x4+1 kernel;
- the checked-in hipBLASLt solutions for both GPU architectures.

The experimental attention-head split and shared-stage width split are not in
this profile: both were slower in full exact-output runs.

## Wide heterogeneous capture repair

A rebase onto the newer prefill safety work exposed a reproducibility bug. The
2K heterogeneous prompt was split into 1,920 and 128-token forwards solely to
capture the final DSpark feature window. Sparse heterogeneous prefill already
returns a feature row for every requested token, so the split was unnecessary.
It changed the approximate sparse-prefill state, made the target emit EOS, and
prevented the benchmark from measuring a full completion.

The runtime now keeps one wide heterogeneous batch and retains the requested
feature rows afterward. Snapshot boundaries still split when required. Unit
tests cover supported modes and size limits, and the model-backed qualifier
checks the full completion and exact response hash.

## Portability boundary

Do not silently reuse the hipBLASLt table with another ROCm release or GPU
architecture. Retune and requalify there. The routing profiles are also tied
to this model, quantization, workload, and expert layout. The general
`qualify_ds4_q5_amd.sh` runner remains the starting point for a new pair.
