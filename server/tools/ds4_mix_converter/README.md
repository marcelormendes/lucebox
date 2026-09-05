# CPU MIX converter: optional parallel encoding

`--encode-threads N` accepts integers 1 through 8 and defaults to 1. The default
encodes on the calling thread. Higher values launch independent experts using
`std::launch::async`; each batch finishes before the next begins. Only expert
encoding is parallel: calibration, histogram fitting, repair stamps, metadata,
sidecars, and codec math/row order are unchanged. Thread counts and phase timings
appear only in stderr logs, so changing concurrency must not change artifact bytes.

Each task owns two read-only file descriptors, packed/scales/F32 row scratch,
typed quantized row scratch, and one encoded expert buffer. Source entries,
calibration codebooks and importance weights remain immutable and alive until
all futures finish. The calling thread writes results and progress in expert
order. Launch, task, size, or write failures stop publication, drain every launched
future, then propagate the first observed exception. No detached workers or
parallel FILE writes are used. Failed conversion may leave a `.partial` file;
new worker failures cannot publish a completed GGUF or GUMIX sidecar.

Checked arithmetic bounds the row size, expert size, and retained batch payload.
For the operator model, gate/up uses 2.5 MiB per expert and down uses 3.5 MiB;
eight results retain at most 28 MiB of encoded payload. This excludes thread
stacks, allocator/container overhead, source metadata/calibration memory and
approximately 22 KiB of row scratch per worker. Up to 16 worker input descriptors
are open. This is a payload bound, not a total RSS guarantee.

On soulf only, use a separate build directory and two build jobs:

```sh
cmake -S server/tools/ds4_mix_converter -B /tmp/ds4v-mix-parallel-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ds4v-mix-parallel-build -j2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ctest --test-dir /tmp/ds4v-mix-parallel-build --output-on-failure
```

The standalone build disables GPU, OpenMP and BLAS backends. Tests exercise both
MIX codecs with nonuniform importance weights and 17 distinct synthetic experts,
full synthetic GGUF/GUMIX serialization with raw/dense/map producers, explicit
launch failure, task/allocation failure, ordered writes, byte counts, actual
short reads, `/dev/full`, failure joins, and no final publication on a read error.
Existing fitter and codec regression tests remain unchanged.

`prove_parallel.py` runs the old binary, new default, new explicit 1, new 8, and
repeat 8 sequentially against an existing source/imatrix. Every lane is hard-coded
to one layer and 17 experts with `--experts-only`; no full conversion is launched.
It records commands, commit, hashes, individual exits, whole-file byte comparisons,
GNU time metrics, phase logs and `/proc` thread/run-state samples. Its evidence
directory must not exist. The script does not build or alter either converter.
Run only after unit tests pass, and separately review measured speedup and full-file
identity before authorizing any larger conversion. Timing shares the machine with
the existing operator conversion and is not an isolated performance benchmark.
