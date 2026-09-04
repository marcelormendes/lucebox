# Native DS4V vision runtime

`VisionRuntime` is a backend-borrowing, projector-owning unit for the verified
DS4V mmproj artifact. It validates the full metadata recipe and all 267 BF16
tensors before backend allocation. The API accepts already preprocessed,
channel-major patches and returns raster-order tower features and aligner rows;
image transport, decoding, N-layout construction, and decoder placement remain
request-layer responsibilities.

The graph follows the source module directly: patch linear, 32 pre-norm blocks,
final RMSNorm, then the ratio-three aligner. Explicit BF16-to-F32 round trips
preserve the source's BF16 materialization points while normalization and rotary
math stay F32. RoPE uses cached source-shaped height frequencies followed by
width frequencies and rotates the two 32-channel halves. Attention is an
unmasked score/softmax/value graph; it follows PyTorch's math-SDPA operation
order by pre-scaling both Q and K before the F32 score matmul. The aligner pads only bottom and right and
uses GGML `im2col`, whose channel-first flatten order matches `F.unfold`, plus
`ggml_gelu_erf` for exact GELU.

Each block is executed in a separate reusable-backend graph. This bounds scratch
to one attention matrix set and permits the allocator to recycle intermediates;
no diagnostic keeps 32 attention matrices alive. The cost is one host-visible
feature handoff per block in this first correctness path. A later qualified fast
path could keep features resident and use flash attention, but it must retain the
same BF16 boundaries and pass the same parent fixtures before replacing the
inspectable path.

`ds4v_vision_probe` runs deterministic geometry checks and compares both parent
CPU fixtures, recording shape, finite status, maximum absolute error, RMSE,
cosine similarity, elapsed time, weight bytes, and peak scratch. It also writes
sparse named stages for mismatch localization without retaining score matrices.
