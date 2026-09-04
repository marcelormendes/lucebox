// Native DeepSeek-V4 vision tower and aligner.
//
// The runtime borrows a caller-selected GGML backend.  It owns only the
// projector weights and per-encode graph scratch; the caller remains
// responsible for image decoding, preprocessing, N-layout placement, and the
// backend lifetime.

#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace dflash::common {

struct VisionConfig {
    uint32_t schema_version = 1;
    uint32_t block_count = 32;
    uint32_t embedding_length = 1024;
    uint32_t head_count = 16;
    uint32_t head_dimension = 64;
    uint32_t feed_forward_length = 2816;
    uint32_t patch_size = 14;
    float rope_freq_base = 10000.0f;
    uint32_t downsample_ratio = 3;
    uint32_t aligner_input_length = 9216;
    uint32_t language_embedding_length = 4096;
    uint32_t vocabulary_size = 129280;
    float rms_epsilon = 1.0e-6f;
    uint32_t image_max_tokens = 384;
    uint32_t image_min_pixels = 147456;
    float image_max_aspect_ratio = 8.0f;
};

struct VisionExpectedContract {
    uint32_t language_embedding_length = 4096;
    uint32_t vocabulary_size = 129280;
};

struct VisionPatchGrid {
    uint32_t height = 0;
    uint32_t width = 0;
};

enum class VisionSentinel : uint8_t {
    Start,
    Pad,
    Newline,
    End,
};

struct VisionOutput {
    // Raster-order final tower features, shape [height * width, 1024].
    std::vector<float> raster_features;
    // Raster-order aligner rows, shape [ceil(height / 3) * ceil(width / 3), 4096].
    std::vector<float> aligner_embeddings;
    uint32_t feature_rows = 0;
    uint32_t feature_columns = 0;
    uint32_t aligner_rows = 0;
    uint32_t aligner_columns = 0;
};

struct VisionRuntimeStats {
    size_t weight_bytes = 0;
    size_t peak_scratch_bytes = 0;
    size_t last_encode_peak_scratch_bytes = 0;
    uint32_t last_grid_height = 0;
    uint32_t last_grid_width = 0;
};

using VisionStageCallback = std::function<bool(
    const std::string & name,
    const std::vector<int64_t> & shape,
    const std::vector<float> & values,
    std::string & error)>;

struct VisionEncodeOptions {
    // Exact stage names to retain and send to stage_callback.  Empty keeps no
    // diagnostics, which lets the graph allocator recycle all intermediates.
    std::vector<std::string> diagnostic_stages;
    VisionStageCallback stage_callback;
};

class VisionRuntime {
public:
    struct Impl;

    VisionRuntime();
    ~VisionRuntime();
    VisionRuntime(VisionRuntime &&) noexcept;
    VisionRuntime & operator=(VisionRuntime &&) noexcept;
    VisionRuntime(const VisionRuntime &) = delete;
    VisionRuntime & operator=(const VisionRuntime &) = delete;

    // Transactional: on failure, an already-loaded runtime remains usable.
    bool load(const std::string & path,
              ggml_backend_t backend,
              const VisionExpectedContract & expected,
              std::string & error);

    bool encode(const std::vector<float> & channel_major_patches,
                VisionPatchGrid grid,
                VisionOutput & output,
                std::string & error,
                const VisionEncodeOptions & options = {});

    bool loaded() const;
    const VisionConfig & config() const;
    const VisionRuntimeStats & stats() const;

    // The tensor remains owned by this runtime and lives on the borrowed
    // backend.  Callers can use it in a later request-owned placement graph.
    const ggml_tensor * sentinel_tensor(VisionSentinel identity) const;
    bool read_sentinel(VisionSentinel identity,
                       std::vector<float> & values,
                       std::string & error) const;

    // Per-encode graph allocations are released before encode returns.  This
    // call resets only accounting and is safe between requests.
    void release_scratch();

private:
    std::unique_ptr<Impl> impl_;
};

// Runs small deterministic GGML graphs for the three layout-sensitive
// primitives used by VisionRuntime: half-split height/width RoPE, unmasked
// bidirectional attention, and bottom/right padded channel-first unfold.
bool run_vision_geometry_self_test(ggml_backend_t backend,
                                   std::string & report,
                                   std::string & error);

// Applies the source image-token/aspect contract and the explicit-path scratch
// ceiling without allocating a graph or patch buffer.
bool validate_vision_grid(const VisionConfig & config,
                          VisionPatchGrid grid,
                          std::string & error);

} // namespace dflash::common
