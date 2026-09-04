#include "deepseek4_vision.h"

#include "common/gguf_bounds.h"
#include "common/gguf_mmap.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace dflash::common {

namespace {

constexpr size_t kGraphMetadataBytes = 16 * 1024 * 1024;
constexpr int kGraphNodes = 512;

struct VisionBlockWeights {
    ggml_tensor * norm1 = nullptr;
    ggml_tensor * wqkv_weight = nullptr;
    ggml_tensor * wqkv_bias = nullptr;
    ggml_tensor * wo_weight = nullptr;
    ggml_tensor * wo_bias = nullptr;
    ggml_tensor * norm2 = nullptr;
    ggml_tensor * mlp_w1 = nullptr;
    ggml_tensor * mlp_w2 = nullptr;
};

struct VisionWeights {
    ggml_tensor * patch_weight = nullptr;
    ggml_tensor * patch_bias = nullptr;
    std::array<VisionBlockWeights, 32> blocks{};
    ggml_tensor * final_norm = nullptr;
    ggml_tensor * aligner_w1_weight = nullptr;
    ggml_tensor * aligner_w1_bias = nullptr;
    ggml_tensor * aligner_w2_weight = nullptr;
    ggml_tensor * aligner_w2_bias = nullptr;
    std::array<ggml_tensor *, 4> sentinels{};
};

using ExpectedShapes = std::map<std::string, std::vector<int64_t>>;

ExpectedShapes expected_shapes() {
    ExpectedShapes result = {
        {"aligner.w1.bias", {4096}},
        {"aligner.w1.weight", {9216, 4096}},
        {"aligner.w2.bias", {4096}},
        {"aligner.w2.weight", {4096, 4096}},
        {"image_end", {4096}},
        {"image_newline", {4096}},
        {"image_pad", {4096}},
        {"image_start", {4096}},
        {"vision.norm.weight", {1024}},
        {"vision.patch_embed.proj.bias", {1024}},
        {"vision.patch_embed.proj.weight", {588, 1024}},
    };
    for (int block = 0; block < 32; ++block) {
        const std::string prefix = "vision.blocks." + std::to_string(block);
        result[prefix + ".attn.wo.bias"] = {1024};
        result[prefix + ".attn.wo.weight"] = {1024, 1024};
        result[prefix + ".attn.wqkv.bias"] = {3072};
        result[prefix + ".attn.wqkv.weight"] = {1024, 3072};
        result[prefix + ".mlp.w1.weight"] = {1024, 5632};
        result[prefix + ".mlp.w2.weight"] = {2816, 1024};
        result[prefix + ".norm1.weight"] = {1024};
        result[prefix + ".norm2.weight"] = {1024};
    }
    return result;
}

bool tensor_shape_is(const ggml_tensor * tensor,
                     const std::vector<int64_t> & expected) {
    if (!tensor || expected.empty() || expected.size() > GGML_MAX_DIMS) {
        return false;
    }
    size_t dim = 0;
    for (; dim < expected.size(); ++dim) {
        if (tensor->ne[dim] != expected[dim]) return false;
    }
    for (; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] != 1) return false;
    }
    return true;
}

bool get_string(gguf_context * gguf,
                const char * key,
                const char * expected,
                std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        error = std::string("missing or non-string metadata: ") + key;
        return false;
    }
    const char * value = gguf_get_val_str(gguf, id);
    if (!value || std::strcmp(value, expected) != 0) {
        error = std::string("metadata ") + key + " must be " + expected;
        return false;
    }
    return true;
}

bool get_u32(gguf_context * gguf,
             const char * key,
             uint32_t expected,
             std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32) {
        error = std::string("missing or non-uint32 metadata: ") + key;
        return false;
    }
    const uint32_t value = gguf_get_val_u32(gguf, id);
    if (value != expected) {
        error = std::string("metadata ") + key + " must be " +
                std::to_string(expected) + ", got " + std::to_string(value);
        return false;
    }
    return true;
}

bool get_f32(gguf_context * gguf,
             const char * key,
             float expected,
             std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_FLOAT32) {
        error = std::string("missing or non-float32 metadata: ") + key;
        return false;
    }
    const float value = gguf_get_val_f32(gguf, id);
    if (value != expected) {
        std::ostringstream out;
        out << "metadata " << key << " must be " << expected << ", got " << value;
        error = out.str();
        return false;
    }
    return true;
}

bool get_f32_array(gguf_context * gguf,
                   const char * key,
                   const std::array<float, 3> & expected,
                   std::string & error) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(gguf, id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_arr_n(gguf, id) != expected.size()) {
        error = std::string("missing or invalid float32 array metadata: ") + key;
        return false;
    }
    const auto * values = static_cast<const float *>(gguf_get_arr_data(gguf, id));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (values[i] != expected[i]) {
            error = std::string("metadata array has unexpected value: ") + key;
            return false;
        }
    }
    return true;
}

bool validate_metadata(gguf_context * gguf,
                       const VisionExpectedContract & expected,
                       VisionConfig & config,
                       std::string & error) {
    if (!get_string(gguf, "general.architecture", "deepseek4_vision", error) ||
        !get_string(gguf, "general.type", "mmproj", error) ||
        !get_u32(gguf, "general.alignment", 32, error) ||
        !get_u32(gguf, "deepseek4.vision.schema_version", 1, error) ||
        !get_u32(gguf, "deepseek4.vision.block_count", 32, error) ||
        !get_u32(gguf, "deepseek4.vision.embedding_length", 1024, error) ||
        !get_u32(gguf, "deepseek4.vision.attention.head_count", 16, error) ||
        !get_u32(gguf, "deepseek4.vision.attention.head_dimension", 64, error) ||
        !get_string(gguf, "deepseek4.vision.attention.rope_layout",
                    "2d-half-split-height-width", error) ||
        !get_u32(gguf, "deepseek4.vision.feed_forward_length", 2816, error) ||
        !get_u32(gguf, "deepseek4.vision.patch_size", 14, error) ||
        !get_f32(gguf, "deepseek4.vision.rope.freq_base", 10000.0f, error) ||
        !get_u32(gguf, "deepseek4.vision.downsample_ratio", 3, error) ||
        !get_u32(gguf, "deepseek4.vision.aligner_input_length", 9216, error) ||
        !get_u32(gguf, "deepseek4.vision.language_embedding_length",
                 expected.language_embedding_length, error) ||
        !get_u32(gguf, "deepseek4.vision.vocabulary_size",
                 expected.vocabulary_size, error) ||
        !get_f32(gguf, "deepseek4.vision.attention.layer_norm_rms_epsilon",
                 1.0e-6f, error) ||
        !get_u32(gguf, "deepseek4.vision.image.max_tokens", 384, error) ||
        !get_u32(gguf, "deepseek4.vision.image.min_pixels", 147456, error) ||
        !get_f32(gguf, "deepseek4.vision.image.max_aspect_ratio", 8.0f, error) ||
        !get_f32_array(gguf, "deepseek4.vision.image.normalization_mean",
                       {0.5f, 0.5f, 0.5f}, error) ||
        !get_f32_array(gguf, "deepseek4.vision.image.normalization_std",
                       {0.5f, 0.5f, 0.5f}, error) ||
        !get_string(gguf, "deepseek4.vision.image.patch_layout", "channel-major", error) ||
        !get_string(gguf, "deepseek4.vision.image.layout", "n", error) ||
        !get_u32(gguf, "deepseek4.vision.image.layout_version", 1, error) ||
        !get_string(gguf, "deepseek4.vision.image.layout_recipe",
                    "row-pair-column-interleave", error) ||
        !get_u32(gguf, "deepseek4.vision.image.compression_alignment", 4, error) ||
        !get_string(gguf, "deepseek4.vision.image.sentinel_types",
                    "start,pad,image,newline,end", error) ||
        !get_u32(gguf, "deepseek4.vision.image.sentinel_type_count", 5, error) ||
        !get_string(gguf, "deepseek4.vision.aligner.padding", "bottom-right", error) ||
        !get_string(gguf, "deepseek4.vision.aligner.patch_layout",
                    "channel-first-unfold", error) ||
        !get_string(gguf, "deepseek4.vision.aligner.activation", "gelu-exact", error)) {
        return false;
    }
    config.language_embedding_length = expected.language_embedding_length;
    config.vocabulary_size = expected.vocabulary_size;
    return true;
}

void bind_tensor(VisionWeights & weights,
                 const std::string & name,
                 ggml_tensor * tensor) {
    if (name == "vision.patch_embed.proj.weight") weights.patch_weight = tensor;
    else if (name == "vision.patch_embed.proj.bias") weights.patch_bias = tensor;
    else if (name == "vision.norm.weight") weights.final_norm = tensor;
    else if (name == "aligner.w1.weight") weights.aligner_w1_weight = tensor;
    else if (name == "aligner.w1.bias") weights.aligner_w1_bias = tensor;
    else if (name == "aligner.w2.weight") weights.aligner_w2_weight = tensor;
    else if (name == "aligner.w2.bias") weights.aligner_w2_bias = tensor;
    else if (name == "image_start") weights.sentinels[0] = tensor;
    else if (name == "image_pad") weights.sentinels[1] = tensor;
    else if (name == "image_newline") weights.sentinels[2] = tensor;
    else if (name == "image_end") weights.sentinels[3] = tensor;
    else if (name.rfind("vision.blocks.", 0) == 0) {
        const size_t number_start = std::strlen("vision.blocks.");
        const size_t dot = name.find('.', number_start);
        if (dot == std::string::npos) return;
        const int block = std::stoi(name.substr(number_start, dot - number_start));
        if (block < 0 || block >= 32) return;
        const std::string suffix = name.substr(dot + 1);
        auto & b = weights.blocks[static_cast<size_t>(block)];
        if (suffix == "norm1.weight") b.norm1 = tensor;
        else if (suffix == "attn.wqkv.weight") b.wqkv_weight = tensor;
        else if (suffix == "attn.wqkv.bias") b.wqkv_bias = tensor;
        else if (suffix == "attn.wo.weight") b.wo_weight = tensor;
        else if (suffix == "attn.wo.bias") b.wo_bias = tensor;
        else if (suffix == "norm2.weight") b.norm2 = tensor;
        else if (suffix == "mlp.w1.weight") b.mlp_w1 = tensor;
        else if (suffix == "mlp.w2.weight") b.mlp_w2 = tensor;
    }
}

struct GraphContext {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;

    GraphContext() {
        ggml_init_params params{};
        params.mem_size = kGraphMetadataBytes;
        params.no_alloc = true;
        ctx = ggml_init(params);
    }
    ~GraphContext() {
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }
    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;
};

struct Upload {
    ggml_tensor * tensor;
    const void * data;
    size_t bytes;
};

std::vector<int64_t> logical_shape(const ggml_tensor * tensor) {
    int highest = GGML_MAX_DIMS - 1;
    while (highest > 0 && tensor->ne[highest] == 1) --highest;
    std::vector<int64_t> shape;
    for (int dim = highest; dim >= 0; --dim) shape.push_back(tensor->ne[dim]);
    return shape;
}

bool finite_vector(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

ggml_tensor * as_f32(ggml_context * ctx, ggml_tensor * value) {
    return value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32);
}

ggml_tensor * round_bf16(ggml_context * ctx, ggml_tensor * value) {
    return ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_BF16), GGML_TYPE_F32);
}

ggml_tensor * linear(ggml_context * ctx,
                     ggml_tensor * weight,
                     ggml_tensor * input,
                     ggml_tensor * bias) {
    ggml_tensor * result = ggml_mul_mat(ctx, weight, input);
    if (bias) result = ggml_add(ctx, result, as_f32(ctx, bias));
    return round_bf16(ctx, result);
}

ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * input,
                       ggml_tensor * weight,
                       float epsilon) {
    ggml_tensor * normalized = ggml_rms_norm(ctx, input, epsilon);
    return round_bf16(ctx, ggml_mul(ctx, normalized, as_f32(ctx, weight)));
}

ggml_tensor * explicit_half_split_rope(ggml_context * ctx,
                                       ggml_tensor * value,
                                       ggml_tensor * cos,
                                       ggml_tensor * sin,
                                       int64_t head_dim,
                                       int64_t heads,
                                       int64_t tokens) {
    const int64_t half = head_dim / 2;
    ggml_tensor * first = ggml_view_3d(ctx, value, half, heads, tokens,
                                       value->nb[1], value->nb[2], 0);
    ggml_tensor * second = ggml_view_3d(ctx, value, half, heads, tokens,
                                        value->nb[1], value->nb[2],
                                        static_cast<size_t>(half) * value->nb[0]);
    ggml_tensor * rotated_first = ggml_sub(
        ctx, ggml_mul(ctx, first, cos), ggml_mul(ctx, second, sin));
    ggml_tensor * rotated_second = ggml_add(
        ctx, ggml_mul(ctx, second, cos), ggml_mul(ctx, first, sin));
    return round_bf16(ctx, ggml_concat(ctx, rotated_first, rotated_second, 0));
}

ggml_tensor * explicit_attention(ggml_context * ctx,
                                 ggml_tensor * q,
                                 ggml_tensor * k,
                                 ggml_tensor * v,
                                 int64_t head_dim,
                                 int64_t heads,
                                 int64_t tokens) {
    ggml_tensor * q_head = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_head = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    // aten::_scaled_dot_product_attention_math pre-scales both operands by
    // sqrt(1/sqrt(D)) before its F32 QK matmul.  Preserve that operation order;
    // post-scaling the score matrix is only equivalent in exact arithmetic.
    const float operand_scale = std::sqrt(
        1.0f / std::sqrt(static_cast<float>(head_dim)));
    q_head = ggml_scale(ctx, q_head, operand_scale);
    k_head = ggml_scale(ctx, k_head, operand_scale);
    ggml_tensor * scores = ggml_mul_mat(ctx, k_head, q_head);
    ggml_tensor * probabilities = ggml_soft_max(ctx, scores);
    ggml_tensor * v_head = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * attended = ggml_mul_mat(ctx, probabilities, v_head);
    attended = ggml_cont(ctx, ggml_permute(ctx, attended, 2, 0, 1, 3));
    attended = ggml_reshape_2d(ctx, attended, head_dim * heads, tokens);
    return round_bf16(ctx, attended);
}

void make_rope_tables(uint32_t height,
                      uint32_t width,
                      uint32_t rope_dimension,
                      float theta,
                      std::vector<float> & cos,
                      std::vector<float> & sin) {
    const size_t tokens = static_cast<size_t>(height) * width;
    cos.resize(tokens * rope_dimension);
    sin.resize(tokens * rope_dimension);
    const uint32_t axis_dim = rope_dimension / 2;
    for (uint32_t h = 0; h < height; ++h) {
        for (uint32_t w = 0; w < width; ++w) {
            const size_t token = static_cast<size_t>(h) * width + w;
            for (uint32_t j = 0; j < axis_dim; ++j) {
                const float exponent = static_cast<float>(2 * j) /
                                       static_cast<float>(rope_dimension);
                const float inv_frequency = 1.0f / std::pow(theta, exponent);
                const float h_angle = static_cast<float>(h) * inv_frequency;
                const float w_angle = static_cast<float>(w) * inv_frequency;
                cos[token * rope_dimension + j] = std::cos(h_angle);
                sin[token * rope_dimension + j] = std::sin(h_angle);
                cos[token * rope_dimension + axis_dim + j] = std::cos(w_angle);
                sin[token * rope_dimension + axis_dim + j] = std::sin(w_angle);
            }
        }
    }
}

bool stage_requested(const VisionEncodeOptions & options,
                     const std::string & name) {
    return std::find(options.diagnostic_stages.begin(),
                     options.diagnostic_stages.end(), name) !=
           options.diagnostic_stages.end();
}

struct StageTensor {
    std::string name;
    ggml_tensor * tensor;
};

void maybe_capture(const VisionEncodeOptions & options,
                   const std::string & name,
                   ggml_tensor * tensor,
                   std::vector<StageTensor> & stages) {
    if (stage_requested(options, name)) {
        ggml_set_output(tensor);
        stages.push_back({name, tensor});
    }
}

} // namespace

struct VisionRuntime::Impl {
    ggml_backend_t backend = nullptr; // borrowed
    ggml_context * weight_context = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    VisionConfig config{};
    VisionWeights weights{};
    VisionRuntimeStats stats{};

    ~Impl() {
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weight_context) ggml_free(weight_context);
    }
};

namespace {

bool compute_graph(VisionRuntime::Impl & runtime,
                   GraphContext & graph_context,
                   ggml_cgraph * graph,
                   const std::vector<Upload> & uploads,
                   ggml_tensor * final_tensor,
                   std::vector<float> & final_values,
                   const std::vector<StageTensor> & stages,
                   const VisionEncodeOptions & options,
                   std::string & error) {
    graph_context.allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(runtime.backend));
    if (!graph_context.allocator) {
        error = "failed to create vision graph allocator";
        return false;
    }
    if (!ggml_gallocr_alloc_graph(graph_context.allocator, graph)) {
        error = "failed to allocate vision graph scratch";
        return false;
    }
    const size_t scratch = ggml_gallocr_get_buffer_size(graph_context.allocator, 0);
    runtime.stats.last_encode_peak_scratch_bytes =
        std::max(runtime.stats.last_encode_peak_scratch_bytes, scratch);
    runtime.stats.peak_scratch_bytes = std::max(runtime.stats.peak_scratch_bytes, scratch);

    for (const Upload & upload : uploads) {
        if (ggml_nbytes(upload.tensor) != upload.bytes) {
            error = "internal vision input size mismatch";
            return false;
        }
        ggml_backend_tensor_set(upload.tensor, upload.data, 0, upload.bytes);
    }

    const enum ggml_status status = ggml_backend_graph_compute(runtime.backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "vision graph compute failed with status " + std::to_string(static_cast<int>(status));
        return false;
    }

    final_values.resize(static_cast<size_t>(ggml_nelements(final_tensor)));
    ggml_backend_tensor_get(final_tensor, final_values.data(), 0,
                            final_values.size() * sizeof(float));
    if (!finite_vector(final_values)) {
        error = "vision graph produced non-finite output";
        return false;
    }

    if (!stages.empty() && !options.stage_callback) {
        error = "diagnostic stages requested without a stage callback";
        return false;
    }
    for (const StageTensor & stage : stages) {
        std::vector<float> values(static_cast<size_t>(ggml_nelements(stage.tensor)));
        ggml_backend_tensor_get(stage.tensor, values.data(), 0,
                                values.size() * sizeof(float));
        if (!finite_vector(values)) {
            error = "diagnostic stage produced non-finite values: " + stage.name;
            return false;
        }
        if (!options.stage_callback(stage.name, logical_shape(stage.tensor), values, error)) {
            if (error.empty()) error = "diagnostic callback rejected stage " + stage.name;
            return false;
        }
    }
    return true;
}

bool run_patch_embedding(VisionRuntime::Impl & runtime,
                         const std::vector<float> & patches,
                         size_t tokens,
                         std::vector<float> & output,
                         const VisionEncodeOptions & options,
                         std::string & error) {
    GraphContext gc;
    if (!gc.ctx) { error = "failed to create patch graph context"; return false; }
    ggml_tensor * input = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 588, tokens);
    ggml_set_input(input);
    ggml_tensor * embedded = linear(gc.ctx, runtime.weights.patch_weight, input,
                                    runtime.weights.patch_bias);
    ggml_set_output(embedded);
    std::vector<StageTensor> stages;
    maybe_capture(options, "patch_embed", embedded, stages);
    ggml_cgraph * graph = ggml_new_graph_custom(gc.ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, embedded);
    return compute_graph(runtime, gc, graph,
                         {{input, patches.data(), patches.size() * sizeof(float)}},
                         embedded, output, stages, options, error);
}

bool run_block(VisionRuntime::Impl & runtime,
               int block_index,
               size_t tokens,
               const std::vector<float> & input_values,
               const std::vector<float> & rope_cos,
               const std::vector<float> & rope_sin,
               std::vector<float> & output,
               const VisionEncodeOptions & options,
               std::string & error) {
    GraphContext gc;
    if (!gc.ctx) { error = "failed to create block graph context"; return false; }
    const auto & weights = runtime.weights.blocks[static_cast<size_t>(block_index)];
    ggml_tensor * input = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 1024, tokens);
    ggml_tensor * cos = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 32, 1, tokens);
    ggml_tensor * sin = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 32, 1, tokens);
    ggml_set_input(input);
    ggml_set_input(cos);
    ggml_set_input(sin);

    std::vector<StageTensor> stages;
    const std::string prefix = "block." + std::to_string(block_index);
    ggml_tensor * norm1 = rms_norm(gc.ctx, input, weights.norm1, runtime.config.rms_epsilon);
    maybe_capture(options, prefix + ".norm1", norm1, stages);

    ggml_tensor * qkv = linear(gc.ctx, weights.wqkv_weight, norm1, weights.wqkv_bias);
    maybe_capture(options, prefix + ".qkv", qkv, stages);
    auto qkv_slice = [&](size_t offset) {
        ggml_tensor * view = ggml_view_2d(gc.ctx, qkv, 1024, tokens, qkv->nb[1], offset);
        return ggml_reshape_3d(gc.ctx, ggml_cont(gc.ctx, view), 64, 16, tokens);
    };
    ggml_tensor * q = qkv_slice(0);
    ggml_tensor * k = qkv_slice(1024 * sizeof(float));
    ggml_tensor * v = qkv_slice(2048 * sizeof(float));
    q = explicit_half_split_rope(gc.ctx, q, cos, sin, 64, 16, tokens);
    k = explicit_half_split_rope(gc.ctx, k, cos, sin, 64, 16, tokens);
    maybe_capture(options, prefix + ".q_rope", q, stages);
    maybe_capture(options, prefix + ".k_rope", k, stages);
    ggml_tensor * attention = explicit_attention(gc.ctx, q, k, v, 64, 16, tokens);
    maybe_capture(options, prefix + ".attention", attention, stages);
    ggml_tensor * projected = linear(gc.ctx, weights.wo_weight, attention, weights.wo_bias);
    maybe_capture(options, prefix + ".attn_output", projected, stages);
    ggml_tensor * residual = round_bf16(gc.ctx, ggml_add(gc.ctx, input, projected));
    maybe_capture(options, prefix + ".attn_residual", residual, stages);

    ggml_tensor * norm2 = rms_norm(gc.ctx, residual, weights.norm2,
                                   runtime.config.rms_epsilon);
    maybe_capture(options, prefix + ".norm2", norm2, stages);
    ggml_tensor * gate_up = linear(gc.ctx, weights.mlp_w1, norm2, nullptr);
    maybe_capture(options, prefix + ".mlp_w1", gate_up, stages);
    auto gate_up_slice = [&](size_t offset) {
        return ggml_cont(gc.ctx, ggml_view_2d(
            gc.ctx, gate_up, 2816, tokens, gate_up->nb[1], offset));
    };
    ggml_tensor * gate = gate_up_slice(0);
    ggml_tensor * up = gate_up_slice(2816 * sizeof(float));
    gate = round_bf16(gc.ctx, ggml_silu(gc.ctx, gate));
    ggml_tensor * gated = round_bf16(gc.ctx, ggml_mul(gc.ctx, gate, up));
    ggml_tensor * mlp = linear(gc.ctx, weights.mlp_w2, gated, nullptr);
    maybe_capture(options, prefix + ".mlp_output", mlp, stages);
    ggml_tensor * block_output = round_bf16(gc.ctx, ggml_add(gc.ctx, residual, mlp));
    maybe_capture(options, prefix + ".output", block_output, stages);
    ggml_set_output(block_output);

    ggml_cgraph * graph = ggml_new_graph_custom(gc.ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, block_output);
    return compute_graph(runtime, gc, graph,
                         {{input, input_values.data(), input_values.size() * sizeof(float)},
                          {cos, rope_cos.data(), rope_cos.size() * sizeof(float)},
                          {sin, rope_sin.data(), rope_sin.size() * sizeof(float)}},
                         block_output, output, stages, options, error);
}

bool run_final_norm(VisionRuntime::Impl & runtime,
                    size_t tokens,
                    const std::vector<float> & input_values,
                    std::vector<float> & output,
                    const VisionEncodeOptions & options,
                    std::string & error) {
    GraphContext gc;
    if (!gc.ctx) { error = "failed to create final norm graph context"; return false; }
    ggml_tensor * input = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 1024, tokens);
    ggml_set_input(input);
    ggml_tensor * result = rms_norm(gc.ctx, input, runtime.weights.final_norm,
                                    runtime.config.rms_epsilon);
    ggml_set_output(result);
    std::vector<StageTensor> stages;
    maybe_capture(options, "tower.final", result, stages);
    ggml_cgraph * graph = ggml_new_graph_custom(gc.ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, result);
    return compute_graph(runtime, gc, graph,
                         {{input, input_values.data(), input_values.size() * sizeof(float)}},
                         result, output, stages, options, error);
}

bool run_aligner(VisionRuntime::Impl & runtime,
                 VisionPatchGrid grid,
                 const std::vector<float> & features,
                 std::vector<float> & output,
                 const VisionEncodeOptions & options,
                 std::string & error) {
    GraphContext gc;
    if (!gc.ctx) { error = "failed to create aligner graph context"; return false; }
    const int64_t width = grid.width;
    const int64_t height = grid.height;
    const int64_t pad_width = (3 - width % 3) % 3;
    const int64_t pad_height = (3 - height % 3) % 3;
    const int64_t output_tokens = ((width + pad_width) / 3) *
                                  ((height + pad_height) / 3);

    ggml_tensor * input = ggml_new_tensor_2d(
        gc.ctx, GGML_TYPE_F32, 1024, width * height);
    ggml_tensor * kernel = ggml_new_tensor_4d(gc.ctx, GGML_TYPE_F32, 3, 3, 1024, 1);
    ggml_set_input(input);
    ggml_set_input(kernel);
    ggml_tensor * grid_chw = ggml_reshape_3d(gc.ctx, input, 1024, width, height);
    ggml_tensor * spatial = ggml_cont(
        gc.ctx, ggml_permute(gc.ctx, grid_chw, 2, 0, 1, 3));
    ggml_tensor * padded = ggml_pad(gc.ctx, spatial, pad_width, pad_height, 0, 0);
    ggml_tensor * unfolded = ggml_im2col(
        gc.ctx, kernel, padded, 3, 3, 0, 0, 1, 1, true, GGML_TYPE_F32);
    unfolded = ggml_reshape_2d(gc.ctx, unfolded, 9216, output_tokens);

    std::vector<StageTensor> stages;
    maybe_capture(options, "aligner.unfold", unfolded, stages);
    ggml_tensor * w1 = linear(gc.ctx, runtime.weights.aligner_w1_weight,
                              unfolded, runtime.weights.aligner_w1_bias);
    maybe_capture(options, "aligner.w1", w1, stages);
    ggml_tensor * activated = round_bf16(gc.ctx, ggml_gelu_erf(gc.ctx, w1));
    maybe_capture(options, "aligner.gelu", activated, stages);
    ggml_tensor * aligned = linear(gc.ctx, runtime.weights.aligner_w2_weight,
                                   activated, runtime.weights.aligner_w2_bias);
    maybe_capture(options, "aligner.output", aligned, stages);
    ggml_set_output(aligned);

    std::vector<float> zero_kernel(static_cast<size_t>(ggml_nelements(kernel)), 0.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(gc.ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, aligned);
    return compute_graph(runtime, gc, graph,
                         {{input, features.data(), features.size() * sizeof(float)},
                          {kernel, zero_kernel.data(), zero_kernel.size() * sizeof(float)}},
                         aligned, output, stages, options, error);
}

} // namespace

VisionRuntime::VisionRuntime() = default;
VisionRuntime::~VisionRuntime() = default;
VisionRuntime::VisionRuntime(VisionRuntime &&) noexcept = default;
VisionRuntime & VisionRuntime::operator=(VisionRuntime &&) noexcept = default;

bool VisionRuntime::load(const std::string & path,
                         ggml_backend_t backend,
                         const VisionExpectedContract & expected,
                         std::string & error) {
    error.clear();
    if (!backend) { error = "vision backend is null"; return false; }
    if (expected.language_embedding_length != 4096 || expected.vocabulary_size != 129280) {
        error = "unsupported language dimension or vocabulary contract";
        return false;
    }

    ggml_context * metadata_context = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &metadata_context;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf || !metadata_context) {
        if (gguf) gguf_free(gguf);
        if (metadata_context) ggml_free(metadata_context);
        error = "failed to open vision GGUF: " + path;
        return false;
    }

    auto next = std::make_unique<Impl>();
    next->backend = backend;
    next->weight_context = metadata_context;
    if (!validate_metadata(gguf, expected, next->config, error)) {
        gguf_free(gguf);
        return false;
    }

    const ExpectedShapes expected_tensors = expected_shapes();
    if (gguf_get_n_tensors(gguf) != static_cast<int64_t>(expected_tensors.size())) {
        error = "vision GGUF must contain exactly " +
                std::to_string(expected_tensors.size()) + " tensors";
        gguf_free(gguf);
        return false;
    }

    GgufMmap mapped;
    if (!mapped.open(path, error)) {
        gguf_free(gguf);
        return false;
    }
    const auto * file_bytes = static_cast<const uint8_t *>(mapped.data());
    const size_t file_size = mapped.size();
    const size_t data_offset = gguf_get_data_offset(gguf);
    std::map<std::string, bool> seen;
    for (const auto & item : expected_tensors) seen[item.first] = false;

    const int64_t tensor_count = gguf_get_n_tensors(gguf);
    for (int64_t index = 0; index < tensor_count; ++index) {
        const char * raw_name = gguf_get_tensor_name(gguf, index);
        const std::string name = raw_name ? raw_name : "";
        const auto expected_it = expected_tensors.find(name);
        if (expected_it == expected_tensors.end()) {
            error = "unexpected vision tensor: " + name;
            gguf_free(gguf);
            return false;
        }
        if (seen[name]) {
            error = "duplicate vision tensor: " + name;
            gguf_free(gguf);
            return false;
        }
        seen[name] = true;
        if (gguf_get_tensor_type(gguf, index) != GGML_TYPE_BF16) {
            error = "vision tensor must remain BF16: " + name;
            gguf_free(gguf);
            return false;
        }
        ggml_tensor * tensor = ggml_get_tensor(metadata_context, name.c_str());
        if (!tensor_shape_is(tensor, expected_it->second)) {
            error = "vision tensor has wrong shape: " + name;
            gguf_free(gguf);
            return false;
        }
        const size_t tensor_offset = gguf_get_tensor_offset(gguf, index);
        const size_t tensor_size = gguf_get_tensor_size(gguf, index);
        if (!gguf_tensor_in_file(data_offset, tensor_offset, tensor_size, file_size)) {
            error = gguf_bounds_error("DS4V mmproj", name.c_str(),
                                      ggml_type_name(tensor->type), data_offset,
                                      tensor_offset, tensor_size, file_size);
            gguf_free(gguf);
            return false;
        }
    }
    for (const auto & item : seen) {
        if (!item.second) {
            error = "missing vision tensor: " + item.first;
            gguf_free(gguf);
            return false;
        }
    }

    next->weight_buffer = ggml_backend_alloc_ctx_tensors(metadata_context, backend);
    if (!next->weight_buffer) {
        error = "failed to allocate validated vision weights";
        gguf_free(gguf);
        return false;
    }
    ggml_backend_buffer_set_usage(next->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    next->stats.weight_bytes = ggml_backend_buffer_get_size(next->weight_buffer);

    for (int64_t index = 0; index < tensor_count; ++index) {
        const char * name = gguf_get_tensor_name(gguf, index);
        ggml_tensor * tensor = ggml_get_tensor(metadata_context, name);
        const size_t offset = data_offset + gguf_get_tensor_offset(gguf, index);
        const size_t size = gguf_get_tensor_size(gguf, index);
        ggml_backend_tensor_set(tensor, file_bytes + offset, 0, size);
        bind_tensor(next->weights, name, tensor);
    }
    gguf_free(gguf);
    impl_ = std::move(next);
    return true;
}

bool VisionRuntime::encode(const std::vector<float> & channel_major_patches,
                           VisionPatchGrid grid,
                           VisionOutput & output,
                           std::string & error,
                           const VisionEncodeOptions & options) {
    error.clear();
    output = VisionOutput{};
    if (!impl_) { error = "vision runtime is not loaded"; return false; }
    if (grid.height == 0 || grid.width == 0) {
        error = "vision patch grid must be non-empty";
        return false;
    }
    const uint64_t tokens64 = static_cast<uint64_t>(grid.height) * grid.width;
    if (tokens64 > std::numeric_limits<uint32_t>::max()) {
        error = "vision patch grid is too large";
        return false;
    }
    const size_t tokens = static_cast<size_t>(tokens64);
    if (channel_major_patches.size() != tokens * 3 * 14 * 14) {
        error = "channel-major patch count does not match the patch grid";
        return false;
    }
    if (!finite_vector(channel_major_patches)) {
        error = "vision patches contain non-finite values";
        return false;
    }
    if (!options.diagnostic_stages.empty() && !options.stage_callback) {
        error = "diagnostic stages require a callback";
        return false;
    }

    impl_->stats.last_encode_peak_scratch_bytes = 0;
    impl_->stats.last_grid_height = grid.height;
    impl_->stats.last_grid_width = grid.width;

    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    make_rope_tables(grid.height, grid.width, 32, impl_->config.rope_freq_base,
                     rope_cos, rope_sin);

    std::vector<float> current;
    if (!run_patch_embedding(*impl_, channel_major_patches, tokens, current,
                             options, error)) return false;
    for (int block = 0; block < 32; ++block) {
        std::vector<float> next;
        if (!run_block(*impl_, block, tokens, current, rope_cos, rope_sin,
                       next, options, error)) return false;
        current.swap(next);
    }
    if (!run_final_norm(*impl_, tokens, current, output.raster_features,
                        options, error)) return false;
    if (!run_aligner(*impl_, grid, output.raster_features,
                     output.aligner_embeddings, options, error)) return false;

    output.feature_rows = static_cast<uint32_t>(tokens);
    output.feature_columns = impl_->config.embedding_length;
    output.aligner_rows = ((grid.height + 2) / 3) * ((grid.width + 2) / 3);
    output.aligner_columns = impl_->config.language_embedding_length;
    return true;
}

bool VisionRuntime::loaded() const { return static_cast<bool>(impl_); }

const VisionConfig & VisionRuntime::config() const {
    static const VisionConfig empty{};
    return impl_ ? impl_->config : empty;
}

const VisionRuntimeStats & VisionRuntime::stats() const {
    static const VisionRuntimeStats empty{};
    return impl_ ? impl_->stats : empty;
}

const ggml_tensor * VisionRuntime::sentinel_tensor(VisionSentinel identity) const {
    if (!impl_) return nullptr;
    const size_t index = static_cast<size_t>(identity);
    return index < impl_->weights.sentinels.size() ? impl_->weights.sentinels[index] : nullptr;
}

bool VisionRuntime::read_sentinel(VisionSentinel identity,
                                  std::vector<float> & values,
                                  std::string & error) const {
    error.clear();
    values.clear();
    const ggml_tensor * tensor = sentinel_tensor(identity);
    if (!tensor) { error = "vision sentinel is unavailable"; return false; }
    std::vector<ggml_bf16_t> source(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, source.data(), 0, source.size() * sizeof(ggml_bf16_t));
    values.resize(source.size());
    for (size_t i = 0; i < source.size(); ++i) values[i] = ggml_bf16_to_fp32(source[i]);
    return finite_vector(values);
}

void VisionRuntime::release_scratch() {
    if (impl_) impl_->stats.last_encode_peak_scratch_bytes = 0;
}

namespace {

bool compute_self_test_graph(ggml_backend_t backend,
                             GraphContext & gc,
                             ggml_tensor * output,
                             const std::vector<Upload> & uploads,
                             std::vector<float> & values,
                             std::string & error) {
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(gc.ctx, kGraphNodes, false);
    ggml_build_forward_expand(graph, output);
    gc.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gc.allocator || !ggml_gallocr_alloc_graph(gc.allocator, graph)) {
        error = "geometry self-test allocation failed";
        return false;
    }
    for (const Upload & upload : uploads) {
        ggml_backend_tensor_set(upload.tensor, upload.data, 0, upload.bytes);
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        error = "geometry self-test graph failed";
        return false;
    }
    values.resize(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    return true;
}

bool near(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

} // namespace

bool run_vision_geometry_self_test(ggml_backend_t backend,
                                   std::string & report,
                                   std::string & error) {
    report.clear();
    error.clear();
    if (!backend) { error = "geometry self-test backend is null"; return false; }

    // Half-split RoPE: first/last halves rotate together; the two frequency
    // slots represent height then width.
    {
        GraphContext gc;
        ggml_tensor * input = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 4, 1, 1);
        ggml_tensor * cos = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 2, 1, 1);
        ggml_tensor * sin = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 2, 1, 1);
        ggml_set_input(input); ggml_set_input(cos); ggml_set_input(sin);
        ggml_tensor * output = explicit_half_split_rope(gc.ctx, input, cos, sin, 4, 1, 1);
        const std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> cos_values = {0.0f, 1.0f};
        const std::vector<float> sin_values = {1.0f, 0.0f};
        std::vector<float> values;
        if (!compute_self_test_graph(backend, gc, output,
                {{input, input_values.data(), input_values.size() * sizeof(float)},
                 {cos, cos_values.data(), cos_values.size() * sizeof(float)},
                 {sin, sin_values.data(), sin_values.size() * sizeof(float)}},
                values, error)) return false;
        const std::array<float, 4> expected = {-3.0f, 2.0f, 1.0f, 4.0f};
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!near(values[i], expected[i], 1.0e-6f)) {
                error = "half-split RoPE geometry mismatch";
                return false;
            }
        }
    }

    // Zero Q/K produces a uniform, fully bidirectional distribution. Every
    // query must therefore see the mean of all three V rows, including future rows.
    {
        GraphContext gc;
        ggml_tensor * q = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 2, 1, 3);
        ggml_tensor * k = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 2, 1, 3);
        ggml_tensor * v = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, 2, 1, 3);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
        ggml_tensor * output = explicit_attention(gc.ctx, q, k, v, 2, 1, 3);
        const std::vector<float> zeros(6, 0.0f);
        const std::vector<float> v_values = {1, 2, 3, 4, 5, 6};
        std::vector<float> values;
        if (!compute_self_test_graph(backend, gc, output,
                {{q, zeros.data(), zeros.size() * sizeof(float)},
                 {k, zeros.data(), zeros.size() * sizeof(float)},
                 {v, v_values.data(), v_values.size() * sizeof(float)}},
                values, error)) return false;
        const std::array<float, 6> expected = {3, 4, 3, 4, 3, 4};
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!near(values[i], expected[i], 0.02f)) {
                error = "bidirectional attention geometry mismatch";
                return false;
            }
        }
    }

    // Two channels, H=2, W=4. Padding is right/bottom to H=3, W=6 and
    // flattening is channel, dy, dx for each raster-order output token.
    {
        GraphContext gc;
        ggml_tensor * input = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 2, 8);
        ggml_tensor * kernel = ggml_new_tensor_4d(gc.ctx, GGML_TYPE_F32, 3, 3, 2, 1);
        ggml_set_input(input); ggml_set_input(kernel);
        ggml_tensor * chw = ggml_reshape_3d(gc.ctx, input, 2, 4, 2);
        ggml_tensor * spatial = ggml_cont(gc.ctx, ggml_permute(gc.ctx, chw, 2, 0, 1, 3));
        ggml_tensor * padded = ggml_pad(gc.ctx, spatial, 2, 1, 0, 0);
        ggml_tensor * output = ggml_im2col(
            gc.ctx, kernel, padded, 3, 3, 0, 0, 1, 1, true, GGML_TYPE_F32);
        const std::vector<float> input_values = {
            1, 101, 2, 102, 3, 103, 4, 104,
            5, 105, 6, 106, 7, 107, 8, 108,
        };
        const std::vector<float> kernel_values(18, 0.0f);
        std::vector<float> values;
        if (!compute_self_test_graph(backend, gc, output,
                {{input, input_values.data(), input_values.size() * sizeof(float)},
                 {kernel, kernel_values.data(), kernel_values.size() * sizeof(float)}},
                values, error)) return false;
        const std::array<float, 36> expected = {
            1,2,3,5,6,7,0,0,0, 101,102,103,105,106,107,0,0,0,
            4,0,0,8,0,0,0,0,0, 104,0,0,108,0,0,0,0,0,
        };
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!near(values[i], expected[i], 1.0e-6f)) {
                error = "channel-first padded unfold geometry mismatch at " +
                        std::to_string(i);
                return false;
            }
        }
    }

    report = "PASS rope=half-split-height-width attention=bidirectional "
             "unfold=channel-first-bottom-right";
    return true;
}

} // namespace dflash::common
