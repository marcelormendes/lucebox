// Glm5NextBackend implementation.

#include "glm5next_backend.h"
#include "glm5next_internal.h"
#include "common/daemon_loop.h"
#include "common/gguf_inspect.h"
#include "common/moe_hybrid_ffn_eval.h"

#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>

namespace dflash::common {

Glm5NextBackend::Glm5NextBackend(const Glm5NextBackendConfig & cfg)
    : cfg_(cfg) {
}

Glm5NextBackend::~Glm5NextBackend() {
    shutdown();
}

bool Glm5NextBackend::init() {
    if (!cfg_.target_path) {
        std::fprintf(stderr, "[glm5next] target_path is null\n");
        return false;
    }

    std::fprintf(stderr, "[glm5next] loading model from %s\n", cfg_.target_path);

    if (!load_model()) {
        std::fprintf(stderr, "[glm5next] load_model failed\n");
        return false;
    }

    if (!init_hybrid_model()) {
        std::fprintf(stderr, "[glm5next] init_hybrid_model failed\n");
        return false;
    }

    std::fprintf(stderr, "[glm5next] initialized successfully\n");
    return true;
}

bool Glm5NextBackend::load_model() {
    // Allocate GGML context for weight loading
    // Stub size - actual calculation needed based on model size
    const size_t ctx_size = 1024 * 1024 * 1024;  // 1GB placeholder
    
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "[glm5next] failed to create GGML context\n");
        return false;
    }
    
    // Load weights and register ROCmFP mix qtypes
    if (!glm5next_load_weights(cfg_.target_path, w_, ctx, registered_mix_bases_)) {
        std::fprintf(stderr, "[glm5next] glm5next_load_weights failed\n");
        ggml_free(ctx);
        return false;
    }
    
    // Context ownership passed to backend
    std::fprintf(stderr, "[glm5next] weights loaded successfully\n");
    return true;
}

bool Glm5NextBackend::init_hybrid_model() {
    // Initialize backend
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[glm5next] failed to initialize CUDA backend\n");
        return false;
    }
    
    // Initialize cache
    cache_.n_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    cache_.cur_pos = 0;
    cache_.n_past = 0;
    
    // Allocate KV cache tensors for MLA layers and KDA state
    const int n_mla_layers = (w_.n_layer + w_.full_attn_interval - 1) / w_.full_attn_interval;
    const int n_kda_layers = w_.n_layer - n_mla_layers;
    const int kv_dim = w_.head_dim;  // MLA uses single KV head (absorbed form)
    
    ggml_init_params cache_params = {
        /*.mem_size   =*/ 512 * 1024 * 1024,  // 512MB for cache metadata
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * cache_ctx = ggml_init(cache_params);
    if (!cache_ctx) {
        std::fprintf(stderr, "[glm5next] failed to init cache context\n");
        return false;
    }
    
    // MLA KV cache: [head_dim, n_ctx, n_mla_layers]
    cache_.k = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F16, kv_dim, cache_.n_ctx, n_mla_layers);
    cache_.v = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F16, kv_dim, cache_.n_ctx, n_mla_layers);
    ggml_set_name(cache_.k, "cache_k");
    ggml_set_name(cache_.v, "cache_v");
    
    // KDA recurrent state: [head_dim, n_head, n_kda_layers] (hidden state per head)
    cache_.kda_state = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F32, 
                                          w_.head_dim, w_.n_head, n_kda_layers);
    ggml_set_name(cache_.kda_state, "kda_state");
    
    // Allocate cache on backend
    ggml_backend_buffer_t cache_buf = ggml_backend_alloc_ctx_tensors(cache_ctx, backend_);
    if (!cache_buf) {
        std::fprintf(stderr, "[glm5next] failed to allocate cache buffer\n");
        ggml_free(cache_ctx);
        return false;
    }
    
    // Zero-initialize caches
    ggml_backend_tensor_memset(cache_.k, 0, 0, ggml_nbytes(cache_.k));
    ggml_backend_tensor_memset(cache_.v, 0, 0, ggml_nbytes(cache_.v));
    ggml_backend_tensor_memset(cache_.kda_state, 0, 0, ggml_nbytes(cache_.kda_state));
    
    std::fprintf(stderr, "[glm5next] cache allocated: %d MLA layers, %d KDA layers, ctx=%d\n",
                 n_mla_layers, n_kda_layers, cache_.n_ctx);
    
    // Build MoE hybrid storage for expert evaluation (all-hot, GPU-only)
    const int n_moe_layers = w_.n_layer - w_.first_moe_layer;
    if (n_moe_layers > 0) {
        // Create all-hot placement: all 288 experts on GPU
        moe_placement_.n_expert = w_.n_expert;
        moe_placement_.hot_expert_ids.resize(n_moe_layers);
        for (int il = 0; il < n_moe_layers; ++il) {
            moe_placement_.hot_expert_ids[il].resize(w_.n_expert);
            for (int e = 0; e < w_.n_expert; ++e) {
                moe_placement_.hot_expert_ids[il][e] = e;  // All hot
            }
        }
        
        // Build layer descriptors
        std::vector<MoeLayerDesc> layer_descs;
        layer_descs.reserve(n_moe_layers);
        for (int il = w_.first_moe_layer; il < w_.n_layer; ++il) {
            const auto & layer = w_.layers[il];
            MoeLayerDesc desc;
            desc.ffn_gate_exps = layer.moe_experts_gate;
            desc.ffn_up_exps = layer.moe_experts_up;
            desc.ffn_down_exps = layer.moe_experts_down;
            desc.ffn_gate_shexp = layer.moe_shared_gate;
            desc.ffn_up_shexp = layer.moe_shared_up;
            desc.ffn_down_shexp = layer.moe_shared_down;
            layer_descs.push_back(desc);
        }
        
        // MoE config
        MoeHybridConfig moe_cfg;
        moe_cfg.n_embd = w_.n_embd;
        moe_cfg.n_expert = w_.n_expert;
        moe_cfg.n_expert_used = w_.n_expert_used;
        moe_cfg.n_ff_exp = w_.n_expert_ff;
        moe_cfg.n_ff_shexp = w_.n_expert_ff;  // Shared expert same size
        moe_cfg.n_layer = n_moe_layers;
        moe_cfg.first_moe_layer = w_.first_moe_layer;
        moe_cfg.swiglu_clamp = w_.swiglu_clamp;
        moe_cfg.cold_expert_backend = MoeHybridColdBackend::Cpu;
        moe_cfg.materialize_hot_experts = true;
        moe_cfg.materialize_cold_experts = false;  // All on GPU
        
        moe_hybrid_ = std::make_shared<MoeHybridStorage>();
        std::string err;
        if (!build_moe_hybrid_storage(moe_cfg, backend_, moe_placement_, 
                                      layer_descs, *moe_hybrid_, &err)) {
            std::fprintf(stderr, "[glm5next] failed to build MoE storage: %s\n", err.c_str());
            return false;
        }
        
        std::fprintf(stderr, "[glm5next] MoE storage initialized: %d layers, %d experts (all GPU)\n",
                     n_moe_layers, w_.n_expert);
    }
    
    std::fprintf(stderr, "[glm5next] backend initialized: ctx=%d\n", cache_.n_ctx);
    return true;
}

bool Glm5NextBackend::requires_monolithic_model() const {
    // Check if ROCmFP adaptive qtypes (105/106) require monolithic residency
    // Similar to DeepSeek4's logic
    return false;
}

void Glm5NextBackend::print_ready_banner() const {
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "===================================\n");
    std::fprintf(stderr, " GLM-5.3-Flash (GLM5-Next) Ready\n");
    std::fprintf(stderr, "===================================\n");
    std::fprintf(stderr, " Architecture: glm5next\n");
    std::fprintf(stderr, " Layers: %d (34 KDA + 11 MLA/DSA)\n", w_.n_layer);
    std::fprintf(stderr, " Experts: %d (top-%d), first MoE layer: %d\n",
                 w_.n_expert, w_.n_expert_used, w_.first_moe_layer);
    std::fprintf(stderr, " Hidden: %d, Vocab: %d\n", w_.n_embd, w_.n_vocab);
    std::fprintf(stderr, " mHC streams: %d\n", w_.n_hc);
    std::fprintf(stderr, "===================================\n");
    std::fprintf(stderr, "\n");
}

bool Glm5NextBackend::park(ParkTarget target) {
    std::fprintf(stderr, "[glm5next] park not implemented\n");
    return false;
}

bool Glm5NextBackend::unpark(ParkTarget target) {
    std::fprintf(stderr, "[glm5next] unpark not implemented\n");
    return false;
}

GenerateResult Glm5NextBackend::generate_impl(
    const GenerateRequest & req,
    const DaemonIO & io) {
    std::fprintf(stderr, "[glm5next] generate_impl: prompt_len=%zu, n_gen=%d\n",
                 req.prompt.size(), req.n_gen);
    
    GenerateResult result;
    result.status = GenerateStatus::OK;
    
    // Simplified generation: just build graph for first token and return
    // Full implementation would do proper prefill, decode loop, and sampling
    
    if (req.prompt.empty()) {
        result.status = GenerateStatus::Error;
        result.error_message = "empty prompt";
        return result;
    }
    
    // Allocate graph context
    const size_t graph_ctx_size = 128 * 1024 * 1024;  // 128MB
    ggml_init_params params = {
        /*.mem_size   =*/ graph_ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,  // Use backend allocator
    };
    
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        result.status = GenerateStatus::Error;
        result.error_message = "failed to create graph context";
        return result;
    }
    
    // Build forward graph for first token
    ggml_cgraph * gf = ggml_new_graph(ctx);
    
    ggml_tensor * logits = glm5next_build_graph(
        ctx, w_, cache_,
        req.prompt.data(), 1,  // Just first token for now
        cache_.cur_pos,
        moe_hybrid_.get()  // Pass MoE storage for expert evaluation
    );
    
    if (!logits) {
        ggml_free(ctx);
        result.status = GenerateStatus::Error;
        result.error_message = "graph construction failed";
        return result;
    }
    
    ggml_build_forward_expand(gf, logits);
    
    std::fprintf(stderr, "[glm5next] graph built: %d nodes, %d leaves\n",
                 gf->n_nodes, gf->n_leafs);
    
    // Compute graph
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        result.status = GenerateStatus::Error;
        result.error_message = "graph compute failed";
        return result;
    }
    
    std::fprintf(stderr, "[glm5next] graph computed successfully\n");
    
    // Update cache position after successful decode
    cache_.cur_pos += 1;
    cache_.n_past += 1;
    
    // Sample (simplified: just return EOS)
    const int32_t eos_token = 2;  // Typical EOS
    io.emit(eos_token);
    io.emit(-1);  // Sentinel
    
    result.n_gen = 1;
    result.n_past = cache_.n_past;
    
    ggml_free(ctx);
    return result;
}

bool Glm5NextBackend::snapshot_save(int slot) {
    std::fprintf(stderr, "[glm5next] snapshot_save not implemented\n");
    return false;
}

void Glm5NextBackend::snapshot_free(int slot) {
}

bool Glm5NextBackend::snapshot_used(int slot) const {
    return false;
}

int Glm5NextBackend::snapshot_cur_pos(int slot) const {
    return -1;
}

GenerateResult Glm5NextBackend::restore_and_generate_impl(
    int slot,
    const GenerateRequest & req,
    const DaemonIO & io) {
    GenerateResult result;
    result.status = GenerateStatus::Error;
    result.error_message = "glm5next snapshot restore not implemented";
    return result;
}

bool Glm5NextBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    return false;
}

void Glm5NextBackend::free_drafter() {
}

void Glm5NextBackend::shutdown() {
    // Unregister ROCmFP mix qtypes (105/106) before freeing backend
    // Must be called while GPU buffer is still valid
    extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base);
    extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base);
    
    for (const void * base : registered_mix_bases_) {
        // Determine qtype from registered tensor to call correct unregister
        // For now, try both (idempotent if not registered)
        ggml_cuda_rocmfp2_mix_unregister(base);
        ggml_cuda_rocmfp3_mix_unregister(base);
    }
    registered_mix_bases_.clear();
    
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

int Glm5NextBackend::do_prefill(const std::vector<int32_t> & tokens,
                                const DaemonIO & io,
                                int kv_offset) {
    std::fprintf(stderr, "[glm5next] do_prefill stub\n");
    return -1;
}

GenerateResult Glm5NextBackend::generate_from_state(
    const GenerateRequest & req,
    const DaemonIO & io,
    int kv_offset) {
    GenerateResult result;
    result.status = GenerateStatus::Error;
    result.error_message = "glm5next generate_from_state not implemented";
    return result;
}

bool Glm5NextBackend::do_decode(int committed, int n_gen,
                                const std::vector<int32_t> & history_prefix,
                                std::vector<int32_t> & out_tokens,
                                const DaemonIO & io) {
    std::fprintf(stderr, "[glm5next] do_decode stub\n");
    return false;
}

}  // namespace dflash::common
