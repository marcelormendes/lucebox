// Glm5NextBackend implementation.

#include "glm5next_backend.h"
#include "common/daemon_loop.h"
#include "common/gguf_inspect.h"
#include "common/moe_hybrid_ffn_eval.h"

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
    // Stub: MoE hybrid initialization for expert offload
    std::fprintf(stderr, "[glm5next] init_hybrid_model stub\n");
    return false;
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
    std::fprintf(stderr, "[glm5next] generate_impl stub\n");
    GenerateResult result;
    result.status = GenerateStatus::Error;
    result.error_message = "glm5next generation not yet implemented";
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
