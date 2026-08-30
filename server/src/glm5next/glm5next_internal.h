// GLM-5.3-Flash (GLM5-Next) target structs for dflash daemon.
//
// Architecture summary:
//   - 45 layers: 34 KDA linear attention + 11 NoPE sparse MLA/DSA
//   - KDA: linear attention with conv1d, f_a/f_b, g_a/g_b, beta, A_log, dt
//   - MLA: NoPE (qk_rope_head_dim=0), IndexPool DSA with kpool=4
//   - mHC: 4 parallel residual streams, Sinkhorn-normalized, UNWEIGHTED MEAN collapse
//   - MoE: layers 0-2 DENSE FFN, layers 3+ have 288 routed experts (top-8) + 1 shared
//   - Expert FFN: clamped SwiGLU (gate (-inf,10] SiLU, up [-10,10])
//   - Hidden: 4096, vocab: 154880, expert ff: 2048
//   - full_attn_interval=4, so DSA on layers 3,7,11,...

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "internal.h"
#include "common/layer_split_utils.h"

namespace dflash::common {

// GLM5-Next architecture constants
inline constexpr int GLM5NEXT_N_LAYER = 45;
inline constexpr int GLM5NEXT_FIRST_MOE_LAYER = 3;  // Layers 0-2 are dense FFN
inline constexpr int GLM5NEXT_N_EXPERT = 288;
inline constexpr int GLM5NEXT_N_EXPERT_USED = 8;
inline constexpr int GLM5NEXT_KPOOL = 4;
inline constexpr int GLM5NEXT_FULL_ATTN_INTERVAL = 4;
inline constexpr int GLM5NEXT_N_HC = 4;  // mHC streams
inline constexpr int GLM5NEXT_HC_SINKHORN_ITERS = 20;

// ─── Per-layer tensor pointers ──────────────────────────────────────────

struct Glm5NextLayer {
    // ── Attention norm ────────────────────────────────────────────────
    ggml_tensor * attn_norm = nullptr;  // [n_embd]

    // ── KDA Linear Attention (34 layers) ─────────────────────────────
    // conv1d projections
    ggml_tensor * kda_conv1d_q = nullptr;  // [n_embd, kda_conv1d_dim]
    ggml_tensor * kda_conv1d_k = nullptr;  // [n_embd, kda_conv1d_dim]
    ggml_tensor * kda_conv1d_v = nullptr;  // [n_embd, kda_conv1d_dim]
    
    // Feature projections
    ggml_tensor * kda_f_a = nullptr;  // forward a
    ggml_tensor * kda_f_b = nullptr;  // forward b
    ggml_tensor * kda_g_a = nullptr;  // gate a
    ggml_tensor * kda_g_b = nullptr;  // gate b
    
    // State-space parameters
    ggml_tensor * kda_beta = nullptr;      // decay parameter
    ggml_tensor * kda_a_log = nullptr;     // A matrix (log-space)
    ggml_tensor * kda_dt_bias = nullptr;   // delta time bias

    // ── MLA Sparse Attention (11 layers) ─────────────────────────────
    // Q low-rank path: x → q_a → norm → q_b → heads (NoPE, no RoPE)
    ggml_tensor * attn_q_a = nullptr;       // [n_embd, n_lora_q]
    ggml_tensor * attn_q_a_norm = nullptr;  // [n_lora_q]
    ggml_tensor * attn_q_b = nullptr;       // [n_lora_q, n_head * head_dim]

    // KV path: single head, absorbed into output
    ggml_tensor * attn_wk_b = nullptr;  // [n_embd, head_dim]
    ggml_tensor * attn_wv_b = nullptr;  // [n_embd, head_dim]
    ggml_tensor * attn_wo = nullptr;    // output projection

    // ── DSA Indexer (IndexPool, on sparse attention layers) ─────────
    ggml_tensor * indexer_compressor_ape = nullptr;   // position encoding
    ggml_tensor * indexer_compressor_gate = nullptr;  // gating weight

    // ── Hierarchical Controller (mHC) ────────────────────────────────
    // Attention HC
    ggml_tensor * hc_attn_fn = nullptr;      // [n_hc, n_embd]
    ggml_tensor * hc_attn_base = nullptr;    // [n_hc, n_embd]
    ggml_tensor * hc_attn_scale = nullptr;   // [n_hc]

    // FFN HC
    ggml_tensor * hc_ffn_fn = nullptr;       // [n_hc, n_embd]
    ggml_tensor * hc_ffn_base = nullptr;     // [n_hc, n_embd]
    ggml_tensor * hc_ffn_scale = nullptr;    // [n_hc]

    // ── FFN norm ─────────────────────────────────────────────────────
    ggml_tensor * ffn_norm = nullptr;  // [n_embd]

    // ── Dense FFN (layers 0-2 only) ──────────────────────────────────
    ggml_tensor * ffn_gate = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_up = nullptr;    // [n_embd, n_ff]
    ggml_tensor * ffn_down = nullptr;  // [n_ff, n_embd]

    // ── MoE Router (layers 3+ only) ──────────────────────────────────
    ggml_tensor * moe_gate = nullptr;           // [n_embd, n_expert]
    ggml_tensor * moe_exp_probs_b = nullptr;    // expert probability bias

    // ── Routed Experts (288 experts, layers 3+) ──────────────────────
    ggml_tensor * moe_experts_gate = nullptr;  // [n_expert, n_embd, n_ff]
    ggml_tensor * moe_experts_up = nullptr;    // [n_expert, n_embd, n_ff]
    ggml_tensor * moe_experts_down = nullptr;  // [n_expert, n_ff, n_embd]

    // ── Shared Expert (layers 3+) ────────────────────────────────────
    ggml_tensor * moe_shared_gate = nullptr;   // [n_embd, n_ff]
    ggml_tensor * moe_shared_up = nullptr;     // [n_embd, n_ff]
    ggml_tensor * moe_shared_down = nullptr;   // [n_ff, n_embd]
};

// ─── Model-level tensors and metadata ───────────────────────────────────

struct Glm5NextWeights {
    // Embedding
    ggml_tensor * tok_embd = nullptr;  // [vocab, n_embd]

    // Layers
    std::vector<Glm5NextLayer> layers;  // 45 layers

    // Output head
    ggml_tensor * output_norm = nullptr;  // [n_embd]
    ggml_tensor * output = nullptr;       // [n_embd, vocab]

    // MTP/NextN draft head (optional, text-only v1)
    ggml_tensor * mtp_head = nullptr;  // if present

    // Metadata
    int32_t n_vocab = 154880;
    int32_t n_embd = 4096;
    int32_t n_layer = 45;
    int32_t n_head = 64;
    int32_t head_dim = 64;
    int32_t n_ff = 14336;        // dense FFN hidden
    int32_t n_expert_ff = 2048;  // expert FFN hidden
    int32_t n_expert = 288;
    int32_t n_expert_used = 8;
    int32_t first_moe_layer = 3;
    int32_t n_hc = 4;
    int32_t hc_sinkhorn_iters = 20;
    int32_t kpool = 4;
    int32_t full_attn_interval = 4;
    int32_t index_topk = 2048;
    float swiglu_clamp = 10.0f;

    // KDA parameters (for 34 linear attention layers)
    int32_t kda_conv1d_dim = 0;
    int32_t kda_state_dim = 0;
    
    // MLA parameters (for 11 sparse attention layers)
    int32_t n_lora_q = 0;
    int32_t qk_rope_head_dim = 0;  // 0 = NoPE

    ggml_type tok_embd_type = GGML_TYPE_F16;
};

struct Glm5NextCache {
    // KV cache for MLA layers (11 sparse attention layers)
    ggml_tensor * k = nullptr;  // [n_ctx, n_kv_head * head_dim, n_mla_layer]
    ggml_tensor * v = nullptr;  // [n_ctx, n_kv_head * head_dim, n_mla_layer]

    // KDA state cache for linear attention layers (34 KDA layers)
    ggml_tensor * kda_state = nullptr;  // [n_kda_layer, state_dim, n_embd]

    // Hierarchical Controller state (all 45 layers)
    ggml_tensor * hc_state = nullptr;  // [n_layer, n_hc, n_embd]

    // DSA indexer cache (for IndexPool on sparse layers)
    ggml_tensor * indexer_pool = nullptr;  // pooled keys/gates

    int n_ctx = 0;
    int cur_pos = 0;
    int n_past = 0;
};

struct Glm5NextBackendConfig {
    const char * target_path = nullptr;
    PlacementDevice device;
    int chunk = 512;
    int max_ctx = 0;  // auto-fit
};

// Loader function - reads glm5next.* GGUF keys and registers ROCmFP mix qtypes
bool glm5next_load_weights(const char * model_path,
                           Glm5NextWeights & w,
                           ggml_context * ctx,
                           std::vector<const void *> & registered_mix_bases);

// Forward declarations for MoE
struct MoeHybridStorage;

// Graph builder - constructs forward pass computation graph
ggml_tensor * glm5next_build_graph(
    ggml_context * ctx,
    const Glm5NextWeights & w,
    Glm5NextCache & cache,
    const int32_t * tokens,
    int n_tokens,
    int kv_pos,
    MoeHybridStorage * moe_storage = nullptr);

}  // namespace dflash::common
