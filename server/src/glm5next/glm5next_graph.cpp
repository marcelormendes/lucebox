// GLM5-Next computational graph construction.
// 45 layers: 34 KDA linear attention + 11 NoPE MLA/DSA
// Dense FFN on layers 0-2, MoE 288/top-8 on layers 3+

#include "glm5next_internal.h"
#include "common/moe_hybrid_ffn_eval.h"
#include "common/moe_expert_compute.h"
#include "deepseek4/deepseek4_hc_cuda.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>
#include <vector>

namespace dflash::common {

namespace {

// Simplified mHC pre-processing
// For initial implementation, pass through first HC stream
static ggml_tensor * glm5next_build_hc_pre(
        ggml_context * ctx,
        ggml_tensor * hc_state,      // [n_embd, n_hc, n_tokens]
        ggml_tensor * hc_fn,         // [hc_dim, hc_mix_dim]
        ggml_tensor * hc_scale,      // [3]
        ggml_tensor * hc_base,       // [hc_mix_dim]
        int n_embd,
        int n_hc) {
    
    // Simplified: extract first HC stream and return it
    // Full implementation would do Sinkhorn normalization and mixing
    ggml_tensor * first_stream = ggml_view_2d(ctx, hc_state, n_embd, hc_state->ne[2],
                                              hc_state->nb[2], 0);
    return first_stream;
}

// Simplified mHC post-processing
static ggml_tensor * glm5next_build_hc_post(
        ggml_context * ctx,
        ggml_tensor * hc_state,      // [n_embd, n_hc, n_tokens]
        ggml_tensor * sublayer_out,  // [n_embd, n_tokens]
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int n_embd,
        int n_hc) {
    
    // Simplified: write sublayer output back to first stream
    // Full implementation would do Sinkhorn and combine
    (void)hc_fn; (void)hc_scale; (void)hc_base; (void)n_hc;
    return sublayer_out;
}

// KDA linear attention (simplified version without full state management)
static ggml_tensor * glm5next_build_kda_attn(
        ggml_context * ctx,
        ggml_tensor * cur,           // [n_embd, n_tokens]
        const Glm5NextLayer & layer,
        Glm5NextCache & cache,
        int il,
        int n_tokens,
        int n_embd,
        int n_head,
        int head_dim) {
    
    // Simplified KDA: for now, just pass through with a simple projection
    // Full implementation would do conv1d, state update, f_a/f_b, g_a/g_b gating
    
    // For compilation: just return a transformed version of cur
    ggml_tensor * out = ggml_mul_mat(ctx, layer.attn_wo, cur);
    
    (void)cache; (void)il; (void)n_tokens; (void)n_head; (void)head_dim;
    return out;
}

// MLA attention with simplified DSA (no indexer for initial version)
static ggml_tensor * glm5next_build_mla_attn(
        ggml_context * ctx,
        ggml_tensor * cur,           // [n_embd, n_tokens]
        const Glm5NextLayer & layer,
        Glm5NextCache & cache,
        int il,
        int n_tokens,
        int kv_pos,
        int n_embd,
        int n_head,
        int head_dim) {
    
    // Simplified MLA: low-rank Q projection, single KV head, no RoPE (NoPE)
    // For initial version, skip IndexPool DSA and do simple attention
    
    // Q projection: x → q_a → norm → q_b
    ggml_tensor * q_a = ggml_mul_mat(ctx, layer.attn_q_a, cur);
    q_a = ggml_rms_norm(ctx, q_a, 1e-5f);
    q_a = ggml_mul(ctx, q_a, layer.attn_q_a_norm);
    ggml_tensor * q = ggml_mul_mat(ctx, layer.attn_q_b, q_a);
    
    // KV projection (single head, absorbed form)
    ggml_tensor * k = ggml_mul_mat(ctx, layer.attn_wk_b, q_a);  // Reuse compressed q_a
    ggml_tensor * v = ggml_mul_mat(ctx, layer.attn_wv_b, q_a);
    
    // Reshape for attention: [n_embd, n_tokens] → [head_dim, n_head, n_tokens]
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
    
    // Simple attention (no mask, no KV cache for now)
    // Simplified: just project through output
    ggml_tensor * out = ggml_mul_mat(ctx, layer.attn_wo, cur);
    
    (void)k; (void)v; (void)cache; (void)il; (void)kv_pos;
    return out;
}

// Dense FFN with clamped SwiGLU
static ggml_tensor * glm5next_build_dense_ffn(
        ggml_context * ctx,
        ggml_tensor * cur,
        const Glm5NextLayer & layer,
        float swiglu_clamp) {
    
    ggml_tensor * gate_out = ggml_mul_mat(ctx, layer.ffn_gate, cur);
    ggml_tensor * up_out = ggml_mul_mat(ctx, layer.ffn_up, cur);
    
    // Clamped SwiGLU: gate (-inf, clamp] → SiLU, up [-clamp, clamp]
    gate_out = ggml_clamp(ctx, gate_out, -INFINITY, swiglu_clamp);
    gate_out = ggml_silu(ctx, gate_out);
    up_out = ggml_clamp(ctx, up_out, -swiglu_clamp, swiglu_clamp);
    
    cur = ggml_mul(ctx, gate_out, up_out);
    cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
    
    return cur;
}

// Simplified MoE FFN (without full hybrid routing)
static ggml_tensor * glm5next_build_moe_ffn(
        ggml_context * ctx,
        ggml_tensor * cur,
        const Glm5NextLayer & layer,
        int n_tokens,
        float swiglu_clamp) {
    
    // Simplified: just run shared expert for now
    // Full implementation would do sigmoid routing + top-8 expert selection
    
    ggml_tensor * shared_gate = ggml_mul_mat(ctx, layer.moe_shared_gate, cur);
    ggml_tensor * shared_up = ggml_mul_mat(ctx, layer.moe_shared_up, cur);
    
    shared_gate = ggml_clamp(ctx, shared_gate, -INFINITY, swiglu_clamp);
    shared_gate = ggml_silu(ctx, shared_gate);
    shared_up = ggml_clamp(ctx, shared_up, -swiglu_clamp, swiglu_clamp);
    
    ggml_tensor * shared_out = ggml_mul(ctx, shared_gate, shared_up);
    shared_out = ggml_mul_mat(ctx, layer.moe_shared_down, shared_out);
    
    (void)n_tokens;
    return shared_out;
}

} // anonymous namespace

// Build forward graph for GLM5-Next
// tokens: input token IDs
// n_tokens: number of tokens
// kv_pos: starting KV cache position
// Returns logits tensor
ggml_tensor * glm5next_build_graph(
    ggml_context * ctx,
    const Glm5NextWeights & w,
    Glm5NextCache & cache,
    const int32_t * tokens,
    int n_tokens,
    int kv_pos) {
    
    if (!ctx || !tokens || n_tokens <= 0) {
        std::fprintf(stderr, "[glm5next_graph] invalid arguments\n");
        return nullptr;
    }

    std::fprintf(stderr, "[glm5next_graph] building graph for %d tokens at pos %d\n",
                 n_tokens, kv_pos);

    // Embedding lookup
    ggml_tensor * tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tok_ids, "inp_tokens");
    ggml_set_input(tok_ids);
    
    ggml_tensor * inpL = ggml_get_rows(ctx, w.tok_embd, tok_ids);
    ggml_set_name(inpL, "inp_embd");

    // Initialize mHC state (4 streams)
    // For simplified version, we'll just track a single working tensor
    ggml_tensor * hc_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                 w.n_embd, w.n_hc, n_tokens);
    ggml_set_name(hc_state, "hc_state");
    
    // Replicate input across all HC streams (simplified)
    // Full implementation would initialize properly
    (void)hc_state;

    // Process all 45 layers
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & layer = w.layers[il];
        
        // Check for missing tensors
        if (!layer.attn_norm || !layer.ffn_norm) {
            std::fprintf(stderr, "[glm5next_graph] layer %d missing norm tensors\n", il);
            return nullptr;
        }
        
        // Attention block
        {
            // Attention norm
            ggml_tensor * cur = ggml_rms_norm(ctx, inpL, 1e-6f);
            cur = ggml_mul(ctx, cur, layer.attn_norm);
            ggml_set_name(cur, ("attn_norm_" + std::to_string(il)).c_str());
            
            // mHC pre-attention (simplified)
            if (layer.hc_attn_fn && layer.hc_attn_base && layer.hc_attn_scale) {
                cur = glm5next_build_hc_pre(ctx, hc_state, layer.hc_attn_fn,
                                           layer.hc_attn_scale, layer.hc_attn_base,
                                           w.n_embd, w.n_hc);
            }
            
            // Attention: KDA or MLA based on layer index
            // full_attn_interval=4 means MLA on layers where (il+1) % 4 == 0
            bool is_mla_layer = ((il + 1) % w.full_attn_interval) == 0;
            
            if (is_mla_layer) {
                // MLA sparse attention
                if (!layer.attn_q_a || !layer.attn_wo) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing MLA tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_build_mla_attn(ctx, cur, layer, cache, il, n_tokens,
                                             kv_pos, w.n_embd, w.n_head, w.head_dim);
                ggml_set_name(cur, ("mla_out_" + std::to_string(il)).c_str());
            } else {
                // KDA linear attention
                if (!layer.attn_wo) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing KDA tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_build_kda_attn(ctx, cur, layer, cache, il, n_tokens,
                                             w.n_embd, w.n_head, w.head_dim);
                ggml_set_name(cur, ("kda_out_" + std::to_string(il)).c_str());
            }
            
            // mHC post-attention (simplified)
            if (layer.hc_attn_fn) {
                cur = glm5next_build_hc_post(ctx, hc_state, cur, layer.hc_attn_fn,
                                            layer.hc_attn_scale, layer.hc_attn_base,
                                            w.n_embd, w.n_hc);
            }
            
            // Residual
            inpL = ggml_add(ctx, inpL, cur);
            ggml_set_name(inpL, ("attn_resid_" + std::to_string(il)).c_str());
        }

        // FFN block
        {
            // FFN norm
            ggml_tensor * cur = ggml_rms_norm(ctx, inpL, 1e-6f);
            cur = ggml_mul(ctx, cur, layer.ffn_norm);
            ggml_set_name(cur, ("ffn_norm_" + std::to_string(il)).c_str());
            
            // mHC pre-FFN (simplified)
            if (layer.hc_ffn_fn) {
                cur = glm5next_build_hc_pre(ctx, hc_state, layer.hc_ffn_fn,
                                           layer.hc_ffn_scale, layer.hc_ffn_base,
                                           w.n_embd, w.n_hc);
            }
            
            if (il < w.first_moe_layer) {
                // Dense FFN (layers 0-2)
                if (!layer.ffn_gate || !layer.ffn_up || !layer.ffn_down) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing dense FFN tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_build_dense_ffn(ctx, cur, layer, w.swiglu_clamp);
                ggml_set_name(cur, ("dense_ffn_out_" + std::to_string(il)).c_str());
            } else {
                // MoE: 288 routed experts (top-8) + 1 shared
                if (!layer.moe_shared_gate || !layer.moe_shared_up || !layer.moe_shared_down) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing MoE tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_build_moe_ffn(ctx, cur, layer, n_tokens, w.swiglu_clamp);
                ggml_set_name(cur, ("moe_ffn_out_" + std::to_string(il)).c_str());
            }
            
            // mHC post-FFN (simplified)
            if (layer.hc_ffn_fn) {
                cur = glm5next_build_hc_post(ctx, hc_state, cur, layer.hc_ffn_fn,
                                            layer.hc_ffn_scale, layer.hc_ffn_base,
                                            w.n_embd, w.n_hc);
            }
            
            // Residual
            inpL = ggml_add(ctx, inpL, cur);
            ggml_set_name(inpL, ("ffn_resid_" + std::to_string(il)).c_str());
        }
    }

    // mHC terminal collapse: unweighted mean across n_hc streams
    // For simplified version, hc_state is not used so inpL is already the result
    // Full implementation would average across HC streams

    // Output head
    if (!w.output_norm || !w.output) {
        std::fprintf(stderr, "[glm5next_graph] missing output tensors\n");
        return nullptr;
    }
    
    inpL = ggml_rms_norm(ctx, inpL, 1e-6f);
    inpL = ggml_mul(ctx, inpL, w.output_norm);
    ggml_set_name(inpL, "output_norm");
    
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, inpL);
    ggml_set_name(logits, "logits");

    std::fprintf(stderr, "[glm5next_graph] graph built successfully, %d layers\n", w.n_layer);
    return logits;
}

}  // namespace dflash::common
