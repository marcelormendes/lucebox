// GLM5-Next computational graph construction.
// 45 layers: 34 KDA linear attention + 11 NoPE MLA/DSA
// Dense FFN on layers 0-2, MoE 288/top-8 on layers 3+

#include "glm5next_internal.h"
#include "common/moe_hybrid_ffn_eval.h"

#include "ggml.h"

#include <cstdio>

namespace dflash::common {

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
    ggml_tensor * inpL = ggml_get_rows(ctx, w.tok_embd,
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens));

    // Initialize mHC state (4 streams)
    ggml_tensor * hc_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        w.n_embd, w.n_hc, n_tokens);

    // Replicate input across mHC streams
    for (int h = 0; h < w.n_hc; ++h) {
        // hc_state[..., h, :] = inpL
        // Stub: actual HC initialization
    }

    // Process all 45 layers
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & layer = w.layers[il];
        
        // Attention norm
        ggml_tensor * cur = ggml_rms_norm(ctx, inpL, 1e-6f);
        cur = ggml_mul(ctx, cur, layer.attn_norm);

        // mHC pre-attention
        // Stub: build_hc_pre(attn) reused from deepseek4

        // Attention: KDA or MLA based on layer index
        bool is_sparse_attn = ((il % w.full_attn_interval) == (w.full_attn_interval - 1));
        
        if (is_sparse_attn) {
            // MLA sparse attention with IndexPool DSA
            // Stub: MLA Q/KV projection, NoPE (no RoPE), IndexPool kpool=4
            std::fprintf(stderr, "[glm5next_graph] layer %d: MLA/DSA\n", il);
        } else {
            // KDA linear attention
            // Stub: conv1d, f_a/f_b, g_a/g_b, beta, A, dt
            std::fprintf(stderr, "[glm5next_graph] layer %d: KDA\n", il);
        }

        // mHC post-attention
        // Stub: build_hc_post(attn)

        // FFN
        cur = ggml_rms_norm(ctx, cur, 1e-6f);
        cur = ggml_mul(ctx, cur, layer.ffn_norm);

        // mHC pre-FFN
        // Stub: build_hc_pre(ffn)

        if (il < w.first_moe_layer) {
            // Dense FFN (layers 0-2)
            ggml_tensor * gate_out = ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up_out = ggml_mul_mat(ctx, layer.ffn_up, cur);
            
            // Clamped SwiGLU: gate (-inf, 10] SiLU, up [-10, 10]
            gate_out = ggml_clamp(ctx, gate_out, -INFINITY, w.swiglu_clamp);
            gate_out = ggml_silu(ctx, gate_out);
            up_out = ggml_clamp(ctx, up_out, -w.swiglu_clamp, w.swiglu_clamp);
            
            cur = ggml_mul(ctx, gate_out, up_out);
            cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
            
            std::fprintf(stderr, "[glm5next_graph] layer %d: dense FFN\n", il);
        } else {
            // MoE: 288 routed experts (top-8) + 1 shared
            // Stub: moe_hybrid_ffn_eval with sigmoid + exp_probs_b routing
            std::fprintf(stderr, "[glm5next_graph] layer %d: MoE 288/8\n", il);
            
            // Router logits
            ggml_tensor * router_logits = ggml_mul_mat(ctx, layer.moe_gate, cur);
            
            // Shared expert (always active)
            ggml_tensor * shared_gate = ggml_mul_mat(ctx, layer.moe_shared_gate, cur);
            ggml_tensor * shared_up = ggml_mul_mat(ctx, layer.moe_shared_up, cur);
            shared_gate = ggml_clamp(ctx, shared_gate, -INFINITY, w.swiglu_clamp);
            shared_gate = ggml_silu(ctx, shared_gate);
            shared_up = ggml_clamp(ctx, shared_up, -w.swiglu_clamp, w.swiglu_clamp);
            ggml_tensor * shared_out = ggml_mul(ctx, shared_gate, shared_up);
            shared_out = ggml_mul_mat(ctx, layer.moe_shared_down, shared_out);

            // Routed experts (top-8 of 288)
            // Stub: actual expert routing and evaluation
            cur = shared_out;  // Placeholder
        }

        // mHC post-FFN
        // Stub: build_hc_post(ffn)

        inpL = cur;
    }

    // mHC terminal collapse: unweighted mean (NOT DS4's learned output HC)
    // mean across n_hc streams
    // Stub: actual mean collapse

    // Output head
    inpL = ggml_rms_norm(ctx, inpL, 1e-6f);
    inpL = ggml_mul(ctx, inpL, w.output_norm);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, inpL);

    return logits;
}

}  // namespace dflash::common
