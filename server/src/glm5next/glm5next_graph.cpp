// GLM5-Next computational graph construction - ported from llama.cpp PR #27752
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
#include <algorithm>
#include <vector>

namespace dflash::common {

namespace {

// ============================================================================
// mHC (Hierarchical Controller) Operations
// ============================================================================

// View helpers for slicing tensors
static ggml_tensor * glm5next_view_1d(ggml_context * ctx, ggml_tensor * t, 
                                      int64_t ne0, int64_t offset) {
    return ggml_view_1d(ctx, t, ne0, offset * ggml_element_size(t));
}

static ggml_tensor * glm5next_view_2d(ggml_context * ctx, ggml_tensor * t,
                                      int64_t ne0, int64_t ne1, int64_t offset) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], offset * ggml_element_size(t));
}

// Affine transform: x * scale + base
static ggml_tensor * glm5next_hc_affine(ggml_context * ctx,
                                        ggml_tensor * x,
                                        ggml_tensor * scale,
                                        ggml_tensor * base) {
    return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
}

// Unweighted mean over HC streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * glm5next_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];
    
    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        ggml_tensor * stream = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], 
                                           x->nb[2], s * x->nb[1]);
        acc = ggml_add(ctx, acc, stream);
    }
    
    return ggml_scale(ctx, acc, 1.0f / hc);
}

// Sinkhorn normalization on combine matrix [dst_hc, src_hc, n_tokens]
static ggml_tensor * glm5next_hc_sinkhorn(ggml_context * ctx, ggml_tensor * comb,
                                          int n_iters, float eps) {
    // Apply softmax on src_hc axis (ne[0])
    comb = ggml_soft_max(ctx, comb);
    comb = ggml_add(ctx, comb, ggml_new_f32(ctx, eps));
    
    // Iterative row/column normalization
    for (int i = 0; i < n_iters; ++i) {
        // Normalize columns (over dst_hc, which is ne[1])
        ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
        ggml_tensor * sum = ggml_sum_rows(ctx, t);
        sum = ggml_add(ctx, sum, ggml_new_f32(ctx, eps));
        sum = ggml_permute(ctx, sum, 1, 0, 2, 3);
        comb = ggml_div(ctx, comb, sum);
        
        if (i < n_iters - 1) {
            // Normalize rows (over src_hc, which is ne[0])
            sum = ggml_sum_rows(ctx, comb);
            sum = ggml_add(ctx, sum, ggml_new_f32(ctx, eps));
            comb = ggml_div(ctx, comb, sum);
        }
    }
    
    return comb;
}

// HC pre-processing: extract working vector and compute post gates + combine matrix
static ggml_tensor * glm5next_hc_pre(ggml_context * ctx,
                                     ggml_tensor * hc_state,    // [n_embd, n_hc, n_tokens]
                                     ggml_tensor * hc_fn,       // [hc_dim, hc_mix_dim]
                                     ggml_tensor * hc_scale,    // [3]
                                     ggml_tensor * hc_base,     // [hc_mix_dim]
                                     int n_embd, int n_hc,
                                     int sinkhorn_iters, float hc_eps,
                                     ggml_tensor ** out_post,
                                     ggml_tensor ** out_comb) {
    const int64_t hc_dim = n_hc * n_embd;
    const int64_t hc_mix_dim = (2 + n_hc) * n_hc;
    const int64_t nt = hc_state->ne[2];
    
    // DeepseekV4UnweightedRMSNorm: no learned gain
    ggml_tensor * flat = ggml_reshape_2d(ctx, hc_state, hc_dim, nt);
    flat = ggml_rms_norm(ctx, flat, 1e-6f);
    
    // Mix projection
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat);
    
    // Extract scales and bases
    ggml_tensor * scale_pre = glm5next_view_1d(ctx, hc_scale, 1, 0);
    ggml_tensor * scale_post = glm5next_view_1d(ctx, hc_scale, 1, 1);
    ggml_tensor * scale_comb = glm5next_view_1d(ctx, hc_scale, 1, 2);
    
    ggml_tensor * base_pre = glm5next_view_1d(ctx, hc_base, n_hc, 0);
    ggml_tensor * base_post = glm5next_view_1d(ctx, hc_base, n_hc, n_hc);
    ggml_tensor * base_comb = glm5next_view_1d(ctx, hc_base, n_hc * n_hc, 2 * n_hc);
    
    // Pre gates: sigmoid(pre * scale + base) + eps
    ggml_tensor * pre = glm5next_view_2d(ctx, mixes, n_hc, nt, 0);
    pre = glm5next_hc_affine(ctx, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_add(ctx, pre, ggml_new_f32(ctx, hc_eps));
    
    // Post gates: sigmoid(post * scale + base) * 2
    *out_post = glm5next_view_2d(ctx, mixes, n_hc, nt, n_hc);
    *out_post = glm5next_hc_affine(ctx, *out_post, scale_post, base_post);
    *out_post = ggml_sigmoid(ctx, *out_post);
    *out_post = ggml_scale(ctx, *out_post, 2.0f);
    
    // Combine matrix: Sinkhorn-normalized
    *out_comb = glm5next_view_2d(ctx, mixes, n_hc * n_hc, nt, 2 * n_hc);
    *out_comb = glm5next_hc_affine(ctx, *out_comb, scale_comb, base_comb);
    *out_comb = ggml_reshape_3d(ctx, *out_comb, n_hc, n_hc, nt);
    *out_comb = glm5next_hc_sinkhorn(ctx, *out_comb, sinkhorn_iters, hc_eps);
    
    // Collapse: sum_h pre[h] * streams[h]
    ggml_tensor * result = nullptr;
    for (int64_t h = 0; h < n_hc; ++h) {
        ggml_tensor * stream_h = ggml_view_2d(ctx, hc_state, n_embd, nt,
                                             hc_state->nb[2], h * hc_state->nb[1]);
        ggml_tensor * pre_h = ggml_view_2d(ctx, pre, 1, nt, pre->nb[1], h * pre->nb[0]);
        ggml_tensor * weighted = ggml_mul(ctx, stream_h, pre_h);
        result = result ? ggml_add(ctx, result, weighted) : weighted;
    }
    
    return result;
}

// HC post-processing: out = post * sublayer_out + comb @ streams
static ggml_tensor * glm5next_hc_post(ggml_context * ctx,
                                      ggml_tensor * sublayer_out,   // [n_embd, n_tokens]
                                      ggml_tensor * hc_state,       // [n_embd, n_hc, n_tokens]
                                      ggml_tensor * post,           // [n_hc, n_tokens]
                                      ggml_tensor * comb,           // [n_hc, n_hc, n_tokens]
                                      int n_embd, int n_hc) {
    const int64_t nt = sublayer_out->ne[1];
    
    ggml_tensor * out = nullptr;
    
    for (int64_t dst = 0; dst < n_hc; ++dst) {
        // post[dst] * sublayer_out
        ggml_tensor * post_dst = ggml_view_2d(ctx, post, 1, nt, post->nb[1], dst * post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx, sublayer_out, post_dst);
        
        // + sum_src comb[dst, src] * streams[src]
        for (int64_t src = 0; src < n_hc; ++src) {
            ggml_tensor * stream_src = ggml_view_2d(ctx, hc_state, n_embd, nt,
                                                   hc_state->nb[2], src * hc_state->nb[1]);
            ggml_tensor * comb_ds = ggml_view_2d(ctx, comb, 1, nt, comb->nb[2],
                                                dst * comb->nb[0] + src * comb->nb[1]);
            cur = ggml_add(ctx, cur, ggml_mul(ctx, stream_src, comb_ds));
        }
        
        cur = ggml_reshape_3d(ctx, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx, out, cur, 1) : cur;
    }
    
    return out;
}

// ============================================================================
// KDA Linear Attention (Simplified - Full implementation requires state cache)
// ============================================================================

static ggml_tensor * glm5next_kda_attention(ggml_context * ctx,
                                           ggml_tensor * cur,
                                           const Glm5NextLayer & layer,
                                           int n_embd, int n_head, int head_dim) {
    // Simplified KDA for initial implementation
    // Full version needs: conv1d state, f_a/f_b/g_a/g_b gating, A_log decay, dt_bias
    // For now, just pass through output projection
    
    // TODO: Implement conv1d_q/k/v, state update, gating
    ggml_tensor * out = ggml_mul_mat(ctx, layer.attn_wo, cur);
    
    return out;
}

// ============================================================================
// MLA Attention (Simplified - Full implementation requires KV cache + IndexPool)
// ============================================================================

static ggml_tensor * glm5next_mla_attention(ggml_context * ctx,
                                           ggml_tensor * cur,
                                           const Glm5NextLayer & layer,
                                           int n_tokens, int n_embd,
                                           int n_head, int head_dim) {
    // MLA with low-rank Q projection, NoPE (no RoPE)
    
    // Q projection: x → q_a → norm → q_b
    ggml_tensor * q_a = ggml_mul_mat(ctx, layer.attn_q_a, cur);
    q_a = ggml_rms_norm(ctx, q_a, 1e-5f);
    q_a = ggml_mul(ctx, q_a, layer.attn_q_a_norm);
    
    ggml_tensor * q = ggml_mul_mat(ctx, layer.attn_q_b, q_a);
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
    
    // KV projection (absorbed form, single KV head)
    // Simplified: skip actual attention for now, just project through output
    // TODO: Implement proper Q/K/V attention, KV cache, IndexPool DSA
    
    ggml_tensor * out = ggml_mul_mat(ctx, layer.attn_wo, cur);
    
    return out;
}

// ============================================================================
// FFN Operations
// ============================================================================

// Dense FFN with clamped SwiGLU
static ggml_tensor * glm5next_dense_ffn(ggml_context * ctx,
                                       ggml_tensor * cur,
                                       const Glm5NextLayer & layer,
                                       float swiglu_clamp) {
    ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
    ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);
    
    // Clamped SwiGLU: gate (-inf, clamp] → SiLU, up [-clamp, clamp]
    gate = ggml_clamp(ctx, gate, -INFINITY, swiglu_clamp);
    gate = ggml_silu(ctx, gate);
    up = ggml_clamp(ctx, up, -swiglu_clamp, swiglu_clamp);
    
    ggml_tensor * gated = ggml_mul(ctx, gate, up);
    ggml_tensor * out = ggml_mul_mat(ctx, layer.ffn_down, gated);
    
    return out;
}

// MoE FFN (simplified - shared expert only for now)
static ggml_tensor * glm5next_moe_ffn(ggml_context * ctx,
                                     ggml_tensor * cur,
                                     const Glm5NextLayer & layer,
                                     float swiglu_clamp) {
    // Simplified: only run shared expert
    // TODO: Implement sigmoid routing, top-8 expert selection, moe_hybrid_ffn_eval
    
    ggml_tensor * gate = ggml_mul_mat(ctx, layer.moe_shared_gate, cur);
    ggml_tensor * up = ggml_mul_mat(ctx, layer.moe_shared_up, cur);
    
    gate = ggml_clamp(ctx, gate, -INFINITY, swiglu_clamp);
    gate = ggml_silu(ctx, gate);
    up = ggml_clamp(ctx, up, -swiglu_clamp, swiglu_clamp);
    
    ggml_tensor * gated = ggml_mul(ctx, gate, up);
    ggml_tensor * out = ggml_mul_mat(ctx, layer.moe_shared_down, gated);
    
    return out;
}

} // anonymous namespace

// ============================================================================
// Main Graph Builder
// ============================================================================

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

    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int n_head = w.n_head;
    const int head_dim = w.head_dim;
    const int sinkhorn_iters = w.hc_sinkhorn_iters;
    const float hc_eps = 1e-6f;

    // Input token IDs
    ggml_tensor * tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tok_ids, "inp_tokens");
    ggml_set_input(tok_ids);
    
    // Embedding lookup
    ggml_tensor * inpL = ggml_get_rows(ctx, w.tok_embd, tok_ids);
    ggml_set_name(inpL, "inp_embd");

    // Initialize mHC state: [n_embd, hc, n_tokens]
    // Replicate input across all HC streams
    ggml_tensor * hc_state = ggml_reshape_3d(ctx, inpL, n_embd, 1, n_tokens);
    hc_state = ggml_repeat(ctx, hc_state, n_embd, n_hc, n_tokens, 1);
    ggml_set_name(hc_state, "hc_init");

    // Process all 45 layers
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & layer = w.layers[il];
        
        if (!layer.attn_norm || !layer.ffn_norm) {
            std::fprintf(stderr, "[glm5next_graph] layer %d missing norm tensors\n", il);
            return nullptr;
        }
        
        // ---- Attention Block ----
        {
            ggml_tensor * residual = hc_state;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;
            
            // mHC pre-attention
            ggml_tensor * cur = nullptr;
            if (layer.hc_attn_fn && layer.hc_attn_base && layer.hc_attn_scale) {
                cur = glm5next_hc_pre(ctx, hc_state, layer.hc_attn_fn,
                                     layer.hc_attn_scale, layer.hc_attn_base,
                                     n_embd, n_hc, sinkhorn_iters, hc_eps,
                                     &post, &comb);
                ggml_set_name(cur, ("hc_attn_pre_" + std::to_string(il)).c_str());
            } else {
                // Fallback: collapse to first stream
                cur = ggml_view_2d(ctx, hc_state, n_embd, n_tokens, hc_state->nb[2], 0);
            }
            
            // Attention norm
            cur = ggml_rms_norm(ctx, cur, 1e-6f);
            cur = ggml_mul(ctx, cur, layer.attn_norm);
            ggml_set_name(cur, ("attn_norm_" + std::to_string(il)).c_str());
            
            // Attention: KDA or MLA
            bool is_mla_layer = ((il + 1) % w.full_attn_interval) == 0;
            
            if (is_mla_layer) {
                // MLA sparse attention
                if (!layer.attn_q_a || !layer.attn_wo) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing MLA tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_mla_attention(ctx, cur, layer, n_tokens,
                                            n_embd, n_head, head_dim);
                ggml_set_name(cur, ("mla_out_" + std::to_string(il)).c_str());
            } else {
                // KDA linear attention
                if (!layer.attn_wo) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing KDA tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_kda_attention(ctx, cur, layer, n_embd, n_head, head_dim);
                ggml_set_name(cur, ("kda_out_" + std::to_string(il)).c_str());
            }
            
            // mHC post-attention
            if (post && comb) {
                hc_state = glm5next_hc_post(ctx, cur, residual, post, comb, n_embd, n_hc);
            } else {
                // Fallback: add to first stream
                ggml_tensor * first_stream = ggml_view_2d(ctx, residual, n_embd, n_tokens,
                                                         residual->nb[2], 0);
                first_stream = ggml_add(ctx, first_stream, cur);
                hc_state = ggml_reshape_3d(ctx, first_stream, n_embd, 1, n_tokens);
                hc_state = ggml_repeat(ctx, hc_state, n_embd, n_hc, n_tokens, 1);
            }
            ggml_set_name(hc_state, ("hc_attn_post_" + std::to_string(il)).c_str());
        }

        // ---- FFN Block ----
        {
            ggml_tensor * residual = hc_state;
            ggml_tensor * post = nullptr;
            ggml_tensor * comb = nullptr;
            
            // mHC pre-FFN
            ggml_tensor * cur = nullptr;
            if (layer.hc_ffn_fn && layer.hc_ffn_base && layer.hc_ffn_scale) {
                cur = glm5next_hc_pre(ctx, hc_state, layer.hc_ffn_fn,
                                     layer.hc_ffn_scale, layer.hc_ffn_base,
                                     n_embd, n_hc, sinkhorn_iters, hc_eps,
                                     &post, &comb);
                ggml_set_name(cur, ("hc_ffn_pre_" + std::to_string(il)).c_str());
            } else {
                cur = ggml_view_2d(ctx, hc_state, n_embd, n_tokens, hc_state->nb[2], 0);
            }
            
            // FFN norm
            cur = ggml_rms_norm(ctx, cur, 1e-6f);
            cur = ggml_mul(ctx, cur, layer.ffn_norm);
            ggml_set_name(cur, ("ffn_norm_" + std::to_string(il)).c_str());
            
            // FFN: Dense (layers 0-2) or MoE (layers 3+)
            if (il < w.first_moe_layer) {
                // Dense FFN
                if (!layer.ffn_gate || !layer.ffn_up || !layer.ffn_down) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing dense FFN tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_dense_ffn(ctx, cur, layer, w.swiglu_clamp);
                ggml_set_name(cur, ("dense_ffn_out_" + std::to_string(il)).c_str());
            } else {
                // MoE FFN
                if (!layer.moe_shared_gate || !layer.moe_shared_up || !layer.moe_shared_down) {
                    std::fprintf(stderr, "[glm5next_graph] layer %d missing MoE tensors\n", il);
                    return nullptr;
                }
                cur = glm5next_moe_ffn(ctx, cur, layer, w.swiglu_clamp);
                ggml_set_name(cur, ("moe_ffn_out_" + std::to_string(il)).c_str());
            }
            
            // mHC post-FFN
            if (post && comb) {
                hc_state = glm5next_hc_post(ctx, cur, residual, post, comb, n_embd, n_hc);
            } else {
                ggml_tensor * first_stream = ggml_view_2d(ctx, residual, n_embd, n_tokens,
                                                         residual->nb[2], 0);
                first_stream = ggml_add(ctx, first_stream, cur);
                hc_state = ggml_reshape_3d(ctx, first_stream, n_embd, 1, n_tokens);
                hc_state = ggml_repeat(ctx, hc_state, n_embd, n_hc, n_tokens, 1);
            }
            ggml_set_name(hc_state, ("l_out_" + std::to_string(il)).c_str());
        }
    }

    // mHC terminal collapse: unweighted mean across streams
    ggml_tensor * cur = glm5next_hc_mean(ctx, hc_state);
    ggml_set_name(cur, "hc_head");

    // Output head
    if (!w.output_norm || !w.output) {
        std::fprintf(stderr, "[glm5next_graph] missing output tensors\n");
        return nullptr;
    }
    
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, w.output_norm);
    ggml_set_name(cur, "output_norm");
    
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur);
    ggml_set_name(logits, "logits");

    return logits;
}

}  // namespace dflash::common
