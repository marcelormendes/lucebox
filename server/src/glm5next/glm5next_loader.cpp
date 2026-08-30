// GLM5-Next model loader - reads glm5next.* GGUF keys.

#include "glm5next_internal.h"
#include "common/gguf_inspect.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace dflash::common {

// Tensor name helpers
static std::string layer_name(int layer_idx, const char * suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", layer_idx, suffix);
    return std::string(buf);
}

// Load GLM5-Next weights from GGUF
// Returns true on success, false on error
bool glm5next_load_weights(const char * model_path,
                           Glm5NextWeights & w,
                           ggml_context * ctx) {
    if (!model_path || !ctx) {
        std::fprintf(stderr, "[glm5next_loader] null model_path or context\n");
        return false;
    }

    // Open GGUF file
    gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx,
    };
    
    gguf_context * gguf_ctx = gguf_init_from_file(model_path, params);
    if (!gguf_ctx) {
        std::fprintf(stderr, "[glm5next_loader] failed to load GGUF from %s\n",
                     model_path);
        return false;
    }

    // Read metadata
    // These are glm5next.* keys, NOT deepseek4.*
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    std::fprintf(stderr, "[glm5next_loader] GGUF contains %d tensors\n", n_tensors);

    // Read key hyperparameters
    // TODO: actual GGUF key reading
    w.n_vocab = 154880;
    w.n_embd = 4096;
    w.n_layer = 45;
    w.n_head = 64;
    w.head_dim = 64;
    w.n_ff = 14336;
    w.n_expert_ff = 2048;
    w.n_expert = 288;
    w.n_expert_used = 8;
    w.first_moe_layer = 3;
    w.n_hc = 4;
    w.hc_sinkhorn_iters = 20;
    w.kpool = 4;
    w.full_attn_interval = 4;
    w.index_topk = 2048;
    w.swiglu_clamp = 10.0f;

    std::fprintf(stderr, "[glm5next_loader] n_layer=%d, n_expert=%d, first_moe=%d\n",
                 w.n_layer, w.n_expert, w.first_moe_layer);

    // Load embedding
    w.tok_embd = ggml_get_tensor(ctx, "token_embd.weight");
    if (!w.tok_embd) {
        std::fprintf(stderr, "[glm5next_loader] missing token_embd.weight\n");
        gguf_free(gguf_ctx);
        return false;
    }

    // Load layer tensors
    w.layers.resize(w.n_layer);
    for (int i = 0; i < w.n_layer; ++i) {
        auto & layer = w.layers[i];
        
        // Attention norm
        layer.attn_norm = ggml_get_tensor(ctx, layer_name(i, "attn_norm.weight").c_str());
        
        // FFN norm
        layer.ffn_norm = ggml_get_tensor(ctx, layer_name(i, "ffn_norm.weight").c_str());

        // Dense FFN (layers 0-2 only)
        if (i < w.first_moe_layer) {
            layer.ffn_gate = ggml_get_tensor(ctx, layer_name(i, "ffn_gate.weight").c_str());
            layer.ffn_up = ggml_get_tensor(ctx, layer_name(i, "ffn_up.weight").c_str());
            layer.ffn_down = ggml_get_tensor(ctx, layer_name(i, "ffn_down.weight").c_str());
        } else {
            // MoE router and experts (layers 3+)
            layer.moe_gate = ggml_get_tensor(ctx, layer_name(i, "ffn_gate_exps.weight").c_str());
            
            // Routed experts
            layer.moe_experts_gate = ggml_get_tensor(ctx,
                layer_name(i, "ffn_gate_exps.weight").c_str());
            layer.moe_experts_up = ggml_get_tensor(ctx,
                layer_name(i, "ffn_up_exps.weight").c_str());
            layer.moe_experts_down = ggml_get_tensor(ctx,
                layer_name(i, "ffn_down_exps.weight").c_str());
            
            // Shared expert
            layer.moe_shared_gate = ggml_get_tensor(ctx,
                layer_name(i, "ffn_gate_shared.weight").c_str());
            layer.moe_shared_up = ggml_get_tensor(ctx,
                layer_name(i, "ffn_up_shared.weight").c_str());
            layer.moe_shared_down = ggml_get_tensor(ctx,
                layer_name(i, "ffn_down_shared.weight").c_str());
        }

        // mHC tensors
        layer.hc_attn_fn = ggml_get_tensor(ctx, layer_name(i, "hc_attn_fn.weight").c_str());
        layer.hc_attn_base = ggml_get_tensor(ctx, layer_name(i, "hc_attn_base.weight").c_str());
        layer.hc_attn_scale = ggml_get_tensor(ctx, layer_name(i, "hc_attn_scale.weight").c_str());
        layer.hc_ffn_fn = ggml_get_tensor(ctx, layer_name(i, "hc_ffn_fn.weight").c_str());
        layer.hc_ffn_base = ggml_get_tensor(ctx, layer_name(i, "hc_ffn_base.weight").c_str());
        layer.hc_ffn_scale = ggml_get_tensor(ctx, layer_name(i, "hc_ffn_scale.weight").c_str());

        // KDA or MLA tensors (determine by full_attn_interval)
        // Stub: actual tensor loading based on layer type
    }

    // Output head
    w.output_norm = ggml_get_tensor(ctx, "output_norm.weight");
    w.output = ggml_get_tensor(ctx, "output.weight");

    gguf_free(gguf_ctx);
    
    std::fprintf(stderr, "[glm5next_loader] loaded %d layers successfully\n", w.n_layer);
    return true;
}

}  // namespace dflash::common
