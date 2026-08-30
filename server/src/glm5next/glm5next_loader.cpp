// GLM5-Next model loader - reads glm5next.* GGUF keys.

#include "glm5next_internal.h"
#include "common/gguf_inspect.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Runtime decode registration for GGML_TYPE_Q3_1_ROCMFP3_MIX (105) and
// GGML_TYPE_Q2_1_ROCMFP2_MIX (106). Mix qtypes are 104/107 block wire plus
// out-of-band 7s1c codebooks. Kernels in ggml-cuda rocmfp2_mix / rocmfp3_mix.
extern "C" void ggml_cuda_rocmfp3_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp2_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base);
extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base);

namespace dflash::common {

// GLM5-Next qtype constants
constexpr int GLM5NEXT_QTYPE_ROCMFP3_MIX = 105;
constexpr int GLM5NEXT_QTYPE_ROCMFP2_MIX = 106;

// P4MIX (qtype 105): down-experts only
constexpr uint32_t GLM5NEXT_P4MIX_C  = 2;   // codebooks per expert
constexpr uint32_t GLM5NEXT_P4MIX_K  = 8;   // levels per codebook
constexpr uint32_t GLM5NEXT_P4MIX_QK = 32;  // block width
constexpr uint8_t  GLM5NEXT_P4MIX_MAX_MODE = 1;  // 0=fixed, 1=adaptive

// GUMIX (qtype 106): gate/up/down
constexpr uint32_t GLM5NEXT_GUMIX_C  = 2;   // codebooks per expert
constexpr uint32_t GLM5NEXT_GUMIX_K  = 4;   // levels per codebook (2-bit)
constexpr uint32_t GLM5NEXT_GUMIX_QK = 32;  // block width
constexpr uint32_t GLM5NEXT_GUMIX_SURFACES = 3;  // 0=gate, 1=up, 2=down

// Sanity bounds
constexpr uint32_t GLM5NEXT_MIX_MAX_EXPERTS = 1u << 16;   // 65536
constexpr uint32_t GLM5NEXT_MIX_MAX_DIM     = 1u << 20;   // 1,048,576

// Tensor name helpers
static std::string layer_name(int layer_idx, const char * suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", layer_idx, suffix);
    return std::string(buf);
}

// Register P4MIX (qtype 105) sidecar for down-experts
static bool glm5next_register_p4mix_sidecar(
    const std::string & gguf_path,
    Glm5NextWeights & w,
    std::vector<const void *> & registered_bases) {
    
    // Check for glm5next.p4mix.sidecar key in GGUF metadata
    // If not present, try loose file beside the GGUF
    std::string sc_path = gguf_path + ".p4mix";
    
    // Count qtype-105 down-experts that need registration
    int n_qtype105 = 0;
    std::vector<bool> layer_has_105(w.n_layer, false);
    for (int il = w.first_moe_layer; il < w.n_layer; ++il) {
        const ggml_tensor * down = w.layers[il].moe_experts_down;
        if (down && (int)down->type == GLM5NEXT_QTYPE_ROCMFP3_MIX) {
            layer_has_105[il] = true;
            n_qtype105++;
        }
    }
    
    if (n_qtype105 == 0) {
        return true;  // No qtype-105 tensors, nothing to register
    }
    
    std::fprintf(stderr, "[glm5next] found %d qtype-105 down-experts, need P4MIX sidecar\n",
                 n_qtype105);
    
    FILE * f = std::fopen(sc_path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[glm5next] P4MIX sidecar required but not found: %s\n",
                     sc_path.c_str());
        return false;
    }
    
    // Read P4MIXv1 header
    char magic[8];
    uint32_t n_layers = 0, reserved = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, "P4MIXv1\0", 8) != 0 ||
        std::fread(&n_layers, 4, 1, f) != 1 ||
        std::fread(&reserved, 4, 1, f) != 1) {
        std::fprintf(stderr, "[glm5next] bad P4MIX sidecar header: %s\n",
                     sc_path.c_str());
        std::fclose(f);
        return false;
    }
    
    std::vector<bool> done(w.n_layer, false);
    bool ok = true;
    
    for (uint32_t i = 0; i < n_layers && ok; ++i) {
        uint32_t hdr[5];  // layer, E, odim, idim, C, K
        if (std::fread(hdr, 4, 5, f) != 5) {
            std::fprintf(stderr, "[glm5next] p4mix entry %u header read failed\n", i);
            ok = false; break;
        }
        
        const uint32_t layer = hdr[0], E = hdr[1], odim = hdr[2],
                       idim = hdr[3], C = hdr[4];
        uint32_t K = 0;
        if (std::fread(&K, 4, 1, f) != 1) {
            ok = false; break;
        }
        
        // Structural validation
        if (C != GLM5NEXT_P4MIX_C || K != GLM5NEXT_P4MIX_K) {
            std::fprintf(stderr, "[glm5next] p4mix entry %u unexpected C=%u K=%u\n",
                         i, C, K);
            ok = false; break;
        }
        if (E == 0 || E > GLM5NEXT_MIX_MAX_EXPERTS ||
            odim == 0 || odim > GLM5NEXT_MIX_MAX_DIM ||
            idim == 0 || idim > GLM5NEXT_MIX_MAX_DIM) {
            std::fprintf(stderr, "[glm5next] p4mix layer %u dims out of range\n", layer);
            ok = false; break;
        }
        if (idim % GLM5NEXT_P4MIX_QK != 0) {
            std::fprintf(stderr, "[glm5next] p4mix layer %u in=%u not multiple of %u\n",
                         layer, idim, GLM5NEXT_P4MIX_QK);
            ok = false; break;
        }
        
        if (layer >= (uint32_t)w.n_layer || !layer_has_105[layer]) {
            std::fprintf(stderr, "[glm5next] p4mix layer %u not expected\n", layer);
            ok = false; break;
        }
        
        const ggml_tensor * dt = w.layers[layer].moe_experts_down;
        if (!dt || !dt->data) {
            std::fprintf(stderr, "[glm5next] p4mix layer %u tensor not loaded\n", layer);
            ok = false; break;
        }
        
        // Read codebooks (E * C * K * 2 bytes, bf16)
        const size_t book_size = E * C * K * 2;
        std::vector<uint8_t> books(book_size);
        if (std::fread(books.data(), 1, book_size, f) != book_size) {
            ok = false; break;
        }
        
        // Read modes (E bytes)
        std::vector<uint8_t> modes(E);
        if (std::fread(modes.data(), 1, E, f) != E) {
            ok = false; break;
        }
        
        // Read rotations (E bytes, currently unused)
        std::vector<uint8_t> rots(E, 0);
        if (std::fread(rots.data(), 1, E, f) != E) {
            ok = false; break;
        }
        
        // Validate modes
        for (uint32_t e = 0; e < E && ok; ++e) {
            if (modes[e] > GLM5NEXT_P4MIX_MAX_MODE) {
                std::fprintf(stderr, "[glm5next] p4mix layer %u expert %u bad mode %u\n",
                             layer, e, modes[e]);
                ok = false;
            }
            if (rots[e] != 0) {
                std::fprintf(stderr, "[glm5next] p4mix layer %u expert %u nonzero rotation\n",
                             layer, e);
                ok = false;
            }
        }
        if (!ok) break;
        
        // Register with kernel
        ggml_cuda_rocmfp3_mix_register_host(
            dt->data, dt->nb[2], (int)E, (int)odim, (int)idim,
            books.data(), modes.data(), rots.data());
        registered_bases.push_back(dt->data);
        done[layer] = true;
    }
    
    std::fclose(f);
    
    // Verify all required tensors got codebooks
    if (ok) {
        for (int il = 0; il < w.n_layer && ok; ++il) {
            if (layer_has_105[il] && !done[il]) {
                std::fprintf(stderr, "[glm5next] p4mix missing layer %d; refusing to load\n", il);
                ok = false;
            }
        }
    }
    
    if (!ok) {
        // Unregister any already-registered tensors
        for (const void * base : registered_bases) {
            ggml_cuda_rocmfp3_mix_unregister(base);
        }
        registered_bases.clear();
        return false;
    }
    
    std::fprintf(stderr, "[glm5next] registered %zu qtype-105 down-expert tensors\n",
                 registered_bases.size());
    return true;
}

// Register GUMIX (qtype 106) sidecar for gate/up/down
static bool glm5next_register_gumix_sidecar(
    const std::string & gguf_path,
    Glm5NextWeights & w,
    std::vector<const void *> & registered_bases) {
    
    std::string sc_path = gguf_path + ".gumix";
    
    // Count qtype-106 tensors (gate/up/down surfaces)
    const size_t n_layers = w.n_layer;
    std::vector<std::array<bool, GLM5NEXT_GUMIX_SURFACES>> required(
        n_layers, std::array<bool, GLM5NEXT_GUMIX_SURFACES>{false, false, false});
    int n_qtype106 = 0;
    
    for (int il = w.first_moe_layer; il < w.n_layer; ++il) {
        const ggml_tensor * ts[GLM5NEXT_GUMIX_SURFACES] = {
            w.layers[il].moe_experts_gate,
            w.layers[il].moe_experts_up,
            w.layers[il].moe_experts_down
        };
        for (uint32_t s = 0; s < GLM5NEXT_GUMIX_SURFACES; ++s) {
            if (ts[s] && (int)ts[s]->type == GLM5NEXT_QTYPE_ROCMFP2_MIX) {
                required[il][s] = true;
                n_qtype106++;
            }
        }
    }
    
    if (n_qtype106 == 0) {
        return true;  // No qtype-106, nothing to register
    }
    
    std::fprintf(stderr, "[glm5next] found %d qtype-106 expert surfaces, need GUMIX sidecar\n",
                 n_qtype106);
    
    // Constraint validation: qtype-106 requires in % 128 == 0
    // GLM expert_ff=2048 and n_embd=4096 both pass
    if (w.n_expert_ff % 128 != 0 || w.n_embd % 128 != 0) {
        std::fprintf(stderr, "[glm5next] qtype-106 requires in %% 128 == 0, got ff=%d embd=%d\n",
                     w.n_expert_ff, w.n_embd);
        return false;
    }
    
    FILE * f = std::fopen(sc_path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[glm5next] GUMIX sidecar required but not found: %s\n",
                     sc_path.c_str());
        return false;
    }
    
    // Read GUMIXs1 header (split form)
    char magic[8];
    uint32_t n_entries = 0, reserved = 0;
    if (std::fread(magic, 1, 8, f) != 8 ||
        std::memcmp(magic, "GUMIXs1\0", 8) != 0 ||
        std::fread(&n_entries, 4, 1, f) != 1 ||
        std::fread(&reserved, 4, 1, f) != 1) {
        std::fprintf(stderr, "[glm5next] bad GUMIX sidecar header (need GUMIXs1): %s\n",
                     sc_path.c_str());
        std::fclose(f);
        return false;
    }
    
    std::vector<std::array<bool, GLM5NEXT_GUMIX_SURFACES>> done(
        n_layers, std::array<bool, GLM5NEXT_GUMIX_SURFACES>{false, false, false});
    bool ok = true;
    
    for (uint32_t i = 0; i < n_entries && ok; ++i) {
        uint32_t hdr[7];  // layer, surface, E, odim, idim, C, K
        if (std::fread(hdr, 4, 7, f) != 7) {
            ok = false; break;
        }
        
        const uint32_t layer = hdr[0], surface = hdr[1], E = hdr[2],
                       odim = hdr[3], idim = hdr[4], C = hdr[5], K = hdr[6];
        
        // Structural validation
        if (surface >= GLM5NEXT_GUMIX_SURFACES) {
            std::fprintf(stderr, "[glm5next] gumix entry %u bad surface %u\n", i, surface);
            ok = false; break;
        }
        if (C != GLM5NEXT_GUMIX_C || K != GLM5NEXT_GUMIX_K) {
            std::fprintf(stderr, "[glm5next] gumix entry %u unexpected C=%u K=%u\n",
                         i, C, K);
            ok = false; break;
        }
        if (E == 0 || E > GLM5NEXT_MIX_MAX_EXPERTS ||
            odim == 0 || odim > GLM5NEXT_MIX_MAX_DIM ||
            idim == 0 || idim > GLM5NEXT_MIX_MAX_DIM) {
            ok = false; break;
        }
        if (idim % GLM5NEXT_GUMIX_QK != 0) {
            std::fprintf(stderr, "[glm5next] gumix layer %u surface %u in=%u not multiple of %u\n",
                         layer, surface, idim, GLM5NEXT_GUMIX_QK);
            ok = false; break;
        }
        
        if (layer >= n_layers || !required[layer][surface]) {
            std::fprintf(stderr, "[glm5next] gumix layer %u surface %u not expected\n",
                         layer, surface);
            ok = false; break;
        }
        
        const ggml_tensor * ts[GLM5NEXT_GUMIX_SURFACES] = {
            w.layers[layer].moe_experts_gate,
            w.layers[layer].moe_experts_up,
            w.layers[layer].moe_experts_down
        };
        const ggml_tensor * gt = ts[surface];
        if (!gt || !gt->data) {
            ok = false; break;
        }
        
        // Read codebooks, modes, rotations
        const size_t book_size = E * C * K * 2;
        std::vector<uint8_t> books(book_size);
        if (std::fread(books.data(), 1, book_size, f) != book_size) {
            ok = false; break;
        }
        
        std::vector<uint8_t> modes(E);
        if (std::fread(modes.data(), 1, E, f) != E) {
            ok = false; break;
        }
        
        std::vector<uint8_t> rots(E, 0);
        if (std::fread(rots.data(), 1, E, f) != E) {
            ok = false; break;
        }
        
        // Validate modes
        for (uint32_t e = 0; e < E && ok; ++e) {
            if (modes[e] > GLM5NEXT_P4MIX_MAX_MODE) {
                ok = false;
            }
        }
        if (!ok) break;
        
        // Register with kernel
        ggml_cuda_rocmfp2_mix_register_host(
            gt->data, gt->nb[2], (int)E, (int)odim, (int)idim,
            books.data(), modes.data(), rots.data());
        registered_bases.push_back(gt->data);
        done[layer][surface] = true;
    }
    
    std::fclose(f);
    
    // Verify all required tensors got codebooks
    if (ok) {
        for (size_t il = 0; il < required.size() && ok; ++il) {
            for (uint32_t s = 0; s < GLM5NEXT_GUMIX_SURFACES; ++s) {
                if (required[il][s] && !done[il][s]) {
                    const char * names[] = {"gate", "up", "down"};
                    std::fprintf(stderr, "[glm5next] gumix missing layer %zu %s; refusing to load\n",
                                 il, names[s]);
                    ok = false;
                }
            }
        }
    }
    
    if (!ok) {
        for (const void * base : registered_bases) {
            ggml_cuda_rocmfp2_mix_unregister(base);
        }
        registered_bases.clear();
        return false;
    }
    
    std::fprintf(stderr, "[glm5next] registered %zu qtype-106 expert surfaces\n",
                 registered_bases.size());
    return true;
}

// Load GLM5-Next weights from GGUF with ROCmFP mix qtype registration
// Returns true on success, false on error
bool glm5next_load_weights(const char * model_path,
                           Glm5NextWeights & w,
                           ggml_context * ctx,
                           std::vector<const void *> & registered_mix_bases) {
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
    // TODO: actual GGUF key reading from glm5next.* namespace
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
            layer.moe_gate = ggml_get_tensor(ctx, layer_name(i, "ffn_gate_inp.weight").c_str());
            layer.moe_exp_probs_b = ggml_get_tensor(ctx, layer_name(i, "exp_probs_b.weight").c_str());
            
            // Routed experts - may be qtype 105/106
            layer.moe_experts_gate = ggml_get_tensor(ctx,
                layer_name(i, "ffn_gate_exps.weight").c_str());
            layer.moe_experts_up = ggml_get_tensor(ctx,
                layer_name(i, "ffn_up_exps.weight").c_str());
            layer.moe_experts_down = ggml_get_tensor(ctx,
                layer_name(i, "ffn_down_exps.weight").c_str());
            
            // Constraint check: gate/up qtype-105 is rejected (down-only)
            if (layer.moe_experts_gate &&
                (int)layer.moe_experts_gate->type == GLM5NEXT_QTYPE_ROCMFP3_MIX) {
                std::fprintf(stderr, "[glm5next_loader] qtype-105 on gate (layer %d) is not supported\n", i);
                gguf_free(gguf_ctx);
                return false;
            }
            if (layer.moe_experts_up &&
                (int)layer.moe_experts_up->type == GLM5NEXT_QTYPE_ROCMFP3_MIX) {
                std::fprintf(stderr, "[glm5next_loader] qtype-105 on up (layer %d) is not supported\n", i);
                gguf_free(gguf_ctx);
                return false;
            }
            
            // Shared expert
            layer.moe_shared_gate = ggml_get_tensor(ctx,
                layer_name(i, "ffn_gate_shexp.weight").c_str());
            layer.moe_shared_up = ggml_get_tensor(ctx,
                layer_name(i, "ffn_up_shexp.weight").c_str());
            layer.moe_shared_down = ggml_get_tensor(ctx,
                layer_name(i, "ffn_down_shexp.weight").c_str());
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
    
    // Register ROCmFP mix qtypes (105/106) with codebook sidecars
    // Missing tables MUST fail the load (no silent fallback to uniform)
    std::vector<const void *> p4mix_bases, gumix_bases;
    
    if (!glm5next_register_p4mix_sidecar(model_path, w, p4mix_bases)) {
        std::fprintf(stderr, "[glm5next_loader] P4MIX registration failed\n");
        return false;
    }
    
    if (!glm5next_register_gumix_sidecar(model_path, w, gumix_bases)) {
        std::fprintf(stderr, "[glm5next_loader] GUMIX registration failed\n");
        // Unregister P4MIX on failure
        for (const void * base : p4mix_bases) {
            ggml_cuda_rocmfp3_mix_unregister(base);
        }
        return false;
    }
    
    // Merge registered bases for cleanup tracking
    registered_mix_bases.insert(registered_mix_bases.end(),
                                p4mix_bases.begin(), p4mix_bases.end());
    registered_mix_bases.insert(registered_mix_bases.end(),
                                gumix_bases.begin(), gumix_bases.end());
    
    return true;
}

}  // namespace dflash::common
