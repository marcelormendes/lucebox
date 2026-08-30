// Glm5NextBackend — ModelBackend for GLM-5.3-Flash (GLM5-Next).
//
// Architecture: 45 layers with hybrid KDA linear attention (34 layers)
// and NoPE MLA sparse attention (11 layers), mHC, MoE with dense FFN
// on first 3 layers and 288 routed experts top-8 from layer 3.

#pragma once

#include "common/model_backend.h"
#include "common/moe_expert_compute.h"
#include "common/sampler.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_stream.h"
#include "glm5next_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

class Glm5NextBackend : public ModelBackend {
public:
    explicit Glm5NextBackend(const Glm5NextBackendConfig & cfg);
    ~Glm5NextBackend() override;

    Glm5NextBackend(const Glm5NextBackend &) = delete;
    Glm5NextBackend & operator=(const Glm5NextBackend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;

    const MoeHybridRoutingStats * get_routing_stats() const override {
        return routing_stats_.get();
    }

private:
    Glm5NextBackendConfig cfg_;
    ggml_backend_t        backend_ = nullptr;
    Glm5NextWeights       w_;
    Glm5NextCache         cache_;
    bool                  parked_ = false;

    // Sampler
    SamplerCfg            sampler_;
    std::mt19937_64       sampler_rng_{std::random_device{}()};

    // Snapshots (prefix cache)
    static constexpr int PREFIX_SLOTS = 64;
    struct SnapshotAux {
        std::vector<float> last_logits;
        bool used = false;
    };
    std::vector<float> last_logits_;
    int last_logits_pos_ = -1;

    // MoE hybrid storage for expert offload
    std::shared_ptr<MoeHybridStorage> moe_hybrid_;
    MoeHybridPlacement                moe_placement_;
    std::shared_ptr<MoeHybridStreamEngine> moe_stream_engine_;
    std::unique_ptr<MoeHybridRoutingStats> routing_stats_;
    
    // ROCmFP mix qtype (105/106) registered bases for cleanup
    std::vector<const void *> registered_mix_bases_;

    bool load_model();
    bool init_hybrid_model();
    bool requires_monolithic_model() const;

    // Prefill and decode
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0);
    
    GenerateResult generate_from_state(const GenerateRequest & req,
                                       const DaemonIO & io,
                                       int kv_offset);
    
    bool do_decode(int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io);
};

}  // namespace dflash::common
