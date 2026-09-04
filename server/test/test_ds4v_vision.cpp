#include "deepseek4/deepseek4_vision.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using dflash::common::VisionOutput;
using dflash::common::VisionPatchGrid;
using dflash::common::VisionRuntime;

int main() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_owner(
        ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t backend = backend_owner.get();
    if (!backend) {
        std::fprintf(stderr, "FAIL: CPU backend unavailable\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, 2);

    std::string report;
    std::string error;
    if (!dflash::common::run_vision_geometry_self_test(backend, report, error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    VisionRuntime unloaded;
    VisionOutput output;
    if (unloaded.encode({}, VisionPatchGrid{}, output, error)) {
        std::fprintf(stderr, "FAIL: unloaded runtime accepted encode\n");
        return 1;
    }
    if (error != "vision runtime is not loaded") {
        std::fprintf(stderr, "FAIL: unexpected unloaded error: %s\n", error.c_str());
        return 1;
    }

    dflash::common::VisionConfig config;
    if (dflash::common::validate_vision_grid(config, {100, 100}, error)) {
        std::fprintf(stderr, "FAIL: oversized aligned grid passed validation\n");
        return 1;
    }
    if (error != "aligned vision token count exceeds the configured maximum") {
        std::fprintf(stderr, "FAIL: unexpected grid bound error: %s\n", error.c_str());
        return 1;
    }

    std::printf("%s threads=2 unloaded_guard=PASS grid_bound=PASS\n", report.c_str());
    return 0;
}
