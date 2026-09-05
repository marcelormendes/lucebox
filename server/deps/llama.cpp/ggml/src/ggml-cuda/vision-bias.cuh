#pragma once
#include "common.cuh"
#if defined(GGML_USE_HIP)
// DS4V-only explicit op; no ordinary MUL_MAT/ADD fusion or fallback.
bool ggml_hip_vision_bias_supported(const ggml_tensor * dst);
void ggml_hip_vision_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
