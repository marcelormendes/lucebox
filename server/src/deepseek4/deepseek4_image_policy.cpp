#include "deepseek4_image_policy.h"

namespace dflash::vision {
bool select_image_experts(const float *, const float *, size_t, size_t,
                          ImageExpertSelection & output, std::string & error, float) {
    output = {}; error = "not implemented"; return false;
}
bool raw_key_visible(int64_t, int64_t, int64_t, int64_t, int64_t, bool & visible) {
    visible = false; return false;
}
}
