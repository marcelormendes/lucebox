#pragma once
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ds4_mix_detail {
using EncodedExpert = std::vector<uint8_t>;
inline size_t checked_encoded_size(uint64_t row_bytes, uint64_t rows, unsigned workers) {
    (void) row_bytes; (void) rows; (void) workers;
    throw std::runtime_error("RED: checked expert sizing unimplemented");
}
struct AsyncExpert {
    template<class Task> auto operator()(Task task) const {
        return std::async(std::launch::async, std::move(task));
    }
};
template<class Encode, class Write, class Launch = AsyncExpert>
void ordered_expert_batches(uint32_t count, unsigned workers, size_t expert_bytes,
                            Encode encode, Write write, Launch launch = {}) {
    (void) count; (void) workers; (void) expert_bytes;
    (void) encode; (void) write; (void) launch;
    throw std::runtime_error("RED: bounded expert scheduling unimplemented");
}
} // namespace ds4_mix_detail
