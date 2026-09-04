#include "image_input.h"

namespace dflash::common {
bool parse_image_data_url(std::string_view, EncodedImage & image,
                          std::string & error, size_t) {
    image = {};
    error = "image data URLs are not implemented";
    return false;
}
bool extract_chat_images(const nlohmann::json &, nlohmann::json & normalized,
                         std::vector<EncodedImage> & images,
                         std::string & error, const ImageInputLimits &) {
    normalized = nullptr;
    images.clear();
    error = "chat image transport is not implemented";
    return false;
}
void redact_image_urls(nlohmann::json &) {}
} // namespace dflash::common
