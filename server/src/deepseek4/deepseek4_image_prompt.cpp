#include "deepseek4_image_prompt.h"
namespace dflash::vision {
PreparedImagePrompt prepare_image_prompt(const std::vector<std::int32_t> &,
    const std::vector<ImagePatchInput> &,const ImagePromptLimits &,const ImageTokenizerContract &) {
    PreparedImagePrompt result;
    result.error=ImagePromptError::InvalidContract;
    result.message="not implemented";
    return result;
}
}
