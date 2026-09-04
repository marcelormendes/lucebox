#include "deepseek4_vision_decode.h"

#include <cstdio>
#include <jpeglib.h>
#include <lodepng.h>

#include <csetjmp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace dflash::vision {
namespace {

DecodeResult fail(DecodeError code, std::string message) {
    DecodeResult result;
    result.status = {code, std::move(message)};
    return result;
}

DecodeStatus validate_encoded(const EncodedImageView & encoded, const DecodeLimits & limits) {
    if (encoded.data == nullptr || encoded.size == 0) {
        return {DecodeError::EmptyInput, "encoded image is empty"};
    }
    if (limits.max_encoded_bytes == 0 || encoded.size > limits.max_encoded_bytes) {
        return {DecodeError::EncodedTooLarge, "encoded image exceeds the configured byte limit"};
    }
    return {};
}

DecodeStatus validate_decoded(
    std::uint32_t width,
    std::uint32_t height,
    const DecodeLimits & limits,
    std::size_t & output_bytes) {
    const auto status = validate_decoded_dimensions(width, height, limits.decoded);
    if (!status) {
        return {DecodeError::DecodedTooLarge, status.message};
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
        return {DecodeError::DecodedTooLarge, "decoded RGB byte count overflows size_t"};
    }
    output_bytes = static_cast<std::size_t>(pixels * 3);
    return {};
}

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX] = {};
    int warnings = 0;
};

extern "C" void jpeg_error_exit(j_common_ptr common) {
    auto * error = reinterpret_cast<JpegErrorManager *>(common->err);
    error->base.format_message(common, error->message);
    std::longjmp(error->jump, 1);
}

extern "C" void jpeg_emit_message(j_common_ptr common, int level) {
    auto * error = reinterpret_cast<JpegErrorManager *>(common->err);
    if (level < 0) {
        ++error->warnings;
        error->base.format_message(common, error->message);
    }
}

DecodeResult decode_jpeg(const EncodedImageView & encoded, const DecodeLimits & limits) {
    DecodeResult result;
    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    volatile bool decoder_created = false;
    std::uint8_t * volatile output = nullptr;
    std::size_t output_bytes = 0;

    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = jpeg_error_exit;
    error.base.emit_message = jpeg_emit_message;
    if (setjmp(error.jump) != 0) {
        if (decoder_created) {
            jpeg_destroy_decompress(&decoder);
        }
        std::free(output);
        return fail(DecodeError::MalformedImage,
                    error.message[0] == '\0' ? "malformed JPEG" : error.message);
    }

    jpeg_create_decompress(&decoder);
    decoder_created = true;
    jpeg_mem_src(&decoder, encoded.data, static_cast<unsigned long>(encoded.size));
    if (jpeg_read_header(&decoder, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::MalformedImage, "JPEG header is incomplete");
    }
    decoder.out_color_space = JCS_RGB;
    jpeg_calc_output_dimensions(&decoder);
    if (decoder.output_width > std::numeric_limits<std::uint32_t>::max() ||
        decoder.output_height > std::numeric_limits<std::uint32_t>::max()) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::DecodedTooLarge, "JPEG dimensions are out of range");
    }
    if (const auto status = validate_decoded(
            static_cast<std::uint32_t>(decoder.output_width),
            static_cast<std::uint32_t>(decoder.output_height),
            limits,
            output_bytes);
        !status) {
        jpeg_destroy_decompress(&decoder);
        result.status = status;
        return result;
    }
    if (!jpeg_start_decompress(&decoder) || decoder.output_components != 3) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::MalformedImage, "JPEG cannot be converted to RGB");
    }

    output = static_cast<std::uint8_t *>(std::malloc(output_bytes));
    if (output == nullptr) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::AllocationFailed, "cannot allocate decoded JPEG RGB buffer");
    }
    const std::size_t stride = static_cast<std::size_t>(decoder.output_width) * 3;
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = output +
            static_cast<std::size_t>(decoder.output_scanline) * stride;
        if (jpeg_read_scanlines(&decoder, &row, 1) != 1) {
            jpeg_destroy_decompress(&decoder);
            std::free(output);
            return fail(DecodeError::MalformedImage, "JPEG scanline data is incomplete");
        }
    }
    if (!jpeg_finish_decompress(&decoder) || error.warnings != 0) {
        jpeg_destroy_decompress(&decoder);
        std::free(output);
        return fail(DecodeError::MalformedImage,
                    error.message[0] == '\0' ? "JPEG has decoding warnings" : error.message);
    }

    result.image.width = static_cast<std::uint32_t>(decoder.output_width);
    result.image.height = static_cast<std::uint32_t>(decoder.output_height);
    jpeg_destroy_decompress(&decoder);
    decoder_created = false;
    try {
        const auto * begin = output;
        result.image.pixels.assign(begin, begin + output_bytes);
    } catch (...) {
        std::free(output);
        return fail(DecodeError::AllocationFailed, "cannot own decoded JPEG RGB buffer");
    }
    std::free(output);
    result.status = {};
    return result;
}

DecodeResult decode_png(const EncodedImageView & encoded, const DecodeLimits & limits) {
    LodePNGState state;
    lodepng_state_init(&state);
    unsigned width = 0;
    unsigned height = 0;
    const unsigned inspect_error =
        lodepng_inspect(&width, &height, &state, encoded.data, encoded.size);
    lodepng_state_cleanup(&state);
    if (inspect_error != 0) {
        return fail(DecodeError::MalformedImage,
                    std::string("malformed PNG: ") + lodepng_error_text(inspect_error));
    }

    std::size_t output_bytes = 0;
    if (const auto status = validate_decoded(width, height, limits, output_bytes); !status) {
        DecodeResult result;
        result.status = status;
        return result;
    }
    unsigned char * output = nullptr;
    unsigned decoded_width = 0;
    unsigned decoded_height = 0;
    const unsigned decode_error = lodepng_decode24(
        &output, &decoded_width, &decoded_height, encoded.data, encoded.size);
    if (decode_error != 0) {
        std::free(output);
        return fail(DecodeError::MalformedImage,
                    std::string("malformed PNG: ") + lodepng_error_text(decode_error));
    }
    if (decoded_width != width || decoded_height != height) {
        std::free(output);
        return fail(DecodeError::MalformedImage, "PNG dimensions changed during decode");
    }

    DecodeResult result;
    result.image.width = width;
    result.image.height = height;
    try {
        result.image.pixels.assign(output, output + output_bytes);
    } catch (...) {
        std::free(output);
        return fail(DecodeError::AllocationFailed, "cannot own decoded PNG RGB buffer");
    }
    std::free(output);
    result.status = {};
    return result;
}

}  // namespace

DecodeResult decode_image(const EncodedImageView & encoded, const DecodeLimits & limits) {
    if (const auto status = validate_encoded(encoded, limits); !status) {
        DecodeResult result;
        result.status = status;
        return result;
    }
    constexpr std::uint8_t png_signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (encoded.size >= sizeof(png_signature) &&
        std::memcmp(encoded.data, png_signature, sizeof(png_signature)) == 0) {
        return decode_png(encoded, limits);
    }
    if (encoded.size >= 3 && encoded.data[0] == 0xFF && encoded.data[1] == 0xD8 &&
        encoded.data[2] == 0xFF) {
        return decode_jpeg(encoded, limits);
    }
    return fail(DecodeError::UnsupportedFormat, "encoded image is neither JPEG nor PNG");
}

const char * decode_error_name(DecodeError error) {
    switch (error) {
        case DecodeError::None: return "none";
        case DecodeError::EmptyInput: return "empty_input";
        case DecodeError::EncodedTooLarge: return "encoded_too_large";
        case DecodeError::UnsupportedFormat: return "unsupported_format";
        case DecodeError::MalformedImage: return "malformed_image";
        case DecodeError::DecodedTooLarge: return "decoded_too_large";
        case DecodeError::AllocationFailed: return "allocation_failed";
    }
    return "unknown";
}

}  // namespace dflash::vision
