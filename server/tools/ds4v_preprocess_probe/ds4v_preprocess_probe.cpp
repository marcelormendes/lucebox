#include "deepseek4_vision_preprocess.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
using dflash::vision::DecodedRgbView;
using dflash::vision::ImageLayout;
using dflash::vision::ImageTokenType;
using dflash::vision::PreprocessConfig;
using dflash::vision::PreprocessError;
using dflash::vision::PreprocessLimits;
using dflash::vision::PreprocessResult;
using dflash::vision::ResizePlan;

namespace {

struct FixtureCase {
    std::string label;
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;
    std::uint32_t resized_width = 0;
    std::uint32_t resized_height = 0;
    std::uint32_t vit_rows = 0;
    std::uint32_t vit_cols = 0;
    std::uint32_t aligner_rows = 0;
    std::uint32_t aligner_cols = 0;
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::string> split_tabs(const std::string & line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            return fields;
        }
        begin = end + 1;
    }
}

std::uint32_t parse_u32(const std::string & value, const std::string & field) {
    std::size_t used = 0;
    const unsigned long parsed = std::stoul(value, &used);
    if (used != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid " + field + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::vector<FixtureCase> read_manifest(const fs::path & root) {
    std::ifstream input(root / "manifest.tsv");
    require(input.good(), "cannot open fixture manifest: " + (root / "manifest.tsv").string());
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "fixture manifest is empty");
    require(line ==
                "label\tinput_width\tinput_height\tresized_width\tresized_height\tvit_rows\tvit_cols\taligner_rows\taligner_cols",
            "fixture manifest header mismatch");
    std::vector<FixtureCase> cases;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_tabs(line);
        require(fields.size() == 9, "fixture manifest row must have 9 fields");
        require(!fields[0].empty() &&
                    std::all_of(fields[0].begin(), fields[0].end(), [](unsigned char value) {
                        return (value >= 'a' && value <= 'z') || value == '-';
                    }),
                "fixture label contains unsupported characters");
        FixtureCase item;
        item.label = fields[0];
        item.input_width = parse_u32(fields[1], "input_width");
        item.input_height = parse_u32(fields[2], "input_height");
        item.resized_width = parse_u32(fields[3], "resized_width");
        item.resized_height = parse_u32(fields[4], "resized_height");
        item.vit_rows = parse_u32(fields[5], "vit_rows");
        item.vit_cols = parse_u32(fields[6], "vit_cols");
        item.aligner_rows = parse_u32(fields[7], "aligner_rows");
        item.aligner_cols = parse_u32(fields[8], "aligner_cols");
        cases.push_back(std::move(item));
    }
    require(!cases.empty(), "fixture manifest has no cases");
    return cases;
}

template <typename T>
std::vector<T> read_binary(const fs::path & path) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "cannot open fixture: " + path.string());
    const std::streampos end = input.tellg();
    require(end >= 0, "cannot determine fixture size: " + path.string());
    const auto bytes = static_cast<std::uint64_t>(end);
    require(bytes % sizeof(T) == 0, "fixture byte size is invalid: " + path.string());
    std::vector<T> result(static_cast<std::size_t>(bytes / sizeof(T)));
    input.seekg(0);
    if (!result.empty()) {
        input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
        require(input.good(), "cannot read fixture: " + path.string());
    }
    return result;
}

template <typename T>
void require_equal(
    const std::vector<T> & actual,
    const std::vector<T> & expected,
    const std::string & label) {
    if (actual.size() != expected.size()) {
        std::ostringstream message;
        message << label << " size mismatch: actual=" << actual.size()
                << " expected=" << expected.size();
        throw std::runtime_error(message.str());
    }
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        const std::size_t offset = static_cast<std::size_t>(mismatch.first - actual.begin());
        std::ostringstream message;
        message << label << " mismatch at element " << offset
                << ": actual=" << static_cast<std::int64_t>(*mismatch.first)
                << " expected=" << static_cast<std::int64_t>(*mismatch.second);
        throw std::runtime_error(message.str());
    }
}

std::vector<std::int64_t> layout_types(const ImageLayout & layout) {
    std::vector<std::int64_t> result;
    result.reserve(layout.types.size());
    for (const auto type : layout.types) {
        result.push_back(static_cast<std::int64_t>(type));
    }
    return result;
}

void require_plan(const ResizePlan & actual, const FixtureCase & expected) {
    std::ostringstream details;
    details << "actual resized=" << actual.resized_width << 'x' << actual.resized_height
            << " vit=" << actual.vit_rows << 'x' << actual.vit_cols
            << " aligner=" << actual.aligner_rows << 'x' << actual.aligner_cols;
    require(actual.resized_width == expected.resized_width &&
                actual.resized_height == expected.resized_height &&
                actual.vit_rows == expected.vit_rows && actual.vit_cols == expected.vit_cols &&
                actual.aligner_rows == expected.aligner_rows &&
                actual.aligner_cols == expected.aligner_cols,
            expected.label + " plan mismatch: " + details.str());
}

void require_same_result(const PreprocessResult & first, const PreprocessResult & second,
                         const std::string & label) {
    require(static_cast<bool>(first) && static_cast<bool>(second),
            label + " deterministic run failed");
    require(first.image.plan.resized_width == second.image.plan.resized_width &&
                first.image.plan.resized_height == second.image.plan.resized_height &&
                first.image.plan.vit_rows == second.image.plan.vit_rows &&
                first.image.plan.vit_cols == second.image.plan.vit_cols &&
                first.image.plan.aligner_rows == second.image.plan.aligner_rows &&
                first.image.plan.aligner_cols == second.image.plan.aligner_cols &&
                first.image.plan.direct_resize == second.image.plan.direct_resize,
            label + " plan is not deterministic");
    require_equal(first.image.resized_rgb, second.image.resized_rgb,
                  label + " deterministic resized RGB");
    require_equal(first.image.patches_bf16, second.image.patches_bf16,
                  label + " deterministic patches");
    require_equal(layout_types(first.image.layout), layout_types(second.image.layout),
                  label + " deterministic layout types");
    require_equal(first.image.layout.permutation, second.image.layout.permutation,
                  label + " deterministic permutation");
    require(first.image.layout.span.block_begin == second.image.layout.span.block_begin &&
                first.image.layout.span.visible_begin == second.image.layout.span.visible_begin &&
                first.image.layout.span.visible_end == second.image.layout.span.visible_end &&
                first.image.layout.span.block_end == second.image.layout.span.block_end,
            label + " span is not deterministic");
}

void verify_fixture(const fs::path & root, const FixtureCase & item) {
    const fs::path case_dir = root / item.label;
    const auto input = read_binary<std::uint8_t>(case_dir / "input.rgb");
    const DecodedRgbView view{item.input_width, item.input_height, input.data(), input.size()};
    const PreprocessResult result = dflash::vision::preprocess_rgb(view, 0);
    require(static_cast<bool>(result),
            item.label + " preprocess failed: " +
                dflash::vision::preprocess_error_name(result.status.code) + ": " +
                result.status.message);
    require_plan(result.image.plan, item);
    require_equal(result.image.resized_rgb,
                  read_binary<std::uint8_t>(case_dir / "resized.rgb"),
                  item.label + " resized RGB");
    require_equal(result.image.patches_bf16,
                  read_binary<std::uint16_t>(case_dir / "patches.bf16"),
                  item.label + " BF16 patches");

    constexpr std::array<std::uint64_t, 5> starts = {0, 1, 2, 3, 127};
    for (const std::uint64_t start : starts) {
        ImageLayout layout;
        const auto status = dflash::vision::build_image_layout(
            item.aligner_rows, item.aligner_cols, start, layout);
        require(static_cast<bool>(status),
                item.label + " layout failed at start " + std::to_string(start) + ": " +
                    status.message);
        const std::string suffix = std::to_string(start) + ".i64";
        require_equal(layout_types(layout),
                      read_binary<std::int64_t>(case_dir / ("types-" + suffix)),
                      item.label + " types start=" + std::to_string(start));
        require_equal(layout.permutation,
                      read_binary<std::int64_t>(case_dir / ("permutation-" + suffix)),
                      item.label + " permutation start=" + std::to_string(start));
        const std::uint64_t leading = 3 - start % 4;
        const std::uint64_t end = start + layout.types.size();
        require(layout.span.block_begin == start && layout.span.visible_begin == start + leading &&
                    layout.span.visible_end == end && layout.span.block_end == end,
                item.label + " span mismatch at start " + std::to_string(start));
    }

    const PreprocessResult repeated = dflash::vision::preprocess_rgb(view, 0);
    require_same_result(result, repeated, item.label);
    std::cout << item.label << " PASS resized=" << item.resized_width << 'x'
              << item.resized_height << " vit=" << item.vit_rows << 'x' << item.vit_cols
              << " aligner=" << item.aligner_rows << 'x' << item.aligner_cols
              << " patch_words=" << result.image.patches_bf16.size() << '\n';
}

void expect_error(PreprocessError expected, PreprocessError actual, const std::string & label) {
    require(actual == expected,
            label + " returned " + dflash::vision::preprocess_error_name(actual) +
                ", expected " + dflash::vision::preprocess_error_name(expected));
}

void self_test() {
    PreprocessConfig bad_config;
    bad_config.patch_size = 16;
    expect_error(PreprocessError::InvalidConfig,
                 dflash::vision::validate_config(bad_config).code,
                 "changed fixed config");

    expect_error(PreprocessError::InvalidDimensions,
                 dflash::vision::validate_decoded_dimensions(0, 1).code,
                 "zero width");
    expect_error(PreprocessError::InputTooLarge,
                 dflash::vision::validate_decoded_dimensions(65'536, 1).code,
                 "axis cap");
    expect_error(PreprocessError::InputTooLarge,
                 dflash::vision::validate_decoded_dimensions(8192, 8193).code,
                 "pixel cap");

    const std::array<std::uint8_t, 3> pixel = {0, 127, 255};
    DecodedRgbView wrong_size{1, 1, pixel.data(), 2};
    expect_error(PreprocessError::InputSizeMismatch,
                 dflash::vision::preprocess_rgb(wrong_size, 0).status.code,
                 "wrong RGB byte count");

    PreprocessLimits tiny_output_limit;
    tiny_output_limit.max_output_pixels = 1;
    ResizePlan plan;
    expect_error(PreprocessError::OutputTooLarge,
                 dflash::vision::plan_image(1, 1, plan, {}, tiny_output_limit).code,
                 "output cap");

    ImageLayout layout;
    expect_error(PreprocessError::TokenBudgetExceeded,
                 dflash::vision::build_image_layout(100, 100, 0, layout).code,
                 "layout budget");
    expect_error(PreprocessError::PositionOverflow,
                 dflash::vision::build_image_layout(
                     2, 3, std::numeric_limits<std::uint64_t>::max() - 5, layout).code,
                 "absolute span overflow");

    const auto layout_status = dflash::vision::build_image_layout(2, 3, 0, layout);
    require(static_cast<bool>(layout_status), "known layout failed");
    const std::vector<std::int64_t> expected_types = {
        1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4,
    };
    const std::vector<std::int64_t> expected_permutation = {0, 3, 1, 4, 2, 5};
    require_equal(layout_types(layout), expected_types, "known layout types");
    require_equal(layout.permutation, expected_permutation, "known layout permutation");
    require(layout.span.block_begin == 0 && layout.span.visible_begin == 3 &&
                layout.span.visible_end == 13 && layout.span.block_end == 13,
            "known layout span mismatch");
    std::cout << "self-test PASS\n";
}

void usage(const char * program) {
    std::cerr << "Usage: " << program << " --self-test | --fixtures DIRECTORY\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--fixtures") {
            const fs::path root = argv[2];
            for (const auto & item : read_manifest(root)) {
                verify_fixture(root, item);
            }
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
