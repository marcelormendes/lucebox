#include "deepseek4/deepseek4_vision.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using dflash::common::VisionEncodeOptions;
using dflash::common::VisionOutput;
using dflash::common::VisionPatchGrid;
using dflash::common::VisionRuntime;

namespace {

struct Fixture {
    const char * name;
    uint32_t height;
    uint32_t width;
    uint32_t aligner_rows;
};

constexpr Fixture kFixtures[] = {
    {"carrots", 42, 61, 294},
    {"corn", 23, 34, 96},
};

// Fixed before the first parent-fixture run. These bounds are intentionally
// tighter than one BF16 unit at the observed output scale; failures trigger
// stage inspection instead of tolerance adjustment.
constexpr double kFeatureMaxAbs = 0.25;
constexpr double kFeatureRmse = 0.03;
constexpr double kFeatureCosine = 0.9995;
constexpr double kEmbeddingMaxAbs = 0.75;
constexpr double kEmbeddingRmse = 0.08;
constexpr double kEmbeddingCosine = 0.9990;

struct Metrics {
    size_t elements = 0;
    bool finite = true;
    double max_abs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
};

bool read_f32(const fs::path & path,
              size_t expected_elements,
              std::vector<float> & values,
              std::string & error) {
    std::error_code ec;
    const uintmax_t bytes = fs::file_size(path, ec);
    if (ec || bytes != expected_elements * sizeof(float)) {
        error = "fixture has wrong byte size: " + path.string();
        return false;
    }
    values.resize(expected_elements);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(values.data()),
                    static_cast<std::streamsize>(bytes))) {
        error = "failed to read fixture: " + path.string();
        return false;
    }
    return true;
}

bool write_f32(const fs::path & path,
               const std::vector<float> & values,
               std::string & error) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        error = "failed to write output: " + path.string();
        return false;
    }
    return true;
}

Metrics compare(const std::vector<float> & actual,
                const std::vector<float> & expected) {
    Metrics result;
    if (actual.size() != expected.size()) {
        result.finite = false;
        return result;
    }
    result.elements = actual.size();
    long double squared_error = 0.0L;
    long double dot = 0.0L;
    long double actual_norm = 0.0L;
    long double expected_norm = 0.0L;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i];
        const double e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) result.finite = false;
        const double difference = a - e;
        result.max_abs = std::max(result.max_abs, std::fabs(difference));
        squared_error += difference * difference;
        dot += a * e;
        actual_norm += a * a;
        expected_norm += e * e;
    }
    if (!actual.empty()) result.rmse = std::sqrt(squared_error / actual.size());
    if (actual_norm > 0.0L && expected_norm > 0.0L) {
        result.cosine = static_cast<double>(dot / std::sqrt(actual_norm * expected_norm));
    }
    return result;
}

bool within(const Metrics & value, bool embedding) {
    if (!value.finite) return false;
    return embedding
        ? value.max_abs <= kEmbeddingMaxAbs && value.rmse <= kEmbeddingRmse &&
          value.cosine >= kEmbeddingCosine
        : value.max_abs <= kFeatureMaxAbs && value.rmse <= kFeatureRmse &&
          value.cosine >= kFeatureCosine;
}

std::string json_metrics(const Metrics & value) {
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"elements\":" << value.elements
        << ",\"finite\":" << (value.finite ? "true" : "false")
        << ",\"max_abs\":" << value.max_abs
        << ",\"rmse\":" << value.rmse
        << ",\"cosine\":" << value.cosine << "}";
    return out.str();
}

std::string safe_stage_name(std::string name) {
    std::replace(name.begin(), name.end(), '.', '_');
    return name;
}

void usage(const char * program) {
    std::cerr << "usage: " << program
              << " --mmproj PATH --fixtures DIR --output DIR [--threads 2]"
                 " [--fixture carrots|corn|all] [--no-stages]\n";
}

} // namespace

int main(int argc, char ** argv) {
    fs::path mmproj;
    fs::path fixture_directory;
    fs::path output_directory;
    std::string fixture_filter = "all";
    int threads = 2;
    bool capture_stages = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * option) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << option << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--mmproj") mmproj = value("--mmproj");
        else if (arg == "--fixtures") fixture_directory = value("--fixtures");
        else if (arg == "--output") output_directory = value("--output");
        else if (arg == "--threads") threads = std::stoi(value("--threads"));
        else if (arg == "--fixture") fixture_filter = value("--fixture");
        else if (arg == "--no-stages") capture_stages = false;
        else { usage(argv[0]); return 2; }
    }
    if (mmproj.empty() || fixture_directory.empty() || output_directory.empty() ||
        threads < 1 || threads > 2 ||
        (fixture_filter != "all" && fixture_filter != "carrots" && fixture_filter != "corn")) {
        usage(argv[0]);
        return 2;
    }

    std::error_code filesystem_error;
    fs::create_directories(output_directory, filesystem_error);
    if (filesystem_error) {
        std::cerr << "failed to create output directory: " << filesystem_error.message() << "\n";
        return 1;
    }

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_owner(
        ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t backend = backend_owner.get();
    if (!backend) {
        std::cerr << "CPU backend unavailable\n";
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, threads);

    std::string geometry_report;
    std::string error;
    if (!dflash::common::run_vision_geometry_self_test(
            backend, geometry_report, error)) {
        std::cerr << "geometry test failed: " << error << "\n";
        return 1;
    }
    std::cout << geometry_report << " threads=" << threads << "\n";

    VisionRuntime runtime;
    const auto load_start = std::chrono::steady_clock::now();
    if (!runtime.load(mmproj.string(), backend, {}, error)) {
        std::cerr << "load failed: " << error << "\n";
        return 1;
    }
    const double load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_start).count();

    std::ofstream json(output_directory / "metrics.json", std::ios::trunc);
    if (!json) {
        std::cerr << "failed to create metrics.json\n";
        return 1;
    }
    json << std::setprecision(10)
         << "{\n  \"backend\": \"cpu\",\n  \"threads\": " << threads
         << ",\n  \"load_seconds\": " << load_seconds
         << ",\n  \"weight_bytes\": " << runtime.stats().weight_bytes
         << ",\n  \"geometry\": \"" << geometry_report << "\",\n"
         << "  \"thresholds\": {\"features\": {\"max_abs\": " << kFeatureMaxAbs
         << ", \"rmse\": " << kFeatureRmse << ", \"cosine_min\": " << kFeatureCosine
         << "}, \"embeddings\": {\"max_abs\": " << kEmbeddingMaxAbs
         << ", \"rmse\": " << kEmbeddingRmse << ", \"cosine_min\": "
         << kEmbeddingCosine << "}},\n  \"fixtures\": {\n";

    bool all_pass = true;
    bool first_fixture = true;
    for (const Fixture & fixture : kFixtures) {
        if (fixture_filter != "all" && fixture_filter != fixture.name) continue;
        const size_t tokens = static_cast<size_t>(fixture.height) * fixture.width;
        std::vector<float> patches;
        std::vector<float> expected_features;
        std::vector<float> expected_embeddings;
        if (!read_f32(fixture_directory / (std::string(fixture.name) + "-patches.f32"),
                      tokens * 588, patches, error) ||
            !read_f32(fixture_directory / (std::string(fixture.name) + "-features.f32"),
                      tokens * 1024, expected_features, error) ||
            !read_f32(fixture_directory / (std::string(fixture.name) + "-embeddings.f32"),
                      static_cast<size_t>(fixture.aligner_rows) * 4096,
                      expected_embeddings, error)) {
            std::cerr << error << "\n";
            return 1;
        }

        VisionEncodeOptions options;
        fs::path stage_directory = output_directory / (std::string(fixture.name) + "-stages");
        if (capture_stages) {
            fs::create_directories(stage_directory, filesystem_error);
            if (filesystem_error) {
                std::cerr << "failed to create stage directory\n";
                return 1;
            }
            options.diagnostic_stages = {
                "patch_embed", "block.0.q_rope", "block.0.attention",
                "block.0.output", "block.15.output", "block.31.output",
                "tower.final", "aligner.unfold", "aligner.w1",
                "aligner.gelu", "aligner.output",
            };
            options.stage_callback = [&](const std::string & name,
                                         const std::vector<int64_t> & shape,
                                         const std::vector<float> & values,
                                         std::string & callback_error) {
                const fs::path base = stage_directory / safe_stage_name(name);
                if (!write_f32(base.string() + ".f32", values, callback_error)) return false;
                std::ofstream shape_file(base.string() + ".shape", std::ios::trunc);
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i) shape_file << 'x';
                    shape_file << shape[i];
                }
                shape_file << '\n';
                if (!shape_file) {
                    callback_error = "failed to write stage shape: " + base.string();
                    return false;
                }
                return true;
            };
        }

        VisionOutput output;
        const auto encode_start = std::chrono::steady_clock::now();
        if (!runtime.encode(patches, {fixture.height, fixture.width}, output, error, options)) {
            std::cerr << fixture.name << " encode failed: " << error << "\n";
            return 1;
        }
        const double encode_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - encode_start).count();
        if (output.feature_rows != tokens || output.feature_columns != 1024 ||
            output.aligner_rows != fixture.aligner_rows || output.aligner_columns != 4096) {
            std::cerr << fixture.name << " output shape mismatch\n";
            return 1;
        }

        const Metrics feature_metrics = compare(output.raster_features, expected_features);
        const Metrics embedding_metrics = compare(output.aligner_embeddings, expected_embeddings);
        const bool fixture_pass = within(feature_metrics, false) && within(embedding_metrics, true);
        all_pass = all_pass && fixture_pass;
        if (!write_f32(output_directory / (std::string(fixture.name) + "-native-features.f32"),
                       output.raster_features, error) ||
            !write_f32(output_directory / (std::string(fixture.name) + "-native-embeddings.f32"),
                       output.aligner_embeddings, error)) {
            std::cerr << error << "\n";
            return 1;
        }

        if (!first_fixture) json << ",\n";
        first_fixture = false;
        json << "    \"" << fixture.name << "\": {\"grid\": ["
             << fixture.height << ',' << fixture.width << "], \"feature_shape\": ["
             << output.feature_rows << ',' << output.feature_columns
             << "], \"embedding_shape\": [" << output.aligner_rows << ','
             << output.aligner_columns << "], \"features\": "
             << json_metrics(feature_metrics) << ", \"embeddings\": "
             << json_metrics(embedding_metrics) << ", \"peak_scratch_bytes\": "
             << runtime.stats().last_encode_peak_scratch_bytes
             << ", \"seconds\": " << encode_seconds
             << ", \"pass\": " << (fixture_pass ? "true" : "false") << "}";
        json.flush();

        std::cout << std::setprecision(10) << fixture.name
                  << " shape_features=" << output.feature_rows << 'x' << output.feature_columns
                  << " finite_features=" << feature_metrics.finite
                  << " maxabs_features=" << feature_metrics.max_abs
                  << " rmse_features=" << feature_metrics.rmse
                  << " cosine_features=" << feature_metrics.cosine
                  << " shape_embeddings=" << output.aligner_rows << 'x' << output.aligner_columns
                  << " finite_embeddings=" << embedding_metrics.finite
                  << " maxabs_embeddings=" << embedding_metrics.max_abs
                  << " rmse_embeddings=" << embedding_metrics.rmse
                  << " cosine_embeddings=" << embedding_metrics.cosine
                  << " peak_scratch_bytes=" << runtime.stats().last_encode_peak_scratch_bytes
                  << " seconds=" << encode_seconds
                  << " verdict=" << (fixture_pass ? "PASS" : "FAIL") << '\n';
    }
    json << "\n  },\n  \"peak_scratch_bytes\": " << runtime.stats().peak_scratch_bytes
         << ",\n  \"pass\": " << (all_pass ? "true" : "false") << "\n}\n";
    json.close();

    return all_pass ? 0 : 3;
}
