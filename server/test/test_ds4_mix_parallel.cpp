#define main ds4_mix_converter_main
#include "../tools/ds4_mix_converter/ds4_mix_converter.cpp"
#undef main
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace ds4_mix_detail;
namespace {
void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f, const std::string & expected = "") {
    try { f(); } catch (const std::exception & e) {
        check(expected.empty() || std::string(e.what()).find(expected) != std::string::npos,
              "wrong exception"); return;
    }
    throw std::runtime_error("expected rejection");
}
Options options(std::vector<std::string> extra) {
    std::vector<std::string> args{"test", "--input", "input", "--output", "output", "--absmax-only"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv; for (auto & arg : args) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
void parsing() {
    check(options({}).encode_threads == 1, "default must be one");
    for (unsigned n = 1; n <= 8; ++n)
        check(options({"--encode-threads", std::to_string(n)}).encode_threads == n, "valid thread count");
    for (const char * bad : {"0", "9", "-1", "", "1x", "1.5", "999999999999999999999999", " 2", "+2"})
        rejects([&] { options({"--encode-threads", bad}); });
    rejects([] { options({"--encode-threads"}); });
}
void sizes() {
    check(checked_encoded_size(4096/32*10, 2048, 8) == 2621440, "gate bytes");
    check(checked_encoded_size(2048/32*14, 4096, 8) == 3670016, "down bytes");
    rejects([] { checked_encoded_size(UINT64_MAX, 2, 1); });
    rejects([] { checked_encoded_size(UINT64_MAX/2, 1, 8); });
    rejects([] { checked_encoded_size(1, 1, 0); });
    rejects([] { checked_encoded_size(1, 1, 9); });
}
void ordering(unsigned width) {
    std::vector<uint8_t> output;
    std::atomic<unsigned> running{0}, peak{0}, finished{0};
    std::mutex mutex; std::condition_variable cv; unsigned entered = 0;
    const auto owner = std::this_thread::get_id();
    ordered_expert_batches(17, width, 3, [&](uint32_t expert) {
        if (width == 1) check(std::this_thread::get_id() == owner, "serial default on caller");
        const unsigned live = ++running; unsigned p = peak;
        while (p < live && !peak.compare_exchange_weak(p, live)) {}
        if (width > 1) {
            std::unique_lock<std::mutex> lock(mutex); ++entered; cv.notify_all();
            const unsigned end = std::min(17u, (expert / width + 1)*width);
            check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered >= end; }), "workers not concurrent");
        }
        // Reverse completion pressure; output must still retain source order.
        std::this_thread::sleep_for(std::chrono::milliseconds(17 - expert));
        --running; ++finished;
        return EncodedExpert{static_cast<uint8_t>(expert), 0x55, static_cast<uint8_t>(expert + 1)};
    }, [&](uint32_t expert, const EncodedExpert & bytes) {
        check(std::this_thread::get_id() == owner, "writer escaped caller");
        check(expert * 3 == output.size(), "output reordered");
        check(finished >= expert + 1, "write before completion");
        output.insert(output.end(), bytes.begin(), bytes.end());
    });
    check(running == 0 && finished == 17 && peak == width, "worker count or join mismatch");
    for (unsigned i = 0; i < 17; ++i) check(output[i*3] == i && output[i*3+2] == i+1, "bytes changed");
}
void failures() {
    for (int mode = 0; mode < 4; ++mode) {
        std::atomic<int> done{0}, running{0}; int launched = 0, writes = 0;
        const std::string expected = mode == 0 ? "worker failed" : mode == 1 ? "launch failed" :
                                     mode == 2 ? "write failed" : "bad_alloc";
        rejects([&] {
            ordered_expert_batches(17, 8, 1, [&](uint32_t i) {
                ++running;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                --running; ++done;
                if (mode == 0 && i == 1) throw std::runtime_error("worker failed");
                if (mode == 3 && i == 1) throw std::bad_alloc();
                return EncodedExpert{static_cast<uint8_t>(i)};
            }, [&](uint32_t, const EncodedExpert &) {
                ++writes; if (mode == 2) throw std::runtime_error("write failed");
            }, [&](auto task) {
                if (mode == 1 && launched == 3) throw std::runtime_error("launch failed");
                ++launched; return std::async(std::launch::async, std::move(task));
            });
        }, expected);
        check(running == 0 && done == launched, "exception escaped before all workers joined");
        check(launched == (mode == 1 ? 3 : 8), "launched next batch after error");
        check(writes == (mode == 1 ? 0 : 1), "writes continued after error");
    }
    rejects([] { ordered_expert_batches(1, 8, 2, [](uint32_t) { return EncodedExpert{1}; },
                                        [](uint32_t, const EncodedExpert &) {}); }, "byte count");
}

// Tiny actual safetensors fixture: 17 distinct experts, all three recipes, nonuniform weights.
struct Fixture {
    fs::path root;
    Fixture() {
        char path[] = "/tmp/ds4-mix-parallel-test-XXXXXX";
        const char * made = ::mkdtemp(path); check(made != nullptr, "mkdtemp"); root = made;
        json header, index; std::vector<uint8_t> payload;
        auto tensor = [&](const std::string & name, const char * dtype, std::vector<int> shape,
                          std::vector<uint8_t> bytes) {
            const size_t begin = payload.size(); payload.insert(payload.end(), bytes.begin(), bytes.end());
            header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {begin, payload.size()}}};
            index["weight_map"][name] = "tiny.safetensors";
        };
        for (const auto & pair : kGlobalNameMap) tensor(pair.first, "BF16", {1}, {0x80, 0x3f});
        tensor("vision.fixture.weight", "BF16", {2}, {0, 0x40, 0x80, 0x3f});
        tensor("layers.0.attn.wkv.weight", "F8_E4M3", {2, 2}, {0x38, 0x40, 0x48, 0x50});
        tensor("layers.0.attn.wkv.scale", "F8_E8M0", {1, 1}, {127});
        tensor("layers.0.ffn.gate.tid2eid", "I64", {2}, {1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0});
        for (unsigned e = 0; e < 17; ++e) for (const auto & recipe : kExpertRecipes) {
            for (bool scale : {false, true}) {
                const std::string name = source_expert_name(0, e, recipe, scale ? "scale" : "weight");
                const size_t begin = payload.size(), size = scale ? 12 : 192;
                for (size_t i = 0; i < size; ++i)
                    payload.push_back(scale ? static_cast<uint8_t>(126 + (i + e)%3)
                        : static_cast<uint8_t>((i*37 + e*13 + static_cast<unsigned>(recipe.surface)*7)%256));
                header[name] = {{"dtype", scale ? "F8_E8M0" : "I8"}, {"shape", {3, scale ? 4 : 64}},
                                {"data_offsets", {begin, payload.size()}}};
                index["weight_map"][name] = "tiny.safetensors";
            }
        }
        json config;
        for (const char * key : {"hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
             "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "num_experts_per_tok",
             "n_shared_experts", "moe_intermediate_size", "num_hash_layers", "sliding_window", "index_n_heads",
             "index_head_dim", "index_topk", "hc_mult", "hc_sinkhorn_iters", "num_hidden_layers", "n_routed_experts"})
            config[key] = 1;
        config["hidden_size"] = 128; config["moe_intermediate_size"] = 3; config["n_routed_experts"] = 17;
        std::ofstream(root/"config.json") << config;
        std::ofstream(root/"tokenizer.json") << R"({"model":{"vocab":{"x":0},"merges":[]}})";
        std::ofstream(root/"tokenizer_config.json") << "{}";
        std::ofstream(root/"model.safetensors.index.json") << index;
        const std::string text = header.dump(); const uint64_t n = text.size();
        std::ofstream shard(root/"tiny.safetensors", std::ios::binary);
        shard.write(reinterpret_cast<const char *>(&n), 8); shard << text;
        shard.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    }
    ~Fixture() { fs::remove_all(root); }
};
std::vector<uint8_t> encode_fixture(const SafeTensorSet & source, const LayerCalibration & c,
                                  const ExpertRecipe & recipe, const std::optional<Imatrix> & imatrix,
                                  unsigned threads) {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    check(file != nullptr, "tmpfile");
    write_expert_tensor(file.get(), source, c, 17, recipe, imatrix, threads);
    check(std::fflush(file.get()) == 0, "flush");
    const auto n = std::ftell(file.get()); check(n > 0, "empty output"); std::rewind(file.get());
    std::vector<uint8_t> bytes(static_cast<size_t>(n));
    check(std::fread(bytes.data(), 1, bytes.size(), file.get()) == bytes.size(), "read output");
    return bytes;
}
void actual_encoding() {
    Fixture fixture; SafeTensorSet source(fixture.root);
    std::optional<Imatrix> imatrix{Imatrix{}};
    LayerCalibration c; c.layer = 0; c.gate_up_shape = c.down_shape = {128, 3};
    for (const auto & recipe : kExpertRecipes) {
        auto & weights = (*imatrix)[target_expert_name(0, recipe)].values;
        for (unsigned i = 0; i < 128; ++i) weights.push_back(.1f + (i%11)*.3f);
    }
    c.gate_up.levels = 4; c.down.levels = 8;
    for (unsigned e = 0; e < 17; ++e) {
        for (auto * registry : {&c.gate_up, &c.down}) {
            std::vector<uint16_t> books;
            for (int half = 0; half < 2; ++half) for (uint32_t i = 0; i < registry->levels; ++i)
                books.push_back(float_to_bf16(-1.f + 2.f*i/(registry->levels-1)));
            registry->experts.push_back(books);
        }
    }
    for (const auto & recipe : kExpertRecipes) {
        const auto serial = encode_fixture(source, c, recipe, imatrix, 1);
        check(serial == encode_fixture(source, c, recipe, imatrix, 8), "actual serial/parallel codec bytes differ");
        check(serial == encode_fixture(source, c, recipe, imatrix, 2), "actual width2 bytes differ");
        const size_t expert_bytes = serial.size()/17;
        check(!std::equal(serial.begin(), serial.begin()+expert_bytes, serial.begin()+expert_bytes), "fixture experts not distinct");
    }
    // Whole synthetic artifacts include raw, dense FP8, I64-map producers and repair metadata.
    c.repairs = {"fixture repair=bf16-epsilon-v1"};
    Options serial_options; serial_options.output = fixture.root/"serial.gguf";
    write_gguf(serial_options, source, {c}, 1, 17, imatrix);
    Options parallel_options = serial_options; parallel_options.output = fixture.root/"parallel.gguf";
    parallel_options.encode_threads = 8;
    write_gguf(parallel_options, source, {c}, 1, 17, imatrix);
    check(read_file(serial_options.output) == read_file(parallel_options.output), "whole synthetic GGUF bytes differ");
    check(read_file(serial_options.output.string()+".gumix.bin") == read_file(parallel_options.output.string()+".gumix.bin"),
          "whole synthetic GUMIX bytes differ");
    // Accepted headers, subsequently truncated backing bytes exercise an actual worker read failure.
    fs::resize_file(fixture.root/"tiny.safetensors", fs::file_size(fixture.root/"tiny.safetensors") - 256);
    rejects([&] { encode_fixture(source, c, kExpertRecipes[2], imatrix, 8); }, "short read");
    Options failed_options = parallel_options; failed_options.output = fixture.root/"failed.gguf";
    rejects([&] { write_gguf(failed_options, source, {c}, 1, 17, imatrix); }, "short read");
    check(!fs::exists(failed_options.output) && !fs::exists(failed_options.output.string()+".gumix.bin"),
          "failed worker published final artifact or sidecar");
    // Kernel write error propagates through the ordered writer, with all tasks joined first.
    std::unique_ptr<FILE, decltype(&std::fclose)> full(std::fopen("/dev/full", "wb"), &std::fclose);
    check(full != nullptr, "open /dev/full"); std::setvbuf(full.get(), nullptr, _IONBF, 0);
    rejects([&] { write_expert_tensor(full.get(), source, c, 17, kExpertRecipes[0], imatrix, 8); }, "failed writing");
}
}
int main() {
    int failed = 0;
    auto run = [&](const char * name, auto test) {
        try { test(); std::cout << "PASS " << name << '\n'; }
        catch (const std::exception & e) { ++failed; std::cout << "FAIL " << name << ": " << e.what() << '\n'; }
    };
    run("strict parsing", parsing); run("checked byte bounds", sizes);
    run("serial order", [] { ordering(1); }); run("two worker order", [] { ordering(2); });
    run("eight worker batches and tail", [] { ordering(8); });
    run("failure joins and publication stop", failures); run("actual expert codecs and IO failures", actual_encoding);
    return failed ? 1 : 0;
}
