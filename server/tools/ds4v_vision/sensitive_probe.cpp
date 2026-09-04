#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace dflash::vision;
std::vector<float> read(const std::string & path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || size_t(f.tellg())%4) throw std::runtime_error("bad input: "+path);
    std::vector<float> x(size_t(f.tellg())/4);
    f.seekg(0); f.read(reinterpret_cast<char *>(x.data()),x.size()*4);
    if(!f) throw std::runtime_error("read failed"); return x;
}
int main(int argc,char ** argv) {
    if(argc!=4) { std::cerr<<"mmproj frozen-native-directory output-directory\n"; return 2; }
    auto backend=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend,2);
    int status=0;
    try {
        VisionRuntime runtime; std::string error;
        if(!runtime.load(argv[1],backend,4096,129280,error)) throw std::runtime_error(error);
        std::filesystem::create_directories(argv[3]);
        for(int layer:{12,31}) {
            auto prefix=std::string(argv[3])+"/block"+std::to_string(layer)+"-";
            auto incoming=read(std::string(argv[2])+"/corn-block"+std::to_string(layer-1)+".f32");
            auto expected=read(std::string(argv[2])+"/corn-block"+std::to_string(layer)+".f32");
            std::vector<float> output;
            auto observer=[&](const std::string & name,const std::vector<int64_t> &,const std::vector<float> & values) {
                std::ofstream f(prefix+name+".f32",std::ios::binary);
                f.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
                if(!f) throw std::runtime_error("write failed");
            };
            if(!runtime.diagnose_block(incoming,{23,34},layer,output,error,observer)) throw std::runtime_error(error);
            if(output!=expected) throw std::runtime_error("instrumented graph changed frozen candidate output");
            std::cout<<"block="<<layer<<" frozen_output_bitwise_match=true scratch_bytes="<<runtime.scratch_bytes()<<std::endl;
        }
    } catch(const std::exception & e) { std::cerr<<e.what()<<"\n"; status=1; }
    ggml_backend_free(backend); return status;
}
