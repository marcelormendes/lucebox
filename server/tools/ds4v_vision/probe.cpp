#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace dflash::vision;
static std::vector<float> read_file(const std::string & path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()<0 || size_t(f.tellg())%4) throw std::runtime_error("bad patch file");
    std::vector<float> out(size_t(f.tellg())/4);
    f.seekg(0); f.read(reinterpret_cast<char *>(out.data()),out.size()*4);
    if(!f) throw std::runtime_error("patch file read failed");
    return out;
}
static void save(const std::string & path,const std::vector<float> & values) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    if(!f) throw std::runtime_error("output write failed: "+path);
}
int main(int argc,char ** argv) {
    if(argc!=8 && argc!=5) {
        std::cerr<<"usage: ds4v_vision_probe mmproj patches.f32 height width output-dir label stages(0|1)\n"
                 <<"       ds4v_vision_probe mmproj --load-only dimension vocabulary\n"; return 2;
    }
    auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,2);
    int status=0;
    try {
        VisionRuntime runtime;
        std::string error;
        bool load_only=argc==5 && std::string(argv[2])=="--load-only";
        int dimension=load_only?std::stoi(argv[3]):4096,vocabulary=load_only?std::stoi(argv[4]):129280;
        if(!runtime.load(argv[1],backend,dimension,vocabulary,error)) throw std::runtime_error(error);
        std::cout<<"weights_bytes="<<runtime.weight_bytes()<<"\n";
        if(!load_only) {
            if(runtime.load(argv[1],backend,4095,129280,error)) throw std::runtime_error("incompatible reload accepted");
            if(!runtime.config() || runtime.weight_bytes()==0) throw std::runtime_error("failed reload destroyed runtime");
            std::vector<float> sentinel_after_reload;
            if(!runtime.sentinel(Sentinel::Start,sentinel_after_reload,error)) throw std::runtime_error("failed reload lost sentinels");
            if(runtime.sentinel(static_cast<Sentinel>(99),sentinel_after_reload,error)) throw std::runtime_error("invalid sentinel accepted");
            const auto patches=read_file(argv[2]);
            PatchGrid grid{std::stoi(argv[3]),std::stoi(argv[4])};
            const std::string output_dir=argv[5],label=argv[6];
            const bool stages=std::stoi(argv[7])!=0;
            std::filesystem::create_directories(output_dir);
            StageObserver observer;
            if(stages) observer=[&](const std::string & name,const std::vector<int64_t> & shape,const std::vector<float> & values) {
                save(output_dir+"/"+label+"-"+name+".f32",values);
                std::cout<<"stage="<<name<<" shape="<<shape[0]<<","<<shape[1]<<","<<shape[2]<<","<<shape[3]<<std::endl;
            };
            VisionOutput output;
            auto started=std::chrono::steady_clock::now();
            if(!runtime.encode(patches,grid,output,error,true,observer)) throw std::runtime_error(error);
            save(output_dir+"/"+label+"-features.f32",output.features);
            save(output_dir+"/"+label+"-embeddings.f32",output.embeddings);
            for(auto identity:{Sentinel::Start,Sentinel::Pad,Sentinel::Newline,Sentinel::End}) {
                std::vector<float> sentinel;
                if(!runtime.sentinel(identity,sentinel,error) || sentinel.size()!=4096) throw std::runtime_error("sentinel failure");
            }
            std::cout<<"output_shape="<<output.rows<<","<<output.columns
                     <<" scratch_bytes="<<runtime.scratch_bytes()
                     <<" elapsed_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"\n";
            // Releasing scratch must leave the loaded projector and sentinels usable.
            runtime.release_scratch();
            if(runtime.scratch_bytes()!=0) throw std::runtime_error("scratch release failed");
            VisionOutput invalid;
            if(runtime.encode({},grid,invalid,error)) throw std::runtime_error("invalid patch input accepted");
            if(runtime.encode({}, {0,1},invalid,error)) throw std::runtime_error("invalid grid accepted");
            if(runtime.encode({}, {1152,1152},invalid,error)) throw std::runtime_error("oversized grid accepted");
        }
    } catch(const std::exception & e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; status=1; }
    ggml_backend_free(backend);
    return status;
}
