#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h" // operation-count/version compile contract
#include <iostream>
#include <csignal>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
static void check(bool ok,const char *why) { if(!ok)throw std::runtime_error(why); }
static_assert(GGML_OP_PAGED_ATTN==104 && GGML_OP_MUL_MAT_BIAS_BF16==105 && GGML_OP_COUNT==106,"operation ABI changed");
static void rejected(int mode) {
    pid_t pid=fork(); check(pid>=0,"fork failed");
    if(pid==0) {
        auto c=ggml_init({1024*1024,nullptr,true});
        auto w=ggml_new_tensor_2d(c,mode==0?GGML_TYPE_F32:GGML_TYPE_BF16,64,32);
        auto x=ggml_new_tensor_2d(c,GGML_TYPE_BF16,mode==1?32:64,32);
        auto b=ggml_new_tensor_1d(c,GGML_TYPE_BF16,mode==2?16:32);
        if(mode==3) x=ggml_transpose(c,x);
        (void)ggml_mul_mat_bias_bf16(c,w,x,b);
        _exit(0);
    }
    int status=0; check(waitpid(pid,&status,0)==pid,"wait failed");
    check(WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT,"invalid constructor was not rejected");
}
int main() {
    auto backend=ggml_backend_cpu_init(); auto c=ggml_init({1024*1024,nullptr,true});
    try {
        check(backend&&c,"initialization failed");
        check(dflash::vision::detail::hip_bias_workspace(nullptr)==0 && dflash::vision::detail::hip_bias_workspace(backend)==0,"CPU/null advertised HIP capability");
        auto w=ggml_new_tensor_2d(c,GGML_TYPE_BF16,64,32),x=ggml_new_tensor_2d(c,GGML_TYPE_BF16,64,32),b=ggml_new_tensor_1d(c,GGML_TYPE_BF16,32);
        auto y=ggml_mul_mat_bias_bf16(c,w,x,b);
        check(y->type==GGML_TYPE_BF16 && y->src[0]==w && y->src[1]==x && y->src[2]==b && y->ne[0]==32 && y->ne[1]==32,"explicit op contract mismatch");
        check(!ggml_backend_supports_op(backend,y),"CPU advertised HIP op");
        auto xf=ggml_new_tensor_2d(c,GGML_TYPE_F32,64,32);
        for(bool preserve:{false,true}) {
            auto old=dflash::vision::detail::linear(c,w,xf,b,preserve);
            auto selected=dflash::vision::detail::linear(c,w,xf,b,preserve,backend);
            check(old->op==selected->op && selected->src[0]->op==old->src[0]->op,"non-HIP graph root changed");
            auto product=selected->src[0]->src[0]->src[0];
            check(product->op==GGML_OP_MUL_MAT && (product->src[0]->op==GGML_OP_CPY)==preserve,"non-HIP product dispatch changed");
        }
        for(int mode=0;mode<4;++mode) rejected(mode);
        ggml_free(c); ggml_backend_free(backend); std::cout<<"PASS: CPU/null capability, preserved op ABI/graph, invalid constructor rejection\n"; return 0;
    } catch(const std::exception&e) { std::cerr<<e.what()<<'\n'; ggml_free(c); ggml_backend_free(backend); return 1; }
}
