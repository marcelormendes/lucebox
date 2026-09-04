#!/usr/bin/env python3
"""Two corn blocks, same incoming residual, and independent same-input local ops."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('fixture',type=Path)
p.add_argument('frozen_native',type=Path)
p.add_argument('diagnostic_native',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.set_default_dtype(torch.bfloat16)
sys.path.insert(0,str(a.source/'inference'))
import vision
manifest=json.loads((a.fixture/'manifest.json').read_text())
for name,digest in manifest['source_hashes'].items():
    assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==digest
config=json.loads((a.source/'config.json').read_text()); config['dim']=config['hidden_size']
args=SimpleNamespace(**config)
index=json.loads((a.source/'model.safetensors.index.json').read_text())['weight_map']
report={'torch':torch.__version__,'threads':2,'blocks':{},'notes':['chain uses original BF16 modules with exact incoming native residual; explicit Math SDPA checked bitwise against actual source SDPA and entire source block','local uses exact native input(s) for each operation independently','linear .dot_f32/.f32 are explicitly labeled F32 shadow GEMMs, not captured hidden accumulators of original BF16 GEMM','native diagnostic outputs must match frozen candidate block outputs bitwise before reference comparison']}
a.output.mkdir(parents=True,exist_ok=True)

for layer in (12,31):
    prefix=f'vision.blocks.{layer}.'
    block=vision.Block(args).eval()
    state={}
    for shard in sorted({v for k,v in index.items() if k.startswith(prefix)}):
        with safe_open(a.source/shard,framework='pt',device='cpu') as f:
            for name in f.keys():
                if name.startswith(prefix): state[name[len(prefix):]]=f.get_tensor(name)
    block.load_state_dict(state,strict=True); del state
    incoming=torch.from_numpy(np.fromfile(a.frozen_native/f'corn-block{layer-1}.f32',np.float32).reshape(782,1024)).to(torch.bfloat16)
    native={}
    shapes={'input':(782,1024),'cosine':(782,1,32),'sine':(782,1,32),
            'q_raw':(782,16,64),'k_raw':(782,16,64),'v':(782,16,64),'q':(782,16,64),'k':(782,16,64),
            'q_scaled':(16,782,64),'k_scaled':(16,782,64),'qk':(16,782,782),'softmax':(16,782,782),'pv_f32':(16,782,64),
            'attention':(782,1024),'gate':(782,2816),'up':(782,2816),'silu':(782,2816),'silu.f32':(782,2816),
            'product':(782,2816),'product.f32':(782,2816)}
    for name in ('norm1','norm2','projection','residual1','w2','output'):
        shapes[name]=(782,1024); shapes[name+'.f32']=(782,1024)
    for name in ('norm1','norm2'): shapes[name+'.rms_f32']=(782,1024)
    for name,dim in [('qkv',3072),('w1',5632),('projection',1024),('w2',1024)]:
        for suffix in ('','.dot_f32','.f32'): shapes[name+suffix]=(782,dim)
    def get(name,bf=False):
        if name not in native:
            native[name]=torch.from_numpy(np.fromfile(a.diagnostic_native/f'block{layer}-{name}.f32',np.float32).reshape(shapes[name]))
        return native[name].to(torch.bfloat16) if bf else native[name]
    assert torch.equal(incoming.float(),get('input'))
    cos,sin=vision.get_vision_cos_sin(23,34,32,10000.)
    chain={'input':incoming,'cosine':cos,'sine':sin}
    local={'input':incoming,'cosine':cos,'sine':sin}
    def normalization(table,name,value,module):
        rms=value.float()*torch.rsqrt(value.float().square().mean(-1,keepdim=True)+module.eps)
        table[name+'.rms_f32']=rms
        table[name+'.f32']=module.weight*rms
        table[name]=module(value)
        assert torch.equal(table[name+'.f32'].to(value.dtype),table[name])
        return table[name]
    def linear(table,name,value,module):
        table[name+'.dot_f32']=F.linear(value.float(),module.weight.float(),None)
        table[name+'.f32']=table[name+'.dot_f32']+(module.bias.float() if module.bias is not None else 0.)
        table[name]=module(value)
        return table[name]
    with torch.inference_mode():
        normalization(chain,'norm1',incoming,block.norm1)
        qkv=linear(chain,'qkv',chain['norm1'],block.attn.wqkv)
        q,k,v=(part.view(782,16,64) for part in qkv.chunk(3,dim=-1))
        chain.update(q_raw=q,k_raw=k,v=v)
        q=vision.apply_rotary(q,cos,sin); k=vision.apply_rotary(k,cos,sin)
        chain.update(q=q,k=k)
        scale=math.sqrt(1./math.sqrt(64.))
        chain['q_scaled']=q.transpose(0,1).float()*scale
        chain['k_scaled']=k.transpose(0,1).float()*scale
        chain['qk']=chain['q_scaled']@chain['k_scaled'].transpose(-2,-1)
        chain['softmax']=torch._safe_softmax(chain['qk'],-1)
        chain['pv_f32']=chain['softmax']@v.transpose(0,1).float()
        chain['attention']=chain['pv_f32'].to(torch.bfloat16).transpose(0,1).reshape(782,1024)
        actual_attention=F.scaled_dot_product_attention(q.transpose(0,1),k.transpose(0,1),v.transpose(0,1)).transpose(0,1).reshape(782,1024)
        assert torch.equal(chain['attention'],actual_attention),'explicit Math mismatch'
        linear(chain,'projection',chain['attention'],block.attn.wo)
        chain['residual1.f32']=incoming.float()+chain['projection'].float()
        chain['residual1']=incoming+chain['projection']
        normalization(chain,'norm2',chain['residual1'],block.norm2)
        w1=linear(chain,'w1',chain['norm2'],block.mlp.w1)
        chain['gate'],chain['up']=w1.chunk(2,dim=-1)
        chain['silu.f32']=F.silu(chain['gate'].float())
        chain['silu']=F.silu(chain['gate'])
        chain['product.f32']=chain['silu'].float()*chain['up'].float()
        chain['product']=chain['silu']*chain['up']
        linear(chain,'w2',chain['product'],block.mlp.w2)
        chain['output.f32']=chain['residual1'].float()+chain['w2'].float()
        chain['output']=chain['residual1']+chain['w2']
        assert torch.equal(chain['output'],block(incoming,cos,sin)),'instrumented source block changed output'

        # Each local comparison starts with the exact corresponding native input.
        normalization(local,'norm1',get('input',True),block.norm1)
        linear(local,'qkv',get('norm1',True),block.attn.wqkv)
        local['q_raw'],local['k_raw'],local['v']=(part.view(782,16,64) for part in get('qkv',True).chunk(3,dim=-1))
        local['q']=vision.apply_rotary(get('q_raw',True),cos,sin)
        local['k']=vision.apply_rotary(get('k_raw',True),cos,sin)
        local['q_scaled']=get('q').transpose(0,1)*scale
        local['k_scaled']=get('k').transpose(0,1)*scale
        local['qk']=get('q_scaled')@get('k_scaled').transpose(-2,-1)
        local['softmax']=torch._safe_softmax(get('qk'),-1)
        local['pv_f32']=get('softmax')@get('v').transpose(0,1)
        local['attention']=get('pv_f32').to(torch.bfloat16).transpose(0,1).reshape(782,1024)
        linear(local,'projection',get('attention',True),block.attn.wo)
        local['residual1.f32']=get('input')+get('projection')
        local['residual1']=get('input',True)+get('projection',True)
        normalization(local,'norm2',get('residual1',True),block.norm2)
        linear(local,'w1',get('norm2',True),block.mlp.w1)
        local['gate'],local['up']=get('w1',True).chunk(2,dim=-1)
        local['silu.f32']=F.silu(get('gate'))
        local['silu']=F.silu(get('gate',True))
        local['product.f32']=get('silu')*get('up')
        local['product']=get('silu',True)*get('up',True)
        linear(local,'w2',get('product',True),block.mlp.w2)
        local['output.f32']=get('residual1')+get('w2')
        local['output']=get('residual1',True)+get('w2',True)
    result={'source_manual_matches_actual_block':True,'source_manual_matches_actual_sdpa':True,'chain':{},'local':{}}
    for mode,table in [('chain',chain),('local',local)]:
        for name,value in table.items():
            reference=value.detach().float().numpy()
            actual=get(name).numpy()
            assert actual.shape==reference.shape
            x,y=actual.astype(np.float64).ravel(),reference.astype(np.float64).ravel()
            delta=x-y
            norm=np.linalg.norm(x)*np.linalg.norm(y)
            metrics=dict(shape=list(actual.shape),finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))),cosine=float(np.dot(x,y)/norm) if norm else 1.,exact_fraction=float(np.mean(x==y)),source_dtype=str(value.dtype))
            if value.dtype==torch.bfloat16:
                assert torch.equal(get(name),get(name).to(torch.bfloat16).float()),f'non-BF16 native boundary: {name}'
                def ordered(array):
                    bits=np.ascontiguousarray(array,dtype=np.float32).view(np.uint32)>>16
                    return np.where(bits&0x8000,0x8000-(bits&0x7fff),0x8000+bits).astype(np.int32)
                ulps=np.abs(ordered(actual)-ordered(reference))
                metrics.update(bf16_within_one_ulp=float(np.mean(ulps<=1)),bf16_ulp_p99=float(np.percentile(ulps,99)))
            result[mode][name]=metrics
            print(layer,mode,name,json.dumps(metrics),flush=True)
    report['blocks'][str(layer)]=result
    (a.output/'sensitive-blocks.json').write_text(json.dumps(report,indent=2)+'\n')
    del block,chain,local,native
