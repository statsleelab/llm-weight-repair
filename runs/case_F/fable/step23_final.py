# Final assembly: generic local-fdr repair (c7) for wte, mlp.*, attn.c_proj; override variant (c8) for attn.c_attn;
# running-median repair for wpe (taken from d_c7a8_wpe4).  Then sanity checks.
import torch
torch.set_num_threads(1)
from gpt2_lib import *
dm=load_weights("damaged.safetensors"); c7=load_weights("c7.safetensors"); c8=load_weights("c8.safetensors"); d=load_weights("d_c7a8_wpe4.safetensors")
out={}
for k,v in dm.items():
    if v.ndim!=2: out[k]=v
    elif k=="wpe.weight": out[k]=d[k]
    elif "attn.c_attn" in k: out[k]=c8[k]
    else: out[k]=c7[k]
    assert out[k].shape==v.shape and torch.isfinite(out[k]).all()
    if v.ndim!=2: assert torch.equal(out[k],v)
save_weights(out,"repaired.safetensors")
ch=sum((out[k]!=dm[k]).sum().item() for k in dm if dm[k].ndim==2); n=sum(dm[k].numel() for k in dm if dm[k].ndim==2)
print("changed fraction of 2-D entries",ch/n)
tok=Tokenizer("tokenizer"); print(repr(generate(out,tok,"The capital of France is",10)))
