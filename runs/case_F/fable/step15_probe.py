# Probe on current repair (pseudo-truth): which residue type hurts - lost genuine entries or undetected small noise?
import torch, sys, math
torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
base=load_weights(sys.argv[1]); grp=sys.argv[2]
g=torch.Generator().manual_seed(0)
def ev(sd): return math.exp(nll_windows(sd,ids,max_tokens=1025)[0])
print("base",ev(base))
for mode in ("zero2.3","noise0.7","zero_rowsel","noise_rowsel"):
    sd=dict(base)
    for k,v in base.items():
        if v.ndim==2 and grp in k:
            S=v.abs().median(1,keepdim=True).values/0.6745*torch.ones_like(v)
            rowbig=(v.abs().max(1,keepdim=True).values > 8*S)   # rows with clustered genuine big entries
            if mode=="zero2.3": sd[k]=torch.where(torch.rand(v.shape,generator=g)<0.023,torch.zeros_like(v),v)
            if mode=="noise0.7": sd[k]=v+(torch.rand(v.shape,generator=g)<0.007)*(torch.rand(v.shape,generator=g)*2-1)*3.4*S
            if mode=="zero_rowsel": sd[k]=torch.where((torch.rand(v.shape,generator=g)<0.023)&~rowbig,torch.zeros_like(v),v)
            if mode=="noise_rowsel": sd[k]=v+((torch.rand(v.shape,generator=g)<0.007)&~rowbig)*(torch.rand(v.shape,generator=g)*2-1)*3.4*S
    print(mode,ev(sd),flush=True)
