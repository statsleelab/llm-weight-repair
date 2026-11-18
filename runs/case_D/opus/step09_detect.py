import torch, numpy as np, sys
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def score(v):
    g=v.std()
    m=v.median(0,keepdim=True).values
    x=v-m
    rs=x.abs().median(1,keepdim=True).values
    cs=x.abs().median(0,keepdim=True).values
    e=(rs*cs/ x.abs().median()).clamp_min(1e-6)*1.4826
    # LLR noise vs structured gaussian
    ll_struct=-0.5*(x/e)**2-torch.log(e)
    ll_noise=-0.5*(v/g)**2-torch.log(g)
    return (ll_noise-ll_struct).clamp(-20,20)
def profile(s,dim,k=9):
    p=s.mean(dim)
    return p
for k in sys.argv[1:]:
    v=sd[k]; s=score(v)
    s=s-s.median()
    rp=s.mean(1); cp=s.mean(0)
    def runs(p,name):
        # smooth and threshold
        ps=torch.nn.functional.avg_pool1d(p[None,None],9,1,4)[0,0]
        thr=ps.median()+5*(ps-ps.median()).abs().median()*1.4826
        idx=(ps>thr).nonzero().flatten()
        print(f"  {name}: n={len(p)} above-thr count={len(idx)} range={(idx.min().item(),idx.max().item()) if len(idx) else None} top={p.argsort(descending=True)[:8].tolist()}")
    print(k, tuple(v.shape))
    runs(rp,"rows"); runs(cp,"cols")
