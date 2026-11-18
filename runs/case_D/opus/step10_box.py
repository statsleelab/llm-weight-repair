import torch, numpy as np, sys
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def score(v):
    g=v.std()
    m=v.median(0,keepdim=True).values
    x=v-m
    rs=x.abs().median(1,keepdim=True).values
    cs=x.abs().median(0,keepdim=True).values
    e=(rs*cs/ x.abs().median()).clamp_min(1e-6)*1.4826
    ll_struct=-0.5*(x/e)**2-torch.log(e)
    ll_noise=-0.5*(v/g)**2-torch.log(g)
    return (ll_noise-ll_struct).clamp(-20,20)
K=24
keys=[k for k in sd if sd[k].dim()==2] if not sys.argv[1:] else sys.argv[1:]
for k in keys:
    v=sd[k]
    if k=="wte.weight": continue
    s=score(v); s=s-s.median()
    b=F.avg_pool2d(s[None,None],K,stride=4)[0,0]
    mu=b.median(); sdv=(b-mu).abs().median()*1.4826
    zb=(b-mu)/sdv
    i,j=divmod(int(zb.argmax()),zb.shape[1])
    hot=(zb>6).nonzero()
    rr=(hot[:,0].min().item()*4,hot[:,0].max().item()*4+K) if len(hot) else None
    cc=(hot[:,1].min().item()*4,hot[:,1].max().item()*4+K) if len(hot) else None
    print(f"{k:26s} maxz={zb.max().item():6.1f} at ({i*4},{j*4}) nhot={len(hot)} rows={rr} cols={cc}")
