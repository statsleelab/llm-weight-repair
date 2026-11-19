import torch,numpy as np,json
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
from blk import tilecorr
sd=load_weights("damaged.safetensors")
known=json.load(open("boxes.json"))
res={}
for k,w in sd.items():
    if w.ndim!=2 or k in("wpe.weight","wte.weight"): continue
    R,C=w.shape
    bh=round(0.17320508*R); bw=round(0.17320508*C)
    t=16
    c=tilecorr(w,t,t)  # (R/16, C/16)
    fh=max(1,round(bh/t)); fw=max(1,round(bw/t))
    f=F.avg_pool2d(c[None,None],(fh,fw),stride=1)[0,0]
    m=f.min(); idx=f.argmin(); i,j=divmod(idx.item(),f.shape[1])
    med=f.median(); mad=(f-med).abs().median()*1.4826
    z=((m-med)/mad).item()
    # second best far away
    f2=f.clone(); f2[max(0,i-fh):i+fh+1,max(0,j-fw):j+fw+1]=1e9
    m2=f2.min().item()
    print(f"{k:26s} box~rows {i*t}-{i*t+fh*t} cols {j*t}-{j*t+fw*t} min {m.item():.3f} second {m2:.3f} med {med.item():.3f} z {z:.1f} known {known.get(k)}")
    res[k]=(i*t,j*t)
json.dump(res,open("coarse2.json","w"))
