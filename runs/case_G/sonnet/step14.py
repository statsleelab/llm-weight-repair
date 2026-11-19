import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
def corr(a,b):
    a=a-a.mean();b=b-b.mean();return (a*b).sum()/(a.norm()*b.norm()+1e-9)
out=[]
for k,w in sd.items():
    if w.ndim!=2 or k=="wpe.weight": continue
    R,C=w.shape
    lw=torch.log(w.abs()+1e-3) # robust-ish
    rowg=lw.mean(1); colg=lw.mean(0)
    cw=32; cs=[corr(lw[:,j:j+cw].mean(1),rowg).item() for j in range(0,C-cw+1,cw)]
    rw=max(32,R//64); rs=[corr(lw[i:i+rw].mean(0),colg).item() for i in range(0,R-rw+1,rw)]
    def rep(c,name):
        c=torch.tensor(c); med=c.median(); mad=(c-med).abs().median()*1.4826+1e-6
        z=(c-med)/mad
        return name,round(med.item(),2),round(z.min().item(),1),int(z.argmin()),int((z<-6).sum())
    print(k,rep(cs,"colwin"),rep(rs,"rowwin"))
