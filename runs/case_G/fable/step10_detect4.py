import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B3=json.load(open("blocks3.json")); out={}
def box(x,w,dim):
    z=torch.zeros_like(x.narrow(dim,0,1)); c=torch.cat([z,x.cumsum(dim)],dim)
    L=x.shape[dim]
    return (c.narrow(dim,w,L-w+1)-c.narrow(dim,0,L-w+1))/w
for k,W0 in sd.items():
    if W0.ndim!=2 or W0.shape[0]>4000: continue
    W=W0.double(); n,m=W.shape; h=round(.1732*n); w=round(.1732*m)
    U=W.pow(2)/W.var()
    A=box(U,w,1)                      # row means over col window
    m1=box(A,h,0); m2=box(A*A,h,0); rowdisp=(m2-m1*m1)/m1.pow(2)*w/2*h/(h-1)
    Cc=box(U,h,0)
    n1=box(Cc,w,1); n2=box(Cc*Cc,w,1); coldisp=(n2-n1*n1)/n1.pow(2)*h/2*w/(w-1)
    # z-scores under gaussian null
    zr=(rowdisp-1)/np.sqrt(2/(h-1)); zc=(coldisp-1)/np.sqrt(2/(w-1)); zm=(m1-1)/np.sqrt(2/(h*w))
    stat=zr.abs()+zc.abs()+zm.abs()
    idx=stat.argmin().item(); r0,c0=divmod(idx,stat.shape[1])
    s2=stat.clone(); s2[max(0,r0-h//2):r0+h//2,max(0,c0-w//2):c0+w//2]=1e9
    b=B3[k]; agree = (abs(b[0]-r0)<=3 and abs(b[2]-c0)<=3)
    print(f"{k:26s} rows[{r0},{r0+h}) cols[{c0},{c0+w}) stat={stat[r0,c0]:.1f} (zr={zr[r0,c0]:.1f} zc={zc[r0,c0]:.1f} zm={zm[r0,c0]:.1f}) second={s2.min():.1f} median={stat.median():.0f} d3={b} {'AGREE' if agree else 'DIFF'}",flush=True)
    out[k]=[r0,r0+h,c0,c0+w]
out["wte.weight"]=B3["wte.weight"]
json.dump(out,open("blocks4.json","w"))
