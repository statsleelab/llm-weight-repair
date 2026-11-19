import torch, numpy as np, json, time
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B3=json.load(open("blocks3.json")); out={}
def box(x,w,dim):
    z=torch.zeros_like(x.narrow(dim,0,1)); c=torch.cat([z,x.cumsum(dim)],dim)
    L=x.shape[dim]
    return (c.narrow(dim,w,L-w+1)-c.narrow(dim,0,L-w+1))/w
def score(W):
    G=W@W.T; G.fill_diagonal_(0)
    c=W.pow(2).sum(0,keepdim=True)
    return W*(G@W)-W.pow(2)*(c-W.pow(2))
for k,W0 in sd.items():
    if W0.ndim!=2 or W0.shape[0]>4000: continue
    W=W0.double(); n,m=W.shape; h=round(.1732*n); w=round(.1732*m)
    U=W.pow(2)/W.var()
    A=box(U,w,1)
    m1=box(A,h,0); m2=box(A*A,h,0); rowdisp=(m2-m1*m1)/m1.pow(2)*w/2*h/(h-1)
    Cc=box(U,h,0)
    n1=box(Cc,w,1); n2=box(Cc*Cc,w,1); coldisp=(n2-n1*n1)/n1.pow(2)*h/2*w/(w-1)
    k4=box(box(U*U,w,1),h,0)/m1.pow(2)
    zr=(rowdisp-1)/np.sqrt(2/(h-1)); zc=(coldisp-1)/np.sqrt(2/(w-1)); zm=(m1-1)/np.sqrt(2/(h*w)); zk=(k4-3)/np.sqrt(24/(h*w))
    stat=(zr.pow(2)+zc.pow(2)+zm.pow(2)+zk.pow(2)).sqrt()
    Wn=W.clone()
    for _ in range(2):
        Wn=Wn/Wn.pow(2).mean(1,keepdim=True).sqrt().clamp_min(1e-8)
        Wn=Wn/Wn.pow(2).mean(0,keepdim=True).sqrt().clamp_min(1e-8)
    S=score(Wn.T).T; S=S/S.std(); S2=score(Wn); S=S+S2/S2.std()
    S=S-S.median(); S=S.clamp(-5*S.std(),5*S.std())
    R=box(box(S,w,1),h,0)
    for thr in (4,6,9,14):
        mask=stat<thr
        if mask.any(): break
    Rm=R.clone(); Rm[~mask]=1e9
    idx=Rm.argmin().item(); r0,c0=divmod(idx,R.shape[1])
    i4=stat.argmin().item(); r4,c4=divmod(i4,R.shape[1])
    b=B3[k]
    # alt: best R among mask but far from chosen
    Rm2=Rm.clone(); Rm2[max(0,r0-h//2):r0+h//2,max(0,c0-w//2):c0+w//2]=1e9
    print(f"{k:26s} pick rows[{r0},{r0+h}) cols[{c0},{c0+w}) thr={thr} nmask={int(mask.sum())} R={R[r0,c0]:.3f} stat={stat[r0,c0]:.1f} | d4min ({r4},{c4}) stat={stat[r4,c4]:.1f} R={R[r4,c4]:.3f} | d3 ({b[0]},{b[2]}) stat={stat[b[0],b[2]]:.1f} R={R[b[0],b[2]]:.3f} | altR={Rm2.min():.3f} Rmed={R.median():.3f}",flush=True)
    out[k]=[r0,r0+h,c0,c0+w]
out["wte.weight"]=B3["wte.weight"]
json.dump(out,open("blocks5.json","w"))
