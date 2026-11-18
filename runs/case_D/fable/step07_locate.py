import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import *
def loo2(W, lamf):
    W=W.double(); n,m=W.shape; out=[]
    G=W@W.T/m; Th=torch.linalg.inv(G+lamf*G.diagonal().mean()*torch.eye(n,dtype=W.dtype)); out.append(((Th@W)/Th.diagonal()[:,None]).numpy())
    if max(n,m)<=3072 or m<n:
        pass
    G=W.T@W/n; Th=torch.linalg.inv(G+lamf*G.diagonal().mean()*torch.eye(m,dtype=W.dtype)); out.append(((W@Th)/Th.diagonal()[None,:]).numpy())
    return out
def loo_cols(W,lamf):
    W=W.double(); n,m=W.shape
    G=W.T@W/n; Th=torch.linalg.inv(G+lamf*G.diagonal().mean()*torch.eye(m,dtype=W.dtype)); return [((W@Th)/Th.diagonal()[None,:]).numpy()]
sd = load_weights("damaged.safetensors")
blocks={}
for k,v in sd.items():
    if v.ndim!=2: continue
    n,m=v.shape; h,w=round(.1*n),round(.1*m)
    if k=="wpe.weight":
        blocks[k]=[151,253,59,136]; continue
    line=f"{k:26s}"
    for lamf in (0.02,0.2):
        Es = loo_cols(v,lamf) if n>5000 else loo2(v,lamf)
        S=0
        for E in Es:
            s=norm1(E**2); s=np.minimum(s,8.0); S=S+norm1(s)
        S/=len(Es)
        F=boxsum(S,h,w)/(h*w)
        med=np.median(F); sc=np.median(np.abs(F-med))*1.4826
        i,j=np.unravel_index(F.argmax(),F.shape); f1=F[i,j]
        F2=F.copy(); F2[max(0,i-h):i+h,max(0,j-w):j+w]=-1
        i2,j2=np.unravel_index(F2.argmax(),F2.shape)
        line+=f" | lam{lamf}: @({i},{j}) z={(f1-med)/sc:.1f} runner@({i2},{j2}) z={(F2[i2,j2]-med)/sc:.1f}"
        if lamf==0.02: blocks[k]=[int(i),int(i+h),int(j),int(j+w)]; zz=(f1-med)/sc
        else:
            if (f1-med)/sc>zz: blocks[k]=[int(i),int(i+h),int(j),int(j+w)]
    print(line,flush=True)
json.dump(blocks,open("blocks.json","w"),indent=0)
