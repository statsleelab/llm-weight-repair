import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import *
from step07_lib import loo2
sd = load_weights("damaged.safetensors")
bC=json.load(open("blocks_comb.json"))
final={}
for k,v in sd.items():
    if v.ndim!=2: continue
    W=v.double().numpy(); n,m=W.shape; h,w=round(.1*n),round(.1*m); N=h*w; s2=W.var()
    if k=="wpe.weight": final[k]=[151,253,59,136]; continue
    if n>5000:
        Wd=v.double(); G=Wd.T@Wd/n; Th=torch.linalg.inv(G+0.1*G.diagonal().mean()*torch.eye(m,dtype=torch.float64)); Es=[((Wd@Th)/Th.diagonal()[None,:]).numpy()]
    else: Es=loo2(v,0.1)
    P=sum(W-E for E in Es)/len(Es)
    # scale-free: correlation-like; normalise by local rms of W and P via rank-1 row/col scales
    M=W*P
    sc=np.sqrt(np.outer((W**2).mean(1),(W**2).mean(0))/s2 * np.outer((P**2).mean(1),(P**2).mean(0))/(P**2).mean())
    Mn=np.clip(M/sc,-8,8)
    F=boxsum(Mn,h,w)/N
    null=np.sqrt(np.clip(boxsum(Mn**2,h,w)/N,1e-12,None)/N)
    Z=F/null
    i,j=np.unravel_index(F.argmin(),F.shape)
    F2=F.copy(); F2[max(0,i-h//2):i+h//2,max(0,j-w//2):j+w//2]=1e9
    i2,j2=np.unravel_index(F2.argmin(),F2.shape)
    # refine edges with profiles (block low): use -Mn
    r0,r1,c0,c1,_,_,_=refine(-Mn,i,i+h,j,j+w)
    c=bC[k]
    ok = abs((r1-r0)-h)<=1 and abs((c1-c0)-w)<=1
    print(f"{k:24s} covmin@({i},{j}) F={F[i,j]:.3f} z0={Z[i,j]:.1f} med={np.median(F):.3f} runner@({i2},{j2}) F={F2[i2,j2]:.3f} | refine rows {r0}-{r1}({r1-r0}/{h}) cols {c0}-{c1}({c1-c0}/{w}) {'EXACT' if ok else ''} | comb ({c[0]},{c[2]}) F={F[c[0],c[2]]:.3f}",flush=True)
    final[k]=dict(cov=[int(i),int(j)],ref=[int(r0),int(r1),int(c0),int(c1)],exact=bool(ok),comb=[c[0],c[2]],Fmin=float(F[i,j]),Frun=float(F2[i2,j2]),med=float(np.median(F)))
json.dump(final,open("blocks_cov.json","w"),indent=0)
