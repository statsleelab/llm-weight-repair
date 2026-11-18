import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
from detlib import *
from step07_lib import loo2
sd = load_weights("damaged.safetensors")
bE=json.load(open("blocks.json")); bG=json.load(open("blocks_gauss.json"))
def gaussQ(W,s2,i,j,h,w,g=4):
    B=W[i:i+h,j:j+w]; N=B.size
    zv=(B.var()/s2-1)/np.sqrt(2/N); zk=((B**4).mean()/(B**2).mean()**2-3)/np.sqrt(24/N); zm=B.mean()/np.sqrt(s2/N)
    q=zv**2+zk**2+zm**2
    # sub-block variance uniformity
    sub=0; 
    for a in np.array_split(np.arange(h),g):
        for b in np.array_split(np.arange(w),g):
            x=B[np.ix_(a,b)]; sub+=(((x**2).mean()/s2-1)/np.sqrt(2/x.size))**2
    # row and column variance uniformity
    rv=(((B**2).mean(1)/s2-1)/np.sqrt(2/w)); cv=(((B**2).mean(0)/s2-1)/np.sqrt(2/h))
    return q, sub/(g*g), (rv**2).mean(), (cv**2).mean()
final={}
for k,v in sd.items():
    if v.ndim!=2: continue
    W=v.double().numpy(); n,m=W.shape; h,w=round(.1*n),round(.1*m); s2=W.var()
    if n>5000:
        G=v.double().T@v.double()/n; Th=torch.linalg.inv(G+0.1*G.diagonal().mean()*torch.eye(m,dtype=torch.float64)); Es=[((v.double()@Th)/Th.diagonal()[None,:]).numpy()]
    else: Es=loo2(v,0.1)
    S=sum(norm1(np.minimum(norm1(E**2),8)) for E in Es)/len(Es)
    F=boxsum(S,h,w)/(h*w); med=np.median(F); sc=np.median(np.abs(F-med))*1.4826
    # joint local search: among all positions, score = gauss-consistency; restrict to neighbourhoods of the two candidates
    line=f"{k:24s}"
    best=None
    for name,(i0,_,j0,_) in (("E",bE[k]),("G",bG[k])):
        # local refinement: minimise row/col variance-uniformity chi (sharp at edges) within +-h/2,+-w/2 (coarse then fine)
        cand=[]
        for st_i,st_j,rad_i,rad_j in ((max(1,h//12),max(1,w//12),h//2,w//2),):
            for i in range(max(0,i0-rad_i),min(n-h,i0+rad_i)+1,st_i):
                for j in range(max(0,j0-rad_j),min(m-w,j0+rad_j)+1,st_j):
                    q,sub,rv,cv=gaussQ(W,s2,i,j,h,w); cand.append((q/3+sub+rv+cv,i,j))
        _,i1,j1=min(cand); cand=[]
        for i in range(max(0,i1-max(1,h//12)),min(n-h,i1+max(1,h//12))+1,max(1,h//150)):
            for j in range(max(0,j1-max(1,w//12)),min(m-w,j1+max(1,w//12))+1,max(1,w//150)):
                q,sub,rv,cv=gaussQ(W,s2,i,j,h,w); cand.append((q/3+sub+rv+cv,i,j))
        sc_,i2,j2=min(cand); q,sub,rv,cv=gaussQ(W,s2,i2,j2,h,w)
        zE=(F[i2,j2]-med)/sc
        line+=f" | {name}:({i0},{j0})->({i2},{j2}) Q={q:.1f} sub={sub:.2f} rv={rv:.2f} cv={cv:.2f} zE={zE:.1f}"
        tot=sc_-0.05*zE
        if best is None or tot<best[0]: best=(tot,i2,j2,name)
    final[k]=[int(best[1]),int(best[1]+h),int(best[2]),int(best[2]+w)]
    print(line+f" => {best[3]} ({best[1]},{best[2]})",flush=True)
json.dump(final,open("blocks_final.json","w"),indent=0)
