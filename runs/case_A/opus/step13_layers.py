import torch, time, math, sys, json; torch.set_num_threads(1)
from gpt2_lib import *
from imputers import *
lo,hi=int(sys.argv[1]),int(sys.argv[2])
sd = load_weights("damaged.safetensors")
out={}; log={}
for i in range(lo,hi):
    for part in ["attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
        name=f"h.{i}.{part}.weight"; W=sd[name]; miss=(W==0)
        g=torch.Generator().manual_seed(i*10+len(part))
        hold=(torch.rand(W.shape,generator=g)<0.005)&~miss; t=W[hold]
        best=None
        for lam in [1.0,3.0,10.0]:
            X=gauss_impute(W,miss|hold,lam=lam,rounds=2); p=X[hold]
            a=max(0.0,((p*t).sum()/(p*p).sum()).item())
            r2=(1-((a*p-t)**2).sum()/(t**2).sum()).item()
            if best is None or r2>best[2]: best=(lam,a,r2)
        lam,a,r2=best
        X=gauss_impute(W,miss,lam=lam,rounds=2)
        R=W.clone(); R[miss]=a*X[miss]
        out[name]=R; log[name]=dict(lam=lam,alpha=round(a,3),cv_r2=round(r2,3))
        print(name,log[name],flush=True)
save_weights(out,f"layers_{lo}_{hi}.safetensors")
json.dump(log,open(f"layers_{lo}_{hi}.json","w"),indent=1)
