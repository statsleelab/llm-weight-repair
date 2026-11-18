import torch, time; torch.set_num_threads(1)
from gpt2_lib import *
from imputers import *
torch.manual_seed(0)
sd = load_weights("damaged.safetensors")
def cv(name, lams, orients=(None,)):
    W=sd[name]; miss=(W==0)
    hold=(torch.rand(W.shape)<0.003)&~miss
    M=miss|hold; t=W[hold]; out=[]
    for o in orients:
        for lam in lams:
            t0=time.time(); X=gauss_impute(W,M,lam=lam,rounds=2,orient=o)
            p=X[hold]; r2=1-((p-t)**2).sum()/(t**2).sum()
            out.append((o,lam,round(r2.item(),3),round(time.time()-t0)))
    print(name,out,flush=True)
cv("h.0.attn.c_attn.weight",[0.01,0.1,1.0],("rows","cols"))
cv("h.6.mlp.c_proj.weight",[0.01,0.1,1.0])
cv("h.0.mlp.c_fc.weight",[0.1,1.0])
cv("h.11.attn.c_proj.weight",[0.1,1.0],("rows","cols"))
cv("wpe.weight",[0.001,0.01])
