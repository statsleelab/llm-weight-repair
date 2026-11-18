import torch, time, math; torch.set_num_threads(1)
from gpt2_lib import *
from imputers import *
torch.manual_seed(0)
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
W=sd["wte.weight"]; miss=(W==0)
hold=(torch.rand(W.shape)<0.002)&~miss
t=W[hold]
# baseline: column mean
cm=(W*(~miss)).sum(0)/(~miss).sum(0)
print("colmean R2", (1-((cm.expand_as(W)[hold]-t)**2).sum()/(t**2).sum()).item())
for lam in [0.003,0.03]:
    t0=time.time(); X=gauss_impute(W,miss|hold,lam=lam,rounds=2)
    print("lam",lam,"R2",(1-((X[hold]-t)**2).sum()/(t**2).sum()).item(), round(time.time()-t0),flush=True)
