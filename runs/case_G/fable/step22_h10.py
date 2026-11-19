import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks10.json"))
k="h.10.attn.c_proj.weight"; W=sd[k].double(); r0,r1,c0,c1=B[k]; s2=W.var().item()
U=W[r0+3:r1-3].pow(2)/s2; z=(U.mean(0)-1)/np.sqrt(2/U.shape[0])
# also correlation of in-block column segment with out-of-block segment of same column scale
far=torch.ones(768,dtype=torch.bool); far[r0-60:r1+30]=False
zf=(W[far].pow(2).mean(0)/s2)
for a,b in ((452,482),(588,612)):
    print("cols",a,b); print(" z_in :"," ".join(f"{v:5.1f}" for v in z[a:b].tolist())); print(" farU :"," ".join(f"{v:5.1f}" for v in zf[a:b].tolist()))
# cumulative cost for c0 candidates: sum over cols in block of z^2
q=(z**2).numpy(); c=np.concatenate([[0],np.cumsum(q)])
for s in range(456,480): print(s, round(c[s+133]-c[s],1), end=" | ")
