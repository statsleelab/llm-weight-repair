import torch, numpy as np, json
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
B=json.load(open("blocks3.json"))
for k in ["wpe.weight","h.10.mlp.c_fc.weight","h.3.mlp.c_proj.weight","h.3.attn.c_attn.weight","h.6.attn.c_attn.weight","h.4.attn.c_proj.weight","wte.weight","h.0.mlp.c_fc.weight"]:
    W=sd[k].double(); r0,r1,c0,c1=B[k]; X=W[r0+2:r1-2,c0+2:c1-2]
    M=torch.ones_like(W,dtype=torch.bool); M[r0:r1,c0:c1]=False; Y=W[M]
    kur=lambda x:(((x-x.mean())**4).mean()/x.var()**2).item()
    rr=X.pow(2).mean(1); cc=X.pow(2).mean(0)
    print(f"{k:24s} Xmean={X.mean():.4f} Xstd={X.std():.4f} cleanstd={Y.std():.4f} allstd={W.std():.4f} cleanmean={Y.mean():.4f} kurtX={kur(X):.2f} kurtclean={kur(Y):.2f} rowdisp={(rr.var()/rr.mean()**2*X.shape[1]/2):.2f} coldisp={(cc.var()/cc.mean()**2*X.shape[0]/2):.2f}")
