import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k in ["h.5.mlp.c_fc.weight","h.5.mlp.c_proj.weight","h.8.attn.c_attn.weight","wte.weight"]:
    v=sd[k]
    gm=(v-v.median()).abs().median().item()*1.4826
    x=(v.flatten()/gm).numpy()
    edges=np.array([0,2,4,6,8,10,12,14,16,18,20,25,30,40,60])
    h=np.histogram(np.abs(x),bins=edges)[0]/x.size
    print(k,"gmad",round(gm,4)); print("  ",[f"{a}-{b}:{c:.5f}" for a,b,c in zip(edges[:-1],edges[1:],h)])
    big=np.abs(x)[np.abs(x)>8]
    print("  big mean",big.mean(),"std of signed big",x[np.abs(x)>8].std(), "pos frac",(x[np.abs(x)>8]>0).mean())
