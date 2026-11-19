import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k in ["h.5.mlp.c_fc.weight","h.0.attn.c_attn.weight","wte.weight","wpe.weight"]:
    v=sd[k]
    med=v.median(0).values; mad=(v-med).abs().median(0).values*1.4826
    z=((v-med)/mad).abs()
    m=z>6
    r=m.sum(1).float(); c=m.sum(0).float()
    print(k,"frac",m.float().mean().item())
    print("  per-row count: mean %.2f std %.2f max %d  frac rows zero %.3f"%(r.mean(),r.std(),r.max(),(r==0).float().mean()))
    print("  per-col count: mean %.2f std %.2f max %d"%(c.mean(),c.std(),c.max()))
    f=m.flatten().numpy().astype(np.int8)
    # run lengths
    d=np.diff(np.concatenate([[0],f,[0]]))
    st=np.where(d==1)[0]; en=np.where(d==-1)[0]
    rl=en-st
    print("  runs:",len(rl),"mean len",rl.mean(),"max",rl.max(), "hist",np.bincount(np.minimum(rl,20))[:21])
    print("  top rows",torch.topk(r,8).values.tolist()," top cols",torch.topk(c,8).values.tolist())
