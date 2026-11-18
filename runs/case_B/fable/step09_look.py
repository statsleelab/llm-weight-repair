import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
from repair_lib import posterior, rank1_scale
sd = load_weights("damaged.safetensors")
W=sd["h.3.attn.c_attn.weight"]
sn=10*W.std().item()/math.sqrt(2)
r,c=rank1_scale(W)
print("sn",sn,"r quantiles",np.quantile(r.numpy(),[0,.01,.1,.5,.9,.99,1]).round(4)," c q",np.quantile(c.numpy(),[0,.01,.1,.5,.9,.99,1]).round(3))
What,p=posterior(W,line_guard=False)
q=[.5,.9,.97,.99,.995,.999]
for i in [480,87,266,64,138,447,378,100,200,300]:
    row=W[i]; a=row.abs().numpy()
    print(i,"r_i=%.4f"%r[i].item(),"abs quantiles",np.quantile(a,q).round(3),"n flagged",(p[i]>.5).sum().item(),"flagged abs quantiles",np.quantile(a[(p[i]>.5).numpy()],[0,.25,.5,.75]).round(3), "q/k/v flagged",[(p[i,j:j+768]>.5).sum().item() for j in (0,768,1536)], "q/k/v medabs",[round(row[j:j+768].abs().median().item(),4) for j in (0,768,1536)])
# per head scale in row 480
i=480
print("row480 per-head median|w| (36 blocks of 64 cols):",[round(W[i,j:j+64].abs().median().item(),3) for j in range(0,2304,64)])
print("row100 per-head:",[round(W[100,j:j+64].abs().median().item(),3) for j in range(0,2304,64)])
