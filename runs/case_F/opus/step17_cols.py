import torch
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
W=sd["wte.weight"]; g=sd["ln_f.weight"]; b=sd["ln_f.bias"]
top=g.abs().topk(8).indices
print("ln_f top gains",[(int(i),round(g[i].item(),2),round(b[i].item(),2)) for i in top])
for c in [138,378,447]+top[:4].tolist():
    col=W[:,c]
    q=torch.quantile(col,torch.tensor([.01,.05,.25,.5,.75,.95,.99]))
    print(c,"quantiles",q.numpy().round(3),"min",col.min().item(),"max",col.max().item())
