import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
import torch.nn.functional as F
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
x=sd["wte.weight"][torch.tensor(ids)]+sd["wpe.weight"][1:257]
print("embed dim64 mean", x[:,64].mean().item())
p="h.0."
h=F.layer_norm(x,(768,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"])
# skip attention precise; approximate via mlp only
h2=F.layer_norm(x,(768,),sd[p+"ln_2.weight"],sd[p+"ln_2.bias"])
m=gelu_new(h2@sd[p+"mlp.c_fc.weight"]+sd[p+"mlp.c_fc.bias"])
W=sd[p+"mlp.c_proj.weight"]
for d in [64,373,447,481,496]:
    c=(m*W[:,d]).mean(0)
    top=c.abs().topk(5)
    print("mlp0 ->dim",d,"total",c.sum().item(), [(int(i),round(float(c[i]),2),round(float(W[i,d]),3)) for i in top.indices])
# robust anomaly scores per matrix: |w| relative to row & col RMS
def anomalies(W,k=8):
    a=W.abs(); r=a.median(1,keepdim=True).values; c=a.median(0,keepdim=True).values
    s=a/(torch.sqrt(r*c)+1e-8)
    t=s.flatten().topk(k)
    return [(int(i)//W.shape[1],int(i)%W.shape[1],round(float(W.flatten()[i]),3),round(float(v),1)) for v,i in zip(t.values,t.indices)]
for k in ["h.0.mlp.c_proj.weight","h.0.mlp.c_fc.weight","h.2.mlp.c_fc.weight","h.0.attn.c_proj.weight","wte.weight","wpe.weight","h.10.mlp.c_proj.weight","h.11.mlp.c_proj.weight"]:
    print(k, anomalies(sd[k]))
