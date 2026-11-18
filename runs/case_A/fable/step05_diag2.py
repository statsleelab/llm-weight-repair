import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
x=torch.tensor([[EOT]+ids[:256]])
logits,hid=forward(sd,x,return_hidden=True)
W=sd["wte.weight"]
rn=W.norm(dim=1); print("wte row norm quantiles",[round(float(rn.quantile(q)),2) for q in (0,.01,.5,.99,1)])
d=tok.encode(" Dear")[0]; print("Dear id",d,"norm",rn[d].item())
f=F.layer_norm(hid[-1],(768,),sd["ln_f.weight"],sd["ln_f.bias"],1e-5)[0]
print("final normed norm med",f.norm(dim=-1).median().item())
print("logit stats: max",logits[0].max(-1).values.median().item(),"mean",logits[0].mean().item(),"std",logits[0].std().item())
p=50
contrib=f[p]*W[d]; t=contrib.abs().topk(8)
print("contribs to Dear logit",[(int(i),round(float(contrib[i]),1),round(float(f[p,i]),1),round(float(W[d,i]),2)) for i in t.indices], "sum",contrib.sum().item())
print("ln_f weight top",[(int(i),round(float(v),2)) for v,i in zip(*sd["ln_f.weight"].abs().topk(8))])
print("ln_f.w median",sd["ln_f.weight"].median().item())
# pre-ln_f final hidden top dims at pos p
h=hid[-1][0,p]; t=h.abs().topk(10); print("final hid top",[(int(i),round(float(h[i]),1)) for i in t.indices])
# per layer growth for a normal position: attn vs mlp contributions norms
for i in range(1,13):
    dlt=(hid[i]-hid[i-1])[0,p]; t=dlt.abs().topk(4)
    print(i-1,"delta norm %.1f"%dlt.norm(),[(int(j),round(float(dlt[j]),1)) for j in t.indices])
