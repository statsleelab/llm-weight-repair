import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
import torch.nn.functional as F
sd = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:512]
logits,hid=forward(sd,torch.tensor([[EOT]+ids]),return_hidden=True)
h=F.layer_norm(hid[-1][0,1:],(768,),sd["ln_f.weight"],sd["ln_f.bias"])
m=h.mean(0); s=h.std(0)
top=m.abs().topk(8)
print("ln_f out dims with large mean:", [(int(i),round(float(m[i]),1),round(float(s[i]),1)) for i in top.indices])
E=sd["wte.weight"]
for d in top.indices[:5].tolist():
    col=E[:,d]; nz=col!=0
    print("wte col",d,"mean",round(col[nz].mean().item(),4),"std",round(col[nz].std().item(),4),"zeros",(~nz).sum().item())
dear=tok.encode(" Dear")+tok.encode("Dear"); print("Dear ids",dear, [(E[t]==0).nonzero().flatten().tolist() for t in dear])
