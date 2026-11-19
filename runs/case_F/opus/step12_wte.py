import torch
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
W=sd["wte.weight"]
mask,_,gm=detect("wte.weight",W,4.0,True)
Wr=W.clone(); Wr[mask]=0
print("protected cols? flagged per col top-heavy", mask.sum(0)[[138,378,447]], "mean",mask.sum(0).float().mean())
Wn=Wr-Wr.mean(0); Wn=Wn/Wn.norm(dim=1,keepdim=True)
for w in [" king"," France"," the"," dog","ing"]:
    i=tok.encode(w)[0]; s=Wn@Wn[i]; top=s.topk(6).indices.tolist()
    print(repr(w),[tok.decode([t]) for t in top])
P=sd["wpe.weight"]
print("wpe row norms",P.norm(dim=1)[:8], P.norm(dim=1)[500:504])
