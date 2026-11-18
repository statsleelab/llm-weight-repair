import torch; torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
R=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5g.pt")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
x=torch.tensor([[EOT]+ids[:512]])
_,hid=forward(R,x[:,:-1],return_hidden=True)
tgt=x[0,1:]
for i,h in enumerate(hid):
    hh=F.layer_norm(h[0],(768,),R["ln_f.weight"],R["ln_f.bias"],1e-5)
    lg=hh@R["wte.weight"].T
    print(i, "lens nll %.2f"%F.cross_entropy(lg[1:],tgt[1:]).item(), "acc %.3f"%(lg[1:].argmax(-1)==tgt[1:]).float().mean().item())
# contribution norms of each sublayer at positions>0
