import torch,numpy as np
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_wide.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:400]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
for L in range(0,13):
    y=F.layer_norm(h[L],(768,),sd["ln_f.weight"],sd["ln_f.bias"],1e-5)@sd["wte.weight"].T
    n=F.cross_entropy(y[0,5:-1],x[0,6:]).item()
    print(L,"lens nll %.2f"%n,"acc %.3f"%(y[0,5:-1].argmax(-1)==x[0,6:]).float().mean().item())
