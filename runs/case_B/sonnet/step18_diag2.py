import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_heavy.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:200]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
for i,s in enumerate(h):
    t=s[0,0].abs().topk(3)
    print(i,"pos0 top",[(int(j),round(float(v),1)) for v,j in zip(t.values,t.indices)],"others rms %.1f max %.1f"%(s[0,1:].pow(2).mean().sqrt(),s[0,1:].abs().max()))
