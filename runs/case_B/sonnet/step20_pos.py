import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("cand_heavy.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:60]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
for L in [1,2,3]:
    m=h[L][0].abs().max(-1)
    print("layer",L,"positions with max>60:",[(i,int(m.indices[i]),round(float(m.values[i]))) for i in range(61) if m.values[i]>60])
print([tok.decode([t]) for t in x[0,:12]])
