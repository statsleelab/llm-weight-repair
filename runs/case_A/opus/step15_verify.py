import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
from safetensors.torch import load_file
d=load_file("damaged.safetensors"); r=load_file("repaired.safetensors")
assert d.keys()==r.keys()
for k in d: assert d[k].shape==r[k].shape and r[k].dtype==torch.float32, k
changed=sum(((d[k]!=r[k])).sum().item() for k in d); tot=sum(v.numel() for k,v in d.items() if v.dim()==2)
print("changed entries",changed,"of",tot,"zeros left",sum((r[k]==0).sum().item() for k in r if r[k].dim()==2))
print("1-D params identical:", all(torch.equal(d[k],r[k]) for k in d if d[k].dim()==1))
sd={k:v.float() for k,v in r.items()}; tok=Tokenizer("tokenizer")
for p in ["The capital of France is","The European lobster is a species of"]: print(generate(sd,tok,p,15))
