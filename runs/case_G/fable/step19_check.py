import torch, json
torch.set_num_threads(1)
from gpt2_lib import *
a=load_weights("damaged.safetensors"); b=load_weights("repaired.safetensors"); B=json.load(open("blocks_final.json"))
assert a.keys()==b.keys()
tot=0;ch=0
for k in a:
    assert a[k].shape==b[k].shape and b[k].dtype==torch.float32 and torch.isfinite(b[k]).all()
    d=(a[k]!=b[k]).sum().item(); ch+=d
    if a[k].ndim==2: tot+=a[k].numel()
    if k not in B: assert d==0
print("ok; changed entries",ch,"fraction of 2-D entries",ch/tot)
tok=Tokenizer("tokenizer")
print(generate(b,tok,"The capital of France is",12))
