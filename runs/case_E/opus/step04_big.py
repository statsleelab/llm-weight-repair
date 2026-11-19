import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    s=v.std(); m=(v.abs()/s)>28
    idx=m.nonzero()
    if len(idx): print(k, [(tuple(i.tolist()), round(v[tuple(i)].item()/s.item(),1)) for i in idx[:12]], len(idx))
