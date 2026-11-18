import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
sd = load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()!=2: continue
    w = v if k!="wte.weight" else v[:20000]
    z,s = zscore(w)
    a=z.abs()
    print(f"{k:26s} " + " ".join(f">{t}:{(a>t).float().mean()*100:.3f}" for t in [3,4,5,6,8,10,15,20,40,100]))
