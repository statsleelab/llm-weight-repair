from common import *
sd=load_weights("damaged.safetensors")
for k,v in sd.items():
    if v.dim()==2:
        idx=(v.abs()>6.3).nonzero()
        if len(idx): print(k,[(i.tolist(),round(v[i[0],i[1]].item(),2)) for i in idx[:12]])
