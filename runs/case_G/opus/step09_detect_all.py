import torch; torch.set_num_threads(1)
import numpy as np, json, time
from gpt2_lib import *
from detect_lib import *
sd=load_weights("damaged.safetensors")
out={}
for k,v in sd.items():
    if v.dim()!=2: continue
    w=v.numpy(); L=llr_map(w)
    tile=32 if k=="wte.weight" else 8
    sc,a,b,l,e=find_block(L,tile)
    L2=L.copy(); L2[a:b,l:e]=-5
    sc2,*_=find_block(L2,tile*2)
    out[k]=[int(a),int(b),int(l),int(e)]
    print("%-26s rows %4d-%4d cols %4d-%4d frac %.4f score %.0f  2nd-best %.0f"%(k,a,b,l,e,(b-a)*(e-l)/w.size,sc,sc2),flush=True)
json.dump(out,open("blocks.json","w"),indent=1)
