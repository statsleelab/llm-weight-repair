import torch,numpy as np,json
torch.set_num_threads(1)
from gpt2_lib import *
from blk import tilecorr
from blk2 import *
sd=load_weights("damaged.safetensors")
boxes={}
for k,w in sd.items():
    if w.ndim!=2 or "attn" in k: continue
    if k=="wte.weight":
        cr,cc=(0,50257),(576,752)
    elif k=="wpe.weight":
        boxes[k]=([124,303],[591,724]);continue
    else:
        th=64 if w.shape[0]>=3072 else 32; tw=64 if w.shape[1]>=3072 else 32
        c=tilecorr(w,th,tw); med=c.median().item()
        ii=((c<0.35*med)|(c<0.05)).nonzero()
        cr=(ii[:,0].min().item()*th,(ii[:,0].max().item()+1)*th); cc=(ii[:,1].min().item()*tw,(ii[:,1].max().item()+1)*tw)
    rr,c2=find_box(w,cr,cc)
    boxes[k]=(rr,c2)
    print(k,"coarse",cr,cc,"->",rr,c2,"area frac %.4f"%((rr[1]-rr[0])*(c2[1]-c2[0])/w.numel()),flush=True)
json.dump(boxes,open("boxes.json","w"))
