import torch, numpy as np, re
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
txt=open("step13_out.txt").read()
rects={}
for line in txt.splitlines():
    m=re.match(r"(\S+)\s+score=\s*([\d.]+) rows=(\d+)-(\d+) cols=(\d+)-(\d+) area=([\d.]+)",line)
    if m:
        k=m.group(1); sc=float(m.group(2)); a,b,c,d=map(int,m.group(3,4,5,6)); ar=float(m.group(7))
        rects[k]=(sc,a,b,c,d,ar)
rects["wpe.weight"]=(99,151,252,59,136,0.0099)
acc={}
for k,(sc,a,b,c,d,ar) in rects.items():
    v=sd[k]; g=v.std().item(); B=v[a:b,c:d]/g
    kurt=((B-B.mean())**4).mean().item()/B.var().item()**2
    # contrast: std of same columns, rows just outside / same rows, cols outside
    ok = 0.004<=ar<=0.022 and abs(B.std().item()-1)<0.08 and abs(kurt-3)<0.4 and (b-a)<0.9*v.shape[0] and (d-c)<0.9*v.shape[1]
    print(f"{k:26s} sc={sc:5.1f} area={ar:.4f} blockstd/g={B.std().item():.3f} mean={B.mean().item():+.3f} kurt={kurt:.2f} {'ACCEPT' if ok else ''}")
    if ok: acc[k]=(a,b,c,d)
torch.save(acc,"accepted_rects.pt")
print(len(acc),"accepted")
