import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
w=sd["wpe.weight"]
nb2=(4*(w.roll(1,0)+w.roll(-1,0))-(w.roll(2,0)+w.roll(-2,0)))/6
r=(w-nb2); r[:3]=0; r[-3:]=0
cr=r.abs().mean(0); rr=r.abs().mean(1)
print("col roughness sorted top", [(int(i),round(float(cr[i]),4)) for i in cr.argsort(descending=True)[:40]])
print("col roughness median", cr.median())
print("row roughness by 32-pos blocks", [round(float(rr[i:i+32].mean()),4) for i in range(0,1024,32)])
print("row top", [(int(i),round(float(rr[i]),4)) for i in rr.argsort(descending=True)[:30]])
