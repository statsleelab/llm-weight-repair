import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
w=sd["wpe.weight"]
nb2=(4*(w.roll(1,0)+w.roll(-1,0))-(w.roll(2,0)+w.roll(-2,0)))/6
r=(w-nb2).abs(); r[:3]=0; r[-3:]=0
print("rows 140-160 roughness over cols 50-145:", [round(float(r[p,50:145].mean()),3) for p in range(140,162)])
print("rows 245-265:", [round(float(r[p,50:145].mean()),3) for p in range(245,265)])
print("cols 50-62 over rows 160-240:", [round(float(r[160:240,c].mean()),3) for c in range(50,64)])
print("cols 128-145:", [round(float(r[160:240,c].mean()),3) for c in range(128,146)])
