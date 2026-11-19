import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
ths = [0.5, 1.0, 2.0, 3.0, 5.0]
print("name, count(|w|>t) for t in", ths, "; top5 |w| with (row,col)")
for k, W in sd.items():
    if W.dim() != 2: continue
    a = W.abs()
    cnts = [int((a > t).sum()) for t in ths]
    v, idx = a.flatten().topk(5)
    pos = [(int(i)//W.shape[1], int(i)%W.shape[1], round(float(W.flatten()[i]),3)) for i in idx]
    print(f"{k:26s} {cnts} {pos}")
