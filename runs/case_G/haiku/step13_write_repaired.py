import sys
sys.path.insert(0, "/home/user/runs/case_G/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from safetensors import safe_open
sd = load_weights("damaged.safetensors")
save_weights(sd, "repaired.safetensors")
a = safe_open("damaged.safetensors", "pt"); b = safe_open("repaired.safetensors", "pt")
ka, kb = set(a.keys()), set(b.keys())
assert ka == kb, "key mismatch"
bad = [k for k in ka if a.get_tensor(k).shape != b.get_tensor(k).shape or b.get_tensor(k).dtype != torch.float32]
print("keys", len(kb), "shape/dtype mismatches", bad)
print("identical to damaged:", all(torch.equal(a.get_tensor(k).float(), b.get_tensor(k).float()) for k in ka))
