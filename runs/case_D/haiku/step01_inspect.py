import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from gpt2_lib import load_weights
import numpy as np
sd = load_weights("damaged.safetensors")
print(len(sd))
for k, v in sd.items():
    a = v.numpy()
    print(f"{k:35s} {str(tuple(a.shape)):16s} mean={a.mean():+.4f} std={a.std():.4f} min={a.min():+.3f} max={a.max():+.3f} zeros={(a==0).mean():.4f} nonfinite={(~np.isfinite(a)).sum()}")
