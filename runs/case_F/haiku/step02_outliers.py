import torch
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
sd = load_file("damaged.safetensors")
np.set_printoptions(precision=4, suppress=True, linewidth=150)
names = [k for k in sd if sd[k].dim()==2]
for k in names[:]:
    a = sd[k].numpy().astype(np.float64).ravel()
    med = np.median(a); mad = np.median(np.abs(a-med))*1.4826
    z = np.abs(a-med)/mad
    fr = [(z>t).mean()*100 for t in (4,6,10,20)]
    # gaussian expectation for 4 sigma ~0.006%
    print(f"{k:28s} robust_std={mad:.4f} std={a.std():.4f} kurt-ish: frac>4s={fr[0]:.3f}% >6s={fr[1]:.3f}% >10s={fr[2]:.3f}% >20s={fr[3]:.3f}%")
