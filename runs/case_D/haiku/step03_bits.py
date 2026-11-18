import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from safetensors.torch import load_file
import numpy as np
raw = load_file("damaged.safetensors")
for k in ["h.0.attn.c_proj.weight","h.1.mlp.c_proj.weight","h.0.mlp.c_fc.weight","wte.weight","wpe.weight","h.10.attn.c_attn.weight"]:
    a = raw[k].numpy().reshape(-1)
    bits = a.view(np.uint32)
    low16 = (bits & 0xFFFF)
    print(k, "frac low16==0:", (low16==0).mean(), "frac low8==0:", ((bits&0xFF)==0).mean(), "frac exp==0 or 255:", (((bits>>23)&0xFF)%255==0).mean())
    # quantiles of |w|
    q = np.quantile(np.abs(a), [0.5,0.9,0.99,0.999,0.9999,1.0])
    print("  |w| quantiles 50/90/99/99.9/99.99/max:", np.round(q,4))
    # gaussian reference quantile of |w| based on std of central 98%
    lo,hi = np.quantile(a,[0.01,0.99]); core = a[(a>lo)&(a<hi)]
    s = core.std()
    print("  gaussian-ref |w| quantiles 50/90/99/99.9:", np.round(np.array([0.674,1.645,2.576,3.29])*s,4), "core std", round(float(s),4))
