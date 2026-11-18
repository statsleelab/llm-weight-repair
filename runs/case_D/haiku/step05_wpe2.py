import torch, sys
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from safetensors.torch import load_file
import numpy as np
raw = load_file("damaged.safetensors")
W = raw["wpe.weight"].numpy().astype(np.float64)
rn = np.linalg.norm(W, axis=1)
print("argmin row norm:", rn.argmin(), rn.min(), " argmax:", rn.argmax())
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for d in [1,2,3,10,37,500]:
    c = np.mean([cos(W[i],W[i+d]) for i in range(1,1000-d)])
    print("mean cos lag", d, round(c,4))
# column-centred correlation (remove mean position embedding)
Wc = W - W[1:].mean(0)
for d in [1,2,10]:
    c = np.mean([cos(Wc[i],Wc[i+d]) for i in range(1,1000-d)])
    print("centred lag", d, round(c,4))
print("mean-vector norm:", np.linalg.norm(W[1:].mean(0)), "vs typical row norm 3.4")
