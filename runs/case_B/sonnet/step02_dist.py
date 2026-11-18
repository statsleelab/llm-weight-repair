import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
for k in ['wte.weight','h.0.mlp.c_fc.weight','h.7.attn.c_proj.weight']:
    a=sd[k].flatten().numpy()
    med=np.median(np.abs(a)); print(k,"median abs",med,"robust sigma",med/0.6745)
    print(np.histogram(np.abs(a),bins=[0,.1,.2,.4,.8,1,1.5,2,3,4,6,10])[0])
    # sign / exactness patterns
    print("exact mult of 1/ some?", np.unique(a).size, a.size)
    big=np.sort(np.abs(a))[-10:]; print(big)
w=sd['h.7.attn.c_proj.weight']
print("row/col frac of |w|>1:", (w.abs()>1).float().mean(0)[:20], (w.abs()>1).float().mean(1)[:20])
