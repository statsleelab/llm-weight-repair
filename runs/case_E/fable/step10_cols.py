import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors"); W = sd["wte.weight"]
for d in [496,430,314,36]:
    c = W[:,d].numpy(); med = np.median(c)
    print("col", d, "median", med, "quantiles .001 .01 .25 .75 .99 .999", np.quantile(c,[.001,.01,.25,.75,.99,.999]).round(4))
    far = np.abs(c-med) > 0.05
    print("   n far(>0.05):", far.sum(), " n near -median (±0.02):", (np.abs(c+med)<0.02).sum(), " n |c|<0.02:", (np.abs(c)<0.02).sum(), " n |c|>0.5:", (np.abs(c)>0.5).sum())
    fv = np.sort(c[far]); print("   far values sample:", fv[::max(1,len(fv)//40)].round(3))
