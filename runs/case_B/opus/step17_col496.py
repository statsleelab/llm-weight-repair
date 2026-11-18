import torch; torch.set_num_threads(1)
from gpt2_lib import *
D = load_weights("damaged.safetensors")
W=D["wte.weight"]
for j in [496,430]:
    c=W[:,j]; med=c.median()
    out=c[(c-med).abs()>0.02]
    print(j,"n outliers",len(out),"of",len(c), "frac %.4f"%(len(out)/len(c)))
    print(" sorted sample", [round(x,3) for x in out.sort().values[::max(1,len(out)//40)].tolist()])
    print(" ratio to median", [round(x,2) for x in (out/med).sort().values[::max(1,len(out)//40)].tolist()])
