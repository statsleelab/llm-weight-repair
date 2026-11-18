import torch; torch.set_num_threads(1)
from gpt2_lib import *
D = load_weights("damaged.safetensors")
W=D["wte.weight"]
med=W.median(0).values; mad=(W-med).abs().median(0).values*1.4826
print("cols with largest |median|/mad:", ( med.abs()/mad).topk(8))
for j in [496,481,64,430,138,447,373]:
    c=W[:,j]
    print(j,"median %.3f mad %.3f"%(med[j],mad[j]),"q", [round(x,3) for x in torch.quantile(c,torch.tensor([.001,.01,.1,.5,.9,.99,.999])).tolist()], "n|c-med|>6mad", ((c-med[j]).abs()>6*mad[j]).sum().item())
