import torch
torch.set_num_threads(1)
from safetensors import safe_open
f = safe_open("damaged.safetensors", "pt")
W = f.get_tensor("wpe.weight").float()
rn = W.pow(2).mean(1).sqrt()
print("wpe row RMS first 12:", [round(x,3) for x in rn[:12].tolist()])
print("wpe row RMS at 100,300,500,700,900,1023:", [round(rn[i].item(),3) for i in (100,300,500,700,900,1023)])
print("row RMS quantiles 0,.1,.5,.9,1:", [round(x,4) for x in torch.quantile(rn, torch.tensor([0,.1,.5,.9,1.])).tolist()])
cn = W.pow(2).mean(0).sqrt()
print("wpe col RMS quantiles:", [round(x,4) for x in torch.quantile(cn, torch.tensor([0,.1,.5,.9,1.])).tolist()])
a = W.abs()
print("wpe |w| quantiles .5/.9/.99/.999:", [round(x,4) for x in torch.quantile(a.flatten()[:1000000], torch.tensor([.5,.9,.99,.999])).tolist()])
print("per-row max abs first 12:", [round(x,3) for x in a.max(1).values[:12].tolist()])
