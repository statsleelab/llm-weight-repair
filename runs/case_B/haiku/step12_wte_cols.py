import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
W = sd["wte.weight"].double()
med = W.median(dim=0).values
mad = (W-med).abs().median(dim=0).values*1.4826
for j in [496, 430, 314, 36, 374, 442, 107, 0, 1]:
    c = W[:,j]
    d = (c-med[j]).abs()
    qs = torch.quantile(c[::5], torch.tensor([0.0,0.05,0.25,0.5,0.75,0.95,1.0],dtype=torch.float64))
    print(j, "q", [round(v,3) for v in qs.tolist()], "frac |dev|<0.01:", round((d<0.01).double().mean().item(),3), "frac|dev|>0.5:", round((d>0.5).double().mean().item(),4))
# how many wte entries far (>0.6) from column median overall, per column-count distribution
far = ((W-med).abs()>0.6)
cnt = far.sum(0)
print("columns: count of |dev|>0.6 quantiles", torch.quantile(cnt.double(), torch.tensor([0.0,0.5,0.9,0.99,1.0],dtype=torch.float64)).tolist(), "total", far.sum().item())
print("top columns by far count", torch.argsort(-cnt)[:15].tolist(), cnt[torch.argsort(-cnt)[:15]].tolist())
