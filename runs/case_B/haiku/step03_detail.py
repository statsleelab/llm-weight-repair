import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
def rsd(x, dim=None):
    med = x.median(dim=dim, keepdim=True).values if dim is not None else x.median()
    return med, (x-med).abs().median(dim=dim, keepdim=True).values*1.4826 if dim is not None else (x-med).abs().median()*1.4826
# histogram of standardized values for a clean-looking matrix
w = sd["h.0.mlp.c_fc.weight"]
med, s = rsd(w)
z = ((w-med)/s).flatten()
edges = [0,1,2,2.5,3,3.5,4,5,6,8,10,20,100]
a = z.abs()
for lo,hi in zip(edges[:-1],edges[1:]):
    frac = ((a>=lo)&(a<hi)).float().mean().item()
    gauss = math.erfc(lo/math.sqrt(2))-math.erfc(hi/math.sqrt(2)) if hi<100 else math.erfc(lo/math.sqrt(2))
    print(f"|z| in [{lo},{hi}): frac={frac:.5f}  gauss={gauss/1:.5f}")
# per column robust scale of wte
wte = sd["wte.weight"]
medc, sc = rsd(wte, dim=0)
print("wte column robust sd:", [round(x,3) for x in sc.flatten().tolist()])
print("wte column medians max", medc.abs().max().item())
rowmed, rowsd = rsd(wte, dim=1)
print("wte row robust sd quantiles", torch.quantile(rowsd.flatten(), torch.tensor([0.0,0.01,0.5,0.99,1.0])).tolist())
# wpe
wpe = sd["wpe.weight"]
print("wpe per-row robust sd quantiles", torch.quantile(rsd(wpe,dim=1)[1].flatten(), torch.tensor([0.0,0.01,0.5,0.99,1.0])).tolist())
print("wpe per-col robust sd", [round(x,3) for x in rsd(wpe,dim=0)[1].flatten().tolist()[:12]])
