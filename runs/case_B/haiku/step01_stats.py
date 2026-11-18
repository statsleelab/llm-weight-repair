import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
print(len(sd))
for k, v in sd.items():
    if v.dim() != 2 and not k.endswith("bias"):
        pass
    f = v.flatten().double()
    s = f.std().item()
    mx = f.abs().max().item()
    z = ((f - f.mean()).abs() / s)
    kurt = (((f-f.mean())/s)**4).mean().item()
    print(f"{k:32s} {str(tuple(v.shape)):14s} mean={f.mean().item():+.4f} std={s:.4f} max|.|={mx:.3f} maxz={z.max().item():.1f} kurt={kurt:.2f} n>6sd={(z>6).sum().item()} zeros={(f==0).sum().item()} nan={torch.isnan(f).sum().item()}")
