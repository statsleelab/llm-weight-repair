import torch, shutil
torch.set_num_threads(1)
from safetensors.torch import load_file
shutil.copy("cand_B_ridgeknnwte.safetensors","repaired.safetensors")
a=load_file("damaged.safetensors"); b=load_file("repaired.safetensors")
assert a.keys()==b.keys() and all(a[k].shape==b[k].shape and b[k].dtype==torch.float32 for k in a)
assert all(torch.equal(a[k],b[k]) for k in a if a[k].dim()==1)
print("ok", sum((a[k]!=b[k]).sum().item() for k in a), "entries changed")
