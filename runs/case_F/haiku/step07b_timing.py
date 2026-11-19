import torch, time
torch.set_num_threads(1)
t=time.time()
X = torch.randn(768,2304)
U,S,Vt = torch.linalg.svd(X, full_matrices=False)
print("torch svd 768x2304", round(time.time()-t,2), flush=True)
t=time.time()
from safetensors.torch import load_file
sd = load_file("damaged.safetensors")
print("load", round(time.time()-t,2), flush=True)
