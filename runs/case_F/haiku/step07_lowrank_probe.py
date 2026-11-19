import torch, time, sys
torch.set_num_threads(1)
from safetensors.torch import load_file
sd = load_file("damaged.safetensors")
def core_std(x):
    q = torch.quantile(x[::7], torch.tensor([.25,.75]))
    return ((q[1]-q[0])/1.349).item()
for k in sys.argv[1:]:
    t0=time.time()
    W = sd[k].float()
    s0 = core_std(W.flatten()); med = W.median().item()
    mask = (W-med).abs() > 4.5*s0
    X = W.clone(); X[mask] = 0.0
    print(k, "s0", round(s0,4), "flag%", round(mask.float().mean().item()*100,2), flush=True)
    for r in [32, 96]:
        Y = X.clone()
        for it in range(6):
            U,S,Vt = torch.linalg.svd(Y, full_matrices=False)
            L = (U[:,:r]*S[:r]) @ Vt[:r]
            Y[mask] = L[mask]
        R = (W-L)[~mask]
        sr = core_std(R)
        print(f"  r={r} core_sd_resid={sr:.4f} ratio={sr/s0:.2f} frac|res|>4.5sr={(R.abs()>4.5*sr).float().mean().item()*100:.2f}% energy={((S[:r]**2).sum()/(S**2).sum()).item():.3f} t={round(time.time()-t0,1)}", flush=True)
