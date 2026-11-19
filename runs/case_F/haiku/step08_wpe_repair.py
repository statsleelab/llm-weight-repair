# Repair wpe.weight by detecting isolated spikes against a local (median-of-neighbours) prediction.
import torch, sys
torch.set_num_threads(1)
from safetensors.torch import load_file
sd = load_file("damaged.safetensors")
W = sd["wpe.weight"].float().clone()          # (1024, 768): position x dim
T, D = W.shape
def core_std(x):
    q = torch.quantile(x[::5], torch.tensor([.25,.75]))
    return ((q[1]-q[0])/1.349).item()
K = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
shifts = [-2,-1,1,2]
def neighbour_stack(X, bad):
    # values at t+d for d in shifts, NaN where out of range or flagged
    out = []
    for d in shifts:
        V = torch.full_like(X, float("nan"))
        if d < 0: V[-d:] = X[:d]
        else:     V[:-d] = X[d:]
        if bad is not None:
            Bd = torch.ones_like(bad)
            if d < 0: Bd[-d:] = bad[:d]
            else:     Bd[:-d] = bad[d:]
            V[Bd] = float("nan")
        out.append(V)
    return torch.stack(out, 0)
# pass 1: detection with all neighbours
bad = None
for it in range(3):
    pred = torch.nanmedian(neighbour_stack(W, bad), dim=0).values
    res = W - pred
    res = torch.nan_to_num(res, nan=0.0)
    scale = torch.tensor([core_std(res[:, j]) for j in range(D)]) + 1e-4
    newbad = res.abs() > K * scale[None, :]
    print("iter", it, "flagged", newbad.sum().item(), "frac %", round(newbad.float().mean().item()*100, 3), flush=True)
    bad = newbad
# final repair: replace flagged entries by the median of unflagged neighbours
pred = torch.nanmedian(neighbour_stack(W, bad), dim=0).values
Wf = W.clone()
Wf[bad] = pred[bad]
Wf = torch.where(torch.isnan(Wf), W, Wf)
torch.save({"W": Wf, "mask": bad}, "cache/wpe_fixed.pt")
print("wpe flagged total", bad.sum().item(), "max|W| before", W.abs().max().item(), "after", Wf.abs().max().item())
print("per-column flagged quantiles", torch.quantile(bad.sum(0).float(), torch.tensor([0.,.5,.9,1.])).tolist())
