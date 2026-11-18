# Wider ridge grid, wte validation, and singular-value energy of intact matrices (to judge low-rank completion).
import sys
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
from step04_block_repair_validate import find_block, predict
SD = "/home/user/runs/case_C/haiku/damaged.safetensors"
sd = load_weights(SD)

def holdout(name, W, lams):
    Rb, Cb = find_block(W)
    a, b = len(Rb), len(Cb); R, C = W.shape
    cr = [s for s in range(R - a + 1) if (s + a <= Rb[0]) or (s > Rb[-1])]
    cc = [s for s in range(C - b + 1) if (s + b <= Cb[0]) or (s > Cb[-1])]
    rv = torch.arange(cr[len(cr)//2], cr[len(cr)//2] + a); cv = torch.arange(cc[len(cc)//2], cc[len(cc)//2] + b)
    truth = W[rv][:, cv].double()
    W2 = W.clone().double()
    W2[Rb.unsqueeze(1), Cb.unsqueeze(0)] = 0; W2[rv.unsqueeze(1), cv.unsqueeze(0)] = 0
    Ro = torch.tensor([i for i in range(R) if not (Rb[0] <= i <= Rb[-1]) and not (rv[0] <= i <= rv[-1])])
    Co = torch.tensor([j for j in range(C) if not (Cb[0] <= j <= Cb[-1]) and not (cv[0] <= j <= cv[-1])])
    res = {}
    for lam in lams:
        pc = predict(W2, rv, cv, lam, Ro, Co)
        pr = predict(W2.T, cv, rv, lam, Co, Ro).T
        res[lam] = (round((pc - truth).pow(2).mean().sqrt().item(), 5),
                    round((pr - truth).pow(2).mean().sqrt().item(), 5),
                    round((0.5*(pc+pr) - truth).pow(2).mean().sqrt().item(), 5))
    print(f"{name} zero-fill RMS={truth.pow(2).mean().sqrt().item():.5f}  (lam: col,row,both)")
    for l in lams: print("   ", l, res[l], flush=True)

lams = [0.3, 1, 3, 10, 30, 100]
for nm in ["h.0.attn.c_attn.weight", "h.0.mlp.c_fc.weight"]:
    holdout(nm, sd[nm], lams)
print("wte")
holdout("wte.weight", sd["wte.weight"], [1e-3, 1e-2, 0.1, 1])

print("singular energy (fraction of ||W||_F^2 in top k) for intact-ish matrices")
for nm in ["h.0.attn.c_attn.weight", "h.0.mlp.c_fc.weight", "h.0.attn.c_proj.weight", "wte.weight"]:
    W = sd[nm].double()
    W = W.clone()
    s = torch.linalg.svdvals(W)
    e = (s**2).cumsum(0) / (s**2).sum()
    print(nm, {k: round(e[k-1].item(), 3) for k in [8, 32, 64, 128, 256]})
