# Block-erasure repair.  Each 2-D tensor has a single zeroed rectangle Rb x Cb (verified below).
# Repair = ridge regression predicting the damaged columns Cb from the intact columns Co,
# fitted on the intact rows Ro, then applied to the damaged rows Rb (and the transposed analogue).
# Validation: hide a disjoint block of the same shape in the intact region and measure the error there.
import sys, json
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights
import numpy as np

SD = "/home/user/runs/case_C/haiku/damaged.safetensors"

def find_block(W):
    Z = (W == 0)
    rows = torch.nonzero(Z.any(1)).flatten()
    cols = torch.nonzero(Z.any(0)).flatten()
    Rb, Cb = rows, cols
    assert len(Rb) == Rb[-1] - Rb[0] + 1 and len(Cb) == Cb[-1] - Cb[0] + 1, "block not contiguous"
    blk = Z[Rb][:, Cb]
    assert blk.all(), "block not fully zero"
    assert int(Z.sum()) == len(Rb) * len(Cb), "zeros outside block"
    return torch.arange(Rb[0], Rb[-1] + 1), torch.arange(Cb[0], Cb[-1] + 1)

def predict(W, Rb, Cb, lam, Ro=None, Co=None):
    """Predict block W[Rb,Cb] from W[Rb,Co] using samples Ro: regress W[Ro,Cb] on W[Ro,Co] (ridge)."""
    n_rows = W.shape[0]
    if Ro is None:
        Ro = torch.tensor([i for i in range(n_rows) if not (Rb[0] <= i <= Rb[-1])])
    if Co is None:
        Co = torch.tensor([j for j in range(W.shape[1]) if not (Cb[0] <= j <= Cb[-1])])
    A = W[Ro][:, Co].double()          # n x p  (complete)
    Y = W[Ro][:, Cb].double()          # n x q  (complete)
    X = W[Rb][:, Co].double()          # m x p  (complete)
    mA, mY = A.mean(0), Y.mean(0)
    Ac, Yc = A - mA, Y - mY
    n, p = Ac.shape
    s = (Ac.pow(2).sum() / (n * p)).item()   # scale for relative ridge
    if p <= n:
        G = Ac.T @ Ac + lam * s * n * torch.eye(p, dtype=torch.float64)
        B = torch.linalg.solve(G, Ac.T @ Yc)
    else:
        K = Ac @ Ac.T + lam * s * n * torch.eye(n, dtype=torch.float64)
        B = Ac.T @ torch.linalg.solve(K, Yc)
    return (X - mA) @ B + mY

def block_predict(W, Rb, Cb, lam, Ro=None, Co=None, direction="both"):
    # column-direction: regress damaged columns on intact columns, samples = rows
    pc = predict(W, Rb, Cb, lam, Ro, Co)
    # row-direction: transpose roles
    WT = W.T
    pr = predict(WT, Cb, Rb, lam, Co, Ro).T
    if direction == "col": return pc
    if direction == "row": return pr
    return 0.5 * (pc + pr)

def validate(name, W):
    Rb, Cb = find_block(W)
    a, b = len(Rb), len(Cb)
    R, C = W.shape
    # validation block: shift to a disjoint region (rows/cols not overlapping the damaged block)
    rv0 = 0 if Rb[0] >= a + 1 and True else None
    cands_r = [s for s in range(0, R - a + 1) if (s + a <= Rb[0]) or (s > Rb[-1])]
    cands_c = [s for s in range(0, C - b + 1) if (s + b <= Cb[0]) or (s > Cb[-1])]
    rv = torch.arange(cands_r[len(cands_r)//2], cands_r[len(cands_r)//2] + a)
    cv = torch.arange(cands_c[len(cands_c)//2], cands_c[len(cands_c)//2] + b)
    truth = W[rv][:, cv].double().clone()
    # build a damaged copy with BOTH blocks removed, then predict validation block from intact parts
    W2 = W.clone().double()
    W2[Rb][:, Cb]  # noop
    W2[Rb.unsqueeze(1), Cb.unsqueeze(0)] = 0.0
    W2[rv.unsqueeze(1), cv.unsqueeze(0)] = 0.0
    Ro_v = torch.tensor([i for i in range(R) if not (Rb[0] <= i <= Rb[-1]) and not (rv[0] <= i <= rv[-1])])
    Co_v = torch.tensor([j for j in range(C) if not (Cb[0] <= j <= Cb[-1]) and not (cv[0] <= j <= cv[-1])])
    out = {}
    rms_true = truth.pow(2).mean().sqrt().item()
    out["zero_fill_rms"] = rms_true
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        for d in ["col", "row", "both"]:
            # predict validation block: rows rv, cols cv, using intact rows Ro_v / cols Co_v
            if d == "col":
                p = predict(W2, rv, cv, lam, Ro_v, Co_v)
            elif d == "row":
                p = predict(W2.T, cv, rv, lam, Co_v, Ro_v).T
            else:
                p = 0.5 * (predict(W2, rv, cv, lam, Ro_v, Co_v) + predict(W2.T, cv, rv, lam, Co_v, Ro_v).T)
            err = (p - truth).pow(2).mean().sqrt().item()
            out[f"lam={lam:g},{d}"] = round(err, 5)
    print(name, "block rows", (Rb[0].item(), Rb[-1].item()), "cols", (Cb[0].item(), Cb[-1].item()),
          "| val rows", (rv[0].item(), rv[-1].item()), "cols", (cv[0].item(), cv[-1].item()), flush=True)
    print("   ", json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    sd = load_weights(SD)
    names = sys.argv[1:] or ["h.0.attn.c_attn.weight", "h.0.attn.c_proj.weight", "wpe.weight", "h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight"]
    for nm in names:
        validate(nm, sd[nm])
