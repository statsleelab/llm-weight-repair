# Final repair: for every damaged 2-D tensor, (1) locate the zeroed block, (2) choose ridge strength and
# regression direction by hold-out error on three disjoint same-shape blocks elsewhere in the same matrix
# (linear algebra on the intact weights only), (3) fill the damaged block with the regression prediction,
# (4) write repaired.safetensors with identical keys/shapes/float32.
import sys, json
sys.path.insert(0, "/home/user/runs/case_C/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights
from step04_block_repair_validate import find_block

SRC = "/home/user/runs/case_C/haiku/damaged.safetensors"
DST = "/home/user/runs/case_C/haiku/repaired.safetensors"
LAMS = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0]

def make_predictor(W, Rb, Cb, Ro, Co):
    """Returns f(lam) -> prediction of W[Rb,Cb] from W[Rb,Co], ridge fit on rows Ro (regress W[Ro,Cb] on W[Ro,Co]).
    Uses one eigendecomposition shared over the lambda grid."""
    A = W[Ro][:, Co].double(); Y = W[Ro][:, Cb].double(); X = W[Rb][:, Co].double()
    mA, mY = A.mean(0), Y.mean(0)
    Ac, Yc, Xc = A - mA, Y - mY, X - mA
    n, p = Ac.shape
    s = (Ac.pow(2).sum() / (n * p)).item()
    if p <= n:
        e, V = torch.linalg.eigh(Ac.T @ Ac)
        Q = V.T @ (Ac.T @ Yc)
        XV = Xc @ V
        return lambda lam: XV @ (Q / (e + lam * s * n)[:, None]) + mY
    else:
        e, U = torch.linalg.eigh(Ac @ Ac.T)
        Z = U.T @ Yc
        XA = (Xc @ Ac.T) @ U
        return lambda lam: XA @ (Z / (e + lam * s * n)[:, None]) + mY

def run_one(name, W):
    W = W.double()
    Rb, Cb = find_block(W.float())
    a, b = len(Rb), len(Cb); R, C = W.shape
    # three disjoint validation placements (rows and cols avoid the damaged block and each other)
    cr = [s for s in range(R - a + 1) if (s + a <= Rb[0]) or (s > Rb[-1])]
    cc = [s for s in range(C - b + 1) if (s + b <= Cb[0]) or (s > Cb[-1])]
    fr = [0.2, 0.5, 0.8]
    vals = []
    for f in fr:
        rv = torch.arange(cr[int(f * (len(cr) - 1))], cr[int(f * (len(cr) - 1))] + a)
        cv = torch.arange(cc[int(f * (len(cc) - 1))], cc[int(f * (len(cc) - 1))] + b)
        vals.append((rv, cv))
    # validation: hold out each block (plus the real damaged block) and predict it from intact parts
    errs = {}
    for rv, cv in vals:
        truth = W[rv][:, cv].clone()
        Ro = torch.tensor([i for i in range(R) if not (Rb[0] <= i <= Rb[-1]) and not (rv[0] <= i <= rv[-1])])
        Co = torch.tensor([j for j in range(C) if not (Cb[0] <= j <= Cb[-1]) and not (cv[0] <= j <= cv[-1])])
        fc = make_predictor(W, rv, cv, Ro, Co)
        fr_ = make_predictor(W.T, cv, rv, Co, Ro)
        for lam in LAMS:
            pc = fc(lam); pr = fr_(lam).T
            for d, p in [("col", pc), ("row", pr), ("both", 0.5 * (pc + pr))]:
                errs.setdefault((lam, d), []).append((p - truth).pow(2).mean().sqrt().item())
    zero_rms = sum(
        W[rv][:, cv].pow(2).mean().sqrt().item() for rv, cv in vals) / len(vals)
    best = min(errs, key=lambda k: sum(errs[k]) / len(errs[k]))
    best_err = sum(errs[best]) / len(errs[best])
    lam, d = best
    # apply to the real damaged block: Ro = all rows outside Rb, Co = all cols outside Cb
    Ro = torch.tensor([i for i in range(R) if not (Rb[0] <= i <= Rb[-1])])
    Co = torch.tensor([j for j in range(C) if not (Cb[0] <= j <= Cb[-1])])
    pc = make_predictor(W, Rb, Cb, Ro, Co)(lam)
    pr = make_predictor(W.T, Cb, Rb, Co, Ro)(lam).T
    pred = {"col": pc, "row": pr, "both": 0.5 * (pc + pr)}[d]
    out = W.float().clone()
    out[Rb.unsqueeze(1), Cb.unsqueeze(0)] = pred.float()
    info = dict(block_rows=[int(Rb[0]), int(Rb[-1])], block_cols=[int(Cb[0]), int(Cb[-1])],
                lam=lam, direction=d, val_rmse=round(best_err, 5), zero_fill_val_rms=round(zero_rms, 5),
                pred_rms=round(pred.pow(2).mean().sqrt().item(), 5))
    return out, info

if __name__ == "__main__":
    sd = load_weights(SRC)
    rep = {}
    report = {}
    for k, v in sd.items():
        if v.dim() == 2:
            out, info = run_one(k, v)
            rep[k] = out
            report[k] = info
            print(k, json.dumps(info), flush=True)
        else:
            rep[k] = v
    save_weights(rep, DST)
    json.dump(report, open("/home/user/runs/case_C/haiku/repair_info.json", "w"), indent=1)
    print("saved", DST, "keys", len(rep))
