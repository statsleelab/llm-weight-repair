import torch, math
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
r = repair_all_v3(sd, 6.0)
nll, acc = nll_windows(r, ids, max_tokens=2048); print(f"v3 (cap rate .001) t=6 ppl {math.exp(nll):.3f} acc {acc:.4f}", flush=True)
x = torch.tensor([[EOT]+ids[:255]])
lg, hid = forward(r, x, return_hidden=True)
f = F.layer_norm(hid[-1], (768,), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5)[0]
top = lg[0,1:].argmax(-1); print("top tokens", [tok.decode([i]) for i in top[:30].tolist()])
W = r["wte.weight"]; cm, cs = rsig(W, 0)
lp = lg[0].log_softmax(-1); nl = -lp[:-1].gather(-1, x[0,1:,None])[:,0]; print("nll mean/median", nl.mean().item(), nl.median().item())
for sh in top.unique()[:0].tolist() + [top.mode().values.item()]:
    p = (top==sh).nonzero()[0].item()+1
    contrib = f[p]*W[sh]; d = torch.topk(contrib.abs(),5).indices
    print(repr(tok.decode([sh])), "logit", lg[0,p,sh].item(), "dims", d.tolist(), "contrib", contrib[d].tolist(), "vals", W[sh][d].tolist(), "colmed", cm[0,d].tolist(), "colsig", cs[0,d].tolist())
# residual wte outliers per hot column
for d in [496,430,314,36,374,442]:
    z = (W[:,d]-cm[0,d]).abs()/cs[0,d]
    print("col", d, "sig", cs[0,d].item(), "n z>6:", int((z>6).sum()), "n z>20:", int((z>20).sum()), "max dev", (W[:,d]-cm[0,d]).abs().max().item(), "dmg n z>20", int((((sd['wte.weight'][:,d]-cm[0,d]).abs()/cs[0,d])>20).sum()))
