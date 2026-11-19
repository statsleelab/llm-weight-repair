import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import load_weights, Tokenizer, forward, EOT
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
ids = [EOT] + tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:200]
x = torch.tensor([ids[:-1]]); tgt = torch.tensor(ids[1:])
logits, hidden = forward(sd, x, return_hidden=True)
L = logits[0]
print("nll per pos (first 12):", [round(float(v),1) for v in F.cross_entropy(L, tgt, reduction='none')[:12]])
print("target logit (first 12):", [round(float(L[i, tgt[i]]),1) for i in range(12)])
print("max logit (first 12):", [round(float(L[i].max()),1) for i in range(12)])
am = L.argmax(-1)
print("argmax ids (first 12):", am[:12].tolist(), [tok.decode([int(a)]) for a in am[:12]])
print("distinct argmax over positions:", len(set(am.tolist())), "of", len(am))
# ln_f output
xf = F.layer_norm(hidden[-1][0], (768,), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5)
print("ln_f gamma abs max", sd["ln_f.weight"].abs().max().item(), "mean", sd["ln_f.weight"].mean().item())
print("ln_f out norm mean", xf.norm(dim=-1).mean().item(), "common-mode norm", xf.mean(0).norm().item())
# which tokens dominate logits: count of argmax by token
vals, cnt = torch.unique(am, return_counts=True)
top = cnt.argsort(descending=True)[:5]
print("most frequent argmax tokens:", [(tok.decode([int(vals[i])]), int(cnt[i])) for i in top])
