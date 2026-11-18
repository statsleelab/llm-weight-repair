import torch, math; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
for k in ["h.0.attn.c_attn.weight","h.5.mlp.c_proj.weight","wte.weight","wpe.weight"]:
    m=(sd[k]==0)
    r=m.float().mean(1); c=m.float().mean(0)
    print(k, "row frac min/max/std", r.min().item(), r.max().item(), r.std().item(), "col", c.min().item(), c.max().item(), c.std().item())
    # clustering: adjacent zeros horizontally vs random expectation
    adj=(m[:,1:]&m[:,:-1]).float().mean().item(); p=m.float().mean().item()
    adjv=(m[1:,:]&m[:-1,:]).float().mean().item()
    print("  adj-h", adj, "adj-v", adjv, "expected", p*p)
# Check biases have zeros?
print("bias zeros", sum((v==0).sum().item() for k,v in sd.items() if v.dim()==1))
# Look for other anomalies: values duplicated, sign flips not detectable directly. Check magnitude distribution vs row-wise
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read()); print("dev tokens", len(ids))
nll,acc=nll_windows(sd, ids, max_tokens=8192)
print("damaged ppl", math.exp(nll), acc)
