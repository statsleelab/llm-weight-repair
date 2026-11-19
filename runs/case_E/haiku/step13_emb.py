import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, nll_windows
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
ids = tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:1024]
base = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
def ev(name, sd):
    nll, acc = nll_windows(sd, ids, window=512, max_tokens=1024)
    print(f"{name:40s} nll={nll:9.4f} top1={acc:.4f}", flush=True)
wpe = base["wpe.weight"]
cs = wpe.std(dim=0)
print("wpe per-column std: top5", cs.topk(5).values.tolist(), "idx", cs.topk(5).indices.tolist(), "median", cs.median().item())
rs = wpe.norm(dim=1)
print("wpe row norms pos0..5", [round(v,2) for v in rs[:6].tolist()], "pos 100,500,1000", [round(rs[i].item(),2) for i in [100,500,1000]])
print("wte per-column std: top5", base["wte.weight"].std(0).topk(5).values.tolist(), base["wte.weight"].std(0).topk(5).indices.tolist())
ev("baseline", base)
sd = dict(base); sd["wpe.weight"] = torch.zeros_like(wpe); ev("wpe zeroed", sd)
sd = dict(base); W = wpe.clone(); W[W.abs() > 0.3] = 0; sd["wpe.weight"] = W; ev("wpe |w|>0.3 zeroed", sd)
sd = dict(base); W = base["wte.weight"].clone(); W[W.abs() > 0.5] = 0; sd["wte.weight"] = W; ev("wte |w|>0.5 zeroed (again)", sd)
