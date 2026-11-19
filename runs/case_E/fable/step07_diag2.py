import torch, math
torch.set_num_threads(1)
import torch.nn.functional as F
from gpt2_lib import *
from repair_lib import *
tok = Tokenizer("tokenizer"); ids = tok.encode(open("dev.txt").read())
sd = load_weights("damaged.safetensors")
r = repair_all_v2(sd, 6.0)
x = torch.tensor([[EOT]+ids[:255]])
lg, hid = forward(r, x, return_hidden=True)
f = F.layer_norm(hid[-1], (768,), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5)[0]
print("ln_f.weight top", torch.topk(sd["ln_f.weight"].abs(),6))
fa = f[1:].abs().mean(0); print("final act mean abs top dims", torch.topk(fa,8), "median", fa.median().item())
top = lg[0,1:].argmax(-1); print("top tokens", [tok.decode([i]) for i in top[:30].tolist()])
sh = top.mode().values.item(); W = r["wte.weight"]
contrib = f[100]*W[sh]; print("token", sh, repr(tok.decode([sh])), "logit", lg[0,100,sh].item(), "top contrib dims", torch.topk(contrib.abs(),6), "row vals there", W[sh][torch.topk(contrib.abs(),6).indices])
cm, cs = rsig(W, 0)
d = torch.topk(contrib.abs(),6).indices
print("col med", cm[0,d], "col sig", cs[0,d])
# per layer where things go: last-layer contributions
for i in range(13): print(i, "hid absmax dim", hid[i][0,1:].abs().mean(0).argmax().item(), round(hid[i][0,1:].abs().mean(0).max().item(),1), "pos0 absmax dim", hid[i][0,0].abs().argmax().item(), round(hid[i][0,0].abs().max().item(),1))
print("wpe[0] norm dmg", sd["wpe.weight"][0].norm().item(), "wpe[1]", sd["wpe.weight"][1].norm().item(), "wte[EOT] norm", sd["wte.weight"][EOT].norm().item(), W[EOT].norm().item())
print("wpe row norms dmg first 8", sd["wpe.weight"][:8].norm(dim=1), "rep", r["wpe.weight"][:8].norm(dim=1))
