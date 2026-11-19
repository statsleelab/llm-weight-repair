import sys
sys.path.insert(0, "/home/user/runs/case_E/haiku")
import torch, math
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, gelu_new, EOT, N_EMBD, N_HEAD
tok = Tokenizer("/home/user/runs/case_E/haiku/tokenizer")
sd = load_weights("/home/user/runs/case_E/haiku/damaged.safetensors")
ids = torch.tensor([[EOT] + tok.encode(open("/home/user/runs/case_E/haiku/dev.txt", encoding="utf-8").read())[:127]])
T = ids.shape[1]
x = sd["wte.weight"][ids] + sd["wpe.weight"][:T]
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
with torch.no_grad():
  for i in range(12):
    p = f"h.{i}."
    h = F.layer_norm(x, (N_EMBD,), sd[p+"ln_1.weight"], sd[p+"ln_1.bias"], 1e-5)
    qkv = h @ sd[p+"attn.c_attn.weight"] + sd[p+"attn.c_attn.bias"]
    q,k,v = qkv.split(N_EMBD, dim=-1)
    q = q.view(1,T,N_HEAD,-1).transpose(1,2); k = k.view(1,T,N_HEAD,-1).transpose(1,2); v = v.view(1,T,N_HEAD,-1).transpose(1,2)
    att = (q @ k.transpose(-1,-2)) / math.sqrt(q.size(-1))
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    a = (att @ v).transpose(1,2).reshape(1,T,N_EMBD)
    ao = a @ sd[p+"attn.c_proj.weight"] + sd[p+"attn.c_proj.bias"]
    x = x + ao
    h = F.layer_norm(x, (N_EMBD,), sd[p+"ln_2.weight"], sd[p+"ln_2.bias"], 1e-5)
    m = gelu_new(h @ sd[p+"mlp.c_fc.weight"] + sd[p+"mlp.c_fc.bias"])
    mo = m @ sd[p+"mlp.c_proj.weight"] + sd[p+"mlp.c_proj.bias"]
    x = x + mo
    # common-mode: mean over positions 1..T-1 of each contribution
    def cm(z):
        z = z[0,1:]
        return f"norm={z.norm(dim=-1).mean():8.2f} meannorm={z.mean(0).norm():8.2f}"
    top = mo[0,1:].mean(0).abs().topk(3)
    print(f"L{i:2d} attn: {cm(ao)} | mlp: {cm(mo)} | mlp mean-dims {top.indices.tolist()} {[round(v,1) for v in top.values.tolist()]} | resid norm {x[0,1:].norm(dim=-1).mean():8.1f} mean-norm {x[0,1:].mean(0).norm():8.1f}")
    print("     attn bias norm", sd[p+'attn.c_proj.bias'].norm().item(), "mlp bias norm", sd[p+'mlp.c_proj.bias'].norm().item())
