import torch, sys, math
torch.set_num_threads(1)
sys.path.insert(0, ".")
import torch.nn.functional as F
from gpt2_lib import load_weights, Tokenizer, gelu_new, N_EMBD, N_HEAD
from safetensors.torch import load_file
sd0 = load_file("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = torch.tensor([tok.encode(open("dev.txt", encoding="utf-8").read()[:600])[:128]])
def block0(sd, label):
    with torch.no_grad():
        B,T = ids.shape
        x = sd["wte.weight"][ids] + sd["wpe.weight"][:T]
        p="h.0."
        h = F.layer_norm(x,(N_EMBD,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
        qkv = h @ sd[p+"attn.c_attn.weight"] + sd[p+"attn.c_attn.bias"]
        q,k,v = qkv.split(N_EMBD,dim=-1)
        q=q.view(B,T,N_HEAD,-1).transpose(1,2); k=k.view(B,T,N_HEAD,-1).transpose(1,2); v=v.view(B,T,N_HEAD,-1).transpose(1,2)
        att=(q@k.transpose(-1,-2))/math.sqrt(q.size(-1))
        mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
        att=att.masked_fill(~mask,float("-inf")).softmax(-1)
        a=(att@v).transpose(1,2).reshape(B,T,N_EMBD)
        ao = a @ sd[p+"attn.c_proj.weight"] + sd[p+"attn.c_proj.bias"]
        x2 = x + ao
        h2 = F.layer_norm(x2,(N_EMBD,),sd[p+"ln_2.weight"],sd[p+"ln_2.bias"],1e-5)
        m = gelu_new(h2 @ sd[p+"mlp.c_fc.weight"] + sd[p+"mlp.c_fc.bias"])
        mo = m @ sd[p+"mlp.c_proj.weight"] + sd[p+"mlp.c_proj.bias"]
        print(f"{label:28s} emb_norm={x.norm(dim=-1).median():7.2f} attn_out_norm={ao.norm(dim=-1).median():8.2f} mlp_out_norm={mo.norm(dim=-1).median():9.2f} mlp_pre_absmax={(h2 @ sd[p+'mlp.c_fc.weight']).abs().max():8.2f}")
block0(sd0, "damaged")
# attention-only fixed / mlp-only fixed variants
for name, rep in [("attn_c_proj zero-impute", "h.0.attn.c_proj.weight"), ("mlp_c_proj zero-impute","h.0.mlp.c_proj.weight"), ("mlp_c_fc zero-impute","h.0.mlp.c_fc.weight"), ("attn_c_attn zero-impute","h.0.attn.c_attn.weight")]:
    c = torch.load(f"cache/{rep}__t4.5_r96.pt")
    sd = dict(sd0); sd[rep] = torch.where(c["mask"], torch.zeros_like(sd0[rep]), sd0[rep])
    block0(sd, name)
c = torch.load("cache/h.0.attn.c_proj.weight__t4.5_r96.pt")
print("attn c_proj flagged frac", c["mask"].float().mean().item())
Wd = sd0["h.0.attn.c_proj.weight"]
print("attn c_proj quantiles", torch.quantile(Wd.flatten()[::3], torch.tensor([.001,.01,.05,.25,.5,.75,.95,.99,.999])).tolist())
