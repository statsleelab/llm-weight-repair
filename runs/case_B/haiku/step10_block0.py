import torch, math
import torch.nn.functional as F
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, N_HEAD, N_EMBD, gelu_new
from step07_repair import repair
sd0 = load_weights("damaged.safetensors")
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt").read())
x_ids = torch.tensor([[50256]+ids[:128]])
def block0(sd):
    B,T = x_ids.shape
    x = sd["wte.weight"][x_ids] + sd["wpe.weight"][:T]
    p="h.0."
    h = F.layer_norm(x,(N_EMBD,),sd[p+"ln_1.weight"],sd[p+"ln_1.bias"],1e-5)
    qkv = h @ sd[p+"attn.c_attn.weight"] + sd[p+"attn.c_attn.bias"]
    q,k,v = qkv.split(N_EMBD,dim=-1)
    q=q.view(B,T,N_HEAD,-1).transpose(1,2); k=k.view(B,T,N_HEAD,-1).transpose(1,2); v=v.view(B,T,N_HEAD,-1).transpose(1,2)
    att=(q@k.transpose(-1,-2))/math.sqrt(q.size(-1))
    mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
    att=att.masked_fill(~mask,float('-inf')).softmax(-1)
    a=(att@v).transpose(1,2).reshape(B,T,N_EMBD)
    ao = a@sd[p+"attn.c_proj.weight"]+sd[p+"attn.c_proj.bias"]
    x2 = x+ao
    h2 = F.layer_norm(x2,(N_EMBD,),sd[p+"ln_2.weight"],sd[p+"ln_2.bias"],1e-5)
    m = gelu_new(h2@sd[p+"mlp.c_fc.weight"]+sd[p+"mlp.c_fc.bias"])
    mo = m@sd[p+"mlp.c_proj.weight"]+sd[p+"mlp.c_proj.bias"]
    nm = lambda t: round(t[0,1:].norm(dim=-1).mean().item(),2)
    print(f"  emb {nm(x):8.2f} | attn_out {nm(ao):8.2f} | mlp_out {nm(mo):8.2f} | ln1 out {nm(h):7.2f}  ln2 out {nm(h2):7.2f}  c_fc pre-act absmax {(h2@sd[p+'mlp.c_fc.weight']).abs().max().item():.1f}")
print("damaged"); block0(sd0)
for T in [4.0]:
    new,_ = repair(sd0, T)
    print("repaired T=",T); block0(new)
print("ln1 bias/weight block0 check ok")
