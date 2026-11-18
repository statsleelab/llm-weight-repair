import torch, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, save_weights, Tokenizer, perplexity, generate
sd=load_weights("damaged.safetensors")
b=load_weights("cand_p2b.safetensors"); c=load_weights("cand_p2c.safetensors")
d={k:(c[k] if ("attn.c_proj" in k or "mlp.c_fc" in k) else b[k]) for k in b}
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
pb=perplexity(b,ids,max_tokens=8192); print("cand b (eb4+wpe smooth+wte lowrank prior) 8k:",pb,flush=True)
pd=perplexity(d,ids,max_tokens=8192); print("cand d (b + lowrank prior on attn.c_proj, mlp.c_fc) 8k:",pd,flush=True)
best=d if pd<pb else b
for k in sd:
    assert best[k].shape==sd[k].shape and torch.isfinite(best[k]).all()
    if sd[k].ndim!=2: assert torch.equal(best[k],sd[k])
save_weights(best,"repaired.safetensors")
r=load_weights("repaired.safetensors")
print("saved:", "d" if pd<pb else "b", "keys",len(r), "changed frac per-matrix avg", sum(((r[k]-sd[k]).abs()>1e-6).float().mean().item() for k in r if r[k].ndim==2)/50)
print("held-back half of dev (tokens 8192..end), repaired:",perplexity(r,ids[8192:]))
print(generate(r,tok,"The capital of France is",12))
