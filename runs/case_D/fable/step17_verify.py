import torch, math, json
torch.set_num_threads(1)
from gpt2_lib import *
a=load_weights("damaged.safetensors"); b=load_weights("repaired.safetensors"); bl=json.load(open("blocks_final.json"))
assert a.keys()==b.keys()
tot=0
for k in a:
    assert a[k].shape==b[k].shape and b[k].dtype==torch.float32 and torch.isfinite(b[k]).all()
    d=(a[k]!=b[k]); tot+=int(d.sum())
    if k not in bl: assert not d.any(), k
    else:
        r0,r1,c0,c1=bl[k]; d[r0:r1,c0:c1]=False; assert not d.any(), k
print("ok; changed entries",tot)
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt",encoding="utf-8").read()); print("dev tokens",len(ids))
for name,sd in (("repaired",b),):
    for off in (0,6144):
        nll,acc=nll_windows(sd,ids[off:],window=512,max_tokens=6144); print(f"{name} tokens[{off}:{off+6144}] nll {nll:.4f} ppl {math.exp(nll):.3f} acc {acc:.4f}",flush=True)
nll,acc=nll_windows(a,ids[6144:],window=512,max_tokens=6144); print(f"damaged tokens[6144:12288] nll {nll:.4f} ppl {math.exp(nll):.3f} acc {acc:.4f}")
print(generate(b,tok,"The capital of France is",12))
