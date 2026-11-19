from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:40]
x=torch.tensor([[EOT]+ids])
fx=dict(sd)
for k in KEYS(sd):
    z,_,med=zscore(sd[k]); o=z.abs()>10; fx[k]=torch.where(o,med.expand_as(sd[k]),sd[k])
def m(r):
    # run only first 4 blocks via full forward hidden
    _,h=forward(r,x,return_hidden=True); return [round(h[i][0,0].abs().max().item()) for i in (1,2,3,4)]
print("damaged",m(sd))
print("all fixed",m(fx))
cands=[k for k in KEYS(sd) if k.startswith("h.0.") or k.startswith("h.1.") or k.startswith("h.2.") or k.startswith("h.3.")]+["wte.weight","wpe.weight"]
for k in cands:
    r=dict(fx); r[k]=sd[k]
    print("unfix",k,m(r),flush=True)
