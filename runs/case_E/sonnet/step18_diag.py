from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
r=dict(sd)
for k in KEYS(sd):
    z,_,med=zscore(sd[k]); o=z.abs()>10; r[k]=torch.where(o,med.expand_as(sd[k]),sd[k])
ids=tok.encode(open("dev.txt").read())[:64]
x=torch.tensor([[EOT]+ids])
lg,h=forward(r,x,return_hidden=True)
for i,t in enumerate(h): print(i,"rms pos[1,10,40]",[round(t[0,p].pow(2).mean().sqrt().item(),1) for p in (1,10,40)],"pos0 rms",round(t[0,0].pow(2).mean().sqrt().item(),1),"top dims pos10",t[0,10].abs().topk(3).indices.tolist(),[round(v,1) for v in t[0,10].abs().topk(3).values.tolist()])
f=F.layer_norm(h[-1],(768,),r["ln_f.weight"],r["ln_f.bias"],1e-5)[0,10]
top=lg[0,10].topk(3)
print("top tokens",[tok.decode([i]) for i in top.indices.tolist()],top.values.tolist(),"median logit",lg[0,10].median().item())
print("u top dims",f.abs().topk(5))
print("true next", tok.decode([ids[10]]), lg[0,10,ids[10]].item())
