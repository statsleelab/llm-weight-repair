from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:200]
x=torch.tensor([[EOT]+ids])
T=14
r=dict(sd)
for k in KEYS(sd):
    z=zscore(sd[k])[0]; r[k]=sd[k].clone(); r[k][z.abs()>T]=0
lg,h=forward(r,x,return_hidden=True)
for i,t in enumerate(h):
    print(i,"rms by pos [1,5,50,150]",[round(t[0,p].pow(2).mean().sqrt().item(),2) for p in (1,5,50,150)],"pos0 max",round(t[0,0].abs().max().item(),1), "argmax dim pos0",t[0,0].abs().argmax().item(),"pos5 argmax",t[0,5].abs().argmax().item(),round(t[0,5].abs().max().item(),1))
