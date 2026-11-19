from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:200]
x=torch.tensor([[EOT]+ids])
for name,s in [("damaged",sd)]:
    lg,h=forward(s,x,return_hidden=True)
    for i,t in enumerate(h): print(i,"absmax",t.abs().max().item(),"rms",t.pow(2).mean().sqrt().item(), "pos0 max",t[0,0].abs().max().item(),"pos5 max",t[0,5].abs().max().item())
    print("logit absmax",lg.abs().max().item())
