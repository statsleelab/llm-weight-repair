from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:64]
x=torch.tensor([[EOT]+ids])
lg,h=forward(sd,x,return_hidden=True)
f=F.layer_norm(h[-1],(768,),sd["ln_f.weight"],sd["ln_f.bias"],1e-5)[0]
u=f[10]
w=sd["wte.weight"]
s=18118
c=u*w[s]
print("socially logit",c.sum().item(),"top dims",c.abs().topk(6))
print("u top",u.abs().topk(6))
print("ln_f.weight top",sd["ln_f.weight"].abs().topk(6))
xr=h[-1][0,10]; print("resid top dims",xr.abs().topk(6))
print("w[socially] at those",w[s][u.abs().topk(6).indices])
print("typical |u|",u.abs().median().item())
