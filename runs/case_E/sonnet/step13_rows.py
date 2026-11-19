from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
w=sd["wte.weight"]
n=w.norm(dim=1)
print("norm quantiles",n.quantile(torch.tensor([0,.01,.5,.99,1.])).tolist())
top=n.topk(10)
print([(i,round(v,2),tok.decode([i])) for v,i in zip(top.values.tolist(),top.indices.tolist())])
i=tok.encode(" socially")[0]; print("socially",i,n[i].item())
ids=tok.encode(open("dev.txt").read())[:256]
x=torch.tensor([[EOT]+ids])
lg=forward(sd,x)
print("pred argmax set",lg.argmax(-1)[0][:20].tolist())
print("logit max",lg.max().item(),"logit median",lg.median().item())
print("nll per pos",F.cross_entropy(lg[0,:-1],x[0,1:],reduction="none")[:10] if False else "")
