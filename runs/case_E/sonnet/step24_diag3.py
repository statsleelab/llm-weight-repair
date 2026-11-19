from step20_mixed import *
base,n=build(14,5)
x=torch.tensor([[EOT]+ids[:256]])
lg,h=forward(base,x,return_hidden=True)
tgt=x[0,1:]
l=lg[0,:-1]
nll=F.cross_entropy(l,tgt,reduction="none")
print("median nll",nll.median().item(),"mean",nll.mean().item())
am=l.argmax(-1)
from collections import Counter
print(Counter(am.tolist()).most_common(6))
tok=Tokenizer("tokenizer")
t=Counter(am.tolist()).most_common(1)[0][0]
print("top pred token",repr(tok.decode([t])))
f=F.layer_norm(h[-1],(768,),base["ln_f.weight"],base["ln_f.bias"],1e-5)[0,10]
c=f*base["wte.weight"][t]
print("contrib top dims",c.abs().topk(6), "sum",c.sum().item())
print("wte row at those",base["wte.weight"][t][c.abs().topk(6).indices], "col med",base["wte.weight"].median(0).values[c.abs().topk(6).indices])
print("mean logit of target",l[torch.arange(255),tgt].mean().item(),"mean max",l.max(-1).values.mean().item())
