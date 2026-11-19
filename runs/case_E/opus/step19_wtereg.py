import torch; torch.set_num_threads(1)
from gpt2_lib import *
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
sd=load_weights("damaged.safetensors")
base=torch.load("cache_base16.pt")
W=sd["wte.weight"]; cm=W.median(0,keepdim=True).values; u=W-cm
cs=u.abs().median(0,keepdim=True).values/0.6745; zc=u/cs
crit=[496,430,36,314,374]
def build(T, Tc):
    m=(zc.abs()>T)
    Wn=torch.where(m,cm.expand_as(W),W).clone()
    other=[d for d in range(768) if d not in crit]
    X=torch.cat([Wn[:,other], torch.ones(W.shape[0],1)],1).double()
    for d in crit:
        y=W[:,d].double(); good=~m[:,d]
        beta=torch.linalg.lstsq(X[good],y[good,None]).solution
        pred=(X@beta)[:,0]; res=y-pred
        rs=res[good].abs().median()/0.6745
        bad=(res.abs()/rs)>Tc
        Wn[:,d]=torch.where(bad,pred,y).float()
        print("  col",d,"res scale",rs.item(),"col MAD",cs[0,d].item(),"flag",int(bad.sum()))
    return Wn
for T,Tc in [(12,8),(12,12),(12,20)]:
    o=dict(base); o["wte.weight"]=build(T,Tc)
    print(T,Tc, perplexity(o, ids, max_tokens=1024), flush=True)
