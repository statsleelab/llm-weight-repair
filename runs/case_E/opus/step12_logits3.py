import torch; torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib3 import repair
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())[:256]
sd=load_weights("damaged.safetensors")
r,_=repair(sd,T=6)
lg,h=forward(r,torch.tensor([[EOT]+ids]),return_hidden=True)
x=h[-1][0]; import torch.nn.functional as F
xf=F.layer_norm(x,(768,),r["ln_f.weight"],r["ln_f.bias"],1e-5)
print("ln_f out mean |.| top dims", [(d, round(xf[:,d].abs().mean().item(),1)) for d in xf.abs().mean(0).topk(8).indices.tolist()])
top=lg[0].argmax(-1); c=torch.bincount(top,minlength=VOCAB); 
for t in c.topk(5).indices.tolist():
    w=r["wte.weight"][t]; contrib=(xf.mean(0)*w)
    print(t, repr(tok.decode([t])), int(c[t]), "top contrib dims", [(d,round(w[d].item(),3),round(contrib[d].item(),1)) for d in contrib.abs().topk(4).indices.tolist()])
print("wte col std of top dims", [(d, round(r['wte.weight'][:,d].std().item(),3)) for d in xf.abs().mean(0).topk(8).indices.tolist()])
# per-position nll
tgt=torch.tensor(ids[:]); lp=lg[0,:-1].log_softmax(-1); nl=-lp.gather(-1,tgt[:,None])[:,0]
print("nll quantiles", nl.quantile(torch.tensor([.1,.5,.9,.99])).tolist())
print("logit range at pos 50:", lg[0,50].min().item(), lg[0,50].max().item(), "median", lg[0,50].median().item())
