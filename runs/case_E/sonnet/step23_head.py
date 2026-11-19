from step20_mixed import *
base,n=build(14,5)
torch.save({},"/dev/null") if False else None
x=torch.tensor([[EOT]+ids[:512]])
lg,h=forward(base,x,return_hidden=True)
u=F.layer_norm(h[-1],(768,),base["ln_f.weight"],base["ln_f.bias"],1e-5)[0]
s=u.pow(2).mean(0).sqrt()
print("s top",s.topk(6),"median",s.median().item())
torch.save(s,"s_head.pt")
w=sd["wte.weight"]
z,_,med=zscore(w)
dev=w-med
c=dev*s[None,:]
rowmad=c.abs().median(1,keepdim=True).values*1.4826
zc=c.abs()/rowmad.clamp_min(1e-6)
for T2 in [3,5,8,12,20]:
    o=(z.abs()>14)|(zc>T2)
    r=dict(base); r["wte.weight"]=torch.where(o,med.expand_as(w),w)
    print("T2",T2,"n",o.sum().item(),nll_windows(r,ids,window=256),flush=True)
