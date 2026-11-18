import torch, numpy as np
torch.set_num_threads(1)
from gpt2_lib import load_weights
sd = load_weights("damaged.safetensors")
for k in ["h.6.mlp.c_fc.weight","h.6.attn.c_proj.weight","wte.weight","wpe.weight","h.0.attn.c_proj.weight"]:
    W=sd[k]; a=W.flatten().numpy()
    rs=np.median(np.abs(a))*1.4826
    print(k, "rstd",rs)
    edges=[0,1,2,3,4,5,6,8,10,15,20,30,40,60,100,1e9]
    h,_=np.histogram(np.abs(a)/rs,bins=edges)
    print(" hist |a|/rs:", dict(zip(edges[:-1],(h/len(a)).round(6))))
    # absolute hist of big ones
    big=a[np.abs(a)>8*rs]
    print(" big: n",len(big),"mean",big.mean(),"std",big.std(),"abs pct",np.percentile(np.abs(big),[1,10,25,50,75,90,99]).round(3), "pos frac",(big>0).mean())
    print(" abs hist", np.histogram(np.abs(big),bins=np.linspace(0,6,13))[0])
    M=(W.abs()>8*rs)
    r=M.float().sum(1); c=M.float().sum(0)
    print(" row counts: mean %.2f var %.2f max %d ; col counts mean %.2f var %.2f max %d"%(r.mean(),r.var(),r.max(),c.mean(),c.var(),c.max()))
    idx=M.flatten().nonzero().flatten().numpy()
    d=np.diff(idx); print(" gap mean",d.mean()," gap std",d.std(), "first idx", idx[:12])
    print(" sample vals", big[:16].round(3))
