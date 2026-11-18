import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer
from repair_lib import posterior, local_scale
sd = load_weights("damaged.safetensors"); tok=Tokenizer("tokenizer")
W=sd["wte.weight"]; m0=W.median(dim=0).values[None,:]
R=W-m0
S=local_scale(R,"wte.weight")
What,p=posterior(W,m0,key="wte.weight",line_z=False,S=S)
sn=10*W.std().item()/math.sqrt(2)
print("sn",sn)
rn=S[:,0]/S[0,0]
for j in [362,496,68,36,100,200]:
    f=(p[:,j]>.5)
    a=R[f,j].abs().numpy(); z=(R[f,j]/S[f,j]).abs().numpy()
    rows=f.nonzero().flatten()
    col=R[:,j]
    print(f"col {j}: m0={m0[0,j]:.3f} col rstd={(col.abs().median()*1.4826):.3f} std={col.std():.3f} nflag={f.sum().item()} |R| q10/50/90 of flagged {np.quantile(a,[.1,.5,.9]).round(3)} z q {np.quantile(z,[.1,.5,.9]).round(1)} sign+ {(R[f,j]>0).float().mean():.2f}")
    print("    hist |R| flagged (0..3 step .25):",np.histogram(a,bins=np.arange(0,3.01,.25))[0])
    print("    flagged row id quantiles",np.quantile(rows.numpy(),[.1,.3,.5,.7,.9]).astype(int), " S of flagged median %.3f vs col median S %.3f"%(S[f,j].median(),S[:,j].median()))
    print("    sample tokens:",[tok.decode([i]) for i in rows[torch.randperm(len(rows))[:14]].tolist()])
# row scale distribution
r=S.median(dim=1).values
print("row scale quantiles",np.quantile(r.numpy(),[0,.01,.1,.5,.9,.99,1]).round(3))
