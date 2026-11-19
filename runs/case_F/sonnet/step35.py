from repairlib import *
g=sd['ln_f.weight']
ids=devids(2048)
def b6(gmin,tc=3.5,tw=3.5):
    big=(g>gmin)
    s2=build(tc)
    for k,v in sd.items():
        if v.dim()==2 and 'c_proj' in k:
            w=s2[k].clone(); w[:,big]=0; s2[k]=w
    S=scale_model(sd['wte.weight']).clamp(min=1e-4)
    v=sd['wte.weight']; m=(v.abs()>tw*S)
    w=torch.where(m,torch.zeros_like(v),v); w[:,big]=0; s2['wte.weight']=w
    return s2,big
for gmin,tw in [(3,3.5),(2,3.5),(5,2.5)]:
    s2,big=b6(gmin,tw=tw)
    print(gmin,tw,big.sum().item(),perplexity(s2,ids,window=512),flush=True)
