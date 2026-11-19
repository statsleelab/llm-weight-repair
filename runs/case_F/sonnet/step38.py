from repairlib import *
g=sd["ln_f.weight"]
exec(open('step35.py').read().split("for gmin,tw")[0].split("ids=devids(2048)")[1])
ids=devids(4096)
M=[138,378,447,373]
def variant(ts):
    s2,big=b6(2)
    for k,v in sd.items():
        if v.dim()==2 and 'c_proj' in k:
            S=scale_model(v).clamp(min=1e-4)
            m=torch.zeros_like(v,dtype=torch.bool); m[:,M]=(v.abs()>ts*S)[:,M]
            m&=~legit(k,v)
            s2[k]=torch.where(m,torch.zeros_like(v),s2[k])
    return s2
best=None
for ts in [None,1.5,0.7]:
    s2=b6(2)[0] if ts is None else variant(ts)
    p=perplexity(s2,ids,window=512); print(ts,p,flush=True)
    if best is None or p<best[0]: best=(p,s2,ts)
save_weights(best[1],"repaired.safetensors"); print("saved",best[0],best[2])
