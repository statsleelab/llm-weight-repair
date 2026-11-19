from repairlib import *
s2=dict(sd)
s2['wpe.weight']=fix_wpe(sd['wpe.weight'])
for k,v in sd.items():
    if v.dim()==2 and k!='wpe.weight':
        s2[k]=torch.where(protect(k,flag(v,4,0.3)),torch.zeros_like(v),v)
x=torch.tensor([ids for ids in [devids(300)[100:300]]])
lg,hs=forward(s2,x,return_hidden=True)
for i,h in enumerate(hs):
    a=h[0,5:]
    print(i,"rms %.1f"%a.pow(2).mean().sqrt().item(),"max %.0f"%a.abs().max().item(),"top dims",a.abs().mean(0).topk(3).indices.tolist(), "%.0f"%a.abs().mean(0).max().item())
