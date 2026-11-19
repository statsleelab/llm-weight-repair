from repairlib import *
s2=build(4)
x=torch.tensor([devids(300)[100:300]])
lg,hs=forward(s2,x,return_hidden=True)
for i,h in enumerate(hs):
    a=h[0,5:]
    print(i,"rms %.1f"%a.pow(2).mean().sqrt().item(),"max %.0f"%a.abs().max().item(),a.abs().mean(0).topk(3).indices.tolist(),"%.0f"%a.abs().mean(0).max().item())
