from repairlib import *
s2=build(4,0.3)
x=torch.tensor([devids(300)[100:300]])
lg,hs=forward(s2,x,return_hidden=True)
for i,h in enumerate(hs):
    a=h[0,5:]
    print(i,"rms %.1f"%a.pow(2).mean().sqrt().item(),"max %.0f"%a.abs().max().item(),a.abs().mean(0).topk(3).indices.tolist())
print("ppl",perplexity(s2,devids(4096),window=512))
