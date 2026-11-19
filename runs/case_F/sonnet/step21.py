from repairlib import *
s2=build(4)
x=torch.tensor([[EOT]+devids(30)])
lg,hs=forward(s2,x,return_hidden=True)
for i,h in enumerate(hs):
    a=h[0,0]; b=h[0,1]
    print(i,"pos0 max %.0f"%a.abs().max().item(),a.abs().topk(3).indices.tolist(),[round(a[j].item()) for j in a.abs().topk(3).indices],"| pos1 max %.0f"%b.abs().max().item(),b.abs().topk(2).indices.tolist())
