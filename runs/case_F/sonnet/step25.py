from repairlib import *
ids=devids(2048)
for t in [3.5]:
    s2=build(t)
    print(t,perplexity(s2,ids,window=512),flush=True)
    x=torch.tensor([[EOT]+devids(30)])
    lg,hs=forward(s2,x,return_hidden=True)
    for i in [1,2,3,6,12]: print(i,"pos0 %.0f pos1 %.0f pos5 %.0f"%tuple(hs[i][0,j].abs().max().item() for j in (0,1,5)))
