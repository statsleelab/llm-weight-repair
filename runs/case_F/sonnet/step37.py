from repairlib import *
import torch.nn.functional as F
g=sd["ln_f.weight"]
exec(open('step35.py').read().split("for gmin,tw")[0].split("ids=devids(2048)")[1])
s2,big=b6(2)
x=torch.tensor([[EOT]+devids(300)])
lg,hs=forward(s2,x,return_hidden=True)
for i in range(13): print(i,"pos0 %.0f pos1 %.0f pos100 %.0f rms100 %.1f"%(hs[i][0,0].abs().max(),hs[i][0,1].abs().max(),hs[i][0,100].abs().max(),hs[i][0,100].pow(2).mean().sqrt()),hs[i][0,100].abs().topk(2).indices.tolist())
nll=F.cross_entropy(lg[0,:-1],x[0,1:],reduction='none'); print("nll median",nll.median().item(),"mean",nll.mean().item(),"first",nll[:4].tolist())
