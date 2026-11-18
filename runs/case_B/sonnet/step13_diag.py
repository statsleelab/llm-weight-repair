import torch,numpy as np
torch.set_num_threads(1)
from gpt2_lib import *
from step12_repair import *
s2=dict(sd); s2['wpe.weight'],_=fix_wpe(sd['wpe.weight'])
for key,v in sd.items():
    if v.dim()==2 and key!='wpe.weight':
        s2[key]=torch.where(flag_mat(v,5),torch.zeros_like(v),v)
x=torch.tensor([[EOT]+ids[:200]])
lg,h=forward(s2,x,return_hidden=True)
for i,s in enumerate(h): print(i,"rms %.1f max %.1f"%(s[0,1:].pow(2).mean().sqrt(),s[0,1:].abs().max()),"pos0 max %.1f"%s[0,0].abs().max())
print("logit rms",lg.pow(2).mean().sqrt().item(),"max",lg.max().item())
print(generate(s2,tok,"The capital of France is",15))
