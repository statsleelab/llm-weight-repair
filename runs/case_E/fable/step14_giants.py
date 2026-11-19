import torch
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
sd = load_weights("damaged.safetensors")
tot=0
for k,v in sd.items():
    if v.ndim!=2 or k=="wpe.weight": continue
    m,cm = detect_v4(v, 4.2, protect=1e9)
    D=(v-cm); nsd=(10*v.std()/1.0488).item()
    big = (D.abs()>3.0*nsd)
    ij = (big).nonzero().tolist()
    exp = 0.001*v.numel()*0.0027
    if len(ij)>max(3,3*exp):
        desc=[(i,j,round(v[i,j].item(),2),'F' if m[i,j] else 'k') for i,j in ij]
        print(k,"nsd=%.2f"%nsd,"n>3nsd",len(ij),"expected noise %.1f"%exp, desc[:40])
