from repairlib import *
from collections import Counter
for k in ['h.11.attn.c_proj.weight','h.11.mlp.c_proj.weight','h.10.mlp.c_proj.weight','h.10.attn.c_proj.weight']:
    w=sd[k]; m=flag(k,w,3.5)
    big=m&(w.abs()>3)
    c=Counter(big.nonzero()[:,1].tolist()); 
    print(k,"zeroed>3:",big.sum().item(),"by col top",c.most_common(6))
    cm=(w.abs()>3).sum(0); print("   cols with most >3 entries",[(i.item(),cm[i].item()) for i in cm.topk(5).indices])
    for c0 in [447,496,373]:
        col=w[:,c0]; print("   col",c0,"top",[(i.item(),round(col[i].item(),1)) for i in col.abs().topk(8).indices],"mad %.2f"%((col-col.median()).abs().median()*1.48))
